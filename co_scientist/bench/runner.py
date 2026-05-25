"""Bench runner — see co_scientist/bench/__init__.py."""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from .. import ids
from ..agents.base import AgentDeps
from ..agents.generation import GenerationAgent
from ..agents.ranking import _parse_better_idea
from ..config import Config
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.budgets import TokenBudget
from ..llm.prompts import render
from ..llm.provider import get_provider
from ..llm.routing import ModelRoute
from ..logging import bind, get_logger
from ..models import Hypothesis, ResearchPlan, Session, Task
from ..orchestrator.elo import update_elo
from ..safety.quoting import quote_hypothesis
from ..storage import db as db_mod
from ..storage.artifacts import write_json
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import tasks as task_repo
from ..tools.registry import ToolRegistry

log = get_logger("bench")


# Structured verdict tool — far more reliable than asking the model to
# emit a `better idea: <N>` line. Every modern provider supports function
# calling, and a forced single-tool call cuts response tokens drastically.
RECORD_VERDICT_TOOL: dict = {
    "name": "record_verdict",
    "description": "Record the winner of a head-to-head hypothesis comparison.",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {
                "type": "integer", "enum": [1, 2],
                "description": "1 if hypothesis_1 is stronger, 2 if hypothesis_2 is stronger.",
            },
            "rationale": {
                "type": "string",
                "description": "One paragraph: why the winner is stronger.",
            },
        },
        "required": ["winner", "rationale"],
    },
}


# --------------------------------------------------------------------------- #
# Public types

@dataclass
class BenchCandidate:
    """One model to evaluate in the bench."""

    label: str
    provider: str          # anthropic | openai | openrouter | gemini | ...
    model: str             # provider-specific model id


@dataclass
class _CandidateState:
    """Internal: per-candidate working state during a bench run."""

    candidate_id: str
    spec: BenchCandidate
    hypotheses: list[Hypothesis] = field(default_factory=list)
    elos: dict[str, float] = field(default_factory=dict)   # hyp_id -> Elo
    matches_played: dict[str, int] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    cost_usd: float = 0.0
    input_tok: int = 0
    output_tok: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchOutcome:
    """Result returned from `run_bench`."""

    bench_id: str
    candidates: list[dict[str, Any]]
    matches_played: int
    total_cost_usd: float
    artifact_path: str


# --------------------------------------------------------------------------- #
# Entry point

async def run_bench(
    base_cfg: Config,
    *,
    goal: str,
    candidates: list[BenchCandidate],
    n_hyps_per_candidate: int = 2,
    matches_per_pair: int = 2,
    judge_provider: str = "anthropic",
    judge_model: str = "",
    per_candidate_budget_usd: float = 5.0,
    judge_budget_usd: float = 5.0,
    preferences_text: str | None = None,
) -> BenchOutcome:
    """Execute a bench. See module docstring for semantics."""
    if not candidates:
        raise ValueError("bench needs at least one candidate")
    if len(candidates) < 2 and matches_per_pair > 0:
        raise ValueError("Elo tournament needs at least two candidates")
    judge_model = judge_model or base_cfg.models.judge

    bench_id_ = ids.bench_id()
    bind(bench_id=bench_id_)
    log.info(
        "bench_started",
        goal=goal, n_candidates=len(candidates),
        n_hyps_per_candidate=n_hyps_per_candidate,
        judge=f"{judge_provider}:{judge_model}",
    )

    conn = await db_mod.connect(base_cfg)
    try:
        await _insert_bench_row(
            conn, bench_id_, goal=goal,
            judge_provider=judge_provider, judge_model=judge_model,
            base_cfg=base_cfg,
        )

        # 1. Spin up a private "bench session" in the sessions table so the
        #    Generation agent's existing dependencies (which expect a real
        #    Session row + ResearchPlan) keep working unchanged.
        plan = ResearchPlan(
            objective=goal,
            preferences=([preferences_text] if preferences_text else []),
        )
        ses = Session(
            id=ids.session_id(),
            created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
            status="running", research_goal=goal, research_plan=plan,
            config_snapshot={"bench_id": bench_id_},
            budget_tokens=base_cfg.run.budget_tokens,
            budget_usd=per_candidate_budget_usd * max(1, len(candidates)),
        )
        await sess_repo.insert(conn, ses)

        # 2. Generate hypotheses for each candidate in parallel.
        states = await _generate_for_all_candidates(
            base_cfg, conn, bench_id_, ses, candidates,
            n_hyps_per_candidate=n_hyps_per_candidate,
            per_candidate_budget_usd=per_candidate_budget_usd,
        )

        # 3. Cross-tournament — pair every candidate-pair `matches_per_pair`
        #    times, randomly drawing one hypothesis from each side. Judged
        #    by a single fixed model.
        n_matches = 0
        if matches_per_pair > 0:
            judge_cfg = _candidate_cfg(base_cfg, judge_provider, judge_model)
            judge_budget = TokenBudget(
                cfg=judge_cfg,
                budget_tokens=base_cfg.run.budget_tokens,
                budget_usd=judge_budget_usd,
            )
            judge_llm = get_provider(judge_cfg, db=conn, budget=judge_budget)

            n_matches = await _run_cross_tournament(
                conn, bench_id_, ses, states,
                judge_llm=judge_llm, judge_cfg=judge_cfg,
                matches_per_pair=matches_per_pair,
            )

        # 4. Aggregate stats per candidate + write to bench_candidates.
        for st in states:
            await _persist_candidate_stats(conn, st)

        total_cost = sum(s.cost_usd for s in states)

        # 5. Write a JSON artifact + flip status.
        summary = _build_summary(bench_id_, goal, states, judge_provider, judge_model, n_matches)
        artifact_path = await write_json(
            base_cfg, ses.id, "bench", bench_id_, summary
        )
        await conn.execute(
            "UPDATE bench_runs SET status='done', artifact_path=?, updated_at=? WHERE id=?",
            (artifact_path, datetime.now(UTC).isoformat(), bench_id_),
        )
        await conn.commit()

        log.info(
            "bench_done",
            bench_id=bench_id_, n_matches=n_matches,
            total_cost_usd=round(total_cost, 4),
        )

        return BenchOutcome(
            bench_id=bench_id_,
            candidates=summary["candidates"],
            matches_played=n_matches,
            total_cost_usd=total_cost,
            artifact_path=artifact_path,
        )
    except Exception as e:
        await conn.execute(
            "UPDATE bench_runs SET status='failed', updated_at=? WHERE id=?",
            (datetime.now(UTC).isoformat(), bench_id_),
        )
        await conn.commit()
        log.exception("bench_failed", err=str(e))
        raise
    finally:
        await conn.close()


# --------------------------------------------------------------------------- #
# Generation phase

async def _generate_for_all_candidates(
    base_cfg: Config,
    conn: aiosqlite.Connection,
    bench_id_: str,
    ses: Session,
    candidates: list[BenchCandidate],
    *,
    n_hyps_per_candidate: int,
    per_candidate_budget_usd: float,
) -> list[_CandidateState]:
    """Run Generation N times per candidate, in parallel across candidates."""
    states: list[_CandidateState] = []
    for c in candidates:
        cand_id = ids.bench_candidate_id()
        await conn.execute(
            """INSERT INTO bench_candidates(id, bench_id, label, provider, model)
               VALUES (?, ?, ?, ?, ?)""",
            (cand_id, bench_id_, c.label, c.provider, c.model),
        )
        states.append(_CandidateState(candidate_id=cand_id, spec=c))
    await conn.commit()

    async def _one_candidate(st: _CandidateState) -> None:
        try:
            await _generate_for_candidate(
                base_cfg, conn, ses, st, n_hyps_per_candidate, per_candidate_budget_usd
            )
        except Exception as e:
            st.error = str(e)
            log.exception("candidate_generation_failed",
                          candidate=st.spec.label, err=str(e))

    await asyncio.gather(*(_one_candidate(st) for st in states))
    return states


async def _generate_for_candidate(
    base_cfg: Config,
    conn: aiosqlite.Connection,
    ses: Session,
    st: _CandidateState,
    n_hyps: int,
    budget_usd: float,
) -> None:
    """Run GenerationAgent.execute() N times under the candidate's config."""
    cfg = _candidate_cfg(base_cfg, st.spec.provider, st.spec.model)
    cfg.models.generation = st.spec.model

    budget = TokenBudget(
        cfg=cfg, budget_tokens=cfg.run.budget_tokens, budget_usd=budget_usd,
    )
    llm = get_provider(cfg, db=conn, budget=budget)
    tools = ToolRegistry(cfg).discover()
    deps = AgentDeps(cfg=cfg, db=conn, llm=llm, tools=tools)
    agent = GenerationAgent(deps)

    initial_cost = 0.0
    for i in range(n_hyps):
        task = Task(
            id=ids.task_id(), session_id=ses.id,
            created_at=datetime.now(UTC),
            agent="generation", action="CreateInitialHypotheses",
            payload={"strategy": "literature", "n": 1},
            priority=100, status="pending",
            idempotency_key=f"bench::{st.candidate_id}::gen::{i}",
        )
        # The Anthropic/OpenAI client persists a transcript row whose
        # task_id FKs into the tasks table; enqueue the task so the FK is
        # satisfied. We own this "worker" so we can flip status manually.
        await task_repo.enqueue(conn, task)
        await task_repo.mark_in_progress(conn, task.id)
        t0 = time.monotonic()
        try:
            result = await agent.execute(task)
            await task_repo.complete(conn, task.id)
        except Exception as e:
            await task_repo.fail(
                conn, task.id, error=str(e),
                max_attempts=cfg.lease.max_attempts,
            )
            log.warning("bench_generation_failed",
                        candidate=st.spec.label, idx=i, err=str(e))
            continue
        latency = int((time.monotonic() - t0) * 1000)
        st.latencies_ms.append(latency)

        for hid in result.hypothesis_ids:
            h = await hyp_repo.fetch(conn, hid)
            if h is None:
                continue
            st.hypotheses.append(h)
            st.elos[hid] = float(base_cfg.ranking.elo_initial)
            st.matches_played[hid] = 0

    # Budget accounting: pull the post-run snapshot.
    snap = budget.snapshot().get("_global", {})
    st.cost_usd = float(snap.get("used_usd", 0.0)) - initial_cost
    st.input_tok = int(snap.get("used_tokens", 0))


def _candidate_cfg(base_cfg: Config, provider: str, model: str) -> Config:
    """Deep-copy base_cfg + apply per-candidate provider + model.

    Anthropic-only knobs (thinking budgets, batch) get zeroed when the
    candidate isn't Anthropic, so the OpenAI translator doesn't try to
    map something that won't help.
    """
    cfg = base_cfg.model_copy(deep=True)
    cfg.llm.provider = provider
    # Point every agent role at the candidate's model so generation +
    # any downstream call inside Generation uses the same one.
    for attr in ("generation", "reflection", "evolution",
                 "ranking_pairwise", "ranking_debate", "ranking_priority",
                 "metareview_feedback", "metareview_final",
                 "parse_goal", "classifier"):
        setattr(cfg.models, attr, model)
    if provider != "anthropic":
        # Thinking / cache features only work on Anthropic.
        for attr in cfg.thinking.__class__.model_fields:
            setattr(cfg.thinking, attr, 0)
    return cfg


# --------------------------------------------------------------------------- #
# Tournament phase

async def _run_cross_tournament(
    conn: aiosqlite.Connection,
    bench_id_: str,
    ses: Session,
    states: list[_CandidateState],
    *,
    judge_llm,
    judge_cfg: Config,
    matches_per_pair: int,
) -> int:
    """Pair every candidate-pair `matches_per_pair` times. Per match, pick
    one random hypothesis from each side and judge head-to-head."""
    pairs = list(itertools.combinations(states, 2))
    n_matches = 0
    for a_st, b_st in pairs:
        if not a_st.hypotheses or not b_st.hypotheses:
            continue
        for _ in range(matches_per_pair):
            a_hyp = random.choice(a_st.hypotheses)
            b_hyp = random.choice(b_st.hypotheses)
            try:
                winner, rationale, jcost, jms = await _judge_match(
                    judge_llm, judge_cfg, ses, a_hyp, b_hyp
                )
            except Exception as e:
                log.warning("bench_match_failed",
                            a=a_hyp.id, b=b_hyp.id, err=str(e))
                continue
            n_matches += 1
            if winner is None:
                # invalid verdict — record but don't update Elo
                await _insert_match(
                    conn, bench_id_,
                    a_st, b_st, a_hyp, b_hyp,
                    winner=None,
                    elo_a_before=a_st.elos[a_hyp.id], elo_b_before=b_st.elos[b_hyp.id],
                    elo_a_after=None, elo_b_after=None,
                    rationale=rationale, judge_cost_usd=jcost, judge_latency_ms=jms,
                )
                continue

            elo_a_before = a_st.elos[a_hyp.id]
            elo_b_before = b_st.elos[b_hyp.id]
            min_mp = min(a_st.matches_played[a_hyp.id], b_st.matches_played[b_hyp.id])
            upd = update_elo(elo_a_before, elo_b_before, winner, matches_played_min=min_mp)
            a_st.elos[a_hyp.id] = upd.elo_a_after
            b_st.elos[b_hyp.id] = upd.elo_b_after
            a_st.matches_played[a_hyp.id] += 1
            b_st.matches_played[b_hyp.id] += 1
            if winner == "a":
                a_st.wins += 1
                b_st.losses += 1
            else:
                a_st.losses += 1
                b_st.wins += 1
            await _insert_match(
                conn, bench_id_,
                a_st, b_st, a_hyp, b_hyp,
                winner=winner,
                elo_a_before=elo_a_before, elo_b_before=elo_b_before,
                elo_a_after=upd.elo_a_after, elo_b_after=upd.elo_b_after,
                rationale=rationale, judge_cost_usd=jcost, judge_latency_ms=jms,
            )
    return n_matches


async def _judge_match(
    judge_llm,
    judge_cfg: Config,
    ses: Session,
    a: Hypothesis,
    b: Hypothesis,
) -> tuple[str | None, str, float, int]:
    """One head-to-head judgement. Returns (winner|None, rationale, cost, latency_ms)."""
    plan = ses.research_plan
    # Anchor on lower id so cache hits cluster (no effect across providers
    # but keeps the test surface deterministic).
    anchor, opponent = (a, b) if a.id <= b.id else (b, a)
    anchor_is_a = anchor is a
    prompt = render(
        "ranking.pairwise",
        goal=plan.objective,
        preferences="; ".join(plan.preferences),
        idea_attributes="; ".join(plan.idea_attributes),
        hypothesis_1_id=anchor.id,
        hypothesis_1=quote_hypothesis(anchor.full_text, id_=anchor.id),
        hypothesis_2_id=opponent.id,
        hypothesis_2=quote_hypothesis(opponent.full_text, id_=opponent.id),
        review_1="(no review)", review_2="(no review)",
        notes="Call record_verdict exactly once with your choice. Do not output free-text reasoning before the tool call.",
    )
    system = [
        CachedBlock(
            "You are a calibrated scientific reviewer. Pick the stronger "
            "hypothesis by mechanism, specificity, and testability. You must "
            "call the `record_verdict` tool exactly once. Do not respond with "
            "any other text.",
            cache=True,
        ),
    ]
    spec = AgentCallSpec(
        route=ModelRoute(
            agent="ranking", mode="pairwise",
            model=judge_cfg.models.ranking_pairwise or judge_cfg.models.judge,
        ),
        system_blocks=system,
        user_blocks=[CachedBlock(prompt, cache=False)],
        tools=[RECORD_VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "record_verdict"},
        max_output_tokens=1024,
    )
    # Account against the "ranking" budget slot rather than a synthetic
    # "bench" agent — the judge work IS ranking work and "bench" has 0%
    # share in cfg.budget_shares, so naming it would only get the reserve
    # buffer.
    ctx = CallContext(
        session_id=ses.id, task_id=None,
        agent="ranking", action="judge_match", mode="pairwise",
    )
    t0 = time.monotonic()
    resp = await judge_llm.call(spec, ctx)
    latency = int((time.monotonic() - t0) * 1000)

    # Look for the record_verdict tool call. Fall back to text parsing if
    # the provider didn't honor tool_choice (some smaller models won't).
    verdict_input: dict | None = None
    for b in getattr(resp.raw, "content", None) or []:
        if (
            getattr(b, "type", None) == "tool_use"
            and getattr(b, "name", "") == "record_verdict"
        ):
            inp = getattr(b, "input", None)
            if isinstance(inp, dict):
                verdict_input = inp
                break

    if verdict_input is not None:
        try:
            choice = int(verdict_input.get("winner", 0))
        except (TypeError, ValueError):
            choice = 0
        rationale = str(verdict_input.get("rationale", ""))
    else:
        # Fallback: hunt for `better idea: 1|2` in the assistant text.
        rationale = _extract_text(resp.raw)
        choice = _parse_better_idea(rationale) or 0

    if choice not in (1, 2):
        return None, rationale, resp.cost_usd, latency
    # Map anchor/opponent choice back to (a, b).
    winner = ("a" if choice == 1 else "b") if anchor_is_a else ("b" if choice == 1 else "a")
    return winner, rationale, resp.cost_usd, latency


def _extract_text(raw) -> str:
    parts = []
    for b in getattr(raw, "content", None) or []:
        if getattr(b, "type", None) == "text":
            parts.append(getattr(b, "text", ""))
    return "\n".join(parts).strip()


async def _insert_match(
    conn: aiosqlite.Connection,
    bench_id_: str,
    a_st: _CandidateState, b_st: _CandidateState,
    a_hyp: Hypothesis, b_hyp: Hypothesis,
    *,
    winner: str | None,
    elo_a_before: float, elo_b_before: float,
    elo_a_after: float | None, elo_b_after: float | None,
    rationale: str, judge_cost_usd: float, judge_latency_ms: int,
) -> None:
    await conn.execute(
        """INSERT INTO bench_matches(
               id, bench_id, created_at, cand_a, cand_b,
               hyp_a_text, hyp_b_text, winner,
               elo_a_before, elo_b_before, elo_a_after, elo_b_after,
               rationale, judge_cost_usd, judge_latency_ms
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ids.bench_match_id(), bench_id_, datetime.now(UTC).isoformat(),
            a_st.candidate_id, b_st.candidate_id,
            (a_hyp.summary or "")[:4000], (b_hyp.summary or "")[:4000],
            winner,
            elo_a_before, elo_b_before, elo_a_after, elo_b_after,
            (rationale or "")[:4000], judge_cost_usd, judge_latency_ms,
        ),
    )
    await conn.commit()


# --------------------------------------------------------------------------- #
# Persistence helpers

async def _insert_bench_row(
    conn: aiosqlite.Connection,
    bench_id_: str,
    *,
    goal: str,
    judge_provider: str,
    judge_model: str,
    base_cfg: Config,
) -> None:
    now = datetime.now(UTC).isoformat()
    snapshot = {
        "provider_judge": judge_provider,
        "model_judge": judge_model,
        "run": base_cfg.run.model_dump(),
    }
    await conn.execute(
        """INSERT INTO bench_runs(id, created_at, updated_at, status,
               research_goal, judge_provider, judge_model, config_snapshot)
           VALUES (?, ?, ?, 'running', ?, ?, ?, ?)""",
        (bench_id_, now, now, goal, judge_provider, judge_model,
         json.dumps(snapshot, default=str)),
    )
    await conn.commit()


async def _persist_candidate_stats(conn: aiosqlite.Connection, st: _CandidateState) -> None:
    elos = list(st.elos.values())
    mean_elo = sum(elos) / len(elos) if elos else None
    top_elo = max(elos) if elos else None
    mean_latency = (sum(st.latencies_ms) // len(st.latencies_ms)) if st.latencies_ms else None
    await conn.execute(
        """UPDATE bench_candidates SET
               n_hypotheses=?, n_matches=?, wins=?, losses=?,
               mean_elo=?, top_elo=?,
               total_cost_usd=?, total_input_tok=?, total_output_tok=?,
               mean_latency_ms=?, error=?
            WHERE id=?""",
        (
            len(st.hypotheses), st.wins + st.losses, st.wins, st.losses,
            mean_elo, top_elo,
            st.cost_usd, st.input_tok, st.output_tok,
            mean_latency, st.error,
            st.candidate_id,
        ),
    )
    await conn.commit()


def _build_summary(
    bench_id_: str,
    goal: str,
    states: list[_CandidateState],
    judge_provider: str,
    judge_model: str,
    n_matches: int,
) -> dict[str, Any]:
    rows = []
    for st in states:
        elos = list(st.elos.values())
        rows.append({
            "candidate_id": st.candidate_id,
            "label": st.spec.label,
            "provider": st.spec.provider,
            "model": st.spec.model,
            "n_hypotheses": len(st.hypotheses),
            "wins": st.wins,
            "losses": st.losses,
            "mean_elo": sum(elos) / len(elos) if elos else None,
            "top_elo": max(elos) if elos else None,
            "cost_usd": round(st.cost_usd, 4),
            "mean_latency_ms": (sum(st.latencies_ms) // len(st.latencies_ms))
                if st.latencies_ms else None,
            "error": st.error,
        })
    # Sort: highest mean_elo first
    rows.sort(key=lambda r: (r["mean_elo"] is None, -(r["mean_elo"] or 0.0)))
    return {
        "bench_id": bench_id_,
        "goal": goal,
        "judge": f"{judge_provider}:{judge_model}",
        "n_matches": n_matches,
        "candidates": rows,
    }
