"""Minimal Supervisor for M3.

This is the first vertical slice:
1. Parse the scientist's goal into a ResearchPlan.
2. Insert the session.
3. Enqueue N parallel Generation tasks (literature strategy).
4. Drain the queue with a bounded asyncio worker pool. Each completed
   Generation enqueues one Reflection follow-up.
5. When the queue empties (no Ranking yet — that lands in M4), synthesize a
   tiny final overview from whatever hypotheses + reviews exist and exit.

Later milestones will replace the linear chain with the full event-driven
scheduler described in the plan.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from .. import ids
from ..config import Config
from ..llm.anthropic_client import (
    AgentCallSpec,
    AnthropicClient,
    CachedBlock,
    CallContext,
)
from ..llm.budgets import TokenBudget
from ..llm.prompts import render
from ..llm.routing import route
from ..logging import bind, get_logger
from ..models import ResearchPlan, Session, Task
from ..storage import db as db_mod
from ..storage.artifacts import write_text
from ..storage.repos import (
    events as events_repo,
)
from ..storage.repos import (
    feedback as fb_repo,
)
from ..storage.repos import (
    hypotheses as hyp_repo,
)
from ..storage.repos import (
    reviews as rev_repo,
)
from ..storage.repos import (
    sessions as sess_repo,
)
from ..storage.repos import (
    tasks as task_repo,
)
from ..tools.registry import ToolRegistry
from .base import AgentDeps
from .generation import GenerationAgent
from .ranking import RankingAgent
from .reflection import ReflectionAgent
from .schemas import RECORD_RESEARCH_PLAN_TOOL

log = get_logger("supervisor")


class Supervisor:
    """One-process Supervisor; CLI invokes via `await supervisor.run_session(...)`."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def run_session(
        self,
        goal: str,
        *,
        preferences_text: str | None = None,
        n_initial: int = 3,
        wall_clock_seconds: int | None = None,
    ) -> str:
        wall = wall_clock_seconds or self.cfg.run.wall_clock_seconds
        conn = await db_mod.connect(self.cfg)
        try:
            session = await self._create_session(conn, goal, preferences_text)
            bind(session_id=session.id)
            log.info(
                "session_started",
                goal=goal[:120], session_id=session.id,
                budget_usd=session.budget_usd, n_initial=n_initial,
            )
            await events_repo.emit(
                conn, session_id=session.id, task_id=None, agent="supervisor",
                event="session_started",
                payload={"goal": goal[:200], "n_initial": n_initial, "budget_usd": session.budget_usd},
            )

            budget = TokenBudget(
                cfg=self.cfg,
                budget_tokens=session.budget_tokens,
                budget_usd=session.budget_usd,
            )
            llm = AnthropicClient(self.cfg, db=conn, budget=budget)
            tools = ToolRegistry(self.cfg).discover()
            deps = AgentDeps(cfg=self.cfg, db=conn, llm=llm, tools=tools)

            # 1. Parse goal → ResearchPlan, persist back on the session row.
            plan = await self._parse_goal(deps, session, goal, preferences_text)
            await self._apply_plan(conn, session, plan)
            session = await sess_repo.fetch(conn, session.id)  # refresh
            assert session is not None

            # 2. Enqueue N parallel Generation tasks.
            for _i in range(n_initial):
                await task_repo.enqueue(conn, Task(
                    id=ids.task_id(), session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="generation", action="CreateInitialHypotheses",
                    payload={"strategy": "literature", "n": 1},
                    priority=100, status="pending",
                    idempotency_key=f"{session.id}::generation::initial::{_i}",
                ))

            # 3. Drain queue with bounded concurrency. M3 keeps it linear:
            # each Generation result triggers one Reflection on the new hypothesis.
            try:
                await asyncio.wait_for(
                    self._drain_queue(conn, deps, session),
                    timeout=wall,
                )
            except TimeoutError:
                await sess_repo.set_status(conn, session.id, "aborted")
                log.warning("session_timed_out", wall_clock_seconds=wall)

            # 4. Synthesize a tiny final overview.
            overview_path = await self._write_simple_overview(conn, session)
            await sess_repo.set_final_overview(conn, session.id, overview_path)
            await events_repo.emit(
                conn, session_id=session.id, task_id=None, agent="supervisor",
                event="session_done", payload={"overview_path": overview_path},
            )
            log.info("session_done", overview_path=overview_path)
            return session.id
        finally:
            await conn.close()

    # -------------------------- internals -------------------------- #

    async def _create_session(
        self, conn, goal: str, preferences_text: str | None
    ) -> Session:
        sid = ids.session_id()
        now = datetime.now(UTC)
        # Empty plan placeholder until parse_goal lands.
        plan = ResearchPlan(objective=goal.strip(), preferences=[], idea_attributes=[])
        snap: dict[str, Any] = json.loads(json.dumps(self.cfg.model_dump(exclude={"secrets"})))
        s = Session(
            id=sid, created_at=now, updated_at=now, status="running",
            research_goal=goal, research_plan=plan,
            config_snapshot=snap,
            budget_tokens=self.cfg.run.budget_tokens, budget_usd=self.cfg.run.budget_usd,
        )
        await sess_repo.insert(conn, s)
        if preferences_text:
            await fb_repo.insert(conn, _human_preference(s.id, preferences_text))
        return s

    async def _parse_goal(
        self,
        deps: AgentDeps,
        session: Session,
        goal: str,
        preferences_text: str | None,
    ) -> ResearchPlan:
        prompt = render(
            "parse_goal", goal=goal,
            preferences_text=preferences_text or "",
        )
        r = route(self.cfg, "parse_goal", None)
        spec = AgentCallSpec(
            route=r,
            system_blocks=[CachedBlock("You parse research goals into structured plans.", cache=True)],
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[RECORD_RESEARCH_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "record_research_plan"},
            max_output_tokens=1024,
        )
        ctx = CallContext(
            session_id=session.id, task_id=None,
            agent="parse_goal", action="parse_goal", mode=None,
        )
        resp = await deps.llm.call(spec, ctx)
        record: dict[str, Any] | None = None
        for b in resp.raw.content:
            if getattr(b, "type", None) == "tool_use" and getattr(b, "name", "") == "record_research_plan":
                inp = getattr(b, "input", None)
                if isinstance(inp, dict):
                    record = inp
                    break
        if record is None:
            log.warning("parse_goal_no_record", note="falling back to bare ResearchPlan")
            return ResearchPlan(objective=goal.strip(), preferences=[], idea_attributes=[])
        return ResearchPlan(
            objective=record.get("objective", goal.strip()),
            preferences=record.get("preferences", []),
            constraints=record.get("constraints", []),
            idea_attributes=record.get("idea_attributes", []),
            domain_hint=record.get("domain_hint") or None,
            notes=record.get("notes") or None,
        )

    async def _apply_plan(self, conn, session: Session, plan: ResearchPlan) -> None:
        await conn.execute(
            "UPDATE sessions SET research_plan=?, updated_at=? WHERE id=?",
            (plan.model_dump_json(), datetime.now(UTC).isoformat(), session.id),
        )
        await conn.commit()

    async def _drain_queue(self, conn, deps: AgentDeps, session: Session) -> None:
        agents = {
            "generation": GenerationAgent(deps),
            "reflection": ReflectionAgent(deps),
            "ranking": RankingAgent(deps),
        }
        sem = asyncio.Semaphore(self.cfg.run.concurrency)
        inflight: set[asyncio.Task] = set()
        worker_seq = 0

        async def _run_task(t: Task) -> None:
            # asyncio contextvars are per-task — bindings here only affect this coro.
            bind(session_id=session.id, task_id=t.id, agent=t.agent)
            async with sem:
                await task_repo.mark_in_progress(conn, t.id)
                log.info("task_started", action=t.action, target=t.target_id)
                agent = agents.get(t.agent)
                if agent is None:
                    await task_repo.fail(conn, t.id, error=f"no agent: {t.agent}")
                    return
                try:
                    result = await agent.execute(t)
                except Exception as e:
                    await task_repo.fail(conn, t.id, error=str(e))
                    log.exception("task_failed", err=str(e))
                    return

                # Follow-up scheduling
                if result.kind == "hypothesis_created":
                    for hid in result.hypothesis_ids:
                        await task_repo.enqueue(conn, Task(
                            id=ids.task_id(), session_id=session.id,
                            created_at=datetime.now(UTC),
                            agent="reflection", action="ReviewHypothesis",
                            target_id=hid, payload={"kind": "full"},
                            priority=100, status="pending",
                            idempotency_key=f"{hid}::review::full",
                        ))
                elif result.kind == "review_completed":
                    for hid in result.hypothesis_ids:
                        await task_repo.enqueue(conn, Task(
                            id=ids.task_id(), session_id=session.id,
                            created_at=datetime.now(UTC),
                            agent="ranking", action="AddToTournament",
                            target_id=hid, payload={}, priority=80, status="pending",
                            idempotency_key=f"{hid}::ranking::add",
                        ))
                elif result.kind == "added_to_tournament":
                    for hid in result.hypothesis_ids:
                        await task_repo.enqueue(conn, Task(
                            id=ids.task_id(), session_id=session.id,
                            created_at=datetime.now(UTC),
                            agent="ranking", action="RunTournamentBatch",
                            target_id=None,
                            payload={"focus": hid}, priority=120, status="pending",
                            idempotency_key=f"{hid}::ranking::focus_batch",
                        ))

                await task_repo.complete(conn, t.id)
                log.info("task_completed", kind=result.kind, follow=len(result.hypothesis_ids))

        # Pump: claim up to (concurrency - len(inflight)) new tasks each iteration.
        while True:
            slots_open = self.cfg.run.concurrency - len(inflight)
            claimed: list[Task] = []
            for _ in range(slots_open):
                t = await task_repo.claim_one(
                    conn, session.id, worker_id=f"w{worker_seq}",
                    lease_seconds=self.cfg.lease.default_seconds,
                )
                if t is None:
                    break
                worker_seq += 1
                claimed.append(t)
            for t in claimed:
                inflight.add(asyncio.create_task(_run_task(t)))

            if not inflight:
                return    # queue empty + no workers running → done

            # Block until at least one completes; then loop and re-fill slots.
            _done, pending = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            inflight = set(pending)

    async def _write_simple_overview(self, conn, session: Session) -> str:
        hyps = await hyp_repo.list_for_session(conn, session.id)
        parts: list[str] = [
            f"# Research overview — session {session.id}",
            f"\n**Goal.** {session.research_goal}\n",
            f"**Status.** {session.status}",
            f"**Hypotheses produced.** {len(hyps)}",
            "",
        ]
        if not hyps:
            parts.append("_No hypotheses were produced. Check the logs and your API key._")
        else:
            for i, h in enumerate(hyps, 1):
                parts.append(f"## {i}. {h.title or h.id}")
                parts.append(f"`{h.id}` — strategy `{h.strategy}` — state `{h.state}`")
                parts.append(h.summary or "(no summary)")
                reviews = await rev_repo.list_for_hypothesis(conn, h.id)
                if reviews:
                    parts.append("\n**Reviews:**")
                    for r in reviews:
                        parts.append(
                            f"- *{r.kind}* — verdict `{r.verdict or '?'}` "
                            f"(n={r.scores.novelty}, c={r.scores.correctness}, "
                            f"t={r.scores.testability})"
                        )
                parts.append("")

        parts.append(
            "\n> Note: this M3 overview is a stub. The full tournament-driven research "
            "overview lands once Ranking (M4) and Meta-review (M6) are wired in."
        )
        body = "\n".join(parts)
        return await write_text(self.cfg, session.id, "final", "overview", ".md", body)


def _human_preference(session_id: str, text: str):
    from ..models import SystemFeedback

    return SystemFeedback(
        id=ids.feedback_id(), session_id=session_id,
        created_at=datetime.now(UTC),
        source="human", kind="preference",
        target_id=None, text=text, active=True,
    )
