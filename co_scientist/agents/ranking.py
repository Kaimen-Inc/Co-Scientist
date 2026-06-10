"""Ranking agent — manages the Elo tournament.

Two actions:
- `AddToTournament(hypothesis_id)` — initialize Elo + state. No LLM call.
- `RunTournamentBatch(focus_id?)` — pick a pair, debate, parse verdict, apply Elo.

Pair selection mixes new-arrival pairings, similar-Elo pairs (weighted toward
embedding-distant ones for information gain), and an occasional random pull.
Debate mode is preferred when matches are new or Elo gap is small; pairwise
otherwise.
"""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from .. import ids
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..logging import get_logger
from ..models import Hypothesis, Task, TaskResult, TournamentMatch
from ..orchestrator.elo import update_elo
from ..safety.quoting import quote_hypothesis
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
    tournaments as tourney_repo,
)
from ..vectors.embedder import make_embedder
from ..vectors.store import FaissStore
from .base import BaseAgent

log = get_logger("ranking")


PairMode = Literal["pairwise", "debate"]


class RankingAgent(BaseAgent):
    name = "ranking"

    async def execute(self, task: Task) -> TaskResult:
        if task.action == "AddToTournament":
            return await self._add_to_tournament(task)
        if task.action == "RunTournamentBatch":
            return await self._run_tournament_batch(task)
        raise ValueError(f"RankingAgent does not handle action {task.action!r}")

    # ----------------------------- AddToTournament ----------------------------- #

    async def _add_to_tournament(self, task: Task) -> TaskResult:
        hypothesis_id = task.target_id
        if not hypothesis_id:
            raise ValueError("AddToTournament requires target_id")
        changed = await hyp_repo.init_tournament(
            self.deps.db, hypothesis_id,
            initial_elo=float(self.deps.cfg.ranking.elo_initial),
        )
        return TaskResult(
            kind="added_to_tournament",
            hypothesis_ids=[hypothesis_id] if changed else [],
            extra={"already_in_tournament": not changed},
        )

    # ----------------------------- RunTournamentBatch -------------------------- #

    async def _run_tournament_batch(self, task: Task) -> TaskResult:
        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        candidates = await hyp_repo.list_for_session(
            self.deps.db, session.id, state="in_tournament"
        )
        if len(candidates) < 2:
            return TaskResult(kind="noop", extra={"reason": "fewer than 2 candidates"})

        focus_id = task.payload.get("focus")
        pair = await self._select_pair(session.id, candidates, focus_id=focus_id)
        if pair is None:
            return TaskResult(kind="noop", extra={"reason": "no pair available"})
        hyp_a, hyp_b, similarity = pair

        mode = self._select_mode(hyp_a, hyp_b)
        verdict, rationale, transcript_id = await self._run_debate(
            session, hyp_a, hyp_b, mode=mode
        )
        # Derive the round_id deterministically from the task id so that a
        # crash-then-retry computes the *same* match_id. `apply_elo_update`
        # below is idempotent on `match_id` — a non-deterministic round_id
        # (e.g. a wall-clock timestamp) defeats that and would double-apply
        # the Elo delta on retry.
        round_id = task.id
        if verdict is None:
            # Parsing failed — record an invalid match and don't update Elo.
            mid_invalid = ids.match_id(hyp_a.id, hyp_b.id, round_id)
            await tourney_repo.insert_match(self.deps.db, TournamentMatch(
                id=mid_invalid, session_id=session.id,
                created_at=datetime.now(UTC),
                hyp_a=hyp_a.id, hyp_b=hyp_b.id, mode="invalid", winner=None,
                elo_a_before=hyp_a.elo or 1200.0, elo_b_before=hyp_b.elo or 1200.0,
                rationale=rationale, transcript_id=transcript_id, similarity=similarity,
            ))
            log.warning("ranking_invalid_verdict", a=hyp_a.id, b=hyp_b.id)
            return TaskResult(kind="noop", extra={"reason": "unparseable verdict"})

        # Compute the Elo update.
        elo_a_before = float(hyp_a.elo or self.deps.cfg.ranking.elo_initial)
        elo_b_before = float(hyp_b.elo or self.deps.cfg.ranking.elo_initial)
        min_matches = min(hyp_a.matches_played, hyp_b.matches_played)
        upd = update_elo(
            elo_a_before, elo_b_before, verdict,
            matches_played_min=min_matches,
            k_new=self.deps.cfg.ranking.k_factor_new,
            k_warm=self.deps.cfg.ranking.k_factor_warm,
        )

        mid = ids.match_id(hyp_a.id, hyp_b.id, round_id)
        await tourney_repo.insert_match(self.deps.db, TournamentMatch(
            id=mid, session_id=session.id,
            created_at=datetime.now(UTC),
            hyp_a=hyp_a.id, hyp_b=hyp_b.id, mode=mode, winner=verdict,
            elo_a_before=elo_a_before, elo_b_before=elo_b_before,
            elo_a_after=upd.elo_a_after, elo_b_after=upd.elo_b_after,
            rationale=rationale, transcript_id=transcript_id, similarity=similarity,
        ))
        applied = await tourney_repo.apply_elo_update(
            self.deps.db,
            match_id=mid, hyp_a=hyp_a.id, hyp_b=hyp_b.id, winner=verdict,
            elo_a_before=elo_a_before, elo_b_before=elo_b_before,
            elo_a_after=upd.elo_a_after, elo_b_after=upd.elo_b_after,
        )
        log.info(
            "match_complete",
            mode=mode, hyp_a=hyp_a.id, hyp_b=hyp_b.id, winner=verdict,
            elo_a=upd.elo_a_after, elo_b=upd.elo_b_after,
            applied=applied, similarity=similarity,
        )
        return TaskResult(
            kind="tournament_match_complete",
            match_ids=[mid],
            hypothesis_ids=[hyp_a.id, hyp_b.id],
            extra={"mode": mode, "winner": verdict, "elo_applied": applied},
        )

    # ----------------------------- pair selection ----------------------------- #

    async def _select_pair(
        self,
        session_id: str,
        candidates: list[Hypothesis],
        *,
        focus_id: str | None,
    ) -> tuple[Hypothesis, Hypothesis, float | None] | None:
        # Build the FAISS store once for this pair selection — every prior
        # iteration re-instantiated the embedder, re-read index.faiss + JSON
        # off disk, and reconstructed the entire index just to dot-product two
        # rows. With ~20 pair candidates per RunTournamentBatch that was
        # ~20 full-index reloads and reconstructions for a single match.
        store = await self._load_store(session_id)

        if focus_id:
            focus = next((h for h in candidates if h.id == focus_id), None)
            if focus is not None:
                opp = self._nearest_elo(focus, [h for h in candidates if h.id != focus_id])
                if opp is not None:
                    sim = self._similarity(store, focus, opp)
                    return focus, opp, sim

        new_hyps = [h for h in candidates if h.matches_played < 3]
        warm = [h for h in candidates if h.matches_played >= 3]

        cfg = self.deps.cfg.ranking
        r = random.random()
        # Bucket 1: pair a new hypothesis with nearest-Elo warm/stable.
        if r < cfg.p_new and new_hyps and warm:
            a = random.choice(new_hyps)
            b = self._nearest_elo(a, warm)
            if b is not None:
                return a, b, self._similarity(store, a, b)

        # Bucket 2: similar-Elo pair within the warm set, weighted by (1 - cosine_similarity)
        # so we prefer pairs that are *distant in idea-space* — debate over differing approaches.
        if r < cfg.p_new + cfg.p_close and len(warm) >= 2:
            pair = self._sample_close_elo(store, warm)
            if pair is not None:
                return pair

        # Bucket 3: random Elo-weighted (top-heavy)
        if len(candidates) >= 2:
            sorted_by_elo = sorted(candidates, key=lambda h: -(h.elo or 1200))
            top = sorted_by_elo[: max(2, len(candidates) // 2)]
            if len(top) >= 2:
                a, b = random.sample(top, 2)
                return a, b, self._similarity(store, a, b)
        return None

    def _nearest_elo(
        self, target: Hypothesis, pool: list[Hypothesis]
    ) -> Hypothesis | None:
        if not pool:
            return None
        return min(pool, key=lambda h: abs((h.elo or 1200) - (target.elo or 1200)))

    def _sample_close_elo(
        self, store: FaissStore | None, pool: list[Hypothesis]
    ) -> tuple[Hypothesis, Hypothesis, float | None] | None:
        """Among pairs with |Δelo|<200, sample weighted by exp(-Δelo/200)*(1-sim)."""
        if len(pool) < 2:
            return None
        # Build a small candidate list of pairs (cap to keep cost low)
        weights: list[float] = []
        pairs: list[tuple[Hypothesis, Hypothesis, float | None]] = []
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                d_elo = abs((a.elo or 1200) - (b.elo or 1200))
                if d_elo > 200:
                    continue
                sim = self._similarity(store, a, b)
                w_sim = 1.0 - (sim if sim is not None else 0.0)
                w = float(np.exp(-d_elo / 200.0)) * max(w_sim, 0.05)
                weights.append(w)
                pairs.append((a, b, sim))
                if len(pairs) >= 20:    # cap
                    break
            if len(pairs) >= 20:
                break
        if not pairs:
            return None
        total = sum(weights)
        if total <= 0:
            return random.choice(pairs)
        r = random.uniform(0, total)
        cum = 0.0
        for w, pair in zip(weights, pairs, strict=True):
            cum += w
            if cum >= r:
                return pair
        return pairs[-1]

    async def _load_store(self, session_id: str) -> FaissStore | None:
        """Instantiate + load the session FAISS store once for pair selection."""
        try:
            embedder = make_embedder(self.deps.cfg)
        except (RuntimeError, ValueError):
            return None
        store = FaissStore(self.deps.cfg, session_id, dim=embedder.dim)
        await store.load_or_create()
        if store.n == 0:
            return None
        return store

    def _similarity(
        self, store: FaissStore | None, a: Hypothesis, b: Hypothesis
    ) -> float | None:
        """Cosine via the session's FAISS store (already L2-normalized).

        Reconstructs only the two rows we need (O(2·dim)) — the previous
        version called `reconstruct_n(0, n)` for every pair, materialising
        the full N×dim matrix just to read two rows.
        """
        if store is None or store.index is None or store.n == 0:
            return None
        i = store.offset_of(a.id)
        j = store.offset_of(b.id)
        if i is None or j is None:
            return None
        vec_i = store.index.reconstruct(int(i))
        vec_j = store.index.reconstruct(int(j))
        return float(vec_i @ vec_j)

    # ----------------------------- mode selection ----------------------------- #

    def _select_mode(self, a: Hypothesis, b: Hypothesis) -> PairMode:
        cfg = self.deps.cfg.ranking
        if min(a.matches_played, b.matches_played) < cfg.debate_when_matches_lt:
            return "debate"
        if abs((a.elo or 1200) - (b.elo or 1200)) < cfg.debate_when_elo_delta_lt:
            return "debate"
        return "pairwise"

    # ----------------------------- the debate / pairwise call ----------------- #

    async def _run_debate(
        self,
        session,
        a: Hypothesis,
        b: Hypothesis,
        *,
        mode: PairMode,
    ) -> tuple[Literal["a", "b"] | None, str, str | None]:
        plan = session.research_plan
        # Anchor on the lower-ID hypothesis so cache hits cluster on it.
        anchor, opponent = (a, b) if a.id <= b.id else (b, a)
        anchor_is_a = anchor is a
        review_anchor = await self._best_review(anchor.id)
        review_opp = await self._best_review(opponent.id)

        template = "ranking.debate" if mode == "debate" else "ranking.pairwise"
        prompt_kwargs = {
            "goal": plan.objective,
            "preferences": "; ".join(plan.preferences),
            "idea_attributes": "; ".join(plan.idea_attributes),
            "hypothesis_1_id": anchor.id,
            "hypothesis_1": quote_hypothesis(anchor.full_text, id_=anchor.id),
            "hypothesis_2_id": opponent.id,
            "hypothesis_2": quote_hypothesis(opponent.full_text, id_=opponent.id),
            "review_1": review_anchor or "(no review)",
            "review_2": review_opp or "(no review)",
            "notes": "Be decisive. End your response with the line: better idea: <1 or 2>",
        }
        prompt = render(template, **prompt_kwargs)

        r = route(self.deps.cfg, "ranking", "debate" if mode == "debate" else "pairwise")

        system = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                f"# Research goal\n{session.research_goal}\n\n"
                f"# Preferences\n{'; '.join(plan.preferences)}\n\n"
                "Conclude every response with the exact line `better idea: 1` or "
                "`better idea: 2`. No other format. Do not call any tools.",
                cache=True,
            ),
        ]
        spec = AgentCallSpec(
            route=r,
            system_blocks=system,
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[],
            tool_choice=None,
            max_output_tokens=2048,
            stop_sequences=None,
        )
        ctx = CallContext(
            session_id=session.id, task_id=None,
            agent="ranking", action="RunTournamentBatch", mode=mode,
        )
        resp = await self.deps.llm.call(spec, ctx)
        text = self._final_text(resp)
        choice = _parse_better_idea(text)
        if choice is None:
            return None, text, resp.transcript_id

        # Map anchor/opponent choice back to (a, b)
        # "1" means anchor, "2" means opponent.
        if anchor_is_a:
            winner: Literal["a", "b"] = "a" if choice == 1 else "b"
        else:
            winner = "b" if choice == 1 else "a"
        return winner, text, resp.transcript_id

    async def _best_review(self, hypothesis_id: str) -> str | None:
        rs = await rev_repo.list_for_hypothesis(self.deps.db, hypothesis_id)
        if not rs:
            return None
        # Prefer 'full' kind if present.
        rs_sorted = sorted(rs, key=lambda r: (r.kind != "full", -(r.scores.novelty or 0)))
        return rs_sorted[0].body


_VERDICT_DIGIT_RE = re.compile(r"^[\W_]*\**\s*([12])\b")


def _parse_better_idea(text: str) -> int | None:
    """Find the trailing 'better idea: 1|2' marker (case-insensitive, any line).

    The previous implementation used `"1" in tail.split()[0:1]`, which is
    `True` only when the first whitespace-token *equals* "1" exactly. That
    rejected valid replies like 'better idea: option 1' or 'better idea: **1
    because...'. The regex anchors at the start and matches the first 1 or 2
    as a word boundary so we accept all those forms while still rejecting
    'better idea: 12' (which the boundary check excludes).
    """
    if not text:
        return None
    lines = text.strip().splitlines()
    for line in reversed(lines):
        low = line.strip().lower()
        if "better idea" in low and ":" in low:
            tail = low.split(":", 1)[1].strip()
            m = _VERDICT_DIGIT_RE.match(tail)
            if m:
                return int(m.group(1))
            # Common phrasing: "option 1", "hypothesis 1", "hyp 1"
            for keyword in ("option", "hypothesis", "hyp"):
                if tail.startswith(keyword):
                    rest = tail[len(keyword):].lstrip()
                    m2 = _VERDICT_DIGIT_RE.match(rest)
                    if m2:
                        return int(m2.group(1))
    return None
