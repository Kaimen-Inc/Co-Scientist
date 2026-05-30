"""Evolution agent — combines, simplifies, and reimagines top hypotheses.

Four strategies:
- `combine`     — merge two distant top hypotheses into a stronger one.
- `simplify`    — strip a hypothesis to its load-bearing claim.
- `feasibility` — make it implementable with current tech.
- `out_of_box`  — out-of-box synthesis inspired by top-K.

Each produces a *new* hypothesis row with `parent_ids` populated, which then
cascades into Reflection → Ranking like any fresh idea.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from .. import ids
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..llm.tool_loop import ToolLoopExhausted, run_tool_loop
from ..logging import get_logger
from ..models import CitedPaper, Hypothesis, Task, TaskResult
from ..safety.quoting import quote_hypothesis
from ..storage.artifacts import write_json
from ..storage.repos import embeddings as emb_repo
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..vectors.embedder import make_embedder
from ..vectors.store import FaissStore
from .base import BaseAgent
from .schemas import RECORD_HYPOTHESIS_TOOL

log = get_logger("evolution")

EvoStrategy = Literal["combine", "simplify", "feasibility", "out_of_box"]


class EvolutionAgent(BaseAgent):
    name = "evolution"

    async def execute(self, task: Task) -> TaskResult:
        strategies: list[EvoStrategy] = task.payload.get("strategies") or [
            "combine", "simplify", "out_of_box"
        ]
        top_k = int(task.payload.get("top_k", 5))

        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        top = await hyp_repo.top_by_elo(self.deps.db, session.id, k=top_k)
        if len(top) < 2:
            return TaskResult(kind="noop", extra={"reason": "need at least 2 top hypotheses"})

        new_ids: list[str] = []
        for strat in strategies:
            try:
                hid = await self._evolve_one(session, top, strategy=strat)
            except Exception as e:
                log.warning("evolution_strategy_failed", strategy=strat, err=str(e))
                continue
            if hid:
                new_ids.append(hid)

        return TaskResult(
            kind="hypothesis_created",
            hypothesis_ids=new_ids,
            extra={"strategies_used": strategies},
        )

    # ----------------------------- one strategy ----------------------------- #

    async def _evolve_one(
        self, session, top: list[Hypothesis], *, strategy: EvoStrategy
    ) -> str | None:
        if strategy == "combine":
            return await self._combine(session, top)
        if strategy == "out_of_box":
            return await self._out_of_box(session, top)
        return await self._unary(session, top, strategy=strategy)

    async def _combine(self, session, top: list[Hypothesis]) -> str | None:
        # Pick the most idea-distant pair within the top set.
        pair = await self._most_distant_pair(session.id, top)
        if pair is None:
            return None
        a, b = pair
        review_a = await self._best_review(a.id)
        review_b = await self._best_review(b.id)
        prompt = render(
            "evolution.combine",
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            hypothesis_a_id=a.id, hypothesis_a=quote_hypothesis(a.full_text, id_=a.id),
            hypothesis_b_id=b.id, hypothesis_b=quote_hypothesis(b.full_text, id_=b.id),
            review_a=review_a, review_b=review_b,
        )
        return await self._run_and_persist(
            session, prompt, strategy="combine",
            mode_for_route="combine", parent_ids=[a.id, b.id],
        )

    async def _out_of_box(self, session, top: list[Hypothesis]) -> str | None:
        inspirations = top[:5]
        prompt = render(
            "evolution.out_of_box",
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            hypotheses=[
                {"id": h.id, "text": quote_hypothesis(h.full_text, id_=h.id)}
                for h in inspirations
            ],
        )
        return await self._run_and_persist(
            session, prompt, strategy="out_of_box",
            mode_for_route="out_of_box",
            parent_ids=[h.id for h in inspirations],
        )

    async def _unary(
        self, session, top: list[Hypothesis], *, strategy: EvoStrategy
    ) -> str | None:
        h = top[0]
        review = await self._best_review(h.id)
        template = f"evolution.{strategy}"
        prompt = render(
            template,
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            hypothesis_id=h.id, hypothesis=quote_hypothesis(h.full_text, id_=h.id),
            review=review,
        )
        return await self._run_and_persist(
            session, prompt, strategy=strategy,
            mode_for_route=strategy,
            parent_ids=[h.id],
        )

    # ----------------------------- run + persist ----------------------------- #

    async def _run_and_persist(
        self,
        session,
        prompt: str,
        *,
        strategy: EvoStrategy,
        mode_for_route: str,
        parent_ids: list[str],
    ) -> str | None:
        sys_blocks = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                _build_session_context(session.research_goal, session.research_plan,
                                       await self._latest_feedback(session.id)),
                cache=True,
            ),
        ]
        user_blocks = [CachedBlock(prompt, cache=False)]

        r = route(self.deps.cfg, "evolution", mode_for_route)
        tools = [*self.deps.tools.anthropic_tools_for("evolution"), RECORD_HYPOTHESIS_TOOL]
        spec = AgentCallSpec(
            route=r,
            system_blocks=sys_blocks,
            user_blocks=user_blocks,
            tools=tools,
            tool_choice={"type": "auto"},
            max_output_tokens=4096,
        )
        ctx = CallContext(
            session_id=session.id, task_id=None,
            agent="evolution", action="EvolveTopHypotheses", mode=mode_for_route,
        )
        try:
            result = await run_tool_loop(
                self.deps.llm, spec=spec, ctx=ctx,
                registry=self.deps.tools,
                max_iters=self.deps.cfg.tool_loop.evolution_max_iters,
                parallel_cap=self.deps.cfg.tool_loop.parallel_cap,
                tool_timeout_s=self.deps.cfg.tool_loop.tool_timeout_seconds,
                force_terminal_tool="record_hypothesis",
            )
        except ToolLoopExhausted as e:
            log.warning("evolution_tool_loop_exhausted", err=str(e))
            return None

        record = self._final_tool_use(result.response, "record_hypothesis")
        if record is None:
            log.warning("evolution_no_record")
            return None

        # Citation URL filter (same as Generation).
        record["citations"] = [
            c for c in record.get("citations", [])
            if isinstance(c, dict) and c.get("url") in result.seen_urls
        ]
        record["strategy"] = strategy
        record["parent_ids"] = parent_ids

        hid, was_new = await self._persist(session.id, record, strategy=strategy)
        return hid if was_new else None

    async def _persist(
        self, session_id: str, record: dict[str, Any], *, strategy: str
    ) -> tuple[str, bool]:
        statement = record.get("statement") or record.get("title") or ""
        if not statement:
            raise ValueError("evolution: record_hypothesis is missing statement")
        origin = f"evolution/{strategy}"
        hid = ids.hypothesis_id(session_id, origin, statement)
        summary = (record.get("statement") or "") + "\n\n" + (record.get("mechanism") or "")
        full_text = _render_hypothesis_md(record)

        artifact_path = await write_json(
            self.deps.cfg, session_id, "hypotheses", hid,
            {"strategy": strategy, "record": record},
        )
        citations = [
            CitedPaper(
                title=c.get("title", ""), url=c.get("url", ""),
                excerpt=c.get("excerpt"), doi=c.get("doi"), year=c.get("year"),
            )
            for c in record.get("citations", [])
            if isinstance(c, dict) and c.get("url")
        ]

        # Dedup: cheap nearest-neighbour query. Same pattern as Generation.
        try:
            dup_id, embed_payload = await self._dedup_query(session_id, summary)
        except Exception as e:
            log.warning("evolution_dedup_query_failed", err=str(e))
            dup_id, embed_payload = None, None

        if dup_id is not None and dup_id != hid:
            return dup_id, False

        h = Hypothesis(
            id=hid, session_id=session_id, created_at=datetime.now(UTC),
            created_by="evolution", strategy=strategy,        # type: ignore[arg-type]
            parent_ids=record.get("parent_ids") or [],
            title=record.get("title", "")[:300],
            summary=(record.get("statement") or "")[:1000],
            full_text=full_text,
            citations=citations,
            artifact_path=artifact_path,
            state="draft",
        )
        inserted = await hyp_repo.insert(self.deps.db, h)

        if inserted and embed_payload is not None:
            try:
                await self._dedup_commit(session_id, hid, embed_payload)
            except Exception as e:
                log.warning("evolution_dedup_commit_failed", hypothesis_id=hid, err=str(e))

        return hid, inserted

    # ----------------------------- helpers ----------------------------- #

    async def _dedup_query(
        self, session_id: str, text: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        try:
            embedder = make_embedder(self.deps.cfg)
        except (RuntimeError, ValueError):
            return None, None
        vec = await embedder.embed([text])
        if vec.size == 0:
            return None, None
        v = vec[0]
        store = FaissStore(self.deps.cfg, session_id, dim=embedder.dim)
        await store.load_or_create()
        nearest = await store.search(np.asarray(v), k=1)
        thr = self.deps.cfg.vectors.dedup_cosine_threshold
        if nearest and nearest[0][1] >= thr:
            return nearest[0][0], None
        return None, {
            "vector": np.asarray(v),
            "model": embedder.model,
            "dim": embedder.dim,
            "text_hash": ids.text_hash(text),
        }

    async def _dedup_commit(
        self, session_id: str, hypothesis_id: str, payload: dict[str, Any]
    ) -> None:
        store = FaissStore(self.deps.cfg, session_id, dim=payload["dim"])
        await store.load_or_create()
        offset = await store.add(hypothesis_id, payload["vector"])
        await store.save()
        await emb_repo.upsert(
            self.deps.db,
            id_=ids.embedding_id(hypothesis_id, payload["model"]),
            session_id=session_id, hypothesis_id=hypothesis_id,
            model=payload["model"], dim=payload["dim"],
            faiss_offset=offset, text_hash=payload["text_hash"],
        )

    async def _most_distant_pair(
        self, session_id: str, top: list[Hypothesis]
    ) -> tuple[Hypothesis, Hypothesis] | None:
        if len(top) < 2:
            return None
        try:
            embedder = make_embedder(self.deps.cfg)
        except (RuntimeError, ValueError):
            return top[0], top[1]
        store = FaissStore(self.deps.cfg, session_id, dim=embedder.dim)
        await store.load_or_create()
        if store.n == 0:
            return top[0], top[1]
        best: tuple[Hypothesis, Hypothesis] | None = None
        best_sim = 2.0
        vecs = store.index.reconstruct_n(0, store.n)
        for i, a in enumerate(top):
            ia = store.offset_of(a.id)
            if ia is None:
                continue
            for b in top[i + 1:]:
                ib = store.offset_of(b.id)
                if ib is None:
                    continue
                sim = float(vecs[ia] @ vecs[ib])
                if sim < best_sim:
                    best_sim = sim
                    best = (a, b)
        return best or (top[0], top[1])

    async def _best_review(self, hypothesis_id: str) -> str | None:
        rs = await rev_repo.list_for_hypothesis(self.deps.db, hypothesis_id)
        if not rs:
            return None
        rs_sorted = sorted(rs, key=lambda r: (r.kind != "full", -(r.scores.novelty or 0)))
        return rs_sorted[0].body

    async def _latest_feedback(self, session_id: str) -> str | None:
        fb = await fb_repo.latest_system_feedback(self.deps.db, session_id)
        return fb.text if fb is not None else None


# ----------------------------- formatting helpers ----------------------------- #


def _render_hypothesis_md(record: dict[str, Any]) -> str:
    parts: list[str] = []
    if record.get("title"):
        parts.append(f"# {record['title']}")
    parts.append(f"**Hypothesis.** {record.get('statement', '')}")
    if record.get("mechanism"):
        parts.append(f"## Mechanism\n{record['mechanism']}")
    if record.get("entities"):
        parts.append("## Entities\n- " + "\n- ".join(record["entities"]))
    if record.get("anticipated_outcomes"):
        parts.append(f"## Anticipated outcomes\n{record['anticipated_outcomes']}")
    if record.get("novelty_argument"):
        parts.append(f"## Novelty\n{record['novelty_argument']}")
    if record.get("citations"):
        parts.append("## Citations")
        for c in record["citations"]:
            year = f" ({c.get('year')})" if c.get("year") else ""
            parts.append(f"- {c.get('title','(no title)')}{year} — {c.get('url','')}")
    if record.get("parent_ids"):
        parts.append(f"## Parents\n{', '.join(record['parent_ids'])}")
    return "\n\n".join(parts)


def _build_session_context(goal: str, plan, sys_feedback_text: str | None) -> str:
    from ..safety.quoting import quote_untrusted

    fb = ""
    if sys_feedback_text:
        fb = "\n\n# Researcher / Meta-review Feedback\n" + quote_untrusted(
            sys_feedback_text, id_="system_feedback:latest"
        )
    return (
        f"# Research goal\n{goal}\n\n"
        f"# Parsed plan\n"
        f"- Objective: {plan.objective}\n"
        f"- Preferences: {'; '.join(plan.preferences) or '(none)'}\n"
        f"- Idea attributes: {'; '.join(plan.idea_attributes) or '(none)'}\n"
        f"- Constraints: {'; '.join(plan.constraints) or '(none)'}\n"
        f"{fb}"
    )
