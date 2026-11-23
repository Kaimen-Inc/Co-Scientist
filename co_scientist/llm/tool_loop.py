"""The assistant↔tool_use↔tool_result loop, over two kinds of backend.

The agents' contract is unchanged in either mode. They hand over an
`AgentCallSpec`, a `ToolRegistry`, and a `max_iters` cap, and get back a
`ToolLoopResult` carrying the final assistant message, which tools ran, and
every URL those tools surfaced (the allowlist the citation verifier checks
against).

What differs is who drives the turns:

**Driven mode** (`anthropic`, `openai`, and every OpenAI-compatible provider) —
this module drives. Call the API, dispatch the `tool_use` blocks against the
registry, feed `tool_result` back, repeat until the model stops or the cap
trips.

**Captured mode** (`claude_cli`, `codex_cli`, `agy_cli`) — the CLI *is* an agent
harness and runs the cycle internally, so there is nothing here to drive. Tools
run inside the MCP server, which records them; the backend hands that log back
as `LLMResponse.capture` and this module reads it. The analogue of "force the
recording tool on the last iteration" is a single escalated retry, because
headless CLIs have no `tool_choice` flag.

A backend declares which mode it wants with `runs_own_loop`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..ids import tool_run_id
from ..logging import get_logger
from ..tools.base import ToolCtx
from ..tools.registry import ToolRegistry
from ..tools.urls import extract_urls
from .types import AgentCallSpec, CachedBlock, CallContext, LLMResponse

log = get_logger("llm.tool_loop")

# Structured-output capture tools. These are virtual: the assistant's answer is
# already in `tool_use.input`, so there is nothing to dispatch. In driven mode
# seeing one ends the loop; in captured mode a response is not complete until
# one has been recorded.
DEFAULT_TERMINAL_TOOLS: tuple[str, ...] = (
    "record_hypothesis",
    "record_review",
    "record_system_feedback",
    "record_rubric_score",
    "record_research_plan",
    "record_safety_assessment",
)


class ToolLoopExhausted(RuntimeError):
    def __init__(self, agent: str, iters: int):
        super().__init__(f"tool loop for agent {agent!r} exhausted after {iters} iterations")
        self.agent = agent
        self.iters = iters


@dataclass
class ToolLoopResult:
    response: LLMResponse                        # final assistant message
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    """Union of URLs that appeared in any tool_result over the loop.

    Used by structured-output validation to reject hallucinated citations:
    Generation's record_hypothesis.citations[].url must be in this set.
    """


async def run_tool_loop(
    client: Any,
    *,
    spec: AgentCallSpec,
    ctx: CallContext,
    registry: ToolRegistry,
    max_iters: int,
    parallel_cap: int = 4,
    tool_timeout_s: float = 30.0,
    force_terminal_tool: str | None = None,
    terminal_tool_names: tuple[str, ...] = DEFAULT_TERMINAL_TOOLS,
) -> ToolLoopResult:
    """Run one agent call to completion, driving turns if the backend doesn't.

    See the module docstring for the two modes. Backends that run their own
    agentic loop set `runs_own_loop = True`; everything else is driven here.
    """
    # `is True` rather than a truth test on purpose: a Mock — or any duck-typed
    # client — answers every getattr with a truthy object, which would silently
    # route a driven backend into captured mode and return an empty result.
    if getattr(client, "runs_own_loop", False) is True:
        return await _run_captured(
            client,
            spec=spec,
            ctx=ctx,
            max_iters=max_iters,
            force_terminal_tool=force_terminal_tool,
            terminal_tool_names=terminal_tool_names,
        )
    return await _run_driven(
        client,
        spec=spec,
        ctx=ctx,
        registry=registry,
        max_iters=max_iters,
        parallel_cap=parallel_cap,
        tool_timeout_s=tool_timeout_s,
        force_terminal_tool=force_terminal_tool,
        terminal_tool_names=terminal_tool_names,
    )


# --------------------------------------------------------------------------- #
# Driven mode — this module owns the turn cycle


async def _run_driven(
    client: Any,
    *,
    spec: AgentCallSpec,
    ctx: CallContext,
    registry: ToolRegistry,
    max_iters: int,
    parallel_cap: int,
    tool_timeout_s: float,
    force_terminal_tool: str | None,
    terminal_tool_names: tuple[str, ...],
) -> ToolLoopResult:
    """Drive the assistant ↔ tool_use ↔ tool_result loop.

    Loop termination:
    - stop_reason != "tool_use" — the model signalled end_turn.
    - The assistant response contains a `terminal_tool_names` call. The
      assistant has already produced its final answer in tool_use.input, so
      dispatching the tool is unnecessary and we should not invite the model to
      call it again. Claude reliably ends its turn after calling these;
      Gemini / OpenAI-compat models do not, so we short-circuit explicitly.
      Without this short-circuit the loop will repeatedly re-invite the
      recording tool until max_iters and then raise ToolLoopExhausted — even
      though a perfectly good record was emitted on the first call.
    - max_iters reached — raise ToolLoopExhausted.

    `force_terminal_tool`: if set, the *final* allowed iteration forces
    `tool_choice` to that tool so the model must emit a record instead of
    spending its last turn on yet another search. This prevents the
    "looped until exhausted, produced nothing" failure mode where a model
    keeps verifying novelty and never commits.
    """
    seen_urls: set[str] = set()
    tool_calls_log: list[dict[str, Any]] = []
    iterations = 0
    current_spec = spec
    terminal_set = set(terminal_tool_names)

    last: LLMResponse | None = None

    while iterations < max_iters:
        iterations += 1
        # On the final allowed iteration, optionally force the recording tool so
        # the model commits instead of burning its last turn on another search.
        call_spec = current_spec
        if force_terminal_tool and iterations == max_iters:
            call_spec = _respec(
                current_spec, tool_choice={"type": "tool", "name": force_terminal_tool}
            )
        resp = await client.call(call_spec, ctx)
        last = resp
        stop = getattr(resp.raw, "stop_reason", None)

        if stop != "tool_use":
            return ToolLoopResult(
                response=resp,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
            )

        # Extract tool_use blocks from the assistant response
        tool_uses = [
            b for b in resp.raw.content if getattr(b, "type", None) == "tool_use"
        ]
        if not tool_uses:
            return ToolLoopResult(
                response=resp,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
            )

        # Early termination: if any tool_use is a terminal recording tool,
        # treat this response as the final assistant message. We still log
        # the call so observability sees it, but we do NOT dispatch (the
        # registry would return "unknown tool" anyway) and we do NOT loop.
        if any(getattr(b, "name", "") in terminal_set for b in tool_uses):
            for b in tool_uses:
                tool_calls_log.append({
                    "name": getattr(b, "name", ""),
                    "args": dict(getattr(b, "input", {}) or {}),
                    "is_error": False,
                    "duration_ms": 0,
                })
            return ToolLoopResult(
                response=resp,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
            )

        tool_uses = tool_uses[:parallel_cap]
        kept_ids = {getattr(tu, "id", None) for tu in tool_uses}

        # Dispatch in parallel
        results = await asyncio.gather(
            *(_dispatch(registry, tu, ctx, tool_timeout_s) for tu in tool_uses),
            return_exceptions=False,
        )

        # Update url tracking + log
        for tu, r in zip(tool_uses, results, strict=True):
            tool_calls_log.append(
                {
                    "name": tu.name,
                    "args": tu.input,
                    "is_error": r["is_error"],
                    "duration_ms": r.get("duration_ms", 0),
                }
            )
            for u in extract_urls(r.get("content")):
                seen_urls.add(u)

        # Build next-turn spec: append the assistant message + a single user message
        # carrying all tool_result blocks. The assistant message must only carry
        # the tool_use blocks we actually dispatched — Anthropic requires every
        # tool_use to be paired with exactly one tool_result on the next turn.
        assistant_blocks = _content_to_dicts(resp.raw.content)
        assistant_blocks = [
            b for b in assistant_blocks
            if b.get("type") != "tool_use" or b.get("id") in kept_ids
        ]
        next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
        next_messages.append(
            {"role": "assistant", "content": assistant_blocks}
        )
        next_messages.append(
            {
                "role": "user",
                "content": [
                    _tool_result_block(tu, r) for tu, r in zip(tool_uses, results, strict=True)
                ],
            }
        )
        current_spec = _respec(current_spec, extra_messages=next_messages)

    assert last is not None
    raise ToolLoopExhausted(ctx.agent, iterations)


# --------------------------------------------------------------------------- #
# Captured mode — the backend already ran the cycle


async def _run_captured(
    client: Any,
    *,
    spec: AgentCallSpec,
    ctx: CallContext,
    max_iters: int,
    force_terminal_tool: str | None,
    terminal_tool_names: tuple[str, ...],
) -> ToolLoopResult:
    """Read back one CLI call's captured work, escalating once if it recorded nothing.

    `registry`, `parallel_cap`, and `tool_timeout_s` are enforced inside the MCP
    tool server (the process that actually runs the tools), so they play no part
    here.
    """
    terminal = set(terminal_tool_names)
    attempts = max(1, min(2, max_iters))
    last: LLMResponse | None = None

    for attempt in range(1, attempts + 1):
        call_spec = spec if attempt == 1 else _escalate(spec, force_terminal_tool, terminal)
        resp = await client.call(call_spec, ctx)
        last = resp

        capture = resp.capture
        tool_calls = list(capture.tool_calls) if capture else []
        seen_urls = set(capture.seen_urls) if capture else set()
        records = capture.records if capture else []

        for name, payload in records:
            tool_calls.append(
                {"name": name, "args": payload, "is_error": False, "duration_ms": 0}
            )

        got_record = any(name in terminal for name, _ in records)
        if got_record or not _requires_record(spec, terminal):
            return ToolLoopResult(
                response=resp,
                iterations=resp.num_turns or attempt,
                tool_calls=tool_calls,
                seen_urls=seen_urls,
            )

        log.warning(
            "no_record_captured",
            agent=ctx.agent, action=ctx.action, attempt=attempt,
            turns=resp.num_turns, tool_calls=len(tool_calls),
        )

    assert last is not None
    raise ToolLoopExhausted(ctx.agent, attempts)


def _requires_record(spec: AgentCallSpec, terminal: set[str]) -> bool:
    """Does this call expect a structured record at all?"""
    return any(str(t.get("name", "")) in terminal for t in spec.tools)


def _escalate(
    spec: AgentCallSpec, force_terminal_tool: str | None, terminal: set[str]
) -> AgentCallSpec:
    """Re-issue the call, demanding the record and nothing else.

    Dropping the research tools is deliberate: the failure mode this recovers
    from is a model that keeps searching and never commits, so the retry
    removes the option to search again.
    """
    target = force_terminal_tool or next(
        (str(t["name"]) for t in spec.tools if str(t.get("name", "")) in terminal),
        None,
    )
    record_tools = [t for t in spec.tools if str(t.get("name", "")) in terminal]
    demand = (
        "\n\nYour previous attempt ended without recording a result. Do not run "
        "any further searches. Using what you already know, call "
        f"`{target}` now with your best answer."
    )
    user_blocks = list(spec.user_blocks)
    if user_blocks:
        user_blocks[-1] = CachedBlock(
            text=user_blocks[-1].text + demand, cache=user_blocks[-1].cache
        )

    return _respec(
        spec,
        user_blocks=user_blocks,
        tools=record_tools or spec.tools,
        tool_choice={"type": "tool", "name": target} if target else spec.tool_choice,
    )


# --------------------------------------------------------------------------- #
# helpers


def _respec(spec: AgentCallSpec, **overrides: Any) -> AgentCallSpec:
    """Copy a spec with selected fields replaced.

    Both modes rebuild the spec between turns; doing that field-by-field at
    each site is how a newly added `AgentCallSpec` field silently gets dropped.
    """
    return AgentCallSpec(
        route=overrides.get("route", spec.route),
        system_blocks=overrides.get("system_blocks", spec.system_blocks),
        user_blocks=overrides.get("user_blocks", spec.user_blocks),
        tools=overrides.get("tools", spec.tools),
        tool_choice=overrides.get("tool_choice", spec.tool_choice),
        max_output_tokens=overrides.get("max_output_tokens", spec.max_output_tokens),
        stop_sequences=overrides.get("stop_sequences", spec.stop_sequences),
        extra_messages=overrides.get("extra_messages", spec.extra_messages),
    )


async def _dispatch(
    registry: ToolRegistry, tool_use, ctx: CallContext, timeout_s: float
) -> dict[str, Any]:
    """Run one tool call. Returns a dict with content + is_error + duration."""
    t0 = time.monotonic()
    run_id = tool_run_id()
    tctx = ToolCtx(
        cfg=registry._cfg,
        db=None,            # tools use their own write paths; DB writes go via repos
        session_id=ctx.session_id,
        task_id=ctx.task_id,
        run_id=run_id,
    )
    args = dict(tool_use.input) if isinstance(tool_use.input, dict) else {"args": tool_use.input}
    try:
        result = await asyncio.wait_for(
            registry.call(tool_use.name, args, tctx), timeout=timeout_s
        )
    except TimeoutError:
        return {
            "is_error": True,
            "content": {"error": f"tool {tool_use.name!r} timed out"},
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
    return {
        "is_error": bool(result.is_error),
        "content": _tool_result_content(result),
        "duration_ms": result.duration_ms,
    }


def _tool_result_content(result) -> Any:
    if result.is_error:
        return {"error": result.error_message or "unknown error"}
    return result.content if result.content is not None else {"ok": True}


def _tool_result_block(tool_use, r: dict[str, Any]) -> dict[str, Any]:
    body = r["content"]
    return {
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": _content_to_text(body),
        "is_error": r["is_error"],
    }


def _content_to_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    return json.dumps(body, default=str, ensure_ascii=False)[:60_000]


def _content_to_dicts(content) -> list[dict[str, Any]]:
    """Convert SDK content blocks to plain dicts for re-sending.

    Thinking blocks must preserve their `signature` verbatim — Anthropic rejects
    a continuation turn that omits it. `provider_fields` carries the equivalent
    for OpenAI-compatible providers (Gemini's thought signature); the request
    adapter decides which of those are safe to replay to which endpoint.
    """
    out: list[dict[str, Any]] = []
    for b in content:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif t == "tool_use":
            d = {
                "type": "tool_use",
                "id": getattr(b, "id", ""),
                "name": getattr(b, "name", ""),
                "input": getattr(b, "input", {}),
            }
            provider_fields = getattr(b, "provider_fields", None)
            if isinstance(provider_fields, dict) and provider_fields:
                d["provider_fields"] = provider_fields
            out.append(d)
        elif t == "thinking":
            d: dict[str, Any] = {"type": "thinking", "thinking": getattr(b, "thinking", "")}
            sig = getattr(b, "signature", None)
            if sig:
                d["signature"] = sig
            out.append(d)
        elif t == "redacted_thinking":
            data = getattr(b, "data", None)
            if data:
                out.append({"type": "redacted_thinking", "data": data})
    return out


# The URL walker moved to tools/urls.py so the MCP server (a separate process)
# can share it. This alias keeps the long-standing private import working.
_extract_urls = extract_urls
