"""The assistant↔tool_use↔tool_result loop.

The agent gives us:
- An initial AgentCallSpec (system + user blocks, tools, tool_choice).
- A ToolRegistry (or just the subset relevant for this agent).
- A max_iters cap.

We drive turns until the model returns a non-tool-use stop_reason, or we hit
the cap (which surfaces as ToolLoopExhausted to the calling agent).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..ids import tool_run_id
from ..tools.base import ToolCtx
from ..tools.registry import ToolRegistry
from .anthropic_client import AgentCallSpec, AnthropicClient, AnthropicResponse, CallContext


class ToolLoopExhausted(RuntimeError):
    def __init__(self, agent: str, iters: int):
        super().__init__(f"tool loop for agent {agent!r} exhausted after {iters} iterations")
        self.agent = agent
        self.iters = iters


@dataclass
class ToolLoopResult:
    response: AnthropicResponse                  # final assistant message
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    """Union of URLs that appeared in any tool_result over the loop.

    Used by structured-output validation to reject hallucinated citations:
    Generation's record_hypothesis.citations[].url must be in this set.
    """


async def run_tool_loop(
    client: AnthropicClient,
    *,
    spec: AgentCallSpec,
    ctx: CallContext,
    registry: ToolRegistry,
    max_iters: int,
    parallel_cap: int = 4,
    tool_timeout_s: float = 30.0,
    force_terminal_tool: str | None = None,
    terminal_tool_names: tuple[str, ...] = (
        "record_hypothesis",
        "record_review",
        "record_system_feedback",
        "record_rubric_score",
        "record_research_plan",
    ),
) -> ToolLoopResult:
    """Drive the assistant ↔ tool_use ↔ tool_result loop.

    Loop termination:
    - stop_reason != "tool_use" — the model signalled end_turn.
    - The assistant response contains a `terminal_tool_names` call. These are
      virtual "structured output capture" tools (e.g. `record_hypothesis`):
      the assistant has already produced its final answer in tool_use.input,
      so dispatching the tool is unnecessary and we should not invite the
      model to call it again. Claude reliably ends its turn after calling
      these; Gemini / OpenAI-compat models do not, so we short-circuit
      explicitly. Without this short-circuit the loop will repeatedly
      re-invite the recording tool until max_iters and then raise
      ToolLoopExhausted — even though a perfectly good record was emitted
      on the first call.
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

    last: AnthropicResponse | None = None

    while iterations < max_iters:
        iterations += 1
        # On the final allowed iteration, optionally force the recording tool so
        # the model commits instead of burning its last turn on another search.
        call_spec = current_spec
        if force_terminal_tool and iterations == max_iters:
            call_spec = AgentCallSpec(
                route=current_spec.route,
                system_blocks=current_spec.system_blocks,
                user_blocks=current_spec.user_blocks,
                tools=current_spec.tools,
                tool_choice={"type": "tool", "name": force_terminal_tool},
                max_output_tokens=current_spec.max_output_tokens,
                stop_sequences=current_spec.stop_sequences,
                extra_messages=current_spec.extra_messages,
            )
        resp = await client.call(call_spec, ctx)
        last = resp
        stop = getattr(resp.raw, "stop_reason", None)

        if stop != "tool_use":
            # If the model returned a text response early (before calling the
            # terminal tool), force one more call so the result is captured.
            # This handles OpenAI/Gemini models that respond in text instead of
            # committing via the recording tool — Claude reliably doesn't need this.
            if force_terminal_tool and iterations < max_iters:
                already_called = any(
                    e.get("name") == force_terminal_tool for e in tool_calls_log
                )
                if not already_called:
                    forced_spec = AgentCallSpec(
                        route=current_spec.route,
                        system_blocks=current_spec.system_blocks,
                        user_blocks=current_spec.user_blocks,
                        tools=current_spec.tools,
                        tool_choice={"type": "tool", "name": force_terminal_tool},
                        max_output_tokens=current_spec.max_output_tokens,
                        stop_sequences=current_spec.stop_sequences,
                        extra_messages=current_spec.extra_messages,
                    )
                    resp = await client.call(forced_spec, ctx)
                    return ToolLoopResult(
                        response=resp,
                        iterations=iterations + 1,
                        tool_calls=tool_calls_log,
                        seen_urls=seen_urls,
                    )
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
            for u in _extract_urls(r.get("content")):
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
        current_spec = AgentCallSpec(
            route=current_spec.route,
            system_blocks=current_spec.system_blocks,
            user_blocks=current_spec.user_blocks,
            tools=current_spec.tools,
            tool_choice=current_spec.tool_choice,
            max_output_tokens=current_spec.max_output_tokens,
            stop_sequences=current_spec.stop_sequences,
            extra_messages=next_messages,
        )

    assert last is not None
    raise ToolLoopExhausted(ctx.agent, iterations)


# --------------------------------------------------------------------------- #
# helpers


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
    a continuation turn that omits it.
    """
    out: list[dict[str, Any]] = []
    for b in content:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif t == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": getattr(b, "id", ""),
                    "name": getattr(b, "name", ""),
                    "input": getattr(b, "input", {}),
                }
            )
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


_URL_RE_KEYS = ("url", "abs_url", "pdf_url", "pubmed_url")


def _extract_urls(body: Any) -> list[str]:
    """Pull URLs out of nested tool_result content (best effort)."""
    out: list[str] = []
    _walk_urls(body, out)
    return out


def _walk_urls(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _URL_RE_KEYS and isinstance(v, str) and v.startswith(("http://", "https://")):
                out.append(v)
            else:
                _walk_urls(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk_urls(item, out)
