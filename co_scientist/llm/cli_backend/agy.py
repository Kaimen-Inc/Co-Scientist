"""Antigravity CLI backend — `agy` headless mode on subscription auth.

Design notes grounded in `agy` 1.5.x headless behavior:

- **Stdin streaming protocol (`--input-format stream-json --output-format stream-json`).**
  `agy -p` expects the prompt as a flag argument, which risks OS `ARG_MAX`
  limits on large meta-review prompts. Streaming NDJSON over stdin avoids argv
  limits completely while returning structured `init`, `step_update`, and
  terminal `result` events (including token usage and `structured_output`).
- **Per-call ephemeral workspace MCP config.** `agy` discovers workspace MCP
  servers at `<cwd>/.agents/mcp_config.json`. Each attempt gets its own
  ephemeral workspace inside `capture_dir` with eager tool loading enabled on
  our `cosci` server, so parallel calls and retries never collide and the user's
  global `~/.gemini/config/mcp_config.json` is never mutated.
- **Session-only `--settings`.** Passes `trustedWorkspaces`, disables slash
  command expansion, clears `modelProvider` (preventing fallback to metered
  `GEMINI_API_KEY` billing), and — in the default `"scoped"` permission mode —
  allows `mcp(cosci/*)` while denying built-in shell, file-write, and web tools
  so all research tool calls flow through our MCP provenance log.
- **Structured output (`--json-schema` + MCP capture).** When a call expects a
  single `record_*` schema and `use_json_schema` is enabled, `--json-schema`
  enforces the schema on the final `result` event while keeping the MCP capture
  path active as both research-tool transport and fallback.
- **Reasoning effort safety.** Many `agy` model slugs already encode an effort
  tier (e.g. `gemini-4-carbon-high`, `gemini-4-neon-xhigh`), and `agy` rejects
  conflicting `--effort` flags. `--effort` is therefore only emitted when
  `thinking_tokens > 0` and the model slug has no explicit effort suffix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import Config
from ..retry import classify_failure
from ..types import AgentCallSpec, CallContext
from .base import AgentCliProvider, CliInvocation, CliOutcome

FORCED_RECORD_FILE = "_forced_record.txt"

_EFFORT_TIERS: tuple[tuple[int, str], ...] = (
    (1, "low"),
    (4_000, "medium"),
    (8_000, "high"),
    (16_000, "xhigh"),
)
_MAX_EFFORT = "max"
_EFFORT_SUFFIXES: tuple[str, ...] = ("-low", "-medium", "-high", "-xhigh", "-max")


def effort_for_thinking_tokens(thinking_tokens: int) -> str:
    for ceiling, tier in _EFFORT_TIERS:
        if thinking_tokens <= ceiling:
            return tier
    return _MAX_EFFORT


def _has_effort_suffix(model: str) -> bool:
    low = model.strip().lower()
    return any(low.endswith(suffix) for suffix in _EFFORT_SUFFIXES)


def _should_pass_effort(model: str, thinking_tokens: int) -> bool:
    return thinking_tokens > 0 and not _has_effort_suffix(model)


class AgyCliProvider(AgentCliProvider):
    backend_name = "agy_cli"

    def _resolve_backend_cfg(self, cfg: Config) -> Any:
        return cfg.llm.agy_cli

    # ----------------------------- command ----------------------------- #

    def build_invocation(
        self, spec: AgentCallSpec, ctx: CallContext, capture_dir: Path
    ) -> CliInvocation:
        record_tools = [t["name"] for t in spec.tools if _is_record_tool(t)]
        research_tools = [t["name"] for t in spec.tools if not _is_record_tool(t)]
        mcp_agent = ctx.agent if research_tools else ""
        all_tools = [*record_tools, *research_tools]

        workspace = capture_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        if all_tools:
            mcp = self.mcp_config(
                capture_dir=capture_dir,
                record_tools=record_tools,
                agent=mcp_agent,
                ctx=ctx,
            )
            server_cfg = mcp["mcpServers"][self.server_label]
            server_cfg["enabledTools"] = all_tools
            server_cfg["tools"] = {name: {"eager": True} for name in all_tools}
            server_cfg["timeoutSeconds"] = max(
                60, int(self._cfg.tool_loop.tool_timeout_seconds) + 30
            )
            agents_dir = workspace / ".agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / "mcp_config.json").write_text(
                json.dumps(mcp, ensure_ascii=False), encoding="utf-8"
            )

        settings: dict[str, Any] = {
            "disableSlashCommands": True,
            "modelProvider": "",
            "trustedWorkspaces": [str(workspace)],
        }
        if self._backend_cfg.permission_mode == "scoped":
            settings["permissions"] = {
                "allow": [f"mcp({self.server_label}/*)"] if all_tools else [],
                "deny": [
                    "command(*)",
                    "write_file(*)",
                    "read_url(*)",
                    "execute_url(*)",
                    "download_url(*)",
                ],
            }
        settings_path = capture_dir / "agy-settings.json"
        settings_path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")

        argv = [
            self._binary,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--disable-slash-commands",
            "--model", spec.route.model,
            "--settings", str(settings_path),
        ]
        if self._backend_cfg.permission_mode == "skip_permissions":
            argv.append("--dangerously-skip-permissions")
        if self._backend_cfg.sandbox:
            argv.append("--sandbox")
        if _should_pass_effort(spec.route.model, spec.route.thinking_tokens):
            argv += ["--effort", effort_for_thinking_tokens(spec.route.thinking_tokens)]

        forced = _forced_record_tool(spec, record_tools)
        use_schema = forced is not None and bool(self._backend_cfg.use_json_schema)
        if use_schema and forced is not None:
            schema = next(t["input_schema"] for t in spec.tools if t["name"] == forced)
            schema_path = capture_dir / "output-schema.json"
            schema_path.write_text(
                json.dumps(_strict_object_schema(schema), ensure_ascii=False),
                encoding="utf-8",
            )
            (capture_dir / FORCED_RECORD_FILE).write_text(forced, encoding="utf-8")
            argv += ["--json-schema", str(schema_path)]

        prompt = _build_prompt(spec, forced if use_schema else None)
        stdin_line = json.dumps(
            {"event": "user", "message": {"content": prompt}},
            ensure_ascii=False,
        ) + "\n"
        return CliInvocation(argv=argv, stdin=stdin_line, cwd=str(workspace))

    # ----------------------------- parsing ----------------------------- #

    def parse_outcome(
        self, *, stdout: str, stderr: str, returncode: int, capture_dir: Path
    ) -> CliOutcome:
        events = _parse_events(stdout)
        result = _find_result(events)

        if result is None:
            detail = (stderr or stdout or "no output").strip()[:800]
            raise classify_failure(
                f"agy did not emit a parseable result event (exit {returncode}): {detail}"
            )

        status = str(result.get("status") or "").upper()
        if status != "SUCCESS" or returncode != 0:
            err_msg = str(result.get("error") or stderr or result.get("response") or "unknown error")
            raise classify_failure(f"agy error [{status or returncode}]: {err_msg[:800]}")

        text = str(result.get("response") or "").strip() or _streamed_text(events)

        # Normalize native `--json-schema` structured_output into the capture
        # directory when the model did not already call the MCP record_* tool.
        forced_path = capture_dir / FORCED_RECORD_FILE
        if forced_path.exists() and not list(capture_dir.glob("record-*.json")):
            structured = result.get("structured_output")
            payload = structured if isinstance(structured, dict) else _parse_json_object(text)
            if payload is not None:
                tool = forced_path.read_text(encoding="utf-8").strip()
                (capture_dir / f"record-0001-{tool}.json").write_text(
                    json.dumps({"tool": tool, "seq": 1, "input": payload}, ensure_ascii=False),
                    encoding="utf-8",
                )

        usage = _find_usage(result, events)
        return CliOutcome(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read=usage.get("cache_read_tokens", 0),
            num_turns=max(1, int(result.get("num_turns") or 1)),
            reported_cost_usd=None,  # agy reports tokens, not dollars
            stop_reason=_map_stop_reason(result.get("finished_reason")),
            raw={"result": result, "events": events[-50:]},
        )


# --------------------------------------------------------------------------- #
# prompt & schema helpers


def _forced_record_tool(spec: AgentCallSpec, record_tools: list[str]) -> str | None:
    """The single record tool this call must produce, if there is exactly one."""
    if spec.tool_choice and spec.tool_choice.get("type") == "tool":
        name = spec.tool_choice.get("name")
        return name if name in record_tools else None
    return record_tools[0] if len(record_tools) == 1 else None


def _strict_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure top-level object type for `--json-schema`."""
    out = dict(schema)
    out.setdefault("type", "object")
    return out


def _build_prompt(spec: AgentCallSpec, forced_schema_tool: str | None) -> str:
    from .claude_code import _render_thread, _tool_instructions, build_system_prompt

    parts: list[str] = []
    system = build_system_prompt(spec)
    if system:
        parts.append(system)

    parts.extend(b.text for b in spec.user_blocks if b.text)

    thread = _render_thread(spec.extra_messages)
    if thread:
        parts.append(thread)

    if forced_schema_tool is not None:
        parts.append(
            "## Required output\n"
            f"You may call the `{forced_schema_tool}` MCP tool to record your "
            "structured answer, and your final response MUST be a single JSON "
            "object matching the provided output schema (no extra prose or "
            "code fences)."
        )
    else:
        instructions = _tool_instructions(spec)
        if instructions:
            parts.append(instructions)

    return "\n\n".join(p for p in parts if p).strip()


def _is_record_tool(tool: dict[str, Any]) -> bool:
    return str(tool.get("name", "")).startswith("record_")


# --------------------------------------------------------------------------- #
# event-stream parsing


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _find_result(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Locate the terminal `result` payload from stream-json (or a plain json envelope)."""
    for event in reversed(events):
        if event.get("event") == "result" and isinstance(event.get("result"), dict):
            return event["result"]
    # Fallback: single `--output-format json` envelope without the outer wrapper.
    for event in reversed(events):
        if "status" in event and ("response" in event or "error" in event):
            return event
    return None


def _streamed_text(events: list[dict[str, Any]]) -> str:
    """Fallback text assembled from `step_update.text_delta` fragments."""
    chunks: list[str] = []
    for event in events:
        if event.get("event") != "step_update":
            continue
        step = event.get("step_update") or {}
        if step.get("step_type") == "agent_response":
            delta = step.get("text_delta")
            if isinstance(delta, str) and delta:
                chunks.append(delta)
    return "".join(chunks).strip()


def _find_usage(result: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, int]:
    """Read token counts from the terminal result, falling back to step updates."""
    usage = result.get("usage")
    if isinstance(usage, dict) and any(
        isinstance(usage.get(k), int) and usage.get(k, 0) > 0
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "total_tokens")
    ):
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_tokens": int(
                usage.get("cache_read_tokens") or usage.get("cache_read_input_tokens") or 0
            ),
        }

    out = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    for event in events:
        if event.get("event") != "step_update":
            continue
        step = event.get("step_update") or {}
        if step.get("state") != "DONE":
            continue
        step_usage = step.get("usage")
        if not isinstance(step_usage, dict):
            continue
        out["input_tokens"] += int(step_usage.get("input_tokens") or 0)
        out["output_tokens"] += int(step_usage.get("output_tokens") or 0)
        out["cache_read_tokens"] += int(
            step_usage.get("cache_read_tokens") or step_usage.get("cache_read_input_tokens") or 0
        )
    return out


def _map_stop_reason(finished_reason: Any) -> str:
    if not isinstance(finished_reason, str) or not finished_reason:
        return "end_turn"
    low = finished_reason.lower()
    if "max_token" in low or "length" in low:
        return "max_tokens"
    return "end_turn"


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a ```json fence around it."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1]
        if body.endswith("```"):
            body = body[:-3]
        body = body.strip()
    start = body.find("{")
    if start == -1:
        return None
    try:
        parsed = json.loads(body[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = ["AgyCliProvider", "effort_for_thinking_tokens"]
