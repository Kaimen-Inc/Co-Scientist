"""A deterministic stand-in for `agy` headless mode (`--input-format stream-json`).

Reads one NDJSON user message from stdin, inspects the workspace MCP config at
`<cwd>/.agents/mcp_config.json` to recover `COSCI_CAPTURE_DIR`, writes any
configured records/tool provenance entries, and emits `init` + `step_update` +
`result` NDJSON events matching `agy --output-format stream-json`.

Two environment variables drive it:

    FAKE_CLI_BEHAVIOR   JSON describing what to do (see keys below)
    FAKE_CLI_DUMP       path to write {argv, stdin, env, cwd, mcp_config, settings}

BEHAVIOR keys:
    mode               "success" (default) | "error" | "rate_limit" | "garbage"
    text               the assistant's final response text
    structured_output  dict returned in result.structured_output
    records            [[tool_name, payload], ...] written to capture dir
    tools              [{name, args, urls, is_error}, ...] appended to tools.jsonl
    usage              override for the reported token counts
    num_turns          override for result.num_turns (default 2)
    finished_reason    override for result.finished_reason (default "stop")
    status             override for result.status (default "SUCCESS")
    returncode         override process exit code
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_USAGE = {
    "input_tokens": 1200,
    "output_tokens": 350,
    "thinking_tokens": 180,
    "cache_read_tokens": 600,
    "total_tokens": 1550,
}


def _behavior() -> dict[str, Any]:
    raw = os.environ.get("FAKE_CLI_BEHAVIOR")
    return json.loads(raw) if raw else {}


def _read_mcp_config() -> dict[str, Any] | None:
    path = Path.cwd() / ".agents" / "mcp_config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_settings(argv: list[str]) -> dict[str, Any] | None:
    if "--settings" not in argv:
        return None
    raw = argv[argv.index("--settings") + 1]
    p = Path(raw)
    try:
        text = p.read_text(encoding="utf-8") if p.exists() else raw
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None


def _dump(argv: list[str], stdin: str) -> None:
    path = os.environ.get("FAKE_CLI_DUMP")
    if not path:
        return
    Path(path).write_text(
        json.dumps({
            "argv": argv,
            "stdin": stdin,
            "env": dict(os.environ),
            "cwd": str(Path.cwd()),
            "mcp_config": _read_mcp_config(),
            "settings": _read_settings(argv),
        }),
        encoding="utf-8",
    )


def _capture_dir(mcp_cfg: dict[str, Any] | None) -> Path | None:
    if not isinstance(mcp_cfg, dict):
        return None
    for server in (mcp_cfg.get("mcpServers") or {}).values():
        target = (server.get("env") or {}).get("COSCI_CAPTURE_DIR")
        if target:
            return Path(target)
    return None


def _write_capture(capture: Path, behavior: dict[str, Any]) -> None:
    capture.mkdir(parents=True, exist_ok=True)
    for i, (tool, payload) in enumerate(behavior.get("records") or [], start=1):
        (capture / f"record-{i:04d}-{tool}.json").write_text(
            json.dumps({"tool": tool, "seq": i, "input": payload}),
            encoding="utf-8",
        )
    calls = behavior.get("tools") or []
    if calls:
        with (capture / "tools.jsonl").open("a", encoding="utf-8") as f:
            for call in calls:
                f.write(
                    json.dumps({
                        "name": call.get("name", "pubmed_search"),
                        "args": call.get("args", {}),
                        "is_error": bool(call.get("is_error")),
                        "duration_ms": int(call.get("duration_ms", 5)),
                        "urls": call.get("urls", []),
                    })
                    + "\n"
                )


def main() -> int:
    argv = sys.argv[1:]

    if argv == ["--version"]:
        sys.stdout.write("1.5.15-fake\n")
        return 0
    if argv == ["models"]:
        behavior = _behavior()
        if behavior.get("models_error"):
            sys.stderr.write(str(behavior["models_error"]) + "\n")
            return 1
        sys.stdout.write(
            "\rFetching available models...\r"
            "gemini-4-carbon-high           Gemini 4 Carbon (High)\n"
            "gemini-4-neon-xhigh            Gemini 4 Neon (XHigh)\n"
        )
        return 0

    stdin = "" if sys.stdin.isatty() else sys.stdin.read()
    _dump(argv, stdin)

    behavior = _behavior()
    mode = behavior.get("mode", "success")

    if mode == "garbage":
        sys.stdout.write("not json at all\n")
        return 1

    mcp_cfg = _read_mcp_config()
    capture = _capture_dir(mcp_cfg)
    if capture is not None and mode == "success":
        _write_capture(capture, behavior)

    if mode == "rate_limit":
        event = {
            "event": "result",
            "result": {
                "conversation_id": "conv-fake",
                "status": "ERROR",
                "response": "",
                "error": "Quota exceeded (429): rate limit reached. Try again later.",
                "duration_seconds": 0.1,
                "num_turns": 0,
                "usage": DEFAULT_USAGE,
                "finished_reason": "error",
            },
        }
        sys.stdout.write(json.dumps(event) + "\n")
        return int(behavior.get("returncode", 1))

    if mode == "error":
        event = {
            "event": "result",
            "result": {
                "conversation_id": "conv-fake",
                "status": "ERROR",
                "response": "",
                "error": behavior.get("text", "invalid model selection"),
                "duration_seconds": 0.1,
                "num_turns": 0,
                "usage": DEFAULT_USAGE,
                "finished_reason": "error",
            },
        }
        sys.stdout.write(json.dumps(event) + "\n")
        return int(behavior.get("returncode", 1))

    usage = {**DEFAULT_USAGE, **(behavior.get("usage") or {})}
    text = behavior.get("text", "done")
    result: dict[str, Any] = {
        "conversation_id": "conv-fake",
        "status": behavior.get("status", "SUCCESS"),
        "response": text,
        "duration_seconds": 1.2,
        "num_turns": behavior.get("num_turns", 2),
        "usage": usage,
        "finished_reason": behavior.get("finished_reason", "stop"),
    }
    if "structured_output" in behavior:
        result["structured_output"] = behavior["structured_output"]

    events = [
        {
            "event": "init",
            "conversation_id": "conv-fake",
            "init": {"cwd": str(Path.cwd()), "permission_mode": "scoped"},
        },
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv-fake",
                "step_index": 1,
                "state": "DONE",
                "step_type": "agent_response",
                "text_delta": text,
                "usage": usage,
            },
        },
        {"event": "result", "result": result},
    ]
    for ev in events:
        sys.stdout.write(json.dumps(ev) + "\n")
    return int(behavior.get("returncode", 0))


if __name__ == "__main__":
    raise SystemExit(main())
