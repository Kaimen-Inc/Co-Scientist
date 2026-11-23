"""Tests for the Antigravity CLI (`agy_cli`) backend.

Runs deterministically against `fake_agy.py` without making live model calls.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from co_scientist.config import Config, backend_binary, has_llm_key, load_config, provider_key_env
from co_scientist.llm.budgets import TokenBudget
from co_scientist.llm.cli_backend.agy import AgyCliProvider, effort_for_thinking_tokens
from co_scientist.llm.cli_backend.base import BILLING_ENV_VARS
from co_scientist.llm.provider import CLI_BACKENDS, check_backend, check_models, get_provider
from co_scientist.llm.retry import CliBackendError, CliRetryableError
from co_scientist.llm.routing import ModelRoute, degrade_model, estimate_cost_usd
from co_scientist.llm.tool_loop import run_tool_loop
from co_scientist.llm.types import AgentCallSpec, CachedBlock, CallContext

RECORD_TOOL = {
    "name": "record_hypothesis",
    "description": "record it",
    "input_schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    },
}
SEARCH_TOOL = {"name": "pubmed_search", "description": "search", "input_schema": {}}


@pytest.fixture
def fake_agy(tmp_path: Path) -> Path:
    from co_scientist.tests.fakes import fake_agy as module

    launcher = tmp_path / "fake-agy"
    launcher.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{Path(module.__file__)}" "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


@pytest.fixture
def agy_cfg(tmp_cfg: Config, fake_agy: Path) -> Config:
    tmp_cfg.llm.provider = "agy_cli"
    tmp_cfg.llm.agy_cli.binary = str(fake_agy)
    tmp_cfg.llm.agy_cli.timeout_seconds = 30
    return tmp_cfg


@pytest.fixture
async def provider(agy_cfg: Config, conn) -> AgyCliProvider:
    from co_scientist.models import ResearchPlan, Session
    from co_scientist.storage.repos import sessions as sess_repo

    now = datetime.now(UTC)
    await sess_repo.insert(conn, Session(
        id="ses_agy", created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="g"),
        config_snapshot={}, budget_tokens=1_000_000, budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    ))
    budget = TokenBudget(cfg=agy_cfg, budget_tokens=1_000_000, budget_usd=10.0)
    return AgyCliProvider(agy_cfg, db=conn, budget=budget)


def _spec(**kw: Any) -> AgentCallSpec:
    return AgentCallSpec(
        route=kw.pop("route", ModelRoute(
            agent="generation", mode="literature",
            model="gemini-4-carbon-high", thinking_tokens=4000,
        )),
        system_blocks=kw.pop("system_blocks", [CachedBlock("You are the generation agent.")]),
        user_blocks=kw.pop("user_blocks", [CachedBlock("Find a hypothesis.")]),
        tools=kw.pop("tools", [SEARCH_TOOL, RECORD_TOOL]),
        **kw,
    )


def _ctx(task_id: str | None = None) -> CallContext:
    return CallContext(
        session_id="ses_agy", task_id=task_id, agent="generation", action="Create",
    )


def _behave(monkeypatch: pytest.MonkeyPatch, **behavior: Any) -> None:
    monkeypatch.setenv("FAKE_CLI_BEHAVIOR", json.dumps(behavior))


# --------------------------------------------------------------------------- #
# registration, config, preflight


def test_agy_cli_is_registered_keyless_cli_backend(agy_cfg: Config) -> None:
    assert "agy_cli" in CLI_BACKENDS
    assert provider_key_env(agy_cfg) == ""
    assert has_llm_key(agy_cfg) is True
    assert backend_binary(agy_cfg) == agy_cfg.llm.agy_cli.binary


def test_get_provider_returns_agy_cli_provider(agy_cfg: Config) -> None:
    budget = TokenBudget(cfg=agy_cfg, budget_tokens=100_000, budget_usd=5.0)
    client = get_provider(agy_cfg, db=None, budget=budget)
    assert isinstance(client, AgyCliProvider)
    assert client.runs_own_loop is True


def test_preset_config_loads_cleanly_with_valid_models() -> None:
    preset_path = Path(__file__).resolve().parents[3] / "config" / "agy.toml"
    cfg = load_config(preset_path)
    assert cfg.llm.provider == "agy_cli"
    assert check_models(cfg) == []
    # Two-tier Gemini 4 routing
    assert cfg.models.parse_goal == "gemini-4-neon-xhigh"
    assert cfg.models.ranking_pairwise == "gemini-4-neon-xhigh"
    assert cfg.models.metareview_feedback == "gemini-4-neon-xhigh"
    assert cfg.models.classifier == "gemini-4-neon-xhigh"
    assert cfg.models.judge == "gemini-4-neon-xhigh"
    assert cfg.models.generation == "gemini-4-carbon-high"
    assert cfg.models.reflection == "gemini-4-carbon-high"
    assert cfg.models.evolution == "gemini-4-carbon-high"
    assert cfg.models.ranking_debate == "gemini-4-carbon-high"
    assert cfg.models.ranking_priority == "gemini-4-carbon-high"
    assert cfg.models.metareview_final == "gemini-4-carbon-high"


def test_check_backend_probes_version_and_models(agy_cfg: Config) -> None:
    status = check_backend(agy_cfg)
    assert status.ok is True
    assert status.backend == "agy_cli"
    assert status.version == "1.5.15-fake"
    assert status.authenticated is True
    assert "2 models available" in status.detail


def test_check_backend_reports_unauthenticated_when_models_fails(
    agy_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _behave(monkeypatch, models_error="authentication required")
    status = check_backend(agy_cfg)
    assert status.ok is False
    assert status.authenticated is False
    assert "authentication required" in status.detail


def test_gemini_4_pricing_and_degrade_chain() -> None:
    assert degrade_model("gemini-4-carbon-high") == "gemini-4-neon-xhigh"
    assert degrade_model("gemini-4-neon-xhigh") == "gemini-4-neon-xhigh"
    c_high = estimate_cost_usd(
        model="gemini-4-carbon-high", input_tokens=1_000_000, output_tokens=1_000_000
    )
    c_neon = estimate_cost_usd(
        model="gemini-4-neon-xhigh", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert c_high > c_neon > 0


# --------------------------------------------------------------------------- #
# billing safety & invocation construction


async def test_billing_env_vars_including_gemini_key_are_stripped(
    provider: AgyCliProvider, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert "GEMINI_API_KEY" in BILLING_ENV_VARS
    assert "GOOGLE_API_KEY" in BILLING_ENV_VARS

    dump = tmp_path / "dump.json"
    monkeypatch.setenv("GEMINI_API_KEY", "gm-should-not-leak")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("FAKE_CLI_DUMP", str(dump))
    _behave(monkeypatch, records=[["record_hypothesis", {"title": "T"}]])

    await provider.call(_spec(), _ctx())

    data = json.loads(dump.read_text(encoding="utf-8"))
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert key not in data["env"]
    assert data["settings"]["modelProvider"] == ""


def test_command_wires_stream_json_workspace_mcp_and_scoped_settings(
    provider: AgyCliProvider, tmp_path: Path
) -> None:
    inv = provider.build_invocation(_spec(), _ctx(), tmp_path)

    assert inv.argv[inv.argv.index("--input-format") + 1] == "stream-json"
    assert inv.argv[inv.argv.index("--output-format") + 1] == "stream-json"
    assert "--disable-slash-commands" in inv.argv
    assert inv.argv[inv.argv.index("--model") + 1] == "gemini-4-carbon-high"
    # Effort suffix already present on model slug -> must NOT emit conflicting --effort
    assert "--effort" not in inv.argv
    assert "--dangerously-skip-permissions" not in inv.argv
    assert "--json-schema" in inv.argv

    # Ephemeral workspace contains .agents/mcp_config.json with eager tool loading
    ws = Path(inv.cwd or "")
    assert ws == tmp_path / "workspace"
    mcp_cfg = json.loads((ws / ".agents" / "mcp_config.json").read_text(encoding="utf-8"))
    server = mcp_cfg["mcpServers"]["cosci"]
    assert server["env"]["COSCI_CAPTURE_DIR"] == str(tmp_path)
    assert server["env"]["COSCI_RECORD_TOOLS"] == "record_hypothesis"
    assert server["env"]["COSCI_AGENT"] == "generation"
    assert set(server["enabledTools"]) == {"record_hypothesis", "pubmed_search"}
    assert server["tools"]["record_hypothesis"] == {"eager": True}
    assert server["tools"]["pubmed_search"] == {"eager": True}

    # Session-only settings allow cosci MCP and deny shell/write/web built-ins
    settings_path = Path(inv.argv[inv.argv.index("--settings") + 1])
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["trustedWorkspaces"] == [str(ws)]
    assert settings["permissions"]["allow"] == ["mcp(cosci/*)"]
    assert "command(*)" in settings["permissions"]["deny"]
    assert "write_file(*)" in settings["permissions"]["deny"]
    assert "read_url(*)" in settings["permissions"]["deny"]

    # Prompt is delivered via stdin NDJSON, not argv
    msg = json.loads(inv.stdin.strip())
    assert msg["event"] == "user"
    assert "You are the generation agent." in msg["message"]["content"]
    assert "Find a hypothesis." in msg["message"]["content"]


def test_effort_is_emitted_only_for_models_without_effort_suffix(
    provider: AgyCliProvider, tmp_path: Path
) -> None:
    bare_spec = _spec(
        route=ModelRoute(
            agent="generation", mode="literature",
            model="gemini-pro-orchestrator", thinking_tokens=8000,
        )
    )
    inv = provider.build_invocation(bare_spec, _ctx(), tmp_path)
    assert inv.argv[inv.argv.index("--effort") + 1] == "high"

    suffixed_spec = _spec(
        route=ModelRoute(
            agent="generation", mode="literature",
            model="gemini-4-neon-xhigh", thinking_tokens=8000,
        )
    )
    inv2 = provider.build_invocation(suffixed_spec, _ctx(), tmp_path / "run2")
    assert "--effort" not in inv2.argv


def test_skip_permissions_and_sandbox_flags(
    provider: AgyCliProvider, agy_cfg: Config, tmp_path: Path
) -> None:
    agy_cfg.llm.agy_cli.permission_mode = "skip_permissions"
    agy_cfg.llm.agy_cli.sandbox = True
    agy_cfg.llm.agy_cli.use_json_schema = False

    inv = provider.build_invocation(_spec(), _ctx(), tmp_path)
    assert "--dangerously-skip-permissions" in inv.argv
    assert "--sandbox" in inv.argv
    assert "--json-schema" not in inv.argv


def test_large_prompt_goes_over_stdin_not_argv(
    provider: AgyCliProvider, tmp_path: Path
) -> None:
    big = "x" * 200_000
    inv = provider.build_invocation(
        _spec(user_blocks=[CachedBlock(big)]), _ctx(), tmp_path
    )
    assert big in inv.stdin
    assert not any(big in arg for arg in inv.argv)


@pytest.mark.parametrize("tokens,expected", [
    (0, "low"), (1, "low"), (4_000, "medium"), (6_000, "high"),
    (8_000, "high"), (12_000, "xhigh"), (16_000, "xhigh"), (32_000, "max"),
])
def test_effort_mapping(tokens: int, expected: str) -> None:
    assert effort_for_thinking_tokens(tokens) == expected


# --------------------------------------------------------------------------- #
# parsing & roundtrip


async def test_roundtrip_captures_mcp_record_and_urls(
    provider: AgyCliProvider, conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    from co_scientist.agents.base import BaseAgent

    _behave(
        monkeypatch,
        text="Recorded hypothesis.",
        records=[["record_hypothesis", {"title": "Mitochondrial uncoupling"}]],
        tools=[{"name": "pubmed_search", "urls": ["https://pubmed.ncbi.nlm.nih.gov/12345/"]}],
    )

    resp = await provider.call(_spec(), _ctx())

    assert BaseAgent._final_tool_use(resp, "record_hypothesis") == {
        "title": "Mitochondrial uncoupling"
    }
    assert resp.capture is not None
    assert resp.capture.seen_urls == {"https://pubmed.ncbi.nlm.nih.gov/12345/"}
    assert resp.input_tokens == 1200
    assert resp.output_tokens == 350
    assert resp.cache_read == 600
    assert resp.num_turns == 2

    async with conn.execute(
        "SELECT agent, model, input_tokens, output_tokens FROM transcripts"
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["model"] == "gemini-4-carbon-high"


async def test_structured_output_from_json_schema_normalizes_to_record(
    provider: AgyCliProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    from co_scientist.agents.base import BaseAgent

    _behave(
        monkeypatch,
        text='{"title": "Schema-enforced hypothesis"}',
        structured_output={"title": "Schema-enforced hypothesis"},
        records=[],
        tools=[{"name": "arxiv_search", "urls": ["https://arxiv.org/abs/2501.00001"]}],
    )

    resp = await provider.call(
        _spec(tool_choice={"type": "tool", "name": "record_hypothesis"}), _ctx()
    )

    assert BaseAgent._final_tool_use(resp, "record_hypothesis") == {
        "title": "Schema-enforced hypothesis"
    }
    assert resp.capture is not None
    assert resp.capture.seen_urls == {"https://arxiv.org/abs/2501.00001"}


async def test_tool_loop_captured_dispatch_with_agy_cli(
    provider: AgyCliProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    _behave(
        monkeypatch,
        structured_output={"title": "Loop hypothesis"},
        tools=[{"name": "pubmed_search", "urls": ["https://pubmed.gov/99"]}],
    )

    result = await run_tool_loop(
        provider,
        spec=_spec(tool_choice={"type": "tool", "name": "record_hypothesis"}),
        ctx=_ctx(),
        registry=MagicMock(),
        max_iters=4,
        force_terminal_tool="record_hypothesis",
    )

    assert [t["name"] for t in result.tool_calls] == ["pubmed_search", "record_hypothesis"]
    assert result.seen_urls == {"https://pubmed.gov/99"}


def test_rate_limit_and_terminal_error_classification(
    provider: AgyCliProvider, tmp_path: Path
) -> None:
    rl_stdout = json.dumps({
        "event": "result",
        "result": {
            "status": "ERROR",
            "response": "",
            "error": "Quota exceeded (429): too many requests",
        },
    })
    with pytest.raises(CliRetryableError):
        provider.parse_outcome(
            stdout=rl_stdout, stderr="", returncode=1, capture_dir=tmp_path
        )

    bad_model_stdout = json.dumps({
        "event": "result",
        "result": {
            "status": "ERROR",
            "response": "",
            "error": "invalid model selection: unknown model",
        },
    })
    with pytest.raises(CliBackendError) as err:
        provider.parse_outcome(
            stdout=bad_model_stdout, stderr="", returncode=1, capture_dir=tmp_path
        )
    assert not isinstance(err.value, CliRetryableError)
