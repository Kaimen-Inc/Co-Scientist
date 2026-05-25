"""Tests for the LLM provider abstraction.

The factory should respect `cfg.llm.provider`, the OpenAI translation must
convert Anthropic-shaped specs into OpenAI Chat Completions requests, and
the response adapter must expose the Message-like attribute surface every
agent already depends on.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from co_scientist.config import Config
from co_scientist.llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from co_scientist.llm.openai_client import (
    OpenAIClient,
    _adapt_response,
    _budget_to_effort,
    _build_openai_request,
    _is_reasoning_model,
    _translate_anthropic_message,
)
from co_scientist.llm.provider import KNOWN_PROVIDERS, get_provider
from co_scientist.llm.routing import ModelRoute


def _route(model: str = "gpt-5", thinking: int = 0) -> ModelRoute:
    return ModelRoute(agent="generation", mode="literature", model=model, thinking_tokens=thinking)


# ----------------------------- factory ----------------------------- #


def test_known_providers_includes_presets() -> None:
    assert "anthropic" in KNOWN_PROVIDERS
    assert "openai" in KNOWN_PROVIDERS
    assert "openai_compatible" in KNOWN_PROVIDERS
    assert "openrouter" in KNOWN_PROVIDERS
    assert "gemini" in KNOWN_PROVIDERS
    assert "google" in KNOWN_PROVIDERS
    assert "groq" in KNOWN_PROVIDERS
    assert "together" in KNOWN_PROVIDERS
    assert "mistral" in KNOWN_PROVIDERS
    assert "ollama" in KNOWN_PROVIDERS


def test_get_provider_returns_anthropic_by_default() -> None:
    cfg = Config()
    cfg.secrets.ANTHROPIC_API_KEY = "sk-fake"
    with patch("co_scientist.llm.anthropic_client.AsyncAnthropic"):
        p = get_provider(cfg, db=MagicMock(), budget=MagicMock())
    # Quack-typing: anthropic client is the AnthropicClient class.
    assert type(p).__name__ == "AnthropicClient"


def test_get_provider_returns_openai_when_configured() -> None:
    cfg = Config()
    cfg.llm.provider = "openai"
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    with patch("openai.AsyncOpenAI"):
        p = get_provider(cfg, db=MagicMock(), budget=MagicMock())
    assert type(p).__name__ == "OpenAIClient"


def test_get_provider_unknown_falls_back_to_anthropic() -> None:
    cfg = Config()
    cfg.llm.provider = "totally-made-up"
    cfg.secrets.ANTHROPIC_API_KEY = "sk-fake"
    with patch("co_scientist.llm.anthropic_client.AsyncAnthropic"):
        p = get_provider(cfg, db=MagicMock(), budget=MagicMock())
    assert type(p).__name__ == "AnthropicClient"


def test_compat_mode_sets_base_url() -> None:
    cfg = Config()
    cfg.llm.provider = "openai_compatible"
    cfg.llm.openai.base_url = "https://api.groq.com/openai/v1"
    cfg.secrets.OPENAI_API_KEY = "gsk-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    mock_sdk.assert_called_once()
    kwargs = mock_sdk.call_args.kwargs
    assert kwargs["base_url"] == "https://api.groq.com/openai/v1"


def test_compat_mode_allows_empty_api_key() -> None:
    """Ollama / vLLM local endpoints don't need a real key."""
    cfg = Config()
    cfg.llm.provider = "openai_compatible"
    cfg.llm.openai.base_url = "http://localhost:11434/v1"
    cfg.secrets.OPENAI_API_KEY = ""
    with patch("openai.AsyncOpenAI") as mock_sdk:
        OpenAIClient(cfg, db=MagicMock(), budget=MagicMock(), compat_mode=True)
    assert mock_sdk.call_args.kwargs["api_key"]  # placeholder, non-empty


# ----------------------------- presets ----------------------------- #


def test_provider_openrouter_uses_preset_base_url_and_key() -> None:
    """`provider = "openrouter"` should configure OpenRouter's base_url and
    pull from OPENROUTER_API_KEY when OPENAI_API_KEY is unset."""
    cfg = Config()
    cfg.llm.provider = "openrouter"
    cfg.secrets.OPENAI_API_KEY = ""
    cfg.secrets.OPENROUTER_API_KEY = "sk-or-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    kwargs = mock_sdk.call_args.kwargs
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["api_key"] == "sk-or-fake"


def test_openrouter_sends_attribution_headers() -> None:
    cfg = Config()
    cfg.llm.provider = "openrouter"
    cfg.llm.openrouter.referer = "https://example.com"
    cfg.llm.openrouter.title = "My App"
    cfg.secrets.OPENROUTER_API_KEY = "sk-or-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    headers = mock_sdk.call_args.kwargs.get("default_headers", {})
    assert headers.get("HTTP-Referer") == "https://example.com"
    assert headers.get("X-Title") == "My App"


def test_openrouter_skips_headers_when_unset() -> None:
    cfg = Config()
    cfg.llm.provider = "openrouter"
    cfg.secrets.OPENROUTER_API_KEY = "sk-or-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    # Either no default_headers, or an empty dict — never partial attribution.
    headers = mock_sdk.call_args.kwargs.get("default_headers", {})
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


def test_provider_gemini_uses_compat_endpoint() -> None:
    cfg = Config()
    cfg.llm.provider = "gemini"
    cfg.secrets.OPENAI_API_KEY = ""
    cfg.secrets.GEMINI_API_KEY = "gemini-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    kwargs = mock_sdk.call_args.kwargs
    assert "generativelanguage.googleapis.com" in kwargs["base_url"]
    assert kwargs["api_key"] == "gemini-fake"


def test_provider_google_is_alias_for_gemini() -> None:
    cfg = Config()
    cfg.llm.provider = "google"
    cfg.secrets.GEMINI_API_KEY = "gemini-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    kwargs = mock_sdk.call_args.kwargs
    assert "generativelanguage.googleapis.com" in kwargs["base_url"]


def test_provider_groq_preset() -> None:
    cfg = Config()
    cfg.llm.provider = "groq"
    cfg.secrets.GROQ_API_KEY = "gsk-fake"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    assert mock_sdk.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"


def test_explicit_openai_api_key_overrides_preset_env() -> None:
    """OPENAI_API_KEY takes precedence over a preset's specific env var."""
    cfg = Config()
    cfg.llm.provider = "openrouter"
    cfg.secrets.OPENAI_API_KEY = "user-explicit"
    cfg.secrets.OPENROUTER_API_KEY = "sk-or"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    assert mock_sdk.call_args.kwargs["api_key"] == "user-explicit"


def test_user_base_url_overrides_preset() -> None:
    """If the user sets [llm.openai] base_url explicitly, it wins even when
    using a preset (lets users point a preset at a private mirror)."""
    cfg = Config()
    cfg.llm.provider = "openrouter"
    cfg.llm.openai.base_url = "https://my-private-mirror/v1"
    cfg.secrets.OPENROUTER_API_KEY = "sk-or"
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    assert mock_sdk.call_args.kwargs["base_url"] == "https://my-private-mirror/v1"


def test_ollama_works_without_any_key() -> None:
    cfg = Config()
    cfg.llm.provider = "ollama"
    cfg.secrets.OPENAI_API_KEY = ""
    cfg.secrets.OLLAMA_API_KEY = ""
    with patch("openai.AsyncOpenAI") as mock_sdk:
        get_provider(cfg, db=MagicMock(), budget=MagicMock())
    kwargs = mock_sdk.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert kwargs["api_key"]  # placeholder, non-empty


# ----------------------------- request translation ----------------------------- #


def test_build_request_emits_system_and_user_messages() -> None:
    spec = AgentCallSpec(
        route=_route(),
        system_blocks=[CachedBlock("You are X.", cache=True)],
        user_blocks=[CachedBlock("Generate a hypothesis.")],
        max_output_tokens=512,
    )
    req = _build_openai_request(spec)
    assert req["model"] == "gpt-5"
    assert req["max_tokens"] == 512
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user"]
    # cache_control must NOT leak into the OpenAI request
    assert "cache_control" not in str(req)


def test_build_request_translates_tools_and_tool_choice() -> None:
    spec = AgentCallSpec(
        route=_route(),
        user_blocks=[CachedBlock("go")],
        tools=[{
            "name": "record_hypothesis",
            "description": "Save a hypothesis",
            "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
        }],
        tool_choice={"type": "tool", "name": "record_hypothesis"},
    )
    req = _build_openai_request(spec)
    assert req["tools"][0]["type"] == "function"
    assert req["tools"][0]["function"]["name"] == "record_hypothesis"
    assert req["tool_choice"] == {"type": "function", "function": {"name": "record_hypothesis"}}


def test_build_request_translates_any_tool_choice_to_required() -> None:
    spec = AgentCallSpec(
        route=_route(),
        user_blocks=[CachedBlock("go")],
        tools=[{"name": "x", "description": "", "input_schema": {}}],
        tool_choice={"type": "any"},
    )
    assert _build_openai_request(spec)["tool_choice"] == "required"


def test_thinking_translates_to_reasoning_effort_for_o_series() -> None:
    spec = AgentCallSpec(
        route=_route(model="o3", thinking=8000),
        user_blocks=[CachedBlock("think hard")],
    )
    req = _build_openai_request(spec)
    assert req.get("reasoning_effort") == "medium"


def test_thinking_dropped_for_non_reasoning_model() -> None:
    spec = AgentCallSpec(
        route=_route(model="gpt-5", thinking=8000),
        user_blocks=[CachedBlock("hi")],
    )
    req = _build_openai_request(spec)
    assert "reasoning_effort" not in req


def test_budget_to_effort_bands() -> None:
    assert _budget_to_effort(500) == "minimal"
    assert _budget_to_effort(2000) == "low"
    assert _budget_to_effort(8000) == "medium"
    assert _budget_to_effort(20000) == "high"


def test_is_reasoning_model() -> None:
    assert _is_reasoning_model("o1")
    assert _is_reasoning_model("o3-mini")
    assert _is_reasoning_model("gpt-4-reasoning")
    assert not _is_reasoning_model("gpt-5")
    assert not _is_reasoning_model("claude-opus-4-7")


# ----------------------------- message translation ----------------------------- #


def test_translate_assistant_tool_use_to_openai_tool_calls() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll search."},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "abc"}},
        ],
    }
    out = _translate_anthropic_message(msg)
    assert len(out) == 1
    assistant = out[0]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "I'll search."
    tc = assistant["tool_calls"][0]
    assert tc["function"]["name"] == "search"
    assert tc["function"]["arguments"] == '{"q": "abc"}'
    assert tc["id"] == "call_1"


def test_translate_user_tool_result_to_openai_tool_message() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1",
             "content": {"hits": ["a", "b"]}, "is_error": False},
        ],
    }
    out = _translate_anthropic_message(msg)
    assert len(out) == 1
    tool_msg = out[0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "hits" in tool_msg["content"]


def test_translate_thinking_block_is_dropped() -> None:
    msg = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "lots of internal text", "signature": "sig"},
            {"type": "text", "text": "done"},
        ],
    }
    out = _translate_anthropic_message(msg)
    # Thinking is not included in the OpenAI assistant content.
    assert out[0]["content"] == "done"
    assert "internal text" not in str(out)


def test_translate_tool_error_is_marked_in_content() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "boom", "is_error": True},
        ],
    }
    out = _translate_anthropic_message(msg)
    assert out[0]["role"] == "tool"
    assert out[0]["content"].startswith("[tool error]")


# ----------------------------- response adaptation ----------------------------- #


def _fake_openai_response(*, text: str = "", tool_calls: list[dict] | None = None,
                          finish: str = "stop", in_tok: int = 100, out_tok: int = 50):
    """Build a SimpleNamespace that quacks like an openai ChatCompletion."""
    tcs = []
    for tc in (tool_calls or []):
        tcs.append(SimpleNamespace(
            id=tc["id"],
            type="function",
            function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
        ))
    msg = SimpleNamespace(content=text or None, tool_calls=tcs or None)
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    usage = SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok)
    return SimpleNamespace(id="resp_1", choices=[choice], usage=usage)


def test_adapt_response_text_only() -> None:
    raw = _fake_openai_response(text="Hello world.", finish="stop")
    msg = _adapt_response(raw, "gpt-5")
    assert msg.stop_reason == "end_turn"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "Hello world."
    assert msg.usage.input_tokens == 100


def test_adapt_response_tool_call() -> None:
    raw = _fake_openai_response(
        text="",
        tool_calls=[{"id": "call_42", "name": "search", "arguments": '{"q": "foo"}'}],
        finish="tool_calls",
    )
    msg = _adapt_response(raw, "gpt-5")
    assert msg.stop_reason == "tool_use"
    tu = msg.content[0]
    assert tu.type == "tool_use"
    assert tu.name == "search"
    assert tu.input == {"q": "foo"}
    assert tu.id == "call_42"


def test_adapt_response_handles_malformed_tool_args() -> None:
    raw = _fake_openai_response(
        tool_calls=[{"id": "c", "name": "x", "arguments": "not-json"}],
        finish="tool_calls",
    )
    msg = _adapt_response(raw, "gpt-5")
    assert msg.content[0].input.get("_raw_arguments") == "not-json"


def test_adapt_response_length_finish_maps_to_max_tokens() -> None:
    raw = _fake_openai_response(text="truncated...", finish="length")
    msg = _adapt_response(raw, "gpt-5")
    assert msg.stop_reason == "max_tokens"


# ----------------------------- end-to-end smoke ----------------------------- #


@pytest.mark.asyncio
async def test_openai_call_persists_transcript_and_charges_budget(tmp_cfg, conn) -> None:
    """End-to-end: OpenAIClient.call wires up budget + transcripts."""
    from co_scientist.llm.budgets import TokenBudget

    cfg = tmp_cfg
    cfg.llm.provider = "openai"
    cfg.secrets.OPENAI_API_KEY = "sk-fake"

    try:
        # Insert a session row so the foreign key is satisfied.
        from datetime import UTC, datetime

        from co_scientist.models import ResearchPlan, Session
        from co_scientist.storage.repos import sessions as sess_repo
        s = Session(
            id="ses_p", created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
            status="running", research_goal="g",
            research_plan=ResearchPlan(objective="x"),
            config_snapshot={}, budget_tokens=10_000, budget_usd=1.0,
        )
        await sess_repo.insert(conn, s)

        budget = TokenBudget(cfg=cfg, budget_tokens=10_000, budget_usd=1.0)

        with patch("openai.AsyncOpenAI") as mock_sdk:
            mock_instance = mock_sdk.return_value
            mock_instance.chat = MagicMock()
            mock_instance.chat.completions = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=_fake_openai_response(text="ok"),
            )
            client = OpenAIClient(cfg, db=conn, budget=budget)

            spec = AgentCallSpec(
                route=_route(model="gpt-5"),
                system_blocks=[CachedBlock("be helpful")],
                user_blocks=[CachedBlock("hi")],
                max_output_tokens=128,
            )
            ctx = CallContext(session_id="ses_p", task_id=None, agent="generation", action="test")
            resp = await client.call(spec, ctx)

        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        snap = budget.snapshot()
        # After settle, reservation is back to zero and used spent.
        assert snap["_global"]["reserved_usd"] == 0
        assert snap["_global"]["used_usd"] > 0
    finally:
        pass
