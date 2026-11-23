"""Tests for the provider-aware API-key checker.

`has_llm_key(cfg)` decides whether the configured provider has the credentials
it needs. The CLI uses this to gate `run` / `resume` so users don't get
mysterious errors at first call.
"""

from __future__ import annotations

from co_scientist.config import Config, has_llm_key, provider_key_env


def _cfg(provider: str) -> Config:
    cfg = Config()
    cfg.llm.provider = provider
    cfg.secrets.ANTHROPIC_API_KEY = ""
    cfg.secrets.OPENAI_API_KEY = ""
    cfg.secrets.OPENROUTER_API_KEY = ""
    cfg.secrets.GEMINI_API_KEY = ""
    cfg.secrets.GROQ_API_KEY = ""
    cfg.secrets.TOGETHER_API_KEY = ""
    cfg.secrets.MISTRAL_API_KEY = ""
    return cfg


def test_anthropic_provider_requires_anthropic_key() -> None:
    cfg = _cfg("anthropic")
    assert not has_llm_key(cfg)
    cfg.secrets.ANTHROPIC_API_KEY = "sk-fake"
    assert has_llm_key(cfg)


def test_openrouter_provider_uses_openrouter_key() -> None:
    cfg = _cfg("openrouter")
    assert not has_llm_key(cfg)
    cfg.secrets.OPENROUTER_API_KEY = "sk-or-fake"
    assert has_llm_key(cfg)


def test_openrouter_provider_accepts_openai_api_key_as_override() -> None:
    """OPENAI_API_KEY is the universal override for OpenAI-compat presets."""
    cfg = _cfg("openrouter")
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    assert has_llm_key(cfg)


def test_gemini_provider_uses_gemini_key() -> None:
    cfg = _cfg("gemini")
    assert not has_llm_key(cfg)
    cfg.secrets.GEMINI_API_KEY = "gemini-fake"
    assert has_llm_key(cfg)


def test_google_alias_uses_gemini_key() -> None:
    cfg = _cfg("google")
    cfg.secrets.GEMINI_API_KEY = "gemini-fake"
    assert has_llm_key(cfg)


def test_ollama_is_keyless() -> None:
    cfg = _cfg("ollama")
    assert has_llm_key(cfg)


def test_agy_cli_is_keyless() -> None:
    cfg = _cfg("agy_cli")
    assert provider_key_env(cfg) == ""
    assert has_llm_key(cfg)


def test_provider_key_env_returns_expected_var_name() -> None:
    cfg = _cfg("openrouter")
    assert provider_key_env(cfg) == "OPENROUTER_API_KEY"
    cfg.llm.provider = "anthropic"
    assert provider_key_env(cfg) == "ANTHROPIC_API_KEY"
    cfg.llm.provider = "ollama"
    assert provider_key_env(cfg) == ""


def test_anthropic_provider_does_not_accept_openai_key() -> None:
    """OPENAI_API_KEY shouldn't satisfy the Anthropic provider check."""
    cfg = _cfg("anthropic")
    cfg.secrets.OPENAI_API_KEY = "sk-fake"
    assert not has_llm_key(cfg)


# ---------------------- models vs provider family ---------------------- #


def test_default_models_match_the_default_provider() -> None:
    from co_scientist.llm.provider import check_models

    assert check_models(Config()) == []


def test_openrouter_ids_are_flagged_under_a_single_vendor_provider() -> None:
    """The layered-config trap: `[models]` is deep-merged, so switching provider
    without replacing every key leaves the previous family's ids in place. A
    live `claude -p --model anthropic/claude-opus-4-7` 404s, but only once the
    session is already running."""
    from co_scientist.llm.provider import check_models

    cfg = Config()
    cfg.llm.provider = "claude_cli"
    problems = check_models(cfg)
    assert len(problems) == len(vars(cfg.models))
    assert "vendor-prefixed" in problems[0]
    assert "claude_cli" in problems[0]


def test_bare_ids_are_flagged_under_openrouter() -> None:
    from co_scientist.llm.provider import check_models

    cfg = Config()
    cfg.models.generation = "claude-opus-4-7"
    problems = check_models(cfg)
    assert len(problems) == 1
    assert "no vendor prefix" in problems[0]


def test_together_expects_a_prefix_too() -> None:
    """Together's ids look like 'meta-llama/Llama-3.3-70B-Instruct-Turbo'."""
    from co_scientist.llm.provider import check_models

    cfg = Config()
    cfg.llm.provider = "together"
    assert check_models(cfg) == []


def test_openai_compatible_has_no_convention_to_check() -> None:
    """It points at whatever `[llm.openai] base_url` says, so stay quiet."""
    from co_scientist.llm.provider import check_models

    cfg = Config()
    cfg.llm.provider = "openai_compatible"
    cfg.models.generation = "some-local-gguf"
    assert check_models(cfg) == []
