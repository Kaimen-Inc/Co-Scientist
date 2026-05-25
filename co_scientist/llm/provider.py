"""LLMProvider — vendor-agnostic LLM client interface.

The co-scientist began as an Anthropic-only system; the type hints and
intermediate request shapes (`AgentCallSpec`, `CachedBlock`) are
Anthropic-flavored. Rather than rewrite every agent we treat those types as
the canonical normalized form: each provider takes a normalized spec, calls
its vendor SDK, and returns an `AnthropicResponse` whose `.raw` exposes a
Message-like object with `.content`, `.stop_reason`, `.usage`.

Concretely:
- AnthropicProvider: passes through to anthropic.AsyncAnthropic.messages.create.
- OpenAIProvider: translates to openai.chat.completions.create (also supports
  arbitrary OpenAI-compatible base_urls: Groq, Together, OpenRouter, Mistral,
  Ollama, Gemini OpenAI-compat endpoint).

Provider-specific features:
- cache_control: honored only on Anthropic. Stripped before sending elsewhere.
- thinking / extended reasoning: Anthropic for Claude opus; on OpenAI we
  translate to `reasoning_effort` for o-series models, else drop.
- batch API: Anthropic only; the BatchPool still talks to Anthropic directly.

Users select a provider in `[llm] provider = "..."` and per-agent models in
`[models]`. Model strings are passed verbatim to the configured provider.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .anthropic_client import AgentCallSpec, AnthropicResponse, CallContext


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface every LLM client implements."""

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> AnthropicResponse:
        ...


# Provider names accepted in config.
KNOWN_PROVIDERS = frozenset({
    "anthropic",
    "openai",
    "openai_compatible",
    "openrouter",
    "gemini",
    "google",         # alias for "gemini"
    "groq",           # convenience preset
    "together",       # convenience preset
    "mistral",        # convenience preset
    "ollama",         # convenience preset
})


# Built-in presets for OpenAI-compatible endpoints. `api_key_env` is the
# environment variable / cfg.secrets attribute we look up for that vendor;
# `default_headers` are extra HTTP headers passed to AsyncOpenAI for that
# vendor's accounting / attribution conventions.
OPENAI_COMPAT_PRESETS: dict[str, dict[str, str | dict[str, str] | None]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        # OpenRouter recommends Referer + X-Title for attribution.
        # Override via [llm.openrouter] referer = "..." / title = "...".
        "default_headers_factory": "_openrouter_headers",
    },
    "gemini": {
        # Google's Gemini OpenAI-compat endpoint. Speaks chat.completions,
        # accepts tools/function calling, and tracks the same usage shape.
        # Model ids look like "gemini-2.5-pro", "gemini-2.5-flash".
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_headers_factory": None,
    },
    "google": {  # alias for gemini
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_headers_factory": None,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_headers_factory": None,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_headers_factory": None,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "default_headers_factory": None,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",   # usually unset; allow blank
        "default_headers_factory": None,
    },
}


def _openrouter_headers(cfg) -> dict[str, str]:
    """Build OpenRouter attribution headers from `[llm.openrouter]` config."""
    h: dict[str, str] = {}
    or_cfg = getattr(cfg.llm, "openrouter", None)
    if or_cfg is None:
        return h
    referer = getattr(or_cfg, "referer", "") or ""
    title = getattr(or_cfg, "title", "") or ""
    if referer:
        h["HTTP-Referer"] = referer
    if title:
        h["X-Title"] = title
    return h


def get_provider(
    cfg,
    *,
    db,
    budget,
    retry_policy=None,
) -> LLMProvider:
    """Construct the LLM provider configured in `cfg.llm.provider`.

    Selection is case-insensitive. Unknown values fall back to `anthropic`
    with a warning so older configs continue to work.
    """
    from ..logging import get_logger

    log = get_logger("llm.provider")

    name = (getattr(cfg.llm, "provider", "anthropic") or "anthropic").strip().lower()
    if name not in KNOWN_PROVIDERS:
        log.warning("unknown_llm_provider", configured=name, fallback="anthropic")
        name = "anthropic"

    if name == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg, db=db, budget=budget, retry_policy=retry_policy)

    if name in ("openai", "openai_compatible"):
        from .openai_client import OpenAIClient

        return OpenAIClient(
            cfg, db=db, budget=budget, retry_policy=retry_policy,
            compat_mode=(name == "openai_compatible"),
        )

    # Named OpenAI-compat preset (openrouter, gemini, groq, together, ...).
    preset = OPENAI_COMPAT_PRESETS.get(name)
    if preset is not None:
        from .openai_client import OpenAIClient

        # Headers come from a named factory so we can keep the table JSON-ish.
        headers_factory = preset.get("default_headers_factory")
        default_headers: dict[str, str] = {}
        if headers_factory == "_openrouter_headers":
            default_headers = _openrouter_headers(cfg)

        return OpenAIClient(
            cfg,
            db=db, budget=budget, retry_policy=retry_policy,
            compat_mode=True,
            preset_base_url=preset["base_url"],   # type: ignore[arg-type]
            preset_api_key_env=preset["api_key_env"],  # type: ignore[arg-type]
            default_headers=default_headers,
        )

    # Unreachable
    raise ValueError(f"unsupported LLM provider {name!r}")
