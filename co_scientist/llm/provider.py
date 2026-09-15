"""LLMProvider — the interface every backend implements, and its factory.

Three families of backend serve the same six agents. Each takes a normalized
`AgentCallSpec` and returns an `LLMResponse` whose `.raw` is
Anthropic-Message-shaped (`.content` blocks, `.stop_reason`, `.usage`) — see
`llm/types.py` for why that shape is the contract.

**Metered API, Anthropic** (`anthropic`) — passes through to
`anthropic.AsyncAnthropic.messages.create`. The only backend with prompt-cache
breakpoints and a Batch API.

**Metered API, OpenAI-shaped** (`openai`, `openai_compatible`, and the named
presets `atlas` / `openrouter` / `gemini` / `google` / `groq` / `together` /
`mistral` / `ollama`) — translates to `openai.chat.completions.create` against the
appropriate base_url.

**Subscription CLI** (`claude_cli`, `codex_cli`) — drives a locally installed
agent harness (`claude -p`, `codex exec`) over its own OAuth login. No API key
is involved, and any that are present in the environment are stripped from the
subprocess so a call can never silently fall back to metered billing. These
run their own agentic loop, so they set `runs_own_loop` and `tool_loop.py`
reads their captured tool use rather than driving turns.

Provider-specific features:
- cache_control: honored only on Anthropic. Stripped before sending elsewhere;
  advisory under the CLI backends, whose harness manages its own cache.
- thinking / extended reasoning: Anthropic uses numeric `[thinking]` budgets;
  OpenAI reasoning models get `reasoning_effort`; the named Gemini provider
  accepts optional semantic levels under `[llm.gemini]`; the CLI backends map
  the budget onto `--effort` tiers.
- provider-owned tool-call fields: retained as opaque metadata by the
  OpenAI-compatible adapter; only provider-documented request fields are
  replayed (currently Gemini's thought-signature envelope).
- batch API: Anthropic only; the BatchPool still talks to Anthropic directly.

Users select a backend in `[llm] provider = "..."` and per-agent models in
`[models]`. Model strings are passed verbatim to the configured backend, so
they are vendor model ids for the API providers and whatever the CLI's
`--model` accepts for the CLI backends (Claude Code takes both aliases like
`"opus"` and full ids like `"claude-sonnet-4-6"`).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config import DEFAULT_PROVIDER
from .types import AgentCallSpec, CallContext, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface every backend implements."""

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> LLMResponse:
        ...


# Backends that drive a local agent CLI on a subscription login.
CLI_BACKENDS = frozenset({"claude_cli", "codex_cli"})

# Backends that call a metered HTTP API with a key.
API_PROVIDERS = frozenset({
    "anthropic",
    "openai",
    "openai_compatible",
    "atlas",          # convenience preset
    "openrouter",
    "gemini",
    "google",         # alias for "gemini"
    "groq",           # convenience preset
    "together",       # convenience preset
    "mistral",        # convenience preset
    "ollama",         # convenience preset
})

# Everything accepted in `[llm] provider`.
KNOWN_PROVIDERS = API_PROVIDERS | CLI_BACKENDS


class BackendUnavailable(RuntimeError):
    """The configured backend cannot be used (missing CLI, missing key)."""


# Built-in presets for OpenAI-compatible endpoints. `api_key_env` is the
# environment variable / cfg.secrets attribute we look up for that vendor;
# `default_headers` are extra HTTP headers passed to AsyncOpenAI for that
# vendor's accounting / attribution conventions.
OPENAI_COMPAT_PRESETS: dict[str, dict[str, str | dict[str, str] | None]] = {
    "atlas": {
        "base_url": "https://api.atlascloud.ai/v1",
        "api_key_env": "ATLASCLOUD_API_KEY",
        "default_headers_factory": None,
    },
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
        # Model ids look like "gemini-3-flash", "gemini-2.5-pro".
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
    """Construct the backend configured in `cfg.llm.provider`.

    Selection is case-insensitive. Unknown values fall back to `anthropic`
    with a warning so older configs continue to work.
    """
    from ..logging import get_logger

    log = get_logger("llm.provider")

    name = (getattr(cfg.llm, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
    if name not in KNOWN_PROVIDERS:
        log.warning("unknown_llm_provider", configured=name, fallback=DEFAULT_PROVIDER)
        name = DEFAULT_PROVIDER

    # Subscription CLI backends. Imported lazily so an API-only install never
    # pays for the subprocess machinery, and vice versa.
    if name == "claude_cli":
        from .cli_backend.claude_code import ClaudeCliProvider

        return ClaudeCliProvider(cfg, db=db, budget=budget, retry_policy=retry_policy)

    if name == "codex_cli":
        from .cli_backend.codex import CodexCliProvider

        return CodexCliProvider(cfg, db=db, budget=budget, retry_policy=retry_policy)

    if name == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg, db=db, budget=budget, retry_policy=retry_policy)

    if name in ("openai", "openai_compatible"):
        from .openai_client import OpenAIClient

        return OpenAIClient(
            cfg, db=db, budget=budget, retry_policy=retry_policy,
            compat_mode=(name == "openai_compatible"),
            provider_name=name,
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
            provider_name=name,
            preset_base_url=preset["base_url"],   # type: ignore[arg-type]
            preset_api_key_env=preset["api_key_env"],  # type: ignore[arg-type]
            default_headers=default_headers,
        )

    # Unreachable
    raise ValueError(f"unsupported LLM provider {name!r}")


# --------------------------------------------------------------------------- #
# preflight


@dataclass
class BackendStatus:
    """What `co-scientist doctor` and `init` report."""

    backend: str
    binary: str
    binary_path: str | None
    version: str | None
    authenticated: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.authenticated and (self.binary_path is not None or not self.binary)


def check_backend(cfg) -> BackendStatus:
    """Verify the configured backend is usable before a run starts.

    Deliberately cheap: for a CLI backend it probes `--version` and the login
    state, never a model call, so `doctor` costs nothing against the
    subscription. For an API provider it checks that a key is resolvable — the
    only preflight possible without spending money.
    """
    name = (getattr(cfg.llm, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
    if name not in KNOWN_PROVIDERS:
        name = DEFAULT_PROVIDER
    if name in API_PROVIDERS:
        return _check_api_provider(cfg, name)
    return _check_cli_backend(cfg, name)


def _check_api_provider(cfg, name: str) -> BackendStatus:
    """An API provider is ready when a key for its endpoint resolves.

    Key precedence (including the OPENAI_API_KEY-wins rule for the
    OpenAI-compatible presets) lives in `config.has_llm_key`; this only reports
    it, so the two cannot drift.
    """
    from ..config import has_llm_key, provider_key_env

    env_var = provider_key_env(cfg)
    authenticated = has_llm_key(cfg)
    if not env_var:
        detail = "local endpoint, no key required"
    elif authenticated:
        detail = f"key resolved for {env_var}"
    else:
        detail = f"no API key — set {env_var} (or OPENAI_API_KEY for a preset)."

    return BackendStatus(
        backend=name, binary="", binary_path=None, version=None,
        authenticated=authenticated, detail=detail,
    )


def _check_cli_backend(cfg, name: str) -> BackendStatus:
    """A CLI backend is ready when its binary is on PATH and signed in."""
    import subprocess

    if name == "codex_cli":
        binary = cfg.llm.codex_cli.binary
        auth_cmd = [binary, "login", "status"]
        auth_marker = "logged in"
    else:
        name = "claude_cli"
        binary = cfg.llm.claude_cli.binary
        auth_cmd = None
        auth_marker = ""

    path = shutil.which(binary)
    if path is None:
        return BackendStatus(
            backend=name, binary=binary, binary_path=None, version=None,
            authenticated=False,
            detail=f"{binary!r} not found on PATH — install it and sign in.",
        )

    version = _probe(subprocess, [path, "--version"])

    if auth_cmd is not None:
        status_text = _probe(subprocess, [path, *auth_cmd[1:]]) or ""
        authenticated = auth_marker in status_text.lower()
        detail = status_text or "no auth status reported"
    else:
        authenticated = _claude_signed_in()
        detail = (
            "signed in with a Claude subscription"
            if authenticated
            else "not signed in — run `claude` once and complete the login."
        )

    return BackendStatus(
        backend=name, binary=binary, binary_path=path, version=version,
        authenticated=authenticated, detail=detail,
    )


# Whether a backend's model ids are vendor-prefixed ("anthropic/claude-opus-4-7"
# vs "claude-opus-4-7"). `None` means the backend has no convention we can check
# — `openai_compatible` points wherever `[llm.openai] base_url` says.
#
# This is the one cross-cutting misconfiguration the layered config invites:
# `[models]` is deep-merged, so switching `[llm] provider` without replacing
# every key leaves ids from the previous family in place. Nothing notices until
# the first real model call, which 404s after a session has already started.
_EXPECTS_VENDOR_PREFIX: dict[str, bool | None] = {
    "atlas": True,
    "openrouter": True,
    "together": True,           # e.g. "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    "openai_compatible": None,
    "anthropic": False,
    "openai": False,
    "gemini": False,
    "google": False,
    "groq": False,
    "mistral": False,
    "ollama": False,
    "claude_cli": False,
    "codex_cli": False,
}


def check_models(cfg) -> list[str]:
    """Flag `[models]` entries that look like another provider's ids.

    Returns one message per offending entry, empty when everything matches (or
    when the backend has no naming convention to check against). A heuristic, so
    callers should warn rather than refuse to run.
    """
    name = (getattr(cfg.llm, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
    if name not in KNOWN_PROVIDERS:
        name = DEFAULT_PROVIDER
    expects_prefix = _EXPECTS_VENDOR_PREFIX.get(name)
    if expects_prefix is None:
        return []

    problems: list[str] = []
    for field, model in vars(cfg.models).items():
        if not isinstance(model, str) or not model:
            continue
        has_prefix = "/" in model
        if has_prefix and not expects_prefix:
            problems.append(
                f"models.{field} = {model!r} is vendor-prefixed, which "
                f"{name!r} does not accept — drop the prefix"
            )
        elif not has_prefix and expects_prefix:
            problems.append(
                f"models.{field} = {model!r} has no vendor prefix, which "
                f"{name!r} requires — e.g. 'anthropic/{model}'"
            )
    return problems


def _claude_signed_in() -> bool:
    """Detect a Claude Code subscription session without spending quota.

    Claude Code has no `login status` subcommand, and where it keeps the OAuth
    token is platform-dependent: the macOS keychain, or a credentials file
    elsewhere. `~/.claude.json` carries an `oauthAccount` record on every
    platform once login completes, so that is the portable signal; the file
    check covers installs that store credentials on disk.
    """
    import json
    from pathlib import Path

    config = Path.home() / ".claude.json"
    if config.exists():
        try:
            if json.loads(config.read_text(encoding="utf-8")).get("oauthAccount"):
                return True
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return (Path.home() / ".claude" / ".credentials.json").exists()


def _probe(subprocess_mod, argv: list[str]) -> str | None:
    try:
        out = subprocess_mod.run(
            argv, capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, ValueError, subprocess_mod.SubprocessError):
        return None
    return (out.stdout or out.stderr or "").strip() or None
