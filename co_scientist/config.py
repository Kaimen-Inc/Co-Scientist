"""Configuration loader.

Layered: config/default.toml → ~/.co-scientist/config.toml → ./co-scientist.toml → env.
Secrets come from environment variables only.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"

# The backend used when `[llm] provider` is absent, blank, or unrecognized.
# Single-sourced because `ModelsCfg`'s defaults only make sense for one family:
# if this and those ever disagree, the out-of-the-box config sends model ids to
# an endpoint that rejects them.
DEFAULT_PROVIDER = "openrouter"


class RunCfg(BaseModel):
    concurrency: int = 4
    max_ideas: int = 60
    max_matches_per_idea: int = 12
    wall_clock_seconds: int = 7200
    budget_tokens: int = 5_000_000
    budget_usd: float = 25.0


class StorageCfg(BaseModel):
    data_dir: str = "./data"


class ScienceSkillsCfg(BaseModel):
    path: str = "./vendor/science-skills"
    pinned_commit: str = ""


class EmbeddingsCfg(BaseModel):
    """Embeddings are chosen independently of the reasoning backend.

    No agent CLI exposes an embedding endpoint, so under `claude_cli` /
    `codex_cli` this is the one place a hosted key is still useful. With no key
    at all it degrades to the local hash embedder rather than failing — a
    session with weaker dedup beats no session.

    Changing `dim` invalidates every FAISS index already on disk; the store
    detects the mismatch and rebuilds rather than corrupting the index.
    """

    provider: str = "voyage"
    model: str = "voyage-3-large"
    dim: int = 1024


class VectorsCfg(BaseModel):
    backend: str = "faiss"
    dedup_cosine_threshold: float = 0.92
    cluster_threshold: float = 0.15
    full_recluster_every_matches: int = 20


class RankingCfg(BaseModel):
    k_factor_new: int = 32
    k_factor_warm: int = 16
    elo_initial: int = 1200
    debate_when_matches_lt: int = 2
    debate_when_elo_delta_lt: int = 50
    batch_below_decile: bool = True
    batch_submit_every_seconds: int = 1800
    p_new: float = 0.4
    p_close: float = 0.4
    p_random: float = 0.2


class TerminationCfg(BaseModel):
    elo_stability_k: int = 5
    elo_stability_n: int = 3
    elo_stability_eps: float = 25.0
    match_snapshot_every: int = 10
    # Guards that prevent elo_stable from firing on a small pool.
    # Defaults of 0 preserve the original behaviour.
    min_ideas_before_stable: int = 0
    min_matches_before_stable: int = 0


class EvolutionCfg(BaseModel):
    """Controls when the idle-refinement loop triggers evolution."""
    # Minimum number of *mature* hypotheses (>= 3 matches played) required
    # before the supervisor schedules an EvolveTopHypotheses task.
    # Default matches the original hardcoded value.
    min_mature: int = 20
    # How many top hypotheses to evolve per idle pass.
    top_k: int = 5


class BudgetSharesCfg(BaseModel):
    generation: float = 0.20
    reflection: float = 0.20
    ranking: float = 0.25
    evolution: float = 0.15
    metareview: float = 0.10
    proximity: float = 0.02
    reserve: float = 0.08


class ModelsCfg(BaseModel):
    """Per-agent model selection, interpreted by the configured backend.

    Values go verbatim to the vendor SDK, or to the CLI's `--model` flag, so
    they must match the family in `[llm] provider`. The defaults are OpenRouter
    ids to match the default provider; every one is in
    `routing.OPENROUTER_PRICE_TABLE`, so cost estimates and the degrade chain
    both resolve.

    Two tiers is the whole design — a stronger model for the reasoning-heavy
    agents and a cheaper one for the rest. Switching families means replacing
    all of these: `anthropic` wants "claude-opus-4-7", `gemini` wants
    "gemini-3-pro", `claude_cli` takes either that or an alias like "opus",
    `codex_cli` wants a Codex model id.

    Because config is deep-merged, a key left behind after such a switch keeps
    its default and gets sent verbatim to the new endpoint. `provider.check_models`
    flags that in preflight — otherwise nothing notices until the first model
    call 404s, with a session already running.
    """

    parse_goal: str = "anthropic/claude-sonnet-4-6"
    generation: str = "anthropic/claude-opus-4-7"
    reflection: str = "anthropic/claude-opus-4-7"
    evolution: str = "anthropic/claude-opus-4-7"
    ranking_pairwise: str = "anthropic/claude-sonnet-4-6"
    ranking_debate: str = "anthropic/claude-sonnet-4-6"
    ranking_priority: str = "anthropic/claude-opus-4-7"
    metareview_feedback: str = "anthropic/claude-sonnet-4-6"
    metareview_final: str = "anthropic/claude-opus-4-7"
    classifier: str = "anthropic/claude-haiku-4.5"
    judge: str = "anthropic/claude-sonnet-4-6"


class ThinkingCfg(BaseModel):
    generation_literature: int = 4000
    generation_debate: int = 8000
    reflection_full: int = 0
    reflection_verification: int = 12000
    reflection_observation: int = 6000
    ranking_pairwise: int = 4000
    ranking_debate: int = 8000
    evolution_combine: int = 6000
    evolution_out_of_box: int = 6000
    evolution_feasibility: int = 0
    evolution_simplify: int = 0
    metareview_feedback: int = 8000
    metareview_final: int = 16000


class ToolLoopCfg(BaseModel):
    generation_max_iters: int = 8
    reflection_max_iters: int = 8
    ranking_max_iters: int = 3
    evolution_max_iters: int = 6
    metareview_max_iters: int = 12
    parallel_cap: int = 4
    tool_timeout_seconds: int = 30


class RetryCfg(BaseModel):
    max_attempts_429: int = 6
    max_attempts_529: int = 8
    max_attempts_5xx: int = 5
    max_attempts_timeout: int = 3
    base_ms: int = 1000
    cap_ms: int = 60_000
    per_call_timeout_seconds: int = 120
    per_call_timeout_thinking_seconds: int = 300


class LeaseCfg(BaseModel):
    default_seconds: int = 300
    reflection_seconds: int = 600
    metareview_final_seconds: int = 1800
    heartbeat_seconds: int = 60
    max_attempts: int = 3


class WebSearchCfg(BaseModel):
    provider: str = "tavily"
    max_results: int = 8


class WebFetchCfg(BaseModel):
    max_bytes: int = 5_000_000
    timeout_seconds: int = 30
    user_agent: str = "co-scientist/0.1"


class CodeExecCfg(BaseModel):
    provider: str = "local"
    local_cpu_seconds: int = 30
    local_mem_mb: int = 512


class SafetyCfg(BaseModel):
    enable_classifier: bool = True
    enable_citation_verifier: bool = True
    classifier_block_categories: list[str] = Field(
        default_factory=lambda: ["cbrn", "csam", "weapons", "illicit_synthesis"]
    )
    classifier_warn_categories: list[str] = Field(default_factory=lambda: ["dual_use_bio"])


class OpenAIProviderCfg(BaseModel):
    """OpenAI / OpenAI-compatible endpoint settings.

    `base_url` overrides the SDK default. Use it to point at any
    OpenAI-compatible provider (Groq, Together, OpenRouter, Mistral,
    Gemini OpenAI-compat, Ollama local, vLLM, ...). When a named preset
    such as `provider = "openrouter"` is used, this only needs to be set
    if you want to override the preset's base_url.
    """

    base_url: str | None = None


class AnthropicProviderCfg(BaseModel):
    """Anthropic provider settings. `base_url` is rarely used; honored if set."""

    base_url: str | None = None


class OpenRouterProviderCfg(BaseModel):
    """OpenRouter attribution headers.

    OpenRouter ranks apps in its catalog by `HTTP-Referer` + `X-Title`.
    Setting these is optional but recommended for production traffic;
    leave blank for ad-hoc use.
    """

    referer: str = ""
    title: str = ""


class ClaudeCliCfg(BaseModel):
    """Settings for the `claude -p` backend (Claude Code, subscription auth)."""

    binary: str = "claude"
    permission_mode: str = "dontAsk"
    timeout_seconds: int = 900
    max_parallel: int = 3
    """Concurrent CLI processes. Bounded by the subscription's rate limit
    rather than by CPU, so this is deliberately lower than `run.concurrency`."""
    replace_system_prompt: bool = True
    """Pass `--system-prompt` (replace) rather than `--append-system-prompt`.

    Replacing drops Claude Code's own scaffolding, which otherwise costs about
    19k cache-creation plus 24k cache-read tokens on every single call."""


class CodexCliCfg(BaseModel):
    """Settings for the `codex exec` backend (Codex, ChatGPT subscription)."""

    binary: str = "codex"
    sandbox: str = "read-only"
    timeout_seconds: int = 900
    max_parallel: int = 3


ThinkingLevel = Literal["default", "minimal", "low", "medium", "high"]


class GeminiProviderCfg(BaseModel):
    """Gemini reasoning controls for the OpenAI-compatible endpoint.

    ``default`` omits ``reasoning_effort`` and lets the selected Gemini model
    use its own default. Per-mode overrides use routing keys such as
    ``generation.literature`` or ``metareview.final``.
    """

    thinking_level: ThinkingLevel = "default"
    thinking_by_mode: dict[str, ThinkingLevel] = Field(default_factory=dict)


class LLMCfg(BaseModel):
    """Choose what backs the agents: a metered API, or a subscription CLI.

    **Subscription CLI** — drives a locally installed agent harness over its
    own OAuth login. No API key is involved, and any present in the
    environment are stripped from the subprocess so a call cannot silently
    fall back to metered billing.

    - "claude_cli" — `claude -p` (Claude Code). `[models]` values are passed
      to `--model`, so both aliases ("opus", "sonnet", "haiku") and full ids
      ("claude-sonnet-4-6") work. Tune under [llm.claude_cli].
    - "codex_cli" — `codex exec` (Codex). `[models]` values are Codex model
      ids; a ChatGPT account must be entitled to the model you pick.
      Tune under [llm.codex_cli].

    **Metered API** — a vendor SDK with a per-token bill.

    - "openrouter" — OpenRouter (openrouter.ai), the default. 200+ models from
      every major vendor behind one key, which is what makes per-agent
      multi-vendor routing possible in a single session. Set OPENROUTER_API_KEY
      (or OPENAI_API_KEY). Optional attribution in [llm.openrouter].
    - "anthropic" — Claude via the official Anthropic SDK. Cache breakpoints,
      extended thinking, and the Batch API are only available under this
      provider.
    - "openai" — OpenAI Chat Completions. Extended reasoning is translated
      to `reasoning_effort` for the o-series models; cache breakpoints are
      stripped.
    - "gemini" / "google" — Google Gemini via the official OpenAI-compat
      endpoint. Set GEMINI_API_KEY. Thought signatures are preserved
      automatically. Optional semantic reasoning levels are configured under
      `llm.gemini`.
    - "atlas", "groq", "together", "mistral", "ollama" — convenience presets for
      those endpoints; each reads its own API key env var
      (ATLASCLOUD_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY).
    - "openai_compatible" — same client as `openai` but allows
      `llm.openai.base_url` to point at any other OpenAI-compatible
      endpoint not yet covered by a preset.

    Note that `[models]` is interpreted by whichever backend is selected, so
    switching families generally means revisiting it.
    """

    provider: str = DEFAULT_PROVIDER
    openai: OpenAIProviderCfg = Field(default_factory=OpenAIProviderCfg)
    anthropic: AnthropicProviderCfg = Field(default_factory=AnthropicProviderCfg)
    openrouter: OpenRouterProviderCfg = Field(default_factory=OpenRouterProviderCfg)
    gemini: GeminiProviderCfg = Field(default_factory=GeminiProviderCfg)
    claude_cli: ClaudeCliCfg = Field(default_factory=ClaudeCliCfg)
    codex_cli: CodexCliCfg = Field(default_factory=CodexCliCfg)


class WebUICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7878


class Secrets(BaseSettings):
    """Secrets pulled from env only. Empty string means 'not configured'.

    Which of these matter depends on `[llm] provider`. Under an API provider
    the matching LLM key is required. Under `claude_cli` / `codex_cli` none of
    the LLM keys are used — reasoning runs on the CLI's subscription login, and
    `cli_backend.BILLING_ENV_VARS` strips them from every subprocess so a call
    cannot silently fall back to metered billing. The embedding and data-source
    keys apply either way.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ATLASCLOUD_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OLLAMA_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    NCBI_API_KEY: str = ""
    OPENALEX_API_KEY: str = ""


class Config(BaseModel):
    run: RunCfg = Field(default_factory=RunCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    science_skills: ScienceSkillsCfg = Field(default_factory=ScienceSkillsCfg)
    embeddings: EmbeddingsCfg = Field(default_factory=EmbeddingsCfg)
    vectors: VectorsCfg = Field(default_factory=VectorsCfg)
    ranking: RankingCfg = Field(default_factory=RankingCfg)
    termination: TerminationCfg = Field(default_factory=TerminationCfg)
    evolution: EvolutionCfg = Field(default_factory=EvolutionCfg)
    budget_shares: BudgetSharesCfg = Field(default_factory=BudgetSharesCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    thinking: ThinkingCfg = Field(default_factory=ThinkingCfg)
    tool_loop: ToolLoopCfg = Field(default_factory=ToolLoopCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
    lease: LeaseCfg = Field(default_factory=LeaseCfg)
    web_search: WebSearchCfg = Field(default_factory=WebSearchCfg)
    web_fetch: WebFetchCfg = Field(default_factory=WebFetchCfg)
    code_exec: CodeExecCfg = Field(default_factory=CodeExecCfg)
    safety: SafetyCfg = Field(default_factory=SafetyCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    web_ui: WebUICfg = Field(default_factory=WebUICfg)
    secrets: Secrets = Field(default_factory=Secrets)

    @property
    def data_dir(self) -> Path:
        p = Path(self.storage.data_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "co_scientist.db"

    def session_artifact_dir(self, session_id: str) -> Path:
        return self.data_dir / "artifacts" / session_id

    def session_vector_dir(self, session_id: str) -> Path:
        return self.data_dir / "vectors" / session_id

    def session_log_path(self, session_id: str) -> Path:
        return self.data_dir / "logs" / f"session-{session_id}.jsonl"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(extra_path: Path | None = None) -> Config:
    """Layered load: default.toml → ~/.co-scientist/config.toml → ./co-scientist.toml → extra_path → env."""
    merged: dict[str, Any] = _read_toml(DEFAULT_CONFIG)

    for p in (
        Path.home() / ".co-scientist" / "config.toml",
        Path.cwd() / "co-scientist.toml",
        extra_path,
    ):
        if p is not None:
            merged = _deep_merge(merged, _read_toml(p))

    cfg = Config.model_validate(merged)
    # secrets pulled from env via Secrets() — already wired by default_factory above
    return cfg


def backend_binary(cfg: Config) -> str:
    """Name of the agent CLI the configured backend drives, or '' for an API."""
    name = (getattr(cfg.llm, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
    if name == "codex_cli":
        return cfg.llm.codex_cli.binary
    if name == "claude_cli":
        return cfg.llm.claude_cli.binary
    return ""


# Env var names per provider preset (see llm/provider.py KNOWN_PROVIDERS).
# An empty string means the backend needs no LLM key at all: Ollama serves
# locally, and the CLI backends authenticate with a subscription login.
_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic":          "ANTHROPIC_API_KEY",
    "openai":             "OPENAI_API_KEY",
    "openai_compatible":  "OPENAI_API_KEY",
    "atlas":              "ATLASCLOUD_API_KEY",
    "openrouter":         "OPENROUTER_API_KEY",
    "gemini":             "GEMINI_API_KEY",
    "google":             "GEMINI_API_KEY",
    "groq":               "GROQ_API_KEY",
    "together":           "TOGETHER_API_KEY",
    "mistral":            "MISTRAL_API_KEY",
    "ollama":             "",   # keyless — local endpoint
    "claude_cli":         "",   # keyless — Claude Code subscription login
    "codex_cli":          "",   # keyless — ChatGPT subscription login
}


def provider_key_env(cfg: Config) -> str:
    """Env-var name the configured LLM provider expects, or '' if keyless."""
    name = (getattr(cfg.llm, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).strip().lower()
    return _PROVIDER_ENV_VARS.get(name, _PROVIDER_ENV_VARS[DEFAULT_PROVIDER])


def has_llm_key(cfg: Config) -> bool:
    """True if the configured provider's API key is available, OR the provider
    is keyless (Ollama, and the subscription CLI backends)."""
    env_var = provider_key_env(cfg)
    if not env_var:
        return True   # keyless provider
    # Explicit OPENAI_API_KEY is always honored (lets users repurpose presets).
    openai_compat_envs = {
        "OPENAI_API_KEY", "ATLASCLOUD_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
        "GROQ_API_KEY", "TOGETHER_API_KEY", "MISTRAL_API_KEY",
    }
    if env_var in openai_compat_envs and (
        cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    ):
        return True
    return bool(getattr(cfg.secrets, env_var, "") or os.environ.get(env_var))


def has_embeddings_key(cfg: Config) -> bool:
    """True if real embeddings are available for the configured provider.

    Otherwise dedup falls back to the hash embedder (see vectors/embedder.py).
    `make_embedder` walks Voyage → OpenAI → hash, so either key counts.
    """
    def _set(name: str) -> bool:
        return bool(getattr(cfg.secrets, name, "") or os.environ.get(name))

    return _set("VOYAGE_API_KEY") or _set("OPENAI_API_KEY")
