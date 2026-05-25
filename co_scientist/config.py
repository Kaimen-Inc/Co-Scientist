"""Configuration loader.

Layered: config/default.toml → ~/.co-scientist/config.toml → ./co-scientist.toml → env.
Secrets come from environment variables only.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"


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


class BudgetSharesCfg(BaseModel):
    generation: float = 0.20
    reflection: float = 0.20
    ranking: float = 0.25
    evolution: float = 0.15
    metareview: float = 0.10
    proximity: float = 0.02
    reserve: float = 0.08


class ModelsCfg(BaseModel):
    parse_goal: str = "claude-sonnet-4-6"
    generation: str = "claude-opus-4-7"
    reflection: str = "claude-opus-4-7"
    evolution: str = "claude-opus-4-7"
    ranking_pairwise: str = "claude-sonnet-4-6"
    ranking_debate: str = "claude-sonnet-4-6"
    ranking_priority: str = "claude-opus-4-7"
    metareview_feedback: str = "claude-sonnet-4-6"
    metareview_final: str = "claude-opus-4-7"
    classifier: str = "claude-haiku-4-5-20251001"
    judge: str = "claude-sonnet-4-6"


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
    provider: str = "anthropic"
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


class LLMCfg(BaseModel):
    """Choose which LLM vendor backs the agents.

    Supported values:
    - "anthropic" — Claude via the official Anthropic SDK (default). Cache
      breakpoints, extended thinking, and the Batch API are only available
      under this provider.
    - "openai" — OpenAI Chat Completions. Extended reasoning is translated
      to `reasoning_effort` for the o-series models; cache breakpoints are
      stripped.
    - "openrouter" — OpenRouter (openrouter.ai). 200+ models from every
      major vendor in one place. Set OPENROUTER_API_KEY (or
      OPENAI_API_KEY). Optional attribution in [llm.openrouter].
    - "gemini" / "google" — Google Gemini via the official OpenAI-compat
      endpoint. Set GEMINI_API_KEY. Models: "gemini-2.5-pro",
      "gemini-2.5-flash", etc.
    - "groq", "together", "mistral", "ollama" — convenience presets for
      those endpoints; each reads its own API key env var
      (GROQ_API_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY).
    - "openai_compatible" — same client as `openai` but allows
      `llm.openai.base_url` to point at any other OpenAI-compatible
      endpoint not yet covered by a preset.
    """

    provider: str = "anthropic"
    openai: OpenAIProviderCfg = Field(default_factory=OpenAIProviderCfg)
    anthropic: AnthropicProviderCfg = Field(default_factory=AnthropicProviderCfg)
    openrouter: OpenRouterProviderCfg = Field(default_factory=OpenRouterProviderCfg)


class WebUICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7878


class Secrets(BaseSettings):
    """Secrets pulled from env only. Empty string means 'not configured'."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
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


def has_anthropic_key(cfg: Config) -> bool:
    return bool(cfg.secrets.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"))
