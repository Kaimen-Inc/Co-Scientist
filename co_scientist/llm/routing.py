"""Model routing per agent.mode + price table for cost accounting.

Price table is approximate and trivially editable as Anthropic posts updates.
Costs are USD per 1M tokens (input / output / cache write / cache read).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config

# USD per 1M tokens. Cache writes are ~1.25x input; cache reads are ~0.1x input.
# These are placeholders the user should sanity-check against current vendor
# pricing before any production use. Unknown models fall back to a
# conservative sonnet-class default.
PRICE_TABLE: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7":            {"input": 15.0,  "output": 75.0,  "cache_write": 18.75, "cache_read": 1.5},
    "claude-sonnet-4-6":          {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "claude-haiku-4-5-20251001":  {"input":  1.0,  "output":  5.0,  "cache_write":  1.25, "cache_read": 0.1},
    # OpenAI (approximate; verify on platform.openai.com/docs/pricing)
    "gpt-5":                      {"input":  5.0,  "output": 20.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "gpt-4.1":                    {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,  "cache_read": 0.2},
    "gpt-4o":                     {"input":  2.5,  "output": 10.0,  "cache_write":  2.5,  "cache_read": 0.25},
    "gpt-4o-mini":                {"input":  0.15, "output":  0.6,  "cache_write":  0.15, "cache_read": 0.075},
    "o3":                         {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "o3-mini":                    {"input":  1.1,  "output":  4.4,  "cache_write":  1.1,  "cache_read": 0.55},
    "o1":                         {"input": 15.0,  "output": 60.0,  "cache_write": 15.0,  "cache_read": 7.5},
    "o1-mini":                    {"input":  3.0,  "output": 12.0,  "cache_write":  3.0,  "cache_read": 1.5},
    # Google (via OpenAI-compat endpoint)
    "gemini-2.5-pro":             {"input":  1.25, "output": 10.0,  "cache_write":  1.25, "cache_read": 0.3},
    "gemini-2.5-flash":           {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    # Mistral, Meta (via providers)
    "mistral-large-latest":       {"input":  2.0,  "output":  6.0,  "cache_write":  2.0,  "cache_read": 0.2},
    "llama-3.3-70b":              {"input":  0.6,  "output":  0.6,  "cache_write":  0.6,  "cache_read": 0.6},
    # OpenRouter passes through the upstream provider's price; common routes:
    "anthropic/claude-opus-4-7":      {"input": 15.0,  "output": 75.0,  "cache_write": 18.75, "cache_read": 1.5},
    "anthropic/claude-sonnet-4-6":    {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "anthropic/claude-3.5-sonnet":    {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "openai/gpt-5":                   {"input":  5.0,  "output": 20.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "openai/gpt-4o":                  {"input":  2.5,  "output": 10.0,  "cache_write":  2.5,  "cache_read": 0.25},
    "openai/o3":                      {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "google/gemini-2.5-pro":          {"input":  1.25, "output": 10.0,  "cache_write":  1.25, "cache_read": 0.3},
    "google/gemini-2.5-flash":        {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.6, "output": 0.6,  "cache_write":  0.6,  "cache_read": 0.6},
    "mistralai/mistral-large":        {"input":  2.0,  "output":  6.0,  "cache_write":  2.0,  "cache_read": 0.2},
}

# Conservative fallback for any model name not in the table — we'd rather
# over-estimate cost than free-run a misconfigured route.
_FALLBACK_PRICE = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3}


# Soft fallback chain: if a degraded route is requested, walk this list once.
DEGRADE_CHAIN = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


# never-degrade modes — config flag overrides require explicit user override
NEVER_DEGRADE = {"reflection.verification", "metareview.final"}


@dataclass
class ModelRoute:
    agent: str
    mode: str         # e.g. "generation.literature"
    model: str
    thinking_tokens: int = 0


def thinking_budget_for(cfg: Config, mode: str) -> int:
    """Translate a mode key into the configured thinking budget."""
    th = cfg.thinking
    return {
        "generation.literature":   th.generation_literature,
        "generation.debate":       th.generation_debate,
        "reflection.full":         th.reflection_full,
        "reflection.verification": th.reflection_verification,
        "reflection.observation":  th.reflection_observation,
        "ranking.pairwise":        th.ranking_pairwise,
        "ranking.debate":          th.ranking_debate,
        "evolution.combine":       th.evolution_combine,
        "evolution.out_of_box":    th.evolution_out_of_box,
        "evolution.feasibility":   th.evolution_feasibility,
        "evolution.simplify":      th.evolution_simplify,
        "metareview.system":       th.metareview_feedback,
        "metareview.final":        th.metareview_final,
    }.get(mode, 0)


def route(cfg: Config, agent: str, mode: str | None = None, *, degraded: bool = False) -> ModelRoute:
    """Pick a model for a given (agent, mode). If `degraded`, walk one step down."""
    m = cfg.models
    model = {
        ("generation", "literature"):  m.generation,
        ("generation", "debate"):      m.generation,
        ("reflection", "full"):        m.reflection,
        ("reflection", "verification"):m.reflection,
        ("reflection", "observation"): m.reflection,
        ("ranking", "pairwise"):       m.ranking_pairwise,
        ("ranking", "debate"):         m.ranking_debate,
        ("ranking", "priority"):       m.ranking_priority,
        ("evolution", "combine"):      m.evolution,
        ("evolution", "out_of_box"):   m.evolution,
        ("evolution", "feasibility"):  m.evolution,
        ("evolution", "simplify"):     m.evolution,
        ("metareview", "system"):      m.metareview_feedback,
        ("metareview", "final"):       m.metareview_final,
        ("parse_goal", None):          m.parse_goal,
        ("classifier", None):          m.classifier,
        ("judge", None):               m.judge,
    }.get((agent, mode), m.generation)

    full_mode = f"{agent}.{mode}" if mode else agent
    if degraded and full_mode not in NEVER_DEGRADE and model in DEGRADE_CHAIN:
        i = DEGRADE_CHAIN.index(model)
        if i + 1 < len(DEGRADE_CHAIN):
            model = DEGRADE_CHAIN[i + 1]

    th = thinking_budget_for(cfg, full_mode) if not degraded else 0
    # Thinking only on Opus by convention.
    if not model.startswith("claude-opus"):
        th = 0

    return ModelRoute(agent=agent, mode=mode or "", model=model, thinking_tokens=th)


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Convert token usage into USD using PRICE_TABLE.

    Anthropic's `usage.input_tokens` is the uncached input count — cache_read /
    cache_write are reported separately. So all four buckets are summed
    independently; no subtraction. Unknown models use a conservative fallback.
    """
    p = PRICE_TABLE.get(model) or _FALLBACK_PRICE
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_write * p["cache_write"]
        + cache_read * p["cache_read"]
    ) / 1_000_000
