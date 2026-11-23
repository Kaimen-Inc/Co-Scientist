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
    # Claude Code CLI aliases — what `[models]` holds under the claude_cli
    # backend. Nothing is actually billed per token on a subscription; these
    # drive the equivalent-cost gauge and the pre-flight estimate.
    "opus":                       {"input":  5.0,  "output": 25.0,  "cache_write":  6.25, "cache_read": 0.5},
    "sonnet":                     {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "haiku":                      {"input":  1.0,  "output":  5.0,  "cache_write":  1.25, "cache_read": 0.1},
    # Anthropic model ids
    "claude-opus-5":              {"input":  5.0,  "output": 25.0,  "cache_write":  6.25, "cache_read": 0.5},
    "claude-sonnet-5":            {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "claude-haiku-4-5":           {"input":  1.0,  "output":  5.0,  "cache_write":  1.25, "cache_read": 0.1},
    "claude-fable-5":             {"input": 10.0,  "output": 50.0,  "cache_write": 12.5,  "cache_read": 1.0},
    "claude-opus-4-8":            {"input":  5.0,  "output": 25.0,  "cache_write":  6.25, "cache_read": 0.5},
    "claude-opus-4-7":            {"input":  5.0,  "output": 25.0,  "cache_write":  6.25, "cache_read": 0.5},
    "claude-sonnet-4-6":          {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "claude-haiku-4-5-20251001":  {"input":  1.0,  "output":  5.0,  "cache_write":  1.25, "cache_read": 0.1},
    "anthropic/claude-haiku-4.5": {"input":  1.0,  "output":  5.0,  "cache_write":  1.25, "cache_read": 0.1},
    "anthropic/claude-3.5-haiku": {"input":  0.8,  "output":  4.0,  "cache_write":  1.0,  "cache_read": 0.08},
    # OpenAI (approximate; verify on platform.openai.com/docs/pricing).
    # "gpt-5.6" is an alias that routes to gpt-5.6-sol.
    "gpt-5.6":                    {"input":  5.0,  "output": 30.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "gpt-5.6-sol":                {"input":  5.0,  "output": 30.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "gpt-5.6-terra":              {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,  "cache_read": 0.2},
    "gpt-5.6-luna":               {"input":  0.2,  "output":  1.2,  "cache_write":  0.2,  "cache_read": 0.02},
    "gpt-5":                      {"input":  5.0,  "output": 20.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "gpt-4.1":                    {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,  "cache_read": 0.2},
    "gpt-4o":                     {"input":  2.5,  "output": 10.0,  "cache_write":  2.5,  "cache_read": 0.25},
    "gpt-4o-mini":                {"input":  0.15, "output":  0.6,  "cache_write":  0.15, "cache_read": 0.075},
    "o3":                         {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "o3-mini":                    {"input":  1.1,  "output":  4.4,  "cache_write":  1.1,  "cache_read": 0.55},
    "o1":                         {"input": 15.0,  "output": 60.0,  "cache_write": 15.0,  "cache_read": 7.5},
    "o1-mini":                    {"input":  3.0,  "output": 12.0,  "cache_write":  3.0,  "cache_read": 1.5},
    # Google (native short names, Antigravity CLI slugs, AND OpenAI-compat short names)
    "gemini-4-carbon-high":       {"input":  5.0,  "output": 25.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "gemini-4-carbon-medium":     {"input":  4.0,  "output": 20.0,  "cache_write":  4.0,  "cache_read": 0.4},
    "gemini-4-carbon-low":        {"input":  3.0,  "output": 15.0,  "cache_write":  3.0,  "cache_read": 0.3},
    "gemini-4-neon-xhigh":        {"input":  3.0,  "output": 15.0,  "cache_write":  3.0,  "cache_read": 0.3},
    "gemini-4-neon-high":         {"input":  2.5,  "output": 12.5,  "cache_write":  2.5,  "cache_read": 0.25},
    "gemini-4-neon-medium":       {"input":  2.0,  "output": 10.0,  "cache_write":  2.0,  "cache_read": 0.2},
    "gemini-4-neon-low":          {"input":  1.5,  "output":  7.5,  "cache_write":  1.5,  "cache_read": 0.15},
    "gemini-3-pro":               {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "gemini-3-pro-preview":       {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "gemini-3-flash":             {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "gemini-3-flash-preview":     {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "gemini-2.5-pro":             {"input":  1.25, "output": 10.0,  "cache_write":  1.25, "cache_read": 0.3},
    "gemini-2.5-flash":           {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "gemini-2.5-flash-lite":      {"input":  0.1,  "output":  0.4,  "cache_write":  0.1,  "cache_read": 0.025},
    "gemini-1.5-pro":             {"input":  1.25, "output":  5.0,  "cache_write":  1.25, "cache_read": 0.3},
    "gemini-1.5-flash":           {"input":  0.075,"output":  0.3,  "cache_write":  0.075,"cache_read": 0.02},
    # Mistral, Meta (via providers)
    "mistral-large-latest":       {"input":  2.0,  "output":  6.0,  "cache_write":  2.0,  "cache_read": 0.2},
    "llama-3.3-70b":              {"input":  0.6,  "output":  0.6,  "cache_write":  0.6,  "cache_read": 0.6},
    # OpenRouter passes through the upstream provider's price; common routes:
    "anthropic/claude-opus-4-7":      {"input":  5.0,  "output": 25.0,  "cache_write":  6.25, "cache_read": 0.5},
    "anthropic/claude-sonnet-4-6":    {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "anthropic/claude-3.5-sonnet":    {"input":  3.0,  "output": 15.0,  "cache_write":  3.75, "cache_read": 0.3},
    "openai/gpt-5":                   {"input":  5.0,  "output": 20.0,  "cache_write":  5.0,  "cache_read": 0.5},
    "openai/gpt-4o":                  {"input":  2.5,  "output": 10.0,  "cache_write":  2.5,  "cache_read": 0.25},
    "openai/gpt-4o-mini":             {"input":  0.15, "output":  0.6,  "cache_write":  0.15, "cache_read": 0.075},
    "openai/o1":                      {"input": 15.0,  "output": 60.0,  "cache_write": 15.0,  "cache_read": 7.5},
    "openai/o1-mini":                 {"input":  3.0,  "output": 12.0,  "cache_write":  3.0,  "cache_read": 1.5},
    "openai/o3":                      {"input":  2.0,  "output":  8.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "google/gemini-2.0-flash-001":    {"input":  0.1,  "output":  0.4,  "cache_write":  0.1,  "cache_read": 0.025},
    "google/gemini-2.0-flash-lite-001": {"input": 0.075,"output": 0.3, "cache_write":  0.075, "cache_read": 0.02},
    "google/gemini-3-pro":            {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "google/gemini-3-pro-preview":    {"input":  2.0,  "output": 12.0,  "cache_write":  2.0,  "cache_read": 0.5},
    "google/gemini-3-flash":          {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "google/gemini-3-flash-preview":  {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "google/gemini-2.5-pro":          {"input":  1.25, "output": 10.0,  "cache_write":  1.25, "cache_read": 0.3},
    "google/gemini-2.5-flash":        {"input":  0.3,  "output":  2.5,  "cache_write":  0.3,  "cache_read": 0.075},
    "google/gemini-2.5-flash-lite":   {"input":  0.1,  "output":  0.4,  "cache_write":  0.1,  "cache_read": 0.025},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.6, "output": 0.6,  "cache_write":  0.6,  "cache_read": 0.6},
    "mistralai/mistral-large":        {"input":  2.0,  "output":  6.0,  "cache_write":  2.0,  "cache_read": 0.2},
}

# Conservative fallback for any model name not in the table — we'd rather
# over-estimate cost than free-run a misconfigured route.
_FALLBACK_PRICE = {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.3}


# Family hints: substring → representative price. When `estimate_cost_usd` is
# called with a model id we don't recognise (e.g. a brand-new preview), match
# the first family fragment and use its price tier. Order matters: more
# specific patterns first.
_FAMILY_PRICE_HINTS: list[tuple[str, dict[str, float]]] = [
    # Flash / Lite / Mini variants — cheap tier
    ("flash-lite", {"input": 0.1,  "output": 0.4,  "cache_write": 0.1,  "cache_read": 0.025}),
    ("flash",      {"input": 0.3,  "output": 2.5,  "cache_write": 0.3,  "cache_read": 0.075}),
    ("haiku",      {"input": 1.0,  "output": 5.0,  "cache_write": 1.25, "cache_read": 0.1}),
    ("4o-mini",    {"input": 0.15, "output": 0.6,  "cache_write": 0.15, "cache_read": 0.075}),
    ("mini",       {"input": 1.1,  "output": 4.4,  "cache_write": 1.1,  "cache_read": 0.55}),
    ("nano",       {"input": 0.1,  "output": 0.4,  "cache_write": 0.1,  "cache_read": 0.025}),
    ("luna",       {"input": 0.2,  "output": 1.2,  "cache_write": 0.2,  "cache_read": 0.02}),
    # Mid tier
    ("terra",      {"input": 2.0,  "output": 12.0, "cache_write": 2.0,  "cache_read": 0.2}),
    ("sonnet",     {"input": 3.0,  "output": 15.0, "cache_write": 3.75, "cache_read": 0.3}),
    ("gpt-4o",     {"input": 2.5,  "output": 10.0, "cache_write": 2.5,  "cache_read": 0.25}),
    ("gpt-4.1",    {"input": 2.0,  "output":  8.0, "cache_write": 2.0,  "cache_read": 0.2}),
    ("gemini",     {"input": 1.25, "output": 10.0, "cache_write": 1.25, "cache_read": 0.3}),
    # High tier
    ("fable",      {"input": 10.0, "output": 50.0, "cache_write": 12.5, "cache_read": 1.0}),
    ("opus",       {"input": 5.0,  "output": 25.0, "cache_write": 6.25, "cache_read": 0.5}),
    ("gpt-5.6",    {"input": 5.0,  "output": 30.0, "cache_write": 5.0,  "cache_read": 0.5}),
    ("gpt-5",      {"input": 5.0,  "output": 20.0, "cache_write": 5.0,  "cache_read": 0.5}),
    ("o3",         {"input": 2.0,  "output":  8.0, "cache_write": 2.0,  "cache_read": 0.5}),
    ("o1",         {"input": 15.0, "output": 60.0, "cache_write": 15.0, "cache_read": 7.5}),
    ("llama",      {"input": 0.6,  "output":  0.6, "cache_write": 0.6,  "cache_read": 0.6}),
    ("mistral",    {"input": 2.0,  "output":  6.0, "cache_write": 2.0,  "cache_read": 0.2}),
    ("qwen",       {"input": 0.4,  "output":  0.4, "cache_write": 0.4,  "cache_read": 0.4}),
    ("deepseek",   {"input": 0.5,  "output":  1.5, "cache_write": 0.5,  "cache_read": 0.1}),
]


def _price_for_model(model: str) -> dict[str, float]:
    """Look up `model` in PRICE_TABLE; fall back to a family hint by substring."""
    direct = PRICE_TABLE.get(model)
    if direct is not None:
        return direct
    ml = model.lower()
    for needle, price in _FAMILY_PRICE_HINTS:
        if needle in ml:
            return price
    return _FALLBACK_PRICE


# Soft fallback chains: if a degraded route is requested, walk one step down.
#
# One chain per naming style we ship, because `[models]` is interpreted by
# whichever backend is configured — full Anthropic ids under `anthropic`, either
# those or bare aliases under `claude_cli`, vendor-prefixed ids under
# `openrouter`. A single exact-match list in one style would silently match none
# of the others, and degradation would become a no-op instead of an error.
#
# Substring rewriting is deliberately avoided: turning "claude-opus-4-7" into
# "claude-sonnet-4-7" would invent a model id that does not exist.
DEGRADE_CHAINS: tuple[list[str], ...] = (
    ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    ["opus", "sonnet", "haiku"],
    [
        # OpenRouter spells the Haiku tier with a dot; using the dashed form
        # here would degrade onto an id that misses OPENROUTER_PRICE_TABLE.
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4.5",
    ],
    ["gemini-4-carbon-high", "gemini-4-neon-xhigh"],
)

# The canonical chain, for callers that just want the default family.
DEGRADE_CHAIN = DEGRADE_CHAINS[0]


def degrade_model(model: str) -> str:
    """One step down whichever chain holds `model`; unchanged if it's in none.

    Leaving an unrecognized model alone is the safe failure: a degraded route
    that keeps the same model costs more than intended, whereas guessing at a
    cheaper id risks naming a model the endpoint will reject outright.
    """
    for chain in DEGRADE_CHAINS:
        if model in chain:
            i = chain.index(model)
            return chain[i + 1] if i + 1 < len(chain) else model
    return model


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
    if degraded and full_mode not in NEVER_DEGRADE:
        model = degrade_model(model)

    th = thinking_budget_for(cfg, full_mode) if not degraded else 0
    # Thinking only on Opus by convention. Substring rather than prefix so the
    # CLI backends' bare aliases ("opus") match alongside API ids
    # ("claude-opus-5").
    if "opus" not in model.lower():
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
    cache_write are reported separately. Unknown models pick up a family-hint
    price (so "google/gemini-3-flash-preview" prices like a flash even before
    we add it to the table); models with no family hint fall back to a
    conservative sonnet-class default.
    """
    p = _price_for_model(model)
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_write * p["cache_write"]
        + cache_read * p["cache_read"]
    ) / 1_000_000
