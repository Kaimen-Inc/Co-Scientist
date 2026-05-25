"""Model bench: cross-tournament comparison of LLM providers / models.

Given one research goal and a list of `(label, provider, model)` candidates,
the bench:

1. Runs Generation N times for each candidate under a per-candidate Config
   override (its own provider + model + API key resolution).
2. Drops all produced hypotheses into a single round-robin Elo tournament
   judged by ONE fixed `judge_provider` / `judge_model` so the comparison
   is fair (no echo-judge bias and no per-side judge drift).
3. Aggregates per-candidate stats: mean Elo, top Elo, win-rate, $/hyp,
   latency, dollars spent.
4. Persists everything to bench_runs / bench_candidates / bench_matches
   and writes a JSON artifact summary.

This is a *separate* code path from Supervisor / sessions — bench runs do
NOT create a `sessions` row, do NOT enter the main task queue, and do NOT
share Elo state with regular sessions. They reuse the underlying
GenerationAgent and the Elo math.
"""

from .runner import BenchCandidate, BenchOutcome, run_bench

__all__ = ["BenchCandidate", "BenchOutcome", "run_bench"]
