-- Migration 0003: bench tables for cross-model comparisons.
--
-- A bench run takes one research goal and runs Generation under several
-- different (provider, model) configurations. All produced hypotheses then
-- enter a single shared Elo tournament judged by one fixed judge model. The
-- per-candidate aggregate Elo + win-rate matrix lets us compare models on
-- a level field.

CREATE TABLE IF NOT EXISTS bench_runs (
    id              TEXT PRIMARY KEY,                  -- bnc_<ULID>
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    status          TEXT NOT NULL,                     -- running|done|aborted|failed
    research_goal   TEXT NOT NULL,
    judge_provider  TEXT NOT NULL,
    judge_model     TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,                     -- JSON
    artifact_path   TEXT                                  -- final report JSON
);

CREATE TABLE IF NOT EXISTS bench_candidates (
    id              TEXT PRIMARY KEY,                  -- bcd_<ULID>
    bench_id        TEXT NOT NULL REFERENCES bench_runs(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,                     -- short display name
    provider        TEXT NOT NULL,                     -- anthropic|openai|openrouter|...
    model           TEXT NOT NULL,                     -- model id for that provider
    -- aggregate stats; populated by the runner
    n_hypotheses    INTEGER NOT NULL DEFAULT 0,
    n_matches       INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    mean_elo        REAL,
    top_elo         REAL,
    total_cost_usd  REAL NOT NULL DEFAULT 0,
    total_input_tok INTEGER NOT NULL DEFAULT 0,
    total_output_tok INTEGER NOT NULL DEFAULT 0,
    mean_latency_ms INTEGER,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS bench_cand_bench ON bench_candidates(bench_id);

-- Bench-tournament matches are stored separately from the main
-- tournament_matches table so they don't pollute per-session leaderboards.
-- `cand_a` / `cand_b` are bench_candidates.id (so we know which model
-- produced each side of the match).
CREATE TABLE IF NOT EXISTS bench_matches (
    id              TEXT PRIMARY KEY,                  -- bmt_<ULID>
    bench_id        TEXT NOT NULL REFERENCES bench_runs(id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    cand_a          TEXT NOT NULL REFERENCES bench_candidates(id),
    cand_b          TEXT NOT NULL REFERENCES bench_candidates(id),
    hyp_a_text      TEXT NOT NULL,
    hyp_b_text      TEXT NOT NULL,
    winner          TEXT,                              -- a|b
    elo_a_before    REAL NOT NULL,
    elo_b_before    REAL NOT NULL,
    elo_a_after     REAL,
    elo_b_after     REAL,
    rationale       TEXT,
    judge_cost_usd  REAL,
    judge_latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS bench_match_bench ON bench_matches(bench_id);
CREATE INDEX IF NOT EXISTS bench_match_cands ON bench_matches(cand_a, cand_b);
