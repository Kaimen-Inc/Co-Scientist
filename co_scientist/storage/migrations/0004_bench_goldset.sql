-- Migration 0004: gold-set scoring columns for the bench.
--
-- Tracks which curated answer-key entities each candidate surfaced. A
-- bench can be run without a gold set; these columns stay NULL in that
-- case.

ALTER TABLE bench_runs ADD COLUMN goldset_label TEXT;
ALTER TABLE bench_runs ADD COLUMN goldset_size INTEGER;

ALTER TABLE bench_candidates ADD COLUMN gold_hits INTEGER DEFAULT 0;
ALTER TABLE bench_candidates ADD COLUMN gold_hit_names TEXT;
