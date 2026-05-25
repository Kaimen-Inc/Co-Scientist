-- Migration 0005: bench candidate mode.
--
-- `mode` distinguishes the two generation harnesses we expose for a bench:
--   "pipeline" — full co-scientist Generation agent (literature tools,
--                tool loop, record_hypothesis, dedup).
--   "direct"   — single LM call with a forced record_hypothesis function
--                call. No tools, no agent loop. Isolates the model's raw
--                contribution so we can measure the value-add of the
--                multi-agent harness.

ALTER TABLE bench_candidates ADD COLUMN mode TEXT DEFAULT 'pipeline';
