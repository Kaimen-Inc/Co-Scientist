# Bench results

Pilot results from every cross-model bench run on this codebase. See [`../README.md`](../README.md) for what the bench is and how to run it.

_Auto-generated from `data/co_scientist.db` by_ _`python scripts/build_bench_report.py`._ _Re-run after any new `co-scientist bench` to refresh._

## How to read this doc

1. **Index** below lists every bench ever run on this machine, one row per bench. Click a bench-id link to jump to its detail.
2. **Per-bench detail** shows, for each bench:
   - the goal it was given,
   - the candidate result table (Elo, hits, $),
   - **every hypothesis the bench produced** with its full statement,
     attributed to the model that produced it (from the bench-match table),
   - **post-hoc rescore** against every registered gold set — so a bench that ran with `aml-repurposing-paper-top3` at the time can still show whether any hypothesis would have hit the broader `aml-repurposing-paper-5` list, and vice versa,
   - **file pointers** for the artifacts on disk + ready-to-run SQL for the raw DB rows.

**Total benches:** 20 · **With gold-set scoring:** 7

## Headline findings

The `*-vs-raw` benches run each model twice on the same goal: **direct** (a single LM call, no harness) and **pipeline** (the full Generation harness — literature tools + tool loop + dedup), in one shared Elo pool. We ran the paper baselines twice and added Gemini 2.5 + 3.x, which is enough to see how stable the comparison is.

### 1. The harness reliably *produces* a hypothesis; whether it *helps* is not reproducible at this sample size

After the pipeline reliability fixes, pipeline mode completes for essentially every candidate (the only misses were a transient HTTP 429 and gemini-2.5-pro intermittently returning an empty completion on the forced final call — 2 of 3 pipeline attempts). So the harness *finishes*. But the **direct→pipeline Elo delta swings from run to run**, so it does not support a per-model — let alone per-provider — verdict.

Same preset, identical settings, two runs — the delta flips sign for haiku while staying put for o1:

| model | run 1 Δ Elo | run 2 Δ Elo |
| --- | --- | --- |
| claude-haiku-4.5 | **+180** (1-9 → 10-0) | **−28** (10-2 → 8-4) |
| openai-o1 | +43 | +29 |

Haiku's *raw* record alone flipped 1-9 → 10-2 across the two runs — with one hypothesis per candidate and ~2 matches per pair, the tournament is dominated by which single hypothesis got sampled, not by the mode. Single-run deltas elsewhere are all over the map and don't line up by provider or strength:

| model | Δ Elo (single run) |
| --- | --- |
| claude-opus-4.7 | +97 |
| gemini-2.5-flash | +172 |
| gemini-2.5-pro | +47 |
| gpt-5 | +26 |
| gemini-2.0-flash | −48 |
| gemini-3-flash | −36 |
| gemini-3-pro | −89 |

Within Google alone the 2.5 models gain (+172, +47) and the 3.x models lose (−36, −89) — so there is no clean "provider" or "stronger-model" story; an earlier draft that claimed one was reading noise. Openai-o1 had pipeline modestly ahead in both runs. A real per-model verdict needs many more seeds (higher `--n`, more matches) to average out the single-hypothesis variance.

### 2. Consistency: models converge on mechanisms, not specific drugs

Across all 48 AML hypotheses recorded on this codebase, agreement is at the **mechanism** level, not the compound level:

| recurring theme | hypotheses (of 48) |
| --- | --- |
| leukemic-stem-cell (LSC) targeting | 28 |
| OXPHOS / mitochondrial complex I | 8 |
| BCL-2 / MCL-1 (Venetoclax axis) | 7 |
| FLT3-ITD | 6 |
| fatty-acid oxidation | 5 |
| ferroptosis | 3 |

At the **drug** level it's a long tail of one-offs. The only compounds proposed more than once are **Itraconazole** (×5, as an OXPHOS inhibitor, has PhII in combo https://www.sciencedirect.com/science/article/pii/S0145212623006380) and **Auranofin** (×2, thioredoxin-reductase, potentially novel). **Venetoclax** (FDA-Approved Frontline Standard of Care) appears ×6 but as the resistance/combo context, not the novel candidate. No hypothesis across any of these benches hit the paper's original pick after many more generation and oncologist selection. 


## Index of recorded benches

| bench | created | preset / kind | n_cand | n_matches | total $ | goldset | hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`bnc_01KSG616HEJDMHFVVJGP…`](#bench-bnc_01ksg616hejdmhfvvjgprzgab6) | 2026-05-25T18:23:24Z | microbiome smoke | 2 | 0 | $0.0009 | `—` | — |
| [`bnc_01KSG6415G35N653G91G…`](#bench-bnc_01ksg6415g35n653g91g8b9p8z) | 2026-05-25T18:24:57Z | microbiome smoke | 2 | 1 | $0.0140 | `—` | — |
| [`bnc_01KSG693N61YY9CSAKK8…`](#bench-bnc_01ksg693n61yy9csakk86vfevr) | 2026-05-25T18:27:43Z | microbiome smoke | 2 | 0 | $0.0105 | `—` | — |
| [`bnc_01KSG6A6JW4DZ95NDJPK…`](#bench-bnc_01ksg6a6jw4dz95ndjpk9561jq) | 2026-05-25T18:28:19Z | microbiome smoke | 2 | 0 | $0.0328 | `—` | — |
| [`bnc_01KSG6BYTB7KK8NDFJDA…`](#bench-bnc_01ksg6bytb7kk8ndfjdajht3j9) | 2026-05-25T18:29:16Z | microbiome smoke | 2 | 0 | $0.0069 | `—` | — |
| [`bnc_01KSG6FCQPP5T4K5V0PW…`](#bench-bnc_01ksg6fcqpp5t4k5v0pw402dpe) | 2026-05-25T18:31:09Z | microbiome smoke | 2 | 2 | $0.0199 | `—` | — |
| [`bnc_01KSG6GM23ERB68V6BCF…`](#bench-bnc_01ksg6gm23erb68v6bcf9xbs2b) | 2026-05-25T18:31:49Z | microbiome smoke | 3 | 6 | $0.0238 | `—` | — |
| [`bnc_01KSG7AGXVXBV5XSZQ3G…`](#bench-bnc_01ksg7agxvxbv5xszq3g7nafq9) | 2026-05-25T18:45:58Z | microbiome smoke | 4 | 0 | $0.2660 | `—` | — |
| [`bnc_01KSG7HM47116412H3NV…`](#bench-bnc_01ksg7hm47116412h3nv3vkdf8) | 2026-05-25T18:49:50Z | microbiome smoke | 4 | 12 | $0.4032 | `—` | — |
| [`bnc_01KSGCEWXMS7T3M3FJSA…`](#bench-bnc_01ksgcewxms7t3m3fjsa9g1k5t) | 2026-05-25T20:15:44Z | AML repurposing | 2 | 0 | $0.0372 | `—` | — |
| [`bnc_01KSGCHAN1348M7WEDX9…`](#bench-bnc_01ksgchan1348m7wedx9e449nd) | 2026-05-25T20:17:04Z | AML repurposing | 3 | 0 | $0.0099 | `—` | — |
| [`bnc_01KSGCJSN8MGMK6H3KZV…`](#bench-bnc_01ksgcjsn8mgmk6h3kzvvqzjpg) | 2026-05-25T20:17:52Z | AML repurposing | 2 | 2 | $0.0189 | `—` | — |
| [`bnc_01KSGCKSG3MJKVPDZBZX…`](#bench-bnc_01ksgcksg3mjkvpdzbzxm3th2g) | 2026-05-25T20:18:24Z | AML repurposing | 4 | 6 | $0.9852 | `aml-repurposing-paper-5` | 0/5 |
| [`bnc_01KSGD0WKFYAF2X15P99…`](#bench-bnc_01ksgd0wkfyaf2x15p99bfax01) | 2026-05-25T20:25:34Z | AML repurposing | 4 | 12 | $0.1452 | `—` | — |
| [`bnc_01KSGV99YG7D4DXZ8G6P…`](#bench-bnc_01ksgv99yg7d4dxz8g6pejwkv1) | 2026-05-26T00:34:49Z | AML repurposing | 4 | 0 | $0.0000 | `aml-repurposing-paper-5` | 0/5 |
| [`bnc_01KSGVHY16Q0GHNE9ZJW…`](#bench-bnc_01ksgvhy16q0ghne9zjwzygfng) | 2026-05-26T00:39:32Z | AML repurposing | 4 | 6 | $1.8881 | `aml-repurposing-paper-top3` | 0/3 |
| [`bnc_01KSN03HG9VW3CPYD40S…`](#bench-bnc_01ksn03hg9vw3cpyd40sa51neb) | 2026-05-27T15:16:01Z | AML repurposing | 8 | 30 | $0.8309 | `aml-repurposing-paper-top3` | 0/3 |
| [`bnc_01KSN0ZMJZV12F6MDE63…`](#bench-bnc_01ksn0zmjzv12f6mde63h7mw9r) | 2026-05-27T15:31:22Z | AML repurposing | 8 | 56 | $2.0820 | `aml-repurposing-paper-top3` | 0/3 |
| [`bnc_01KSN8WZ9XZP0HSW68YS…`](#bench-bnc_01ksn8wz9xzp0hsw68ysrdbr77) | 2026-05-27T17:49:43Z | AML repurposing | 8 | 42 | $1.1057 | `aml-repurposing-paper-top3` | 0/3 |
| [`bnc_01KSN98V3SAB2X0XDSVV…`](#bench-bnc_01ksn98v3sab2x0xdsvv5pyqkm) | 2026-05-27T17:56:12Z | AML repurposing | 4 | 12 | $0.1612 | `aml-repurposing-paper-top3` | 0/3 |

## Per-bench detail

<a id="bench-bnc_01ksg616hejdmhfvvjgprzgab6"></a>
## Bench `bnc_01KSG616HEJDMHFVVJGPRZGAB6`

- **Created:** 2026-05-25T18:23:24.210826+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-2.5-pro`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0009
- **Matches played:** 0
- **Session:** `ses_01KSG616HKWF8K9GC4MSRQN386`
- **Bench artifact:** `artifacts/ses_01KSG616HKWF8K9GC4MSRQN386/bench/bnc_01KSG616HEJDMHFVVJGPRZGAB6.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash25` | pipeline | 0 | — | — | 0/— | $0.0004 | — | — |  |
| `flash3` | pipeline | 0 | — | — | 0/— | $0.0006 | — | — |  |

_No hypotheses produced (every candidate failed)._

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG616HKWF8K9GC4MSRQN386/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG616HKWF8K9GC4MSRQN386/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG616HKWF8K9GC4MSRQN386/bench/bnc_01KSG616HEJDMHFVVJGPRZGAB6.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG616HEJDMHFVVJGPRZGAB6';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG616HEJDMHFVVJGPRZGAB6';
```

<a id="bench-bnc_01ksg6415g35n653g91g8b9p8z"></a>
## Bench `bnc_01KSG6415G35N653G91G8B9P8Z`

- **Created:** 2026-05-25T18:24:57.012161+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-2.5-pro`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0140
- **Matches played:** 1
- **Session:** `ses_01KSG6415MR30Q45K24E1RN6BF`
- **Bench artifact:** `artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/bench/bnc_01KSG6415G35N653G91G8B9P8Z.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash25` | pipeline | 1 | — | 1200 | 0/— | $0.0056 | 4,138 / 1,753 | 12.4s |  |
| `flash3` | pipeline | 1 | — | 1200 | 0/— | $0.0083 | 18,633 / 1,097 | 12.7s |  |

### Hypotheses surfaced (2 total)

- **Genetically-mediated Bile Acid Dysregulation in Microbiome-driven Systemic Inflammation** — via `flash25 (pipeline)`
  - Genetically-mediated alterations in host bile acid synthesis or enterohepatic circulation lead to a dysbiotic gut microbiome characterized by an overabundance of specific bile acid-metabolizing bacteria, which produce immunomodulatory metab
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/hypotheses/hyp_6e6a238879a8f416.json`](data/artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/hypotheses/hyp_6e6a238879a8f416.json)
- **The Akkermansia Paradox: Mucin-Degradation Thresholds Drive the Shift from Metabolic Symbiosis to Inflammatory Pathogeni** — via `flash3 (pipeline)`
  - The metabolic benefits of Akkermansia muciniphila are governed by a concentration-dependent threshold where excessive mucin degradation, necessitated by low dietary fiber availability, triggers sub-clinical mucosal inflammation and metaboli
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/hypotheses/hyp_aaf15d3f067fbf1f.json`](data/artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/hypotheses/hyp_aaf15d3f067fbf1f.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG6415MR30Q45K24E1RN6BF/bench/bnc_01KSG6415G35N653G91G8B9P8Z.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG6415G35N653G91G8B9P8Z';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG6415G35N653G91G8B9P8Z';
```

<a id="bench-bnc_01ksg693n61yy9csakk86vfevr"></a>
## Bench `bnc_01KSG693N61YY9CSAKK86VFEVR`

- **Created:** 2026-05-25T18:27:43.402528+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-2.5-pro`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0105
- **Matches played:** 0
- **Session:** `ses_01KSG693NBFSS3YQYYH4R41MAH`
- **Bench artifact:** `artifacts/ses_01KSG693NBFSS3YQYYH4R41MAH/bench/bnc_01KSG693N61YY9CSAKK86VFEVR.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3` | pipeline | 1 | — | 1200 | 0/— | $0.0097 | 23,632 / 1,051 | 12.1s |  |
| `flash25` | pipeline | 0 | — | — | 0/— | $0.0008 | 2,192 / 65 | — |  |

### Hypotheses surfaced (1 total)

- **Indole-AHR Signaling Failure in Post-Bypass Mucosal Inflammation** — via _(no match table entry)_
  - Post-gastric bypass intestinal inflammation is driven by the depletion of commensal-derived indolic metabolites (e.g., I3PA), leading to a loss of constitutive Aryl Hydrocarbon Receptor (AHR) signaling and a subsequent rise in mucosal pro-i
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG693NBFSS3YQYYH4R41MAH/hypotheses/hyp_7880fd7bd6aff734.json`](data/artifacts/ses_01KSG693NBFSS3YQYYH4R41MAH/hypotheses/hyp_7880fd7bd6aff734.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG693NBFSS3YQYYH4R41MAH/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG693NBFSS3YQYYH4R41MAH/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG693NBFSS3YQYYH4R41MAH/bench/bnc_01KSG693N61YY9CSAKK86VFEVR.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG693N61YY9CSAKK86VFEVR';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG693N61YY9CSAKK86VFEVR';
```

<a id="bench-bnc_01ksg6a6jw4dz95ndjpk9561jq"></a>
## Bench `bnc_01KSG6A6JW4DZ95NDJPK9561JQ`

- **Created:** 2026-05-25T18:28:19.167294+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-2.5-pro`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0328
- **Matches played:** 0
- **Session:** `ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J`
- **Bench artifact:** `artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/bench/bnc_01KSG6A6JW4DZ95NDJPK9561JQ.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3` | pipeline | 1 | — | 1200 | 0/— | $0.0103 | 26,848 / 905 | 25.1s |  |
| `pro25` | pipeline | 1 | — | 1200 | 0/— | $0.0225 | 3,908 / 1,757 | 17.9s |  |

### Hypotheses surfaced (2 total)

- **Microbial Cross-Feeding and Inflammation** — via _(no match table entry)_
  - The anti-inflammatory effects of fermented foods are mediated by a microbial cross-feeding mechanism initiated by Lactobacillus-derived indole-3-lactic acid (ILA).
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/hypotheses/hyp_5812cff97f605b12.json`](data/artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/hypotheses/hyp_5812cff97f605b12.json)
- **The B. melaninogenicus-TLR4 Axis in Atherosclerotic Inflammation** — via _(no match table entry)_
  - Bacteroides melaninogenicus-derived lipopolysaccharide (LPS) acts as a systemic pro-inflammatory super-agonist of the TLR4 pathway, driving chronic vascular inflammation and accelerating the progression of atherosclerotic plaques.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/hypotheses/hyp_74ec8753bbbce34a.json`](data/artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/hypotheses/hyp_74ec8753bbbce34a.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG6A6K0ZCPSKTQAJ5CYWT4J/bench/bnc_01KSG6A6JW4DZ95NDJPK9561JQ.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG6A6JW4DZ95NDJPK9561JQ';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG6A6JW4DZ95NDJPK9561JQ';
```

<a id="bench-bnc_01ksg6bytb7kk8ndfjdajht3j9"></a>
## Bench `bnc_01KSG6BYTB7KK8NDFJDAJHT3J9`

- **Created:** 2026-05-25T18:29:16.752113+00:00
- **Status:** done
- **Judge:** `openrouter:openai/gpt-4o`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0069
- **Matches played:** 0
- **Session:** `ses_01KSG6BYTG2GR0EGTXV9ZEYFG1`
- **Bench artifact:** `artifacts/ses_01KSG6BYTG2GR0EGTXV9ZEYFG1/bench/bnc_01KSG6BYTB7KK8NDFJDAJHT3J9.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3` | pipeline | 1 | — | 1200 | 0/— | $0.0061 | 10,710 / 1,149 | 11.4s |  |
| `flash25` | pipeline | 0 | — | — | 0/— | $0.0008 | 2,192 / 64 | — |  |

### Hypotheses surfaced (1 total)

- **Bile Acid 12-Oxidation as a Microbial Driver of Post-Stroke Neuroinflammation** — via _(no match table entry)_
  - Microbiome-derived 12-oxobiliary acids, produced by stroke-induced shifts in 12-hydroxysteroid dehydrogenase-expressing gut bacteria, cross the blood-brain barrier to trigger microglial NLRP3 inflammasome activation, thereby exacerbating po
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6BYTG2GR0EGTXV9ZEYFG1/hypotheses/hyp_9961812b5f37d953.json`](data/artifacts/ses_01KSG6BYTG2GR0EGTXV9ZEYFG1/hypotheses/hyp_9961812b5f37d953.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG6BYTG2GR0EGTXV9ZEYFG1/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG6BYTG2GR0EGTXV9ZEYFG1/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG6BYTG2GR0EGTXV9ZEYFG1/bench/bnc_01KSG6BYTB7KK8NDFJDAJHT3J9.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG6BYTB7KK8NDFJDAJHT3J9';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG6BYTB7KK8NDFJDAJHT3J9';
```

<a id="bench-bnc_01ksg6fcqpp5t4k5v0pw402dpe"></a>
## Bench `bnc_01KSG6FCQPP5T4K5V0PW402DPE`

- **Created:** 2026-05-25T18:31:09.306054+00:00
- **Status:** done
- **Judge:** `openrouter:openai/gpt-4o`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0199
- **Matches played:** 2
- **Session:** `ses_01KSG6FCQT953KKFVX05GGEPDT`
- **Bench artifact:** `artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/bench/bnc_01KSG6FCQPP5T4K5V0PW402DPE.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3` | pipeline | 1 | 2-0 | 1231 | 0/— | $0.0091 | 20,108 / 1,214 | 10.7s |  |
| `flash25` | pipeline | 1 | 0-2 | 1169 | 0/— | $0.0108 | 13,090 / 2,759 | 21.2s |  |

### Hypotheses surfaced (2 total)

- **SFB-Induced Gut-Mammary Th17 Trafficking and DNA Damage** — via `flash3 (pipeline)`
  - The gut pathobiont Segmented Filamentous Bacteria (SFB) drives mammary gland inflammation and pre-malignant DNA damage in obesity by inducing systemic trafficking of SFB-specific Th17 cells that secrete IL-17A to trigger local oxidative str
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/hypotheses/hyp_09cb4e3a898e671a.json`](data/artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/hypotheses/hyp_09cb4e3a898e671a.json)
- **Intestinal IL-17R Dysregulation and Microbiome-Mediated Hepatic Inflammation** — via `flash25 (pipeline)`
  - Dysregulation of intestinal IL-17R signaling leads to microbiome dysbiosis and increased translocation of bacterial products, specifically CpG DNA, which subsequently drives IL-18 production in the liver, exacerbating hepatic inflammation a
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/hypotheses/hyp_d93b3a9725bbc24f.json`](data/artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/hypotheses/hyp_d93b3a9725bbc24f.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG6FCQT953KKFVX05GGEPDT/bench/bnc_01KSG6FCQPP5T4K5V0PW402DPE.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG6FCQPP5T4K5V0PW402DPE';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG6FCQPP5T4K5V0PW402DPE';
```

<a id="bench-bnc_01ksg6gm23erb68v6bcf9xbs2b"></a>
## Bench `bnc_01KSG6GM23ERB68V6BCF9XBS2B`

- **Created:** 2026-05-25T18:31:49.575192+00:00
- **Status:** done
- **Judge:** `openrouter:openai/gpt-4o`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0238
- **Matches played:** 6
- **Session:** `ses_01KSG6GM275W3TKGG6ETH9V53X`
- **Bench artifact:** `artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/bench/bnc_01KSG6GM23ERB68V6BCF9XBS2B.json`

**Goal:**

> Identify two hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3` | pipeline | 1 | 3-1 | 1232 | 0/— | $0.0090 | 20,512 / 1,123 | 9.8s |  |
| `flash25` | pipeline | 1 | 3-1 | 1227 | 0/— | $0.0098 | 8,906 / 2,865 | 16.7s |  |
| `gpt4o-mini` | pipeline | 1 | 0-4 | 1142 | 0/— | $0.0050 | 29,999 / 869 | 19.1s |  |

### Hypotheses surfaced (3 total)

- **Akkermansia P9-GLP-1 Axis for Inflammasome Suppression** — via `flash3 (pipeline)`
  - Akkermansia muciniphila mitigates microbiome-driven systemic inflammation by secreting the P9 protein, which triggers L-cell GLP-1 production to systemically inhibit NLRP3 inflammasome activation.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/hyp_0e4ca278a85727ad.json`](data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/hyp_0e4ca278a85727ad.json)
- **Gut Microbiome Dysbiosis Drives Systemic Inflammation Mediated by Immune Responses** — via `gpt4o-mini (pipeline)`
  - Dysbiosis of the gut microbiome leads to increased systemic inflammation through the activation of immune pathways, contributing to various inflammatory conditions.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/hyp_b6d23e6819ba7540.json`](data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/hyp_b6d23e6819ba7540.json)
- **Microbiome-Bile Acid Dysregulation Drives PCOS Inflammation** — via `flash25 (pipeline)`
  - Dysbiosis of the gut microbiota contributes to systemic low-grade inflammation in Polycystic Ovary Syndrome (PCOS) by impairing bile acid metabolism and increasing circulating pro-inflammatory cytokines, leading to insulin resistance and hy
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/hyp_f899791c034ef7ef.json`](data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/hyp_f899791c034ef7ef.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG6GM275W3TKGG6ETH9V53X/bench/bnc_01KSG6GM23ERB68V6BCF9XBS2B.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG6GM23ERB68V6BCF9XBS2B';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG6GM23ERB68V6BCF9XBS2B';
```

<a id="bench-bnc_01ksg7agxvxbv5xszq3g7nafq9"></a>
## Bench `bnc_01KSG7AGXVXBV5XSZQ3G7NAFQ9`

- **Created:** 2026-05-25T18:45:58.334490+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.2660
- **Matches played:** 0
- **Session:** `ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR`
- **Bench artifact:** `artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/bench/bnc_01KSG7AGXVXBV5XSZQ3G7NAFQ9.json`

**Goal:**

> Identify two promising hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-haiku-4.5` | pipeline | 1 | — | 1200 | 0/— | $0.2068 | 183,841 / 4,584 | 81.5s |  |
| `gemini-2-flash-thinking` | pipeline | 1 | — | 1200 | 0/— | $0.0121 | 27,160 / 1,568 | 27.3s |  |
| `gemini-2-pro` | pipeline | 1 | — | 1200 | 0/— | $0.0471 | 5,473 / 4,028 | 40.9s |  |
| `openai-o1` | pipeline | 0 | — | — | 0/— | $0.0000 | — | — |  |

### Hypotheses surfaced (3 total)

- **Microbiome-Inflammation Axis in HIV-Associated Cardiovascular Disease** — via _(no match table entry)_
  - Specific gut microbiome dysbiosis in HIV-infected individuals promotes systemic inflammation, contributing to increased subclinical cardiovascular disease risk.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/hyp_73af166cd168c32b.json`](data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/hyp_73af166cd168c32b.json)
- **Progressive Loss of SCFA-Mediated Immune Tolerance in Dysbiosis-Driven Inflammation** — via _(no match table entry)_
  - Dysbiosis-driven inflammation progresses through a stage-specific cascade in which sequential depletion of SCFA-producing bacteria (Faecalibacterium prausnitzii, Roseburia, Clostridium clusters) progressively diminishes butyrate-dependent F
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/hyp_84f3ac53779b3f4c.json`](data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/hyp_84f3ac53779b3f4c.json)
- **Temporal Dysregulation in Peyer's Patches as a Driver of AIEC-Mediated Inflammation** — via _(no match table entry)_
  - Antibiotic-induced disruption of circadian immune rhythms in Peyer's patch-associated microbiome niches creates a permissive window for Adherent-Invasive E. coli (AIEC) colonization, which in turn establishes a persistent, localized inflamm
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/hyp_ad96328d18db9617.json`](data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/hyp_ad96328d18db9617.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG7AGXZDEEGEZ1VM0ZFPMJR/bench/bnc_01KSG7AGXVXBV5XSZQ3G7NAFQ9.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG7AGXVXBV5XSZQ3G7NAFQ9';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG7AGXVXBV5XSZQ3G7NAFQ9';
```

<a id="bench-bnc_01ksg7hm47116412h3nv3vkdf8"></a>
## Bench `bnc_01KSG7HM47116412H3NV3VKDF8`

- **Created:** 2026-05-25T18:49:50.985049+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.4032
- **Matches played:** 12
- **Session:** `ses_01KSG7HM49E7Q77K0WN5GN28TN`
- **Bench artifact:** `artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/bench/bnc_01KSG7HM47116412H3NV3VKDF8.json`

**Goal:**

> Identify two promising hypotheses about microbiome-driven inflammation

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-2-flash-thinking` | pipeline | 1 | 6-0 | 1284 | 0/— | $0.0026 | 21,971 / 965 | 35.2s |  |
| `gemini-2-pro` | pipeline | 1 | 4-2 | 1229 | 0/— | $0.0342 | 5,783 / 2,695 | 26.6s |  |
| `openai-o1` | pipeline | 1 | 2-4 | 1165 | 0/— | $0.2326 | 3,522 / 2,996 | 28.7s |  |
| `claude-haiku-4.5` | pipeline | 1 | 0-6 | 1123 | 0/— | $0.1338 | 116,434 / 3,477 | 71.3s |  |

### Hypotheses surfaced (4 total)

- **Targeting Oscillibacter valericigenes to reduce adipose inflammation in metabolic syndrome** — via `openai-o1 (pipeline)`
  - Selective suppression of Oscillibacter valericigenes in the gut microbiota decreases macrophage-mediated adipose tissue inflammation and mitigates diet-induced metabolic syndrome.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_38b1964d15cad8b3.json`](data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_38b1964d15cad8b3.json)
- **Microbiome-derived pro-inflammatory lipids and asthma exacerbation** — via `gemini-2-flash-thinking (pipeline)`
  - Specific gut microbes exacerbate asthma by producing pro-inflammatory lipid mediators such as 12,13-diHOME, which translocate to the lungs via the gut-lung axis, promoting airway inflammation.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_b4fac88683f2face.json`](data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_b4fac88683f2face.json)
- **Synergistic bacterial protection against allergic lung inflammation** — via `gemini-2-pro (pipeline)`
  - The synergistic action of gut commensals Akkermansia muciniphila and Parabacteroides distasonis ameliorates allergic lung inflammation by enhancing the gut-lung axis.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_bec78dba6ec909fb.json`](data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_bec78dba6ec909fb.json)
- **Dysbiosis-driven barrier dysfunction and LPS translocation cascade** — via `claude-haiku-4.5 (pipeline)`
  - Dysbiosis-driven dysregulation of tight junction protein expression and epithelial integrity enables increased translocation of lipopolysaccharide (LPS) from gram-negative bacteria, which activates TLR4/NF-κB signaling in macrophages and en
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_ca7310d1ef4a17e5.json`](data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/hyp_ca7310d1ef4a17e5.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSG7HM49E7Q77K0WN5GN28TN/bench/bnc_01KSG7HM47116412H3NV3VKDF8.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSG7HM47116412H3NV3VKDF8';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSG7HM47116412H3NV3VKDF8';
```

<a id="bench-bnc_01ksgcewxms7t3m3fjsa9g1k5t"></a>
## Bench `bnc_01KSGCEWXMS7T3M3FJSA9G1K5T`

- **Created:** 2026-05-25T20:15:44.568652+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0372
- **Matches played:** 0
- **Session:** `ses_01KSGCEWXS9H04GHGNRJR9D790`
- **Bench artifact:** `artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/bench/bnc_01KSGCEWXMS7T3M3FJSA9G1K5T.json`

**Goal:**

> Identify FDA-approved drugs that could be repurposed for AML; name specific drugs, mechanisms, and an experiment.

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3-pipe` | pipeline | 1 | — | 1200 | 0/— | $0.0164 | 43,383 / 1,341 | 12.5s |  |
| `flash3-raw` | pipeline | 1 | — | 1200 | 0/— | $0.0208 | 57,868 / 1,394 | 14.3s |  |

### Hypotheses surfaced (2 total)

- **Repurposing Thioridazine to Target Leukemic Stem Cells in AML via Dopamine Receptor Antagonism** — via _(no match table entry)_
  - The FDA-approved antipsychotic thioridazine can be repurposed for Acute Myeloid Leukemia (AML) by selectively eliminating leukemic stem cells (LSCs) through the antagonism of overexpressed dopamine receptors D2 and D4.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/hypotheses/hyp_25b09255f880e4a3.json`](data/artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/hypotheses/hyp_25b09255f880e4a3.json)
- **Repurposing Selinexor to Overcome Venetoclax Resistance in Monocytic AML via MCL-1 Suppression** — via _(no match table entry)_
  - The combination of the XPO1 inhibitor selinexor and the BCL-2 inhibitor venetoclax will synergistically overcome therapeutic resistance in monocytic and relapsed/refractory AML by modulating the p53-NF-κB axis to downregulate MCL-1 expressi
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/hypotheses/hyp_29e3a8e0dd33eedd.json`](data/artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/hypotheses/hyp_29e3a8e0dd33eedd.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGCEWXS9H04GHGNRJR9D790/bench/bnc_01KSGCEWXMS7T3M3FJSA9G1K5T.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGCEWXMS7T3M3FJSA9G1K5T';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGCEWXMS7T3M3FJSA9G1K5T';
```

<a id="bench-bnc_01ksgchan1348m7wedx9e449nd"></a>
## Bench `bnc_01KSGCHAN1348M7WEDX9E449ND`

- **Created:** 2026-05-25T20:17:04.163262+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0099
- **Matches played:** 0
- **Session:** `ses_01KSGCHAN3MC0Z484E97C3WWEW`
- **Bench artifact:** `artifacts/ses_01KSGCHAN3MC0Z484E97C3WWEW/bench/bnc_01KSGCHAN1348M7WEDX9E449ND.json`

**Goal:**

> Identify FDA-approved drugs that could be repurposed for AML. Name a specific drug INN/brand, the mechanism in AML, and a concrete experiment.

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash25-raw` | direct | 0 | — | — | 0/— | $0.0000 | — | — | 1 validation error for Task
action
  Input should be 'Create |
| `flash3-pipe` | pipeline | 0 | — | — | 0/— | $0.0099 | 31,406 / 202 | — |  |
| `flash3-raw` | direct | 0 | — | — | 0/— | $0.0000 | — | — | 1 validation error for Task
action
  Input should be 'Create |

_No hypotheses produced (every candidate failed)._

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGCHAN3MC0Z484E97C3WWEW/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGCHAN3MC0Z484E97C3WWEW/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGCHAN3MC0Z484E97C3WWEW/bench/bnc_01KSGCHAN1348M7WEDX9E449ND.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGCHAN1348M7WEDX9E449ND';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGCHAN1348M7WEDX9E449ND';
```

<a id="bench-bnc_01ksgcjsn8mgmk6h3kzvvqzjpg"></a>
## Bench `bnc_01KSGCJSN8MGMK6H3KZVVQZJPG`

- **Created:** 2026-05-25T20:17:52.298818+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.0189
- **Matches played:** 2
- **Session:** `ses_01KSGCJSNBQA3C0Y09736DSJVW`
- **Bench artifact:** `artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/bench/bnc_01KSGCJSN8MGMK6H3KZVVQZJPG.json`

**Goal:**

> Identify FDA-approved drugs that could be repurposed for AML. Name a specific drug INN/brand, the mechanism in AML, and a concrete experiment.

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3-pipe` | pipeline | 1 | 2-0 | 1231 | 0/— | $0.0174 | 49,617 / 1,014 | 12.9s |  |
| `flash3-raw` | direct | 1 | 0-2 | 1169 | 0/— | $0.0015 | 389 / 544 | 4.2s |  |

### Hypotheses surfaced (2 total)

- **Mifepristone Repurposing for FLT3-ITD Positive AML** — via `flash3-raw (direct)`
  - Mifepristone (RU486) inhibits the growth of FLT3-ITD mutated Acute Myeloid Leukemia by antagonizing the glucocorticoid receptor-mediated survival signaling and inducing apoptosis.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/hypotheses/hyp_a37e3de6b24daa2b.json`](data/artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/hypotheses/hyp_a37e3de6b24daa2b.json)
- **Repurposing Salicylanilide Anthelmintics to Target the MLL-MYB-OXPHOS Nexus in AML LSCs** — via `flash3-pipe (pipeline)`
  - The FDA-approved anthelmintics Niclosamide and Bithionol can be repurposed to eradicate AML Leukemia Stem Cells (LSCs) by inducing the simultaneous depletion of MLL-fusion proteins and the c-MYB transcription factor, thereby sensitizing res
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/hypotheses/hyp_d577dce14a8b60fd.json`](data/artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/hypotheses/hyp_d577dce14a8b60fd.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGCJSNBQA3C0Y09736DSJVW/bench/bnc_01KSGCJSN8MGMK6H3KZVVQZJPG.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGCJSN8MGMK6H3KZVVQZJPG';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGCJSN8MGMK6H3KZVVQZJPG';
```

<a id="bench-bnc_01ksgcksg3mjkvpdzbzxm3th2g"></a>
## Bench `bnc_01KSGCKSG3MJKVPDZBZXM3TH2G`

- **Created:** 2026-05-25T20:18:24.903406+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-5` (size 5)
- **Total cost:** $0.9852
- **Matches played:** 6
- **Session:** `ses_01KSGCKSG89DHGN780150HFR4C`
- **Bench artifact:** `artifacts/ses_01KSGCKSG89DHGN780150HFR4C/bench/bnc_01KSGCKSG3MJKVPDZBZXM3TH2G.json`

**Goal:**

> Identify FDA-approved drugs that could be repurposed as therapeutic candidates for acute myeloid leukemia (AML). For each hypothesis, name the specific approved drug (its INN or brand name), describe the molecular mechanism by which it would act against AML blasts or leukemic stem cells, and propose a concrete in vitro or in vivo experiment to test it.

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-haiku-4.5` | pipeline | 2 | 3-0 | 1224 | 0/5 | $0.3922 | 350,189 / 8,402 | 54.1s |  |
| `openai-o1` | pipeline | 2 | 2-1 | 1208 | 0/5 | $0.5341 | 10,441 / 6,291 | 19.3s |  |
| `gemini-2-pro` | pipeline | 2 | 1-2 | 1192 | 0/5 | $0.0540 | 15,522 / 3,457 | 29.1s |  |
| `gemini-2-flash-thinking` | pipeline | 2 | 0-3 | 1176 | 0/5 | $0.0049 | 41,386 / 2,001 | 10.6s |  |

### Hypotheses surfaced (8 total)

- **Trifluoperazine-mediated ferroptosis induction in AML** — via `openai-o1 (pipeline)`
  - We hypothesize that the FDA-approved antipsychotic trifluoperazine can be repurposed to selectively induce ferroptosis in AML blasts and leukemic stem cells by targeting the Nrf2/SLC7A11/GPX4 antioxidant axis.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_282added778413c0.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_282added778413c0.json)
- **Belumosudil for AML** — via `gemini-2-flash-thinking (pipeline)`
  - Belumosudil, an FDA-approved ROCK2 inhibitor, can inhibit the proliferation and promote the differentiation of AML blasts and leukemic stem cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_b9ecde9089d61997.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_b9ecde9089d61997.json)
- **Metformin-induced ferroptosis in IDH2-/FLT3-mutant AML** — via `openai-o1 (pipeline)`
  - We hypothesize that repurposing metformin, an FDA-approved antidiabetic drug, will induce ferroptosis and suppress AML blasts, especially IDH2-/FLT3-mutant subtypes, by impairing oxidative phosphorylation and driving lipid metabolic remodel
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_ca4b9050f7f3e680.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_ca4b9050f7f3e680.json)
- **Itraconazole for AML by Targeting OXPHOS** — via `gemini-2-pro (pipeline)`
  - Itraconazole, an FDA-approved antifungal drug, can be repurposed as a therapeutic agent for Acute Myeloid Leukemia (AML) by targeting mitochondrial oxidative phosphorylation in leukemic cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_cdedf3366cef4a73.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_cdedf3366cef4a73.json)
- **Itraconazole-mediated OXPHOS inhibition targets therapy-resistant AML leukemic stem cells** — via `claude-haiku-4.5 (pipeline)`
  - Itraconazole, an FDA-approved azole antifungal targeting CYP51A1, inhibits mitochondrial electron transport chain complex I activity to suppress oxidative phosphorylation and selectively eradicate therapy-resistant leukemic stem cells in AM
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_d6232451243be7b2.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_d6232451243be7b2.json)
- **Itraconazole as a repurposed therapeutic for acute myeloid leukemia.** — via _(no match table entry)_
  - The FDA-approved antifungal drug itraconazole can be repurposed for the treatment of acute myeloid leukemia (AML) by targeting leukemic stem cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_d75698c7976be57f.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_d75698c7976be57f.json)
- **Cabozantinib for FLT3-ITD AML** — via `gemini-2-flash-thinking (pipeline)`
  - Cabozantinib can be repurposed for the treatment of FLT3-ITD positive acute myeloid leukemia.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_eaeca72f89d53fdb.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_eaeca72f89d53fdb.json)
- **Itraconazole as OXPHOS inhibitor for AML LSC targeting** — via `claude-haiku-4.5 (pipeline)`
  - Itraconazole, an FDA-approved azole antifungal, can be repurposed to selectively eradicate therapy-resistant leukemic stem cells in AML by inhibiting CYP51A1-dependent electron transport chain complex I activity, thereby suppressing oxidati
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_ee162a788222e25e.json`](data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/hyp_ee162a788222e25e.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGCKSG89DHGN780150HFR4C/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGCKSG89DHGN780150HFR4C/bench/bnc_01KSGCKSG3MJKVPDZBZXM3TH2G.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGCKSG3MJKVPDZBZXM3TH2G';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGCKSG3MJKVPDZBZXM3TH2G';
```

<a id="bench-bnc_01ksgd0wkfyaf2x15p99bfax01"></a>
## Bench `bnc_01KSGD0WKFYAF2X15P99BFAX01`

- **Created:** 2026-05-25T20:25:34.067046+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `(none)`
- **Total cost:** $0.1452
- **Matches played:** 12
- **Session:** `ses_01KSGD0WKK7XZC5H4BBZT0QX6K`
- **Bench artifact:** `artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/bench/bnc_01KSGD0WKFYAF2X15P99BFAX01.json`

**Goal:**

> Identify FDA-approved drugs that could be repurposed for AML. For each hypothesis, name a specific approved drug (INN/brand), describe the molecular mechanism in AML, and propose a concrete experiment.

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash3-pipe` | pipeline | 1 | 4-2 | 1234 | 0/— | $0.0275 | 78,505 / 1,568 | 14.9s |  |
| `flash3-raw` | direct | 2 | 5-1 | 1228 | 0/— | $0.0033 | 794 / 1,205 | 5.0s |  |
| `gpt4o-pipe` | pipeline | 1 | 3-3 | 1201 | 0/— | $0.1040 | 34,909 / 1,671 | 18.5s |  |
| `gpt4o-raw` | direct | 2 | 0-6 | 1154 | 0/— | $0.0105 | 882 / 833 | 3.7s |  |

### Hypotheses surfaced (6 total)

- **Repurposing Arsenic Trioxide for Targeting p53 Mutations in AML** — via `gpt4o-pipe (pipeline)`
  - Arsenic Trioxide (ATO) can be repurposed to target and rescue structural p53 mutations in AML, thereby restoring its tumor suppressor function and inducing apoptosis in malignant cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_58d446efd9c70dd2.json`](data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_58d446efd9c70dd2.json)
- **Auranofin Repurposing for AML via Thioredoxin Reductase Inhibition** — via `flash3-raw (direct)`
  - Auranofin induces selective apoptosis in acute myeloid leukemia (AML) cells by irreversibly inhibiting thioredoxin reductase 1 (TXNRD1), thereby overwhelming the cell's antioxidant capacity and triggering ROS-mediated programmed cell death.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_7a8cb45f9aab9603.json`](data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_7a8cb45f9aab9603.json)
- **Repurposing Venetoclax for AML targeting BCL-2** — via `gpt4o-raw (direct)`
  - Venetoclax could be repurposed to treat Acute Myeloid Leukemia (AML) by targeting the anti-apoptotic protein BCL-2, leading to increased apoptosis of AML cells.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_80a4bdeff205988e.json`](data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_80a4bdeff205988e.json)
- **Repurposing Pimavanserin for FLT3-ITD Acute Myeloid Leukemia via 5-HT2AR Antagonism** — via `flash3-raw (direct)`
  - Pimavanserin (Nuplazid) inhibits AML progression by acting as an inverse agonist of the 5-HT2A receptor, thereby suppressing the hyperactivated STAT5 and AKT signaling pathways in FLT3-ITD-positive leukemic cells.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_97551af04eee3997.json`](data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_97551af04eee3997.json)
- **Repurposing Itraconazole to Overcome Venetoclax Resistance in AML via Mitochondrial Complex I Inhibition** — via `flash3-pipe (pipeline)`
  - The FDA-approved antifungal Itraconazole sensitizes Acute Myeloid Leukemia (AML) cells and leukemic stem cells to Venetoclax by inhibiting CYP51A1-dependent mitochondrial Complex I activity, thereby disrupting the metabolic compensatory hig
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_af38040d776d69f1.json`](data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_af38040d776d69f1.json)
- **Repurposing Bosutinib for AML Treatment** — via `gpt4o-raw (direct)`
  - Bosutinib, a tyrosine kinase inhibitor, can be repurposed to treat acute myeloid leukemia by inhibiting abnormal signaling pathways.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_b552ad85a384e282.json`](data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/hyp_b552ad85a384e282.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGD0WKK7XZC5H4BBZT0QX6K/bench/bnc_01KSGD0WKFYAF2X15P99BFAX01.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGD0WKFYAF2X15P99BFAX01';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGD0WKFYAF2X15P99BFAX01';
```

<a id="bench-bnc_01ksgv99yg7d4dxz8g6pejwkv1"></a>
## Bench `bnc_01KSGV99YG7D4DXZ8G6PEJWKV1`

- **Created:** 2026-05-26T00:34:49.940013+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-5` (size 5)
- **Total cost:** $0.0000
- **Matches played:** 0
- **Session:** `ses_01KSGV99YMT4S6605JKT61P244`
- **Bench artifact:** `artifacts/ses_01KSGV99YMT4S6605JKT61P244/bench/bnc_01KSGV99YG7D4DXZ8G6PEJWKV1.json`

**Goal:**

> Produce a ranked list of drug repurposing candidates for acute myeloid leukemia (AML), strictly under the following constraints:  (1) Each candidate must NOT have prior published evidence of being repurposed for AML, and there must be no preclinical studies in AML for the proposed compound at the time of writing. (2) Use only your internal knowledge. Do NOT assume access to DepMap dependency scores, gene-essentiality datasets, transcriptomic screens, or human expert curation. No external inputs. (3) Name the specific compound (INN, brand name, or research-code alias) — do not propose generic d…

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-haiku-4.5` | pipeline | 0 | — | — | 0/5 | $0.0000 | — | — |  |
| `gemini-2-flash-thinking` | pipeline | 0 | — | — | 0/5 | $0.0000 | — | — |  |
| `gemini-2-pro` | pipeline | 0 | — | — | 0/5 | $0.0000 | — | — |  |
| `openai-o1` | pipeline | 0 | — | — | 0/5 | $0.0000 | — | — |  |

_No hypotheses produced (every candidate failed)._

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGV99YMT4S6605JKT61P244/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGV99YMT4S6605JKT61P244/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGV99YMT4S6605JKT61P244/bench/bnc_01KSGV99YG7D4DXZ8G6PEJWKV1.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGV99YG7D4DXZ8G6PEJWKV1';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGV99YG7D4DXZ8G6PEJWKV1';
```

<a id="bench-bnc_01ksgvhy16q0ghne9zjwzygfng"></a>
## Bench `bnc_01KSGVHY16Q0GHNE9ZJWZYGFNG`

- **Created:** 2026-05-26T00:39:32.649554+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-top3` (size 3)
- **Total cost:** $1.8881
- **Matches played:** 6
- **Session:** `ses_01KSGVHY1A3Q59WY2RWXP37J4D`
- **Bench artifact:** `artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/bench/bnc_01KSGVHY16Q0GHNE9ZJWZYGFNG.json`

**Goal:**

> Produce a ranked list of drug repurposing candidates for acute myeloid leukemia (AML), strictly under the following constraints:  (1) Each candidate must NOT have prior published evidence of being repurposed for AML, and there must be no preclinical studies in AML for the proposed compound at the time of writing. (2) Use only your internal knowledge. Do NOT assume access to DepMap dependency scores, gene-essentiality datasets, transcriptomic screens, or human expert curation. No external inputs. (3) Name the specific compound (INN, brand name, or research-code alias) — do not propose generic d…

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai-o1` | pipeline | 2 | 2-2 | 1201 | 0/3 | $1.1566 | 14,718 / 15,598 | 30.9s |  |
| `gemini-2-flash-thinking` | pipeline | 2 | 2-2 | 1199 | 0/3 | $0.0015 | 4,905 / 2,583 | 7.9s |  |
| `gemini-2-pro` | pipeline | 1 | 2-2 | 1199 | 0/3 | $0.1540 | 32,906 / 11,285 | 31.1s |  |
| `claude-haiku-4.5` | pipeline | 0 | — | — | 0/3 | $0.5760 | 544,629 / 6,270 | — |  |

### Hypotheses surfaced (5 total)

- **Riluzole-induced redox disruption as a novel AML therapy** — via `openai-o1 (pipeline)`
  - We hypothesize that Riluzole’s blockade of glutamate release and subsequent reduction of cystine uptake can selectively induce oxidative stress and apoptosis in acute myeloid leukemia cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_02bd0ee28ba70b71.json`](data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_02bd0ee28ba70b71.json)
- **Carglumic Acid for the Treatment of Acute Myeloid Leukemia** — via `gemini-2-pro (pipeline)`
  - Carglumic acid, an FDA-approved drug for hyperammonemia, can be repurposed to treat Acute Myeloid Leukemia (AML) by inhibiting pyrimidine biosynthesis in leukemic cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_18395516265ebfe4.json`](data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_18395516265ebfe4.json)
- **Auranofin as TrxR inhibitor in AML** — via `gemini-2-flash-thinking (pipeline)`
  - Auranofin, an inhibitor of thioredoxin reductase (TrxR), can induce oxidative stress and apoptosis in AML cells, including leukemic stem cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_66bf74de0f5a57b8.json`](data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_66bf74de0f5a57b8.json)
- **Targeting SRPK1 in AML with Seclidemstat** — via `gemini-2-flash-thinking (pipeline)`
  - Seclidemstat, an investigational lysine-specific histone demethylase 1A (LSD1) inhibitor, can be repurposed to treat acute myeloid leukemia (AML) by inhibiting serine/arginine-rich protein kinase 1 (SRPK1), thereby disrupting RNA splicing a
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_753d9069756fc70f.json`](data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_753d9069756fc70f.json)
- **Teprotumumab Repurposing for AML via IGF1R Blockade** — via `openai-o1 (pipeline)`
  - Teprotumumab, an IGF1R monoclonal antibody originally developed for thyroid eye disease, can disrupt survival signals in AML blasts and leukemic stem cells by blocking the IGF-1 growth pathway.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_d8b68f4010144129.json`](data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/hyp_d8b68f4010144129.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSGVHY1A3Q59WY2RWXP37J4D/bench/bnc_01KSGVHY16Q0GHNE9ZJWZYGFNG.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSGVHY16Q0GHNE9ZJWZYGFNG';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSGVHY16Q0GHNE9ZJWZYGFNG';
```

<a id="bench-bnc_01ksn03hg9vw3cpyd40sa51neb"></a>
## Bench `bnc_01KSN03HG9VW3CPYD40SA51NEB`

- **Created:** 2026-05-27T15:16:01.677338+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-top3` (size 3)
- **Total cost:** $0.8309
- **Matches played:** 30
- **Session:** `ses_01KSN03HGEWWH78FVR954ZQ5KA`
- **Bench artifact:** `artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/bench/bnc_01KSN03HG9VW3CPYD40SA51NEB.json`

**Goal:**

> Produce a ranked list of drug repurposing candidates for acute myeloid leukemia (AML), strictly under the following constraints:  (1) Each candidate must NOT have prior published evidence of being repurposed for AML, and there must be no preclinical studies in AML for the proposed compound at the time of writing. (2) Use only your internal knowledge. Do NOT assume access to DepMap dependency scores, gene-essentiality datasets, transcriptomic screens, or human expert curation. No external inputs. (3) Name the specific compound (INN, brand name, or research-code alias) — do not propose generic d…

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-haiku-4.5[pipe]` | pipeline | 1 | 10-0 | 1300 | 0/3 | $0.1365 | 118,967 / 3,497 | 54.3s |  |
| `gemini-2-pro[raw]` | direct | 1 | 8-2 | 1277 | 0/3 | $0.0291 | 541 / 2,838 | 31.4s |  |
| `openai-o1[pipe]` | pipeline | 1 | 6-4 | 1221 | 0/3 | $0.3142 | 4,517 / 4,107 | 132.1s |  |
| `openai-o1[raw]` | direct | 1 | 4-6 | 1178 | 0/3 | $0.2905 | 612 / 4,688 | 53.9s |  |
| `claude-haiku-4.5[raw]` | direct | 1 | 1-9 | 1120 | 0/3 | $0.0077 | 1,432 / 1,259 | 11.3s |  |
| `gemini-2-flash-thinking[pipe]` | pipeline | 1 | 1-9 | 1103 | 0/3 | $0.0031 | 28,419 / 760 | 21.4s |  |
| `gemini-2-flash-thinking[raw]` | direct | 0 | — | — | 0/3 | $0.0000 | — | — |  |
| `gemini-2-pro[pipe]` | pipeline | 0 | — | — | 0/3 | $0.0499 | 19,624 / 2,536 | — |  |

### Hypotheses surfaced (6 total)

- **Serine Hydroxymethyltransferase 2 (SHMT2) Inhibition for AML** — via `claude-haiku-4.5 (pipeline)`
  - The small molecule SHMT2 inhibitor SHIN2 will selectively kill AML blasts and leukemic stem cells by disrupting one-carbon metabolism and purine synthesis while sparing normal hematopoietic stem cells.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_086a8cf008dba0af.json`](data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_086a8cf008dba0af.json)
- **Spliceostatin for AML** — via `gemini-2-flash-thinking (pipeline)`
  - Spliceostatin, an inhibitor of SF3b, can induce apoptosis and impair proliferation of AML cells, particularly in subtypes dependent on specific splicing programs.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_1886ac45bfbae3e8.json`](data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_1886ac45bfbae3e8.json)
- **Repurposing Mirabegron for AML via Bone Marrow Neuropathy Modulation** — via `openai-o1 (pipeline)`
  - Mirabegron, a β3-adrenergic receptor agonist, can restore sympathetic innervation in the AML bone marrow niche and thereby reduce leukemic expansion.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_75d7eacb11e5375c.json`](data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_75d7eacb11e5375c.json)
- **Brensocatib targeting neutrophil protease-driven AML progression** — via `openai-o1 (direct)`
  - Brensocatib, a dipeptidyl peptidase 1 (DPP1) inhibitor, prevents the activation of key neutrophil proteases in the bone marrow niche, potentially limiting AML blast survival.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_b13a0ac424bc8729.json`](data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_b13a0ac424bc8729.json)
- **Repurposing MLH1 inhibitors for AML via synthetic lethality with p53 mutations** — via `claude-haiku-4.5 (direct)`
  - MLH1 inhibitors (specifically transient MLH1 pathway inhibition via small molecules) will selectively kill AML blasts harboring TP53 mutations through synthetic lethality mechanisms that exploit mismatch-repair deficiency in the context of 
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_d30978fd338be65b.json`](data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_d30978fd338be65b.json)
- **Repurposing Nitazoxanide to Target Leukemic Stem Cells in AML** — via `gemini-2-pro (direct)`
  - The FDA-approved antiparasitic drug Nitazoxanide will induce apoptosis and inhibit self-renewal in acute myeloid leukemia (AML) leukemic stem cells (LSCs) by disrupting the Wnt/β-catenin signaling pathway.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_d55752d9d8fe110a.json`](data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/hyp_d55752d9d8fe110a.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSN03HGEWWH78FVR954ZQ5KA/bench/bnc_01KSN03HG9VW3CPYD40SA51NEB.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSN03HG9VW3CPYD40SA51NEB';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSN03HG9VW3CPYD40SA51NEB';
```

<a id="bench-bnc_01ksn0zmjzv12f6mde63h7mw9r"></a>
## Bench `bnc_01KSN0ZMJZV12F6MDE63H7MW9R`

- **Created:** 2026-05-27T15:31:22.336476+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-top3` (size 3)
- **Total cost:** $2.0820
- **Matches played:** 56
- **Session:** `ses_01KSN0ZMK1X3V4JD71YZT226A4`
- **Bench artifact:** `artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/bench/bnc_01KSN0ZMJZV12F6MDE63H7MW9R.json`

**Goal:**

> Produce a ranked list of drug repurposing candidates for acute myeloid leukemia (AML), strictly under the following constraints:  (1) Each candidate must NOT have prior published evidence of being repurposed for AML, and there must be no preclinical studies in AML for the proposed compound at the time of writing. (2) Use only your internal knowledge. Do NOT assume access to DepMap dependency scores, gene-essentiality datasets, transcriptomic screens, or human expert curation. No external inputs. (3) Name the specific compound (INN, brand name, or research-code alias) — do not propose generic d…

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-opus-4.7[pipe]` | pipeline | 1 | 14-0 | 1367 | 0/3 | $0.7966 | 27,151 / 5,191 | 79.4s |  |
| `gemini-3-pro[raw]` | direct | 1 | 12-2 | 1275 | 0/3 | $0.0521 | 541 / 11,710 | 133.0s |  |
| `claude-opus-4.7[raw]` | direct | 1 | 10-4 | 1270 | 0/3 | $0.2244 | 2,051 / 2,582 | 40.9s |  |
| `gemini-3-pro[pipe]` | pipeline | 1 | 7-7 | 1186 | 0/3 | $0.1298 | 105,896 / 3,027 | 52.6s |  |
| `gpt-5[pipe]` | pipeline | 1 | 6-8 | 1172 | 0/3 | $0.6786 | 81,730 / 13,497 | 294.5s |  |
| `gpt-5[raw]` | direct | 1 | 5-9 | 1146 | 0/3 | $0.1894 | 615 / 9,314 | 170.9s |  |
| `gemini-3-flash[raw]` | direct | 1 | 2-12 | 1110 | 0/3 | $0.0017 | 582 / 598 | 4.7s |  |
| `gemini-3-flash[pipe]` | pipeline | 1 | 0-14 | 1074 | 0/3 | $0.0095 | 21,472 / 1,212 | 12.7s |  |

### Hypotheses surfaced (8 total)

- **Repurposing Meldonium to Target Fatty Acid Oxidation Addiction in AML Leukemic Stem Cells via Carnitine Depletion** — via `gemini-3-pro (direct)`
  - Repurposing the anti-ischemic drug meldonium will selectively eradicate acute myeloid leukemia (AML) stem cells by competitively inhibiting the carnitine transporter OCTN2 (SLC22A5) and the biosynthetic enzyme BBOX1, thereby enforcing a let
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_40274ef161482bcd.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_40274ef161482bcd.json)
- **Pirfenidone as a TGF-β/p38–axis repurposing candidate to disrupt the AML bone-marrow niche and sensitize LSCs** — via `claude-opus-4.7 (direct)`
  - Pirfenidone, the anti-fibrotic drug approved for idiopathic pulmonary fibrosis, will selectively impair AML leukemic stem cell (LSC) maintenance and chemoresistance by simultaneously dampening TGF-β1/SMAD signaling and p38 MAPK activity in 
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_4c63943c13f5ddb9.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_4c63943c13f5ddb9.json)
- **Pitavastatin-Induced Isoprenylation Depletion in Acute Myeloid Leukemia** — via `gemini-3-flash (direct)`
  - Pitavastatin exerts anti-leukemic activity in AML by depleting geranylgeranyl pyrophosphate pools, thereby disrupting the membrane localization of Rho-family GTPases necessary for AML blast survival and proliferation.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_8899dbe4250d21e0.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_8899dbe4250d21e0.json)
- **Bazedoxifene repurposing for AML via GP130/IL-6/STAT3 blockade in leukemic stem cells** — via `claude-opus-4.7 (pipeline)`
  - Bazedoxifene, an FDA-approved selective estrogen-receptor modulator that also functions as a small-molecule inhibitor of the IL-6/IL-11–GP130 interface and thereby blocks JAK1–STAT3 activation, will selectively impair AML blast survival and
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_92e077ffae6cca81.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_92e077ffae6cca81.json)
- **Repurposing Cefiderocol as an Iron-Depleting Agent to Induce Ferroptosis in Acute Myeloid Leukemia** — via `gemini-3-flash (pipeline)`
  - Cefiderocol acts as a potent anti-leukemic agent in AML by sequestering extracellular iron through its siderophore moiety, thereby inducing intracellular iron depletion and triggering ferroptotic cell death in iron-dependent leukemic blasts
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_b105be77ee34e084.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_b105be77ee34e084.json)
- **Belapectin (GR-MD-02) to disrupt galectin-3–mediated stromal adhesion and survival signaling in AML LSCs** — via `gpt-5 (direct)`
  - The galectin-3 inhibitor belapectin (GR-MD-02) will impair acute myeloid leukemia blasts and leukemic stem cells by blocking galectin-3–dependent adhesion and pro-survival signaling within the bone-marrow niche, thereby mobilizing cells fro
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_c736f06b018da456.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_c736f06b018da456.json)
- **ND-646 for targeting Acetyl-CoA Carboxylase (ACC) in Acute Myeloid Leukemia** — via `gemini-3-pro (pipeline)`
  - ND-646, a potent and specific allosteric inhibitor of acetyl-CoA carboxylase (ACC), will effectively eradicate acute myeloid leukemia blasts and leukemic stem cells by simultaneously blocking de novo lipogenesis and dysregulating fatty acid
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_d924779f909ddc4e.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_d924779f909ddc4e.json)
- **Ranolazine as a partial fatty-acid-oxidation (pFOX) modulator to extinguish OXPHOS-dependent AML stemness** — via `gpt-5 (pipeline)`
  - Ranolazine, an antianginal late-Na+ current inhibitor with partial fatty‑acid‑oxidation (pFOX)–modulating activity, will suppress oxidative‑phosphorylation–dependent acute myeloid leukemia (AML) blasts and leukemic stem cells by limiting β‑
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_e5486ae68bc054fe.json`](data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/hyp_e5486ae68bc054fe.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSN0ZMK1X3V4JD71YZT226A4/bench/bnc_01KSN0ZMJZV12F6MDE63H7MW9R.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSN0ZMJZV12F6MDE63H7MW9R';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSN0ZMJZV12F6MDE63H7MW9R';
```

<a id="bench-bnc_01ksn8wz9xzp0hsw68ysrdbr77"></a>
## Bench `bnc_01KSN8WZ9XZP0HSW68YSRDBR77`

- **Created:** 2026-05-27T17:49:43.617521+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-top3` (size 3)
- **Total cost:** $1.1057
- **Matches played:** 42
- **Session:** `ses_01KSN8WZA25QV3RS8F8VKJNS98`
- **Bench artifact:** `artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/bench/bnc_01KSN8WZ9XZP0HSW68YSRDBR77.json`

**Goal:**

> Produce a ranked list of drug repurposing candidates for acute myeloid leukemia (AML), strictly under the following constraints:  (1) Each candidate must NOT have prior published evidence of being repurposed for AML, and there must be no preclinical studies in AML for the proposed compound at the time of writing. (2) Use only your internal knowledge. Do NOT assume access to DepMap dependency scores, gene-essentiality datasets, transcriptomic screens, or human expert curation. No external inputs. (3) Name the specific compound (INN, brand name, or research-code alias) — do not propose generic d…

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-2-pro[raw]` | direct | 1 | 12-0 | 1340 | 0/3 | $0.0413 | 541 / 4,064 | 43.8s |  |
| `claude-haiku-4.5[raw]` | direct | 1 | 10-2 | 1269 | 0/3 | $0.0078 | 1,432 / 1,280 | 14.3s |  |
| `claude-haiku-4.5[pipe]` | pipeline | 1 | 8-4 | 1241 | 0/3 | $0.1164 | 99,197 / 3,450 | 87.2s |  |
| `openai-o1[pipe]` | pipeline | 1 | 6-6 | 1214 | 0/3 | $0.6521 | 15,418 / 7,013 | 148.1s |  |
| `openai-o1[raw]` | direct | 1 | 4-8 | 1185 | 0/3 | $0.2398 | 612 / 3,844 | 30.8s |  |
| `gemini-2-flash-thinking[raw]` | direct | 1 | 2-10 | 1099 | 0/3 | $0.0003 | 541 / 525 | 5.6s |  |
| `gemini-2-flash-thinking[pipe]` | pipeline | 1 | 0-12 | 1051 | 0/3 | $0.0052 | 47,466 / 1,258 | 18.1s |  |
| `gemini-2-pro[pipe]` | pipeline | 0 | — | — | 0/3 | $0.0427 | 7,842 / 3,288 | — |  |

### Hypotheses surfaced (7 total)

- **Trimetazidine as a Fatty Acid Oxidation Inhibitor for an AML Subgroup** — via `gemini-2-pro (direct)`
  - Trimetazidine, a clinically approved anti-anginal drug, will selectively eliminate acute myeloid leukemia (AML) stem cells by inhibiting their metabolic dependency on fatty acid oxidation.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_056855d4417a9917.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_056855d4417a9917.json)
- **DPP4 inhibition to mobilize and sensitize leukemic stem cells in AML** — via `claude-haiku-4.5 (pipeline)`
  - Sitagliptin (a clinically approved DPP4 inhibitor) will disrupt the CXCL12-DPP4-GPC3 axis at the bone marrow niche, dislodge quiescent leukemic stem cells from protective metaphyseal microenvironments, and re-sensitize them to cytarabine or
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_3c20191c39846f87.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_3c20191c39846f87.json)
- **Repurposing Apomorphine for AML treatment** — via `gemini-2-flash-thinking (direct)`
  - Apomorphine, a dopamine receptor agonist, can induce apoptosis in AML cells by modulating the PI3K/AKT signaling pathway.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_430f01e4e8f2ecbf.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_430f01e4e8f2ecbf.json)
- **Belzutifan (HIF-2α Inhibitor) for AML Repurposing** — via `openai-o1 (direct)`
  - We hypothesize that Belzutifan, a small-molecule inhibitor specific for HIF-2α, can reduce AML blast cell viability and selfsustaining leukemic stem cell activity by blocking hypoxia-driven signaling required for leukemic transformation.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_9a5fe6dd13e209db.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_9a5fe6dd13e209db.json)
- **Repurposing Mitapivat for AML via RBC glycolytic enhancement and metabolic vulnerability** — via `claude-haiku-4.5 (direct)`
  - Mitapivat, a pyruvate kinase-M2 (PKM2) activator approved for sickle cell disease, will selectively induce differentiation and apoptosis in AML blasts by amplifying glycolytic flux and ROS stress, exploiting the Warburg metabolism dependenc
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_9c4e620c196843c7.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_9c4e620c196843c7.json)
- **Roflumilast-mediated PDE4 inhibition as a novel anti-AML strategy** — via `openai-o1 (pipeline)`
  - We hypothesize that roflumilast’s selective inhibition of phosphodiesterase-4 (PDE4) can suppress AML blasts by elevating intracellular cAMP and thereby impairing leukemogenic signaling.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_a6e0a54a55465b0d.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_a6e0a54a55465b0d.json)
- **RepSox as Differentiation Agent in AML** — via `gemini-2-flash-thinking (pipeline)`
  - RepSox, a TGF-beta receptor I inhibitor, can induce differentiation of AML blasts and leukemic stem cells, particularly those resistant to standard chemotherapy, by disrupting TGF-beta signaling.
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_cd306bcde4aef8f5.json`](data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/hyp_cd306bcde4aef8f5.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSN8WZA25QV3RS8F8VKJNS98/bench/bnc_01KSN8WZ9XZP0HSW68YSRDBR77.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSN8WZ9XZP0HSW68YSRDBR77';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSN8WZ9XZP0HSW68YSRDBR77';
```

<a id="bench-bnc_01ksn98v3sab2x0xdsvv5pyqkm"></a>
## Bench `bnc_01KSN98V3SAB2X0XDSVV5PYQKM`

- **Created:** 2026-05-27T17:56:12.541886+00:00
- **Status:** done
- **Judge:** `openrouter:google/gemini-3-flash-preview`
- **Gold set at runtime:** `aml-repurposing-paper-top3` (size 3)
- **Total cost:** $0.1612
- **Matches played:** 12
- **Session:** `ses_01KSN98V3YK9DBTQWK9FRRY0Q4`
- **Bench artifact:** `artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/bench/bnc_01KSN98V3SAB2X0XDSVV5PYQKM.json`

**Goal:**

> Produce a ranked list of drug repurposing candidates for acute myeloid leukemia (AML), strictly under the following constraints:  (1) Each candidate must NOT have prior published evidence of being repurposed for AML, and there must be no preclinical studies in AML for the proposed compound at the time of writing. (2) Use only your internal knowledge. Do NOT assume access to DepMap dependency scores, gene-essentiality datasets, transcriptomic screens, or human expert curation. No external inputs. (3) Name the specific compound (INN, brand name, or research-code alias) — do not propose generic d…

### Candidates

| label | mode | n_hyps | W-L | Elo | hits (runtime) | $ | tokens (in / out) | p50 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-2.5-flash[pipe]` | pipeline | 1 | 6-0 | 1284 | 0/3 | $0.0132 | 16,650 / 3,290 | 26.2s |  |
| `gemini-2.5-pro[pipe]` | pipeline | 1 | 4-2 | 1226 | 0/3 | $0.1071 | 17,197 / 8,558 | 107.1s |  |
| `gemini-2.5-pro[raw]` | direct | 1 | 2-4 | 1179 | 0/3 | $0.0392 | 541 / 3,852 | 50.0s |  |
| `gemini-2.5-flash[raw]` | direct | 1 | 0-6 | 1112 | 0/3 | $0.0017 | 541 / 610 | 16.3s |  |

### Hypotheses surfaced (4 total)

- **Repurposing Niclosamide for AML via STAT3 Inhibition** — via `gemini-2.5-flash (direct)`
  - Niclosamide, an anthelmintic drug, can be repurposed for acute myeloid leukemia (AML) treatment by inhibiting the STAT3 signaling pathway, thereby reducing leukemic cell proliferation and survival.
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_2280caf8fd669dd3.json`](data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_2280caf8fd669dd3.json)
- **Repurposing of the ERAD Inhibitor Eeyarestatin I for the Treatment of Acute Myeloid Leukemia** — via `gemini-2.5-pro (pipeline)`
  - The small molecule Eeyarestatin I, an inhibitor of endoplasmic reticulum-associated degradation (ERAD), will selectively induce apoptosis in acute myeloid leukemia (AML) cells, which are dependent on robust protein quality control pathways 
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_4816ba4892736e1c.json`](data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_4816ba4892736e1c.json)
- **Denifanstat for Acute Myeloid Leukemia via FASN Inhibition** — via `gemini-2.5-flash (pipeline)`
  - Denifanstat, a Fatty Acid Synthase (FASN) inhibitor, will demonstrate therapeutic efficacy against acute myeloid leukemia (AML) blasts and leukemic stem cells (LSCs) by disrupting aberrant lipid metabolism, leading to lipotoxicity, energeti
  - mode: `pipeline` · artifact: [`data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_86ff66d295a2f5b4.json`](data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_86ff66d295a2f5b4.json)
- **RNA Polymerase I Inhibition via CX-5461 for AML Therapy** — via `gemini-2.5-pro (direct)`
  - The small molecule CX-5461, by selectively inhibiting RNA Polymerase I (Pol I)-mediated transcription of ribosomal RNA, will induce potent and selective apoptosis in acute myeloid leukemia (AML) cells, which are addicted to high rates of ri
  - mode: `direct` · artifact: [`data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_e768dbad247673a9.json`](data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/hyp_e768dbad247673a9.json)

### Recall across known gold sets (post-hoc rescore)

- · `aml-repurposing-paper-5` (5 entities): **0/5** → _none_
- · `aml-repurposing-paper-top3` (3 entities): **0/3** → _none_

### Files

- Hypotheses (all `record_hypothesis` payloads): `data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/hypotheses/`
- LLM transcripts (request + response per call): `data/artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/transcripts/generation/`
- Bench summary JSON (per-candidate `gold_hit_detail` with alias / field / hyp): `artifacts/ses_01KSN98V3YK9DBTQWK9FRRY0Q4/bench/bnc_01KSN98V3SAB2X0XDSVV5PYQKM.json`

**SQL to inspect this bench:**

```sql
-- per-candidate detail
SELECT label, mode, n_hypotheses, wins, losses,
       round(mean_elo,0), gold_hits, gold_hit_names,
       round(total_cost_usd, 4),
       total_input_tok, total_output_tok
  FROM bench_candidates
 WHERE bench_id='bnc_01KSN98V3SAB2X0XDSVV5PYQKM';

-- every match with judge rationale
SELECT bc_a.label, bc_b.label, bm.winner,
       round(bm.judge_cost_usd, 4),
       substr(bm.rationale, 1, 200)
  FROM bench_matches bm
  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
 WHERE bm.bench_id='bnc_01KSN98V3SAB2X0XDSVV5PYQKM';
```
