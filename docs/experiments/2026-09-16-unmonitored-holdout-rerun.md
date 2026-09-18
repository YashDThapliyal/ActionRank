# Unmonitored final-checkpoint evaluation

Status: pre-registered, not yet run. This file is committed before `results/06-unmonitored-holdout/` exists.

## What this is, and is not

The headline comparison (README section 4.4, blog post) is on the first 500 steps of the held-out split. Both final training runs logged accuracy on a prefix of that split after every epoch: `train_tier2.py` on steps [0, 200), `baseline_sft.py` on steps [0, 100). No checkpoint was selected on those logs, but the numbers were visible while decisions were made, so the 500 is a development set.

This run evaluates the two frozen final checkpoints, once, on held-out steps [503, 1855): 1,352 steps from 371 trajectories that no training-time log ever scored and that share no trajectory with the development prefix (trajectory 19681 straddles steps 498 to 502, so the cut is at 503, not 500).

It is a **confirmatory evaluation, not an independent test**. The frozen Tier 1 heads (table and span, both pooling modes) were evaluated on the full 1,855-step held-out split during development (`train_tier1.py`, `held_out_eval` in each `tier1_span_history.json`; README section 4.3 quotes those numbers). Those aggregate results informed the choice of span head and last-token pooling, and the final scorer's head is initialised from that stage. So steps 503 onward are untouched by the final checkpoints' epoch monitoring, but not by the broader model-development process. A truly untouched test needs a predeclared train/dev/test trajectory split and retraining; that is listed in the README's next steps.

## Fixed before the run

- **Checkpoints.** Scorer: `checkpoints/last/tier2_span` (LoRA r=8 on q/v + span head, last-token pooling, 3 epochs, head initialised from `checkpoints/last/tier1_span_head.pt`). Generator: `checkpoints/baseline_sft3` (LoRA r=8 on q/v, answer-only loss, 3 epochs from the base model). Neither is retrained or reselected.
- **Steps.** `select_eval(ds.eval, 503, None)`: 1,352 steps, 371 trajectories, from the seed-13 split in the dataset's deterministic order. Composition: 372 `Finish` steps, 651 non-`Finish` steps whose label was a training label, 329 whose label never was. Zero trajectory overlap with `ds.eval[:500]`.
- **Config.** `configs/unmonitored_holdout.yaml`: the final scorer config with the generator checkpoint pointed at the 3-epoch run and `results_dir` set to `results/06-unmonitored-holdout`. Its training fields (`tier2.epochs`, `max_train_examples`, `sft_epochs`) are inert for an evaluation-only run. Backbone, prompt format, truncation, beam count (5), warm-up (3) and device (MPS, fp16, batch size 1) are unchanged from every earlier evaluation.
- **Evaluator changes made for this run** (`eval.py`; `tests/test_eval.py`; 102 tests passing, 1 skipped):
  1. `--offset` and `--all-remaining` flags via `select_eval(steps, offset, limit)`, so the run can start at step 503. No other code path changed.
  2. Top-5 is a strict five-item list: the system's top-1 followed by its remaining distinct ranked entries, cut to five. The old code credited a hit if the label matched top-1 *or* any of the first five `topk` entries, which for the generator (greedy answer plus five beams) could cover six guesses. Recomputed on the existing 500-step predictions this lowers the 3-epoch generator from 93.8% to 93.4%, the prompted generator from 53.4% to 52.8% and random from 85.0% to 83.4%; scorer rows are unchanged. The README already reports the strict numbers.
- **Analysis script.** `scripts/paired_test.py` (tests in `tests/test_paired_test.py`), committed before the run. It reports the top-1 difference, a trajectory-clustered percentile bootstrap 95% CI, a verdict against a predeclared margin, and a secondary step-level exact McNemar test.
- **Command.**

  ```bash
  set -o pipefail
  mkdir -p results/06-unmonitored-holdout
  ACTIONRANK_CONFIG=configs/unmonitored_holdout.yaml .venv/bin/python eval.py \
      --systems baseline_sft,tier2 --offset 503 --all-remaining \
      2>&1 | tee results/06-unmonitored-holdout/eval.log
  ACTIONRANK_CONFIG=configs/unmonitored_holdout.yaml .venv/bin/python scripts/analyse_predictions.py \
      > results/06-unmonitored-holdout/analysis_body.md
  .venv/bin/python scripts/paired_test.py \
      results/06-unmonitored-holdout/predictions_tier2.jsonl \
      results/06-unmonitored-holdout/predictions_baseline_sft.jsonl --margin 3 \
      | tee results/06-unmonitored-holdout/paired.txt
  ```

  One evaluator invocation scores both systems on the identical step list and writes `results.json`, `results.md`, and one `predictions_*.jsonl` per system. Reference rows (random, most-frequent) are computed on the same steps automatically.

- **Runtime estimate.** Scorer about 0.25 s per step; generator about 0.6 s greedy plus about 2.5 s for the untimed beam pass. Roughly 75 to 90 minutes on the laptop with nothing else on the GPU. Latency comes from the same timer as before (prompt build, tokenise, model call; beam pass excluded).

## Primary analysis and decision rule

**Primary:** scorer-minus-generator top-1 difference in percentage points, with a 95% CI from a bootstrap that resamples *trajectories* (371 clusters), not steps, because steps within a trajectory are strongly dependent. Practical-equivalence margin: **±3 points**, declared here.

- CI wholly inside (−3, +3): **equivalent** on top-1.
- CI wholly above +3: scorer better. CI wholly below −3: generator better.
- Anything else: **inconclusive**. Not "a tie".

**Secondary:** step-level exact McNemar p on the discordant pairs, reported but not used for the verdict.

For calibration, the same analysis on the 500-step dev set gives: both right 276, scorer only 57, generator only 54, neither 113; difference +0.6 pt; clustered 95% CI [−3.3, +4.6]; verdict **inconclusive**; McNemar p = 0.85. So the dev-set "tie" was never established under this rule. With 2.7x the trajectories the interval should be roughly 1.6x narrower, so a difference near zero would land inside the margin, and a difference of 2 points or more probably would not.

## Pre-registered expectations

Written before running.

| quantity | 500-step dev | expected on the 1,352 |
|---|---:|---|
| scorer top-1 | 66.6% | 62% to 68%. Below 60% means the dev prefix flattered it. |
| generator top-1 | 66.0% | 60% to 67%. |
| top-1 difference | +0.6 pt (CI [−3.3, +4.6]) | point estimate within ±2; CI inside ±3 if the estimate is near 0. Verdict per the rule above. |
| scorer top-5 (strict) | 98.2% | 96% to 99%. |
| generator top-5 (strict) | 93.4% | 90% to 95%. Gap of at least 3 points; this is the ranking claim. |
| scorer hallucination | 0.0% | exactly 0.0% by construction; anything else is a bug. |
| generator hallucination | 0.6% | 0.3% to 1.5%. |
| latency, scorer / generator | 250 / 575 ms | same ratio, about 2x or more; absolute values within about 10% run-to-run laptop variance. |
| never-label tools (329 steps) | scorer 41.3%, generator 48.8% | generator ahead by 4 to 10 points. Under 3, the "remaining weakness" softens; over 12, it hardens. |
| `Finish` recall (372 steps) | scorer 88%, generator 78% | scorer still ahead. |

## What each outcome does to the write-up

- **Equivalent, other gaps hold:** README headline table and blog table switch to the 1,352-step numbers, described as "a pre-registered evaluation of the frozen final checkpoints on 1,352 steps outside their training-time monitoring prefixes", with the caveat that earlier frozen-head experiments reported aggregates over the full held-out split. The 500-step tables stay as development history.
- **Inconclusive:** report the difference and the CI as the headline, say the accuracy question is open at this sample size, and keep the hallucination, ranking and latency claims, which do not depend on it.
- **Generator better:** headline becomes "scoring buys zero hallucination, a better ranking and half the latency at a top-1 cost of X points (CI)". The blog's "what this means" is rewritten.
- **Scorer better:** reported as is, with the one-seed caveat.
- **Latency ratio changes materially:** investigate before reporting; checkpoints and hardware are unchanged.

## For the reviewer

- `select_eval(ds.eval, 503, None)` is disjoint from every step either training script scored (`ds.eval[:200]`, `ds.eval[:100]`) and from every trajectory in `ds.eval[:500]`. `data.py` builds `ds.eval` deterministically from `split_seed`.
- `configs/unmonitored_holdout.yaml` points at the checkpoints the README calls final.
- The strict top-5 change does not alter scorer rows.
- The expectations and the decision rule above predate the run: this file, the config, `eval.py` and `scripts/paired_test.py` are committed before the results directory exists.
- Whether 371 trajectories is enough for a ±3 margin. It may not be; that is what "inconclusive" is for.

## Outcome (added 2026-09-17, after the run)

Run on 2026-09-16 22:11 to 23:11 PDT on the laptop, exactly as specified above. Results in `results/06-unmonitored-holdout/`.

| quantity | expected | observed | met? |
|---|---|---|---|
| scorer top-1 | 62% to 68% | 63.5% | yes |
| generator top-1 | 60% to 67% | 68.3% | no, above range |
| top-1 difference | point estimate within ±2; CI inside ±3 if near 0 | −4.8, CI [−7.3, −2.4] | no; verdict **inconclusive** under the margin rule, direction clear |
| scorer top-5 (strict) | 96% to 99% | 97.5% | yes |
| generator top-5 (strict) | 90% to 95%, gap ≥ 3 | 92.2%, gap 5.3 | yes |
| scorer hallucination | 0.0% | 0.0% | yes |
| generator hallucination | 0.3% to 1.5% | 0.9% | yes |
| latency ratio | about 2x or more | 251 / 632 ms, 2.5x | yes |
| never-label tools | generator ahead by 4 to 10 | 55.0% vs 46.8%, 8.2 | yes |
| `Finish` recall | scorer ahead | 77.2% vs 78.2%, level | no |

The dev-set tie did not replicate. The scorer's dev-set parity came from a `Finish`-recall edge (88% vs 78% on the 500) that was absent on the 1,352. Per the "generator better" branch above, the README headline and the blog were rewritten.

**Follow-up, not pre-registered:** two more training seeds per system (`configs/seeds/`, `results/07-seeds/`; a `train_seed` field was added since the LoRA initialisation had been unseeded). Top-1 on the same 1,352 steps: scorer 63.5 / 62.4 / 61.8, generator 68.3 / 68.2 / 63.4. Generator seed 3's drop is entirely `Finish` recall (62% vs 78% and 91%); its non-`Finish` accuracy matches the other seeds. Six of nine pairings favour the generator with clustered CIs clear of zero; the three involving generator seed 3 are equivalent or inconclusive. Mean gap 4.1 points overall, about 6 on non-`Finish` steps. The seed evaluations ran on an A100, so their latency columns are not comparable and are not reported.
