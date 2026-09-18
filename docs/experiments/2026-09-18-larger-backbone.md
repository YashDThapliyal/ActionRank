# Experiment B: a larger backbone for both systems

Status: planned, pre-registered. Not yet run. Depends on nothing in Experiment A, but shares its evaluation discipline.

## The question

Does the scorer's accuracy deficit shrink when the backbone reads tool descriptions better? The span head represents each candidate by the hidden states over its one-line description. A 1.5B model reads short descriptions weakly, and the scorer's largest deficit is on tools it never saw as a training label (44.2% vs. 53.8%, a reading problem by construction). GenRec itself runs on a large foundation model; the 1.5B choice was a laptop constraint.

## Design

**Backbone:** Qwen2.5-7B-Instruct in bf16 (about 15 GB of weights; fits LoRA training with gradient checkpointing on the 40 GB A100 the Colab sessions have provided). Same tokenizer family, so the prompt, truncation, and span offsets carry over. 14B in 4-bit is a possible second step but adds a quantization confound; it is not part of this plan.

**Both systems get the same backbone.** Scorer: Tier 1 span cache and head, then Phase 2 LoRA (r=8, q/v, lr 2e-4, effective batch 16, 3 epochs), last-token pooling. Generator: LoRA SFT, answer-only loss, 3 epochs, same adapter config. Plus the prompted 7B generator zero-shot, for context only. If Experiment A's Phase 1 helps at 1.5B, both a Phase-2-only and a two-phase scorer are run at 7B; otherwise Phase 2 only.

**Seeds.** One seed each first. If the gap moves by more than 2 points in either direction, two more seeds each.

**Latency.** Measured on the A100 for both systems at batch size 1, same timer, and reported as A100 numbers next to the laptop 1.5B numbers. The ratio is the claim, not the absolute.

**Budget.** Per the 1.5B timings scaled by about 4.5x: Tier 1 cache about 45 minutes, Phase 2 about 2 hours per seed, generator SFT about 2.5 hours per seed, evaluation with beams about 40 minutes. One seed of everything: about 6 A100-hours. Three seeds: about 16. Sessions have been reclaimed mid-run before; pipelines package after every stage and the laptop must stay awake (`caffeinate`) for the keep-alive.

## Evaluation and decision rule

Same as Experiment A: develop on the 500-step dev set, evaluate each final model once on the 1,352 unmonitored steps, paired comparison with trajectory-clustered CI and the ±3-point margin. Primary quantity: 7B scorer minus 7B generator top-1. Secondary: the same gap on the never-label slice, and how much each system gained over its 1.5B version.

## Pre-registered expectations

| quantity | 1.5B (3 seeds) | expected at 7B |
|---|---:|---|
| generator top-1 | 66.6% | 70% to 76%. |
| scorer top-1 | 62.6% | 68% to 75%. |
| gap, scorer minus generator | −4.1 pt | between −4 and 0. The never-label slice is expected to carry most of the change. |
| never-label gap | −9.6 pt | narrows to −6 or better. If it does not narrow, description reading is not the bottleneck. |
| `Finish` recall | level | level. |
| generator hallucination | 0.9% | 0.2% to 0.7%. Structural gap persists but shrinks. |
| scorer top-5 | 97.2% | 97% to 99%. |
| latency ratio, A100 | 2.5x on laptop | 2x or more. Both prefill and decode scale with model size; decode remains memory-bound per token. |
| prompted 7B generator, zero-shot | 31.8% at 1.5B | 45% to 60%, and it will output `Finish` sometimes. Context only. |

## What each outcome does to the write-up

- **Gap within the margin at 7B:** the story becomes "at 1.5B scoring costs 4 points; at 7B it is a tie", and the scale dependence is the finding.
- **Gap unchanged:** the deficit is not about reading descriptions; report it, and point at the catalog head design.
- **Gap widens:** report it. It would mean the generator benefits more from scale, which is itself worth knowing.

## Implementation notes

- No new model code: `model.py` is backbone-agnostic through `cfg.model.backbone`, `head_hidden` is set from `hidden_size` at load. Confirm the span offset logic on the 7B tokenizer with the existing tests.
- Configs under `configs/7b/`. `ACTIONRANK_DEVICE=cuda`. Batch 4 with grad accumulation 4 to keep effective batch 16 within 40 GB.
- Pipeline `09_7b_*.py`, one session per system, packaging after every stage.
