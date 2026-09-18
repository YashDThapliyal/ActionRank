# Experiment A: GenRec's two-phase training for the scorer

Status: planned, pre-registered. Not yet run.

## The gap this targets

With matched LoRA fine-tuning and three seeds, the generator leads the span scorer by 4.1 points top-1 on the 1,352 unmonitored held-out steps (66.6% vs. 62.6%), by about 6 points on non-`Finish` steps, and by about 10 points on tools that were never a training label (53.8% vs. 44.2%). `Finish` recall is level. See README section 4.6.

## Why this is the GenRec-faithful lever

GenRec trains in two phases. **Phase 1, domain adaptation:** continue training the LLM backbone with the ordinary next-token objective on the verbalized histories, so the model learns to read the domain's text. **Phase 2, ranking fine-tuning:** train the catalog-aware head and the backbone with the ranking loss over the catalog. ActionRank so far has only done Phase 2. The generator baseline, by contrast, was fine-tuned with the next-token objective on the verbalized prompt plus the answer, which is Phase 1's signal. The hypothesis is that the accuracy gap is mostly this missing phase: the ranking loss through a small head gives the backbone a weak, indirect gradient for learning which description matters, while the LM objective gives it a direct one.

Everything else stays GenRec: one prefill, last-token pooling for `h`, the span head as the metadata-aware catalog head, softmax over the offered candidates, cross-entropy, prefill-only serving with no decoding at inference.

**Not GenRec, included only as a reference bound:** scoring each candidate name's log-likelihood under the fine-tuned generator (constrained generation). It is the "generate identifiers" path GenRec argues against. It is run once to answer "what does the LM head know that the catalog head does not", and it is reported as a baseline, never as an ActionRank variant.

## Design

Backbone Qwen2.5-1.5B-Instruct, data and split unchanged (`split_seed` 13), same 1,280-token prompt, last-token pooling, span head.

**Phase 1.** LoRA (r=8, q/v, same as every run) trained with the next-token loss over the *full verbalized sequence*: the scorer's own prompt format (`Task / History / Available tools / Next tool:`) followed by the label tool name and EOS. Loss on every token, not answer-only, because Phase 1 is domain adaptation, not answer supervision. 1 epoch over the 10,568 training steps, lr 2e-4, effective batch 16. Checkpoint the adapter.

Two Phase 1 variants, to separate "reading the domain" from "seeing the answer":
- **A1, prompt-only:** loss over the prompt tokens only, answer masked. Pure domain adaptation.
- **A2, prompt + answer:** loss over prompt and answer tokens. Closest to what the generator got.

**Phase 2.** Starting from the Phase 1 adapter, train adapter + span head jointly with the ranking loss for 3 epochs, exactly the recipe in `configs/last_span_3ep.yaml` (head initialised from the frozen Tier 1 span head, which is itself re-cached on the Phase 1 backbone). Three seeds per variant via `tier2.train_seed`.

**Controls.** The existing three-seed span scorer (Phase 2 only) and three-seed generator, already evaluated on the 1,352 steps. No retraining of controls.

**Budget.** Phase 1: about 12 minutes per variant on an A100. Re-cache Tier 1 spans: about 15 minutes per variant. Phase 2: 27 minutes per seed. Total for A1 and A2, three seeds each: about 4 A100-hours. Reference bound (likelihood scorer): evaluation only, about 20 minutes.

## Evaluation and decision rule

Development on the 500-step dev set only (`ds.eval[:500]`); the per-epoch monitors in the training scripts stay as they are. Each final variant is evaluated **once** on the 1,352 unmonitored steps (`--offset 503 --all-remaining`), three seeds. Primary comparison: variant scorer minus the existing three-seed generator, all nine seed pairings, trajectory-clustered bootstrap 95% CI, ±3-point margin, via `scripts/paired_test.py`. Secondary: non-`Finish` top-1, never-label top-1, `Finish` recall, top-5 (strict), hallucination, and laptop latency for the best variant's seed 1 (Phase 1 changes nothing about inference cost, so latency should be unchanged at about 250 ms).

Note on the 1,352 steps: they have now been used to evaluate five final models. Each further use erodes their status. This plan commits to exactly two scorer variants and one reference bound on them, with expectations fixed here, and no iteration against them.

## Pre-registered expectations

| quantity | current (Phase 2 only) | expected with Phase 1 |
|---|---:|---|
| scorer top-1, mean of 3 seeds | 62.6% | A2: 64% to 67%. A1: 63% to 66%. |
| gap to generator, mean | −4.1 pt | A2: within ±3 on at least six of nine pairings. A1: narrower than −4.1 but probably not within the margin. |
| non-`Finish` top-1 | 56.6% | +2 to +5 points. |
| never-label top-1 | 44.2% | +3 to +6 points. This is where Phase 1 should help most, since it is a reading problem. |
| `Finish` recall | 78% | unchanged within seed noise. |
| hallucination | 0.0% | 0.0% by construction. |
| top-5 (strict) | 97.2% | unchanged or slightly up. |
| latency, laptop | 251 ms | unchanged. |
| likelihood-scorer bound | n/a | top-1 within 1 point of the generator's 66.6%, since it is the same model read differently. If A2 gets within 2 points of this bound, the catalog head is not the bottleneck. |

## What each outcome does to the write-up

- **A2 closes to within the margin:** the headline becomes "with GenRec's full two-phase recipe, scoring matches generation on accuracy and keeps every structural win". The 4-point gap is reported as the cost of skipping Phase 1.
- **A2 narrows but stays outside the margin:** report the narrowed gap; Phase 1 is a partial fix and the remaining deficit is attributed to the never-label slice if that is where it stays.
- **No change:** the ranking head, not the backbone's reading, is the bottleneck. That points at Experiment B and at richer catalog heads.
- **A1 ≈ A2:** the benefit is domain reading, not answer supervision, which is the cleaner story for GenRec.

## Implementation notes

- New script `train_phase1.py`: reuses `build_sft_batch` from `baseline_sft.py` with the scorer's verbalizer and a `mask_answer` flag; saves an adapter under `checkpoints/phase1/{a1,a2}`.
- `train_tier1.py` and `train_tier2.py` gain an optional `tier2.init_adapter` path so Phase 2 starts from the Phase 1 adapter (already supported by `load_trainable_adapter` for resume; expose it in config).
- Configs under `configs/phase1/`. Colab pipeline `08_phase1_*.py` following the seed pipelines: train, package, evaluate, package.
- Tests: batch masking for A1 vs A2; adapter init path is honoured; a smoke run on the fixture data.
