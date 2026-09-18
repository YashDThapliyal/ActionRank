# Experiment A: GenRec's full training recipe for the scorer

Status: planned, pre-registered. Not yet run. Revised 2026-09-18 after checking the GenRec post and paper (arXiv:2608.10257) line by line.

## The gap this targets

With matched LoRA fine-tuning and three seeds, the generator leads the span scorer by 4.1 points top-1 on the 1,352 unmonitored held-out steps (66.6% vs. 62.6%), by about 6 points on non-`Finish` steps, and by about 10 points on tools that were never a training label (53.8% vs. 44.2%). `Finish` recall is level. See README section 4.6.

## What GenRec actually trains, and what ActionRank skipped

From the post ("Objectives: Ranking, Language, and Rewards", "Training Data as Conversations") and the paper (sections 4.4 to 4.6):

1. **Phase 1, Netflix-adapted foundation LLM.** An open-source LLM is adapted on proprietary corpora for content understanding, member behaviour patterns, and general language. No ranking labels. Updated infrequently. Ablation: starting Phase 2 from the Phase 1 model instead of the off-the-shelf LLM improves offline ranking metrics by 10 to 20%.
2. **Phase 2, GenRec.** Training data are conversations: a user message (verbalized context, history, item metadata, task) and an assistant message (the member's actual engagement). The loss is multi-objective: a **catalog-aware ranking objective** (cross-entropy over the catalog or candidate set, from the pooled hidden state and learned item embeddings), plus a **language-modeling objective over the verbalized inputs and outputs**, plus reward weighting of the ranking loss. "During Phase-2 training, the LLM learns how assistant messages depend on user messages." Backbone, head and item embeddings are trained jointly. At inference nothing is decoded; the head scores the candidates from one prefill.
3. **Serving.** Prefill-only, context compaction, smaller or distilled backbones.

ActionRank's Tier 2 is Phase 2 with only the ranking objective. It has **no LM term** and **no Phase 1**. The generator baseline, by contrast, was trained with exactly the LM objective over the verbalized input and the answer. So the generator got the part of GenRec's recipe that the scorer did not. That is the hypothesis: the accuracy gap is mostly the missing LM objective (and secondarily the missing Phase 1), not the scoring architecture. Under this reading, the tie-then-gap sequence in the README is the difference between a half-GenRec scorer and a full one.

**Explicitly not GenRec:** scoring each candidate name's log-likelihood under the fine-tuned generator (constrained generation). It is the "generate identifiers" path the post argues against. It runs once as a reference bound on what the LM head knows, and is reported as a baseline, never as an ActionRank variant.

## Design

Backbone Qwen2.5-1.5B-Instruct, data and split unchanged (`split_seed` 13), same 1,280-token prompt, last-token pooling, span head. Three arms, three seeds each:

- **Arm J, joint Phase 2 (the core).** Tier 2 as now (LoRA r=8 on q/v + span head, 3 epochs, head initialised from the frozen Tier 1 span head), with the loss changed to `L = L_rank + λ · L_LM`. `L_rank` is the existing cross-entropy over the candidate set. `L_LM` is next-token cross-entropy over the verbalized sequence: the prompt (user message) followed by `Next tool: <label>` and EOS (assistant message), computed from the same forward pass on the same tokens, so the pooled state and the LM logits come from one prefill. λ = 1 pre-declared; no tuning against the held-out set. This is GenRec's Phase 2 objective minus reward weighting, which ToolBench cannot supply.
- **Arm P, Phase 1 then joint Phase 2.** Phase 1: LoRA continued pretraining with the LM objective on ToolBench text with no labels: every task's tool descriptions (the catalog) and the verbalized trajectories with the `Next tool:` answers masked out. 1 epoch, lr 2e-4, effective batch 16. Then re-cache Tier 1 spans on the Phase 1 backbone, retrain the Tier 1 head, then Arm J on top.
- **Arm R, ranking-only (control).** The existing three-seed scorer. Not retrained.
- **Reference bound (not GenRec):** likelihood scoring of candidate names under the existing three-seed generator. Evaluation only.

Everything at inference stays as it is now: one prefill, last-token pooling, span head over the offered candidates, no decoding. Latency is unchanged by construction and is re-measured once on the laptop for Arm J seed 1 to confirm.

**Budget.** Arm J: about 30 minutes per seed on an A100 (the LM head adds a full-vocabulary logit pass over the answer tokens only if the LM loss is restricted to the answer; over the whole sequence it adds roughly 30% to the step). Arm P: Phase 1 about 15 minutes, re-cache about 15 minutes, then Arm J. Total for three seeds of J and P: about 5 A100-hours. Sessions must be kept alive (`caffeinate` on the laptop) and each stage packages its checkpoint.

## Evaluation and decision rule

Development on the 500-step dev set only (`ds.eval[:500]`); λ and everything else are fixed above, not searched. Each arm is evaluated **once** per seed on the 1,352 unmonitored steps (`--offset 503 --all-remaining`). Primary: arm scorer minus the existing three-seed generator, nine seed pairings, trajectory-clustered bootstrap 95% CI, ±3-point margin (`scripts/paired_test.py`). Secondary: non-`Finish` top-1, never-label top-1, `Finish` recall, strict top-5, hallucination, latency.

The 1,352 steps have judged five models so far. This plan adds two arms and one reference bound, and commits to no iteration against them.

## Pre-registered expectations

| quantity | Arm R (current) | Arm J expected | Arm P expected |
|---|---:|---|---|
| top-1, mean of 3 seeds | 62.6% | 64% to 67% | 65% to 68% |
| gap to generator, mean | −4.1 pt | within ±3 on at least six of nine pairings | within ±3 on at least six of nine; may cross zero |
| non-`Finish` top-1 | 56.6% | +2 to +5 | +3 to +6 |
| never-label top-1 | 44.2% | +3 to +6 | +4 to +8 (Phase 1 sees every description, labelled or not) |
| `Finish` recall | 78% | unchanged within seed noise | unchanged |
| hallucination | 0.0% | 0.0% by construction | 0.0% |
| top-5 (strict) | 97.2% | unchanged or up | unchanged or up |
| latency, laptop | 251 ms | unchanged | unchanged |
| likelihood bound | n/a | within 1 point of the generator's 66.6% | same |
| Arm P over Arm J | n/a | +1 to +3, in line with GenRec's 10 to 20% relative Phase 1 gain on a much smaller corpus | |

## What each outcome does to the write-up

- **J or P within the margin:** headline becomes "with GenRec's full objective, scoring matches generation on accuracy and keeps every structural win"; the 4-point gap is reported as the cost of the ranking-only objective, and the GenRec mapping table in the README is corrected to show which parts of the recipe each result used.
- **J narrows but stays outside, P closes:** Phase 1 is the decisive piece; report both.
- **Neither moves:** the objective is not the bottleneck; the catalog head or the backbone size is (Experiment B).
- **J ≈ P:** the LM objective is what matters, Phase 1 adds little at this corpus size.

## Implementation notes

- `train_tier2.py`: add `tier2.lm_weight` (default 0.0, so existing configs are unchanged). When > 0, `tier2_logits` returns the LM logits alongside the pooled scores; the batch builder appends `Next tool: <label>` + EOS to each prompt and returns LM labels (IGNORE over padding). Loss = rank CE + lm_weight × token CE. The span offsets are unaffected because the answer is appended after the candidate lines.
- `train_phase1.py`: LM-only LoRA on the label-masked verbalized trajectories plus one document per task listing its tools and descriptions. Saves an adapter under `checkpoints/phase1/`. `tier2.init_adapter` (new, optional) makes Tier 1 caching and Tier 2 start from it.
- Configs under `configs/genrec/`: `joint_s{1,2,3}.yaml`, `phase1.yaml`, `phase1_joint_s{1,2,3}.yaml`. Pipelines `08_joint_seeds.py`, `08b_phase1_then_joint.py`.
- Tests: LM labels align with the appended answer; `lm_weight: 0` reproduces the current loss exactly on the fixture; Phase 1 masking hides every label token; adapter init path is honoured.
