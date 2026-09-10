# ActionRank

**LLM-native ranking for agent tool/action selection.** Instead of *generating* a tool call token by
token, ActionRank scores every tool in a known catalog with a single prefill pass of the LLM and ranks
them, in the spirit of Netflix's GenRec (verbalize → LLM backbone → catalog-aware scoring head →
ranking loss → prefill-only serving). See `spec.md` for the full design.

```
query + compressed history + catalog ──verbalize──▶ prompt
prompt ──Qwen2.5-1.5B (prefill only)──▶ hidden states ──mean-pool──▶ h
h ──MLP──▶ q ;  score(tool) = cos(q, E[tool]) / τ  ──mask to candidates──▶ softmax ──▶ ranked tools
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest            # unit tests (fast, no model download)
```

All scripts read `config.yaml`. Set `model.device` to `mps` (Apple Silicon), `cuda` (Colab) or `cpu`.

## Pipeline

| step | command | output |
|---|---|---|
| download + parse ToolBench G1 | `.venv/bin/python data.py` | `data/processed/{catalog.json,train.jsonl,eval.jsonl}` |
| Tier 1: cache pooled vectors, train head | `.venv/bin/python train_tier1.py` | `cache/*.pt`, `checkpoints/tier1_head.pt` |
| Tier 2: LoRA + head (small local run) | `.venv/bin/python train_tier2.py` | `checkpoints/tier2/` |
| evaluate vs. generation baseline | `.venv/bin/python eval.py --systems tier1,span,baseline` | `results/results.md`, `results/predictions_*.jsonl` |

Use `--limit N` on any script for a smoke run.

## Files

- `data.py` — parses `answer/G1_answer/*.json` (Hugging Face mirror `Adorg/ToolBench`): the winning
  root-to-leaf path of each DFS tree becomes a trajectory, sliced into one example per tool call.
  The catalog is the union of all functions (including `Finish`). 15% of *trajectories* are held out.
- `verbalize.py` — task query, last 3 steps in full (args/observations truncated), older steps
  collapsed to their tool names, then the candidate catalog as `name: description` lines.
- `model.py` — backbone loader, prefill-only encoder (no LM head), pooling, `ScoringHead`
  (tool embedding table + residual MLP projection, cosine scores with temperature, candidate mask).
  The tool table is initialised from the backbone's own encoding of each tool's `name: description`
  (`model.tool_init: text`), so tools that never occur as a training label still get a meaningful vector.
- `model.py` also holds `SpanScoringHead` (`tier1.head: span`): no tool table at all. The candidate
  lines are already in the prompt, so each candidate's vector is the mean of the hidden states over its
  own `- name: description` span (from the same prefill pass), scored against the pooled prompt vector and
  scattered back into catalog-width logits so evaluation code is unchanged. A candidate span that covers
  zero tokens (prompt truncated past the catalog) raises rather than being scored.
- `train_tier1.py` — frozen backbone, cached vectors (fingerprinted against prompts + config), head-only
  training for either head; the checkpoint epoch is chosen on a 10% validation slice of *training*
  trajectories, never on the held-out eval split. `scripts/overnight_span.py` drives the span re-encode
  with latency and throughput guards.
- `train_tier2.py` — LoRA (q/v projections) fine-tuned jointly with the head.
- `baseline.py` — same backbone, chat-prompted to emit a tool name (greedy for top-1 + latency,
  beam-5 for top-5); a hallucination is a top-1 name outside the task's candidate list.
- `eval.py` — top-1 / top-5 accuracy, hallucination rate, per-decision latency, plus
  random-candidate and most-frequent-candidate reference rows.

## Data

- Source: ToolBench G1 answer files (`Adorg/ToolBench`, `answer/G1_answer/*.json`, 5000 files).
- 3394 files have a winning trajectory (root-to-leaf DFS path ending in `Finish` / `give_answer`); the
  other 1606 are skipped (`data.require_win: true`).
- Catalog: 6372 functions (union of every task's API list, plus `Finish`). Tasks list 5.7 candidates on average.
- Step examples: 10568 train (from 2885 trajectories) and 1855 held-out (from 509 trajectories, 15% of
  trajectories, seed 13). 27% of labels are `Finish`.
- Only 3597 of the 6372 catalog tools ever occur as a training label; 24% of held-out labels (34% of the
  non-`Finish` ones) are tools never seen as a training label. This is the main difficulty of the split.
- Prompts average ~410 tokens (p95 ~790, budget 1280 with left-truncation so the catalog survives).

## Results

All numbers are on the first 500 held-out step examples (same examples for every system), batch size 1,
Qwen2.5-1.5B-Instruct in fp16 on an M3 Pro (MPS). Latency covers prompt building, tokenization and the
model call; the beam-search pass that produces a generator's top-5 list is not timed. `random-candidate`
picks uniformly among the task's candidates; `most-frequent-candidate` picks the candidate with the
highest training-label frequency (always `Finish`).

Systems:
- `actionrank-tier1`: frozen backbone, learned per-tool table (text-initialised).
- `actionrank-span`: frozen backbone, each candidate scored from its own line in the prompt (no table).
- `actionrank-tier2`: LoRA (r=8, q/v) fine-tuned jointly with the table head, all 10,568 steps, 2 epochs (A100).
- `baseline-generation`: same backbone, prompted to emit the tool name (zero-shot).
- `baseline-generation-sft`: same backbone, LoRA (same rank/targets/lr/effective batch as Tier 2) fine-tuned to
  emit the tool name, all 10,568 steps, 1 epoch (A100, 12 minutes). This is the fair control.

### Mean pooling (default `model.pooling: mean`)

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 500 | 20.8% | 85.0% | 0.0% | - | - | 19.1% (n=362) |
| most-frequent-candidate | 500 | 27.6% | 88.0% | 0.0% | - | - | 0.0% (n=362) |
| actionrank-tier1 | 500 | 49.4% | 91.4% | 0.0% | 235 | 219 | 37.3% (n=362) |
| actionrank-span | 500 | 57.8% | 97.2% | 0.0% | 231 | 217 | 45.9% (n=362) |
| baseline-generation | 500 | 31.8% | 53.4% | 1.8% | 726 | 653 | 43.9% (n=362) |
| baseline-generation-sft | 500 | 66.8% | 91.4% | 1.4% | 639 | 592 | 60.2% (n=362) |
| actionrank-tier2 | 500 | 50.4% | 91.8% | 0.0% | 251 | 230 | 38.7% (n=362) |

### Last-token pooling (`config_last.yaml`, run via `ACTIONRANK_CONFIG=config_last.yaml`)

Same recipe with the prompt vector taken from the last token instead of the mean over the prompt.

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 500 | 20.8% | 85.0% | 0.0% | - | - | 19.1% (n=362) |
| most-frequent-candidate | 500 | 27.6% | 88.0% | 0.0% | - | - | 0.0% (n=362) |
| actionrank-tier1 | 500 | 50.0% | 90.0% | 0.0% | 249 | 228 | 35.6% (n=362) |
| actionrank-span | 500 | 62.0% | 96.2% | 0.0% | 264 | 239 | 53.0% (n=362) |
| actionrank-tier2 | 500 | 53.0% | 90.6% | 0.0% | 284 | 250 | 40.3% (n=362) |

(Latencies in this block were measured while the machine was under memory pressure from the day's other
runs; the code path and prompts are identical to the block above, where the same heads cost 231-251 ms.)

### Where the accuracy comes from

Per-example predictions are saved next to each results table (`predictions_<system>.jsonl`); the tables
below slice the 500 evaluated steps by label type and by whether the labelled tool occurs as a training
label at all (`scripts/analyse_predictions.py`).

Mean pooling:

| system | predicts Finish | Finish recall (n) | top-1 non-Finish | top-1 seen tools (n) | top-1 unseen tools (n) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0% | 0.0% (138) | 43.9% | 46.5% (241) | 38.8% (121) |
| baseline_sft | 33.6% | 84.1% (138) | 60.2% | 65.6% (241) | 49.6% (121) |
| span | 43.8% | 89.1% (138) | 45.9% | 52.3% (241) | 33.1% (121) |
| tier1 | 38.8% | 81.2% (138) | 37.3% | 47.3% (241) | 17.4% (121) |
| tier2 | 35.4% | 81.2% (138) | 38.7% | 48.5% (241) | 19.0% (121) |

Last-token pooling:

| system | predicts Finish | Finish recall (n) | top-1 non-Finish | top-1 seen tools (n) | top-1 unseen tools (n) |
|---|---:|---:|---:|---:|---:|
| span | 34.0% | 85.5% (138) | 53.0% | 61.0% (241) | 37.2% (121) |
| tier1 | 43.4% | 87.7% (138) | 35.6% | 44.0% (241) | 19.0% (121) |
| tier2 | 34.0% | 86.2% (138) | 40.3% | 51.5% (241) | 18.2% (121) |

- **Fine-tuning the generator is the strongest single thing you can do.** With the same adapter budget and
  data as Tier 2, the generator goes from 31.8% to 66.8% top-1 (sanity subset during training: 31% -> 72%),
  learns to stop (84% `Finish` recall) and generalises best to unseen tools (49.6%). It still names a tool
  that is not offered in 1.4% of decisions and takes 2.8x longer per decision.
- **Frozen-backbone scoring beats prompted generation but not fine-tuned generation.** The span head is the
  best frozen scorer (57.8% mean / 62.0% last pooling) and has the best top-5 of any system (96-97%), with
  zero hallucination by construction and ~230 ms per decision.
- **Pooling decides whether the scorer can be fine-tuned.** With mean pooling, Tier 2 at scale is a null
  (48.0% -> 47.5% -> 48.0% on a 200-step subset over 2 epochs) even though the adapter changes the pooled vector by ~10% and
  receives gradient; the mean over ~400 tokens dilutes it. With last-token pooling the same run learns
  (50.0% -> 53.0% -> 54.5%) and lands at 53.0% on the 500 steps. Two epochs is where we stopped, not where it plateaued.
- **Unseen tools remain the gap.** Table heads are at chance on tools never seen as a training label (18-19%);
  the span head recovers most of it (33-37%); the fine-tuned generator is at 50%.

### Success criteria, revisited

| criterion | vs. prompted generator | vs. fine-tuned generator |
|---|---|---|
| match or exceed top-1 / top-5 | yes (62% vs 32%; 96% vs 53%) | no on top-1 (62% vs 67%); yes on top-5 (96% vs 91%) |
| near-zero hallucinated tool rate | yes, 0.0% vs 1.8% | yes, 0.0% vs 1.4% |
| lower latency per decision (prefill-only) | yes, ~2.9x | yes, ~2.7x |

## Notes and limitations

- Scoring is restricted to each task's candidate list (the API list ToolBench gives the agent), i.e.
  the spec's "true label + hard negatives" option; the embedding table still covers the whole catalog.
- `Finish` is a catalog tool: predicting "stop and answer" is part of next-action prediction. Metrics
  are also reported excluding `Finish`-label steps.
- Latency is measured per decision at batch size 1 on the same device for both systems, including
  tokenization. ActionRank = prefill + head; baseline = prefill + greedy decode of the tool name.
- Candidate sets average 5.6 tools (max 11): this is shortlist ranking, not full-catalog retrieval, so the
  latency advantage of prefill-only scoring is understated relative to a large-catalog setting.
- Held-out trajectories share tools with training trajectories only 76% of the time; results on the
  unseen-tool slice are the honest measure of generalisation, and the fine-tuned generator leads there.
- Next levers, in order: fine-tune the *span* head with last-token pooling (Tier 2 was run with the table
  head); more Tier 2 epochs (it was still improving); a fine-tuned generator with constrained decoding to
  remove its remaining hallucinations, as the strongest possible baseline; larger candidate sets (G3).

## Running on Colab

`colab/` holds a detached pipeline for the Colab CLI (`uv tool install google-colab-cli --with "jupyter-kernel-client<1"`):
upload `actionrank_colab.zip` (code + `data/processed` + Tier 1 head) and `colab/remote_pipeline.py`, launch with
`colab exec -f colab/run_remote.py`, poll with `colab exec -f colab/tail_remote.py`, then `colab download` the
results zip. `colab exec` drops the connection after ~60 s of silence, which is why the pipeline runs detached
and logs to a file. On an A100 the fine-tuned baseline takes 12 minutes and Tier 2 9 minutes per epoch.
