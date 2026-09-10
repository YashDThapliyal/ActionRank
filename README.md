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

All numbers below are on the first 500 held-out step examples (same examples for every system), batch
size 1, Qwen2.5-1.5B-Instruct in fp16 on an M3 Pro (MPS). Latency covers prompt building, tokenization
and the model call; the beam-search pass that produces the baseline's top-5 list is not timed.
`random-candidate` picks uniformly among the task's candidates; `most-frequent-candidate` picks the
candidate with the highest training-label frequency (always `Finish`). `actionrank-tier1` is the
per-tool table head; `actionrank-span` scores each candidate from its own line in the prompt.

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 500 | 20.8% | 85.0% | 0.0% | - | - | 19.1% (n=362) |
| most-frequent-candidate | 500 | 27.6% | 88.0% | 0.0% | - | - | 0.0% (n=362) |
| actionrank-tier1 | 500 | 49.4% | 91.4% | 0.0% | 231 | 217 | 37.3% (n=362) |
| actionrank-span | 500 | 57.8% | 97.2% | 0.0% | 232 | 217 | 45.9% (n=362) |
| baseline-generation | 500 | 31.8% | 53.4% | 1.8% | 664 | 606 | 43.9% (n=362) |

On the full held-out set (1855 steps): table head 46.5% top-1 / 90.8% top-5;
span head 56.7% top-1 / 96.7% top-5 (validation top-1 used for checkpoint
selection: 46.3% and 57.9%). Each head needs one pass of the backbone over the
12.4k prompts (~80 minutes on the M3 Pro); head training then takes a minute or two.

Tier 2 (LoRA r=8 on q/v projections, 400 examples, 1 epoch, 25 optimizer steps, 8 minutes locally) starts
from the Tier 1 table head and neither helps nor hurts (48.0% -> 47.5% on a 200-step validation subset;
per-step loss flat within noise). It validates the joint LoRA + head pipeline rather than measuring what
a full fine-tune would do.

### Where the accuracy comes from

Per-example predictions are saved to `results/predictions_<system>.jsonl`; the table below slices the
500 evaluated steps by label type and by whether the labelled tool occurs as a training label at all
(`scripts/analyse_predictions.py`).

| system | predicts Finish | Finish recall (n) | top-1 non-Finish | top-1 seen tools (n) | top-1 unseen tools (n) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0% | 0.0% (138) | 43.9% | 46.5% (241) | 38.8% (121) |
| span | 43.8% | 89.1% (138) | 45.9% | 52.3% (241) | 33.1% (121) |
| tier1 | 38.8% | 81.2% (138) | 37.3% | 47.3% (241) | 17.4% (121) |

- **Stopping.** The generation baseline never outputs `Finish` (0 of 138), even though it is listed as a
  tool; it always picks an API. Both ActionRank heads learn when the trajectory is complete (81% and 89%
  `Finish` recall), which is a large part of their top-1 lead over the baseline.
- **Seen tools.** On non-`Finish` steps whose tool occurs in training, the table head ties the baseline
  (47% vs. 46%); the span head is ahead (52%).
- **Unseen tools.** On the 121 steps whose tool never appears as a training label, the table head is at
  chance (17%): a table row that was never trained cannot rank a tool. The span head, which builds each
  candidate's vector from its description inside the prompt, recovers most of the gap (33%) but is still
  below the prompted baseline (39%), which reads the same descriptions with the full language model.
  Closing the rest is the case for Tier 2 at scale (fine-tuning the encoder so prompt and tool spans
  align), which the local 400-example run is far too small to show.
- **Top-5.** The span head reaches 97% top-5 with ~5.6 candidates per step (random is 85%). The
  baseline's 53% top-5 is low because beam search mostly returns spelling variants and comma-joined names
  of the same tool rather than five distinct valid tools.
- **Latency.** The span head costs the same as the table head (232 vs. 231 ms): the prompt is identical,
  prefill dominates, and the per-candidate pooling is ~1 ms on CPU. On MPS a naive on-device version was
  +72 ms from kernel-launch overhead (`scripts/prototypes/`), hence the CPU offload after prefill.

### Success criteria

| criterion | outcome |
|---|---|
| match or exceed baseline top-1 / top-5 | yes: 57.8% vs. 31.8% top-1, 97% vs. 53% top-5 (span head); on unseen tools alone the baseline still leads, 39% vs. 33% |
| near-zero hallucinated tool rate | yes: 0.0% by construction (baseline 1.8%, plus it can only ever return one candidate) |
| lower latency per decision, attributable to prefill-only inference | yes: 232 ms vs. 664 ms mean (2.9x); ActionRank runs one forward pass with no decoding |

## Notes and limitations

- Scoring is restricted to each task's candidate list (the API list ToolBench gives the agent), i.e.
  the spec's "true label + hard negatives" option; the embedding table still covers the whole catalog.
- `Finish` is a catalog tool: predicting "stop and answer" is part of next-action prediction. Metrics
  are also reported excluding `Finish`-label steps.
- Latency is measured per decision at batch size 1 on the same device for both systems, including
  tokenization. ActionRank = prefill + head; baseline = prefill + greedy decode of the tool name.
- Tier 2 is a small local run (a few hundred examples, one epoch) to validate the joint LoRA + head
  pipeline; a full run needs a CUDA GPU (`model.device: cuda`).
- The baseline is prompted, not fine-tuned, so the comparison is "trained head on a frozen backbone"
  vs. "zero-shot generation with the same backbone". A fine-tuned generation baseline would be the
  natural next comparison, as would ToolBench G3 (multi-tool tasks, larger candidate sets).
- Held-out trajectories share tools with training trajectories only 76% of the time; results on the
  unseen-tool slice are the honest measure of generalisation. The span head closes most of the table
  head's gap there (17% -> 33%) but the prompted baseline still leads (39%).
