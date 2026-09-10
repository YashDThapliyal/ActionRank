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
| evaluate vs. generation baseline | `.venv/bin/python eval.py --systems tier1,tier2,baseline` | `results/results.md` |

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
- `train_tier1.py` — frozen backbone, cached vectors (fingerprinted against prompts + config), head-only
  training; the checkpoint epoch is chosen on a 10% validation slice of *training* trajectories, never
  on the held-out eval split.
- `train_tier2.py` — LoRA (q/v projections) fine-tuned jointly with the head.
- `baseline.py` — same backbone, chat-prompted to emit a tool name (greedy for top-1 + latency,
  beam-5 for top-5); a hallucination is a top-1 name outside the task's candidate list.
- `eval.py` — top-1 / top-5 accuracy, hallucination rate, per-decision latency, plus
  random-candidate and most-frequent-candidate reference rows.

## Data

<!-- DATA_STATS -->

## Results

<!-- RESULTS -->

## Notes and limitations

- Scoring is restricted to each task's candidate list (the API list ToolBench gives the agent), i.e.
  the spec's "true label + hard negatives" option; the embedding table still covers the whole catalog.
- `Finish` is a catalog tool: predicting "stop and answer" is part of next-action prediction. Metrics
  are also reported excluding `Finish`-label steps.
- Latency is measured per decision at batch size 1 on the same device for both systems, including
  tokenization. ActionRank = prefill + head; baseline = prefill + greedy decode of the tool name.
- Tier 2 is a small local run (a few hundred examples, one epoch) to validate the joint LoRA + head
  pipeline; a full run needs a CUDA GPU (`model.device: cuda`).
