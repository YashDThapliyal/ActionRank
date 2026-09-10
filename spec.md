# ActionRank

**LLM-native ranking for agent tool/action selection**, inspired by Netflix's GenRec (verbalize → LLM backbone → catalog-aware scoring head → reward-weighted ranking loss → prefill-only serving), applied to next-tool-call prediction for agents instead of media recommendations.

## Motivation

Standard agent tool-selection works by having an LLM *generate* a tool call autoregressively (function calling). This is slow when the candidate tool set is large, can hallucinate tools that don't exist, and gives no clean probability distribution over the full action space.

GenRec's insight: when you have a fixed, known catalog, you don't need to generate — you can score every candidate in a single forward pass and rank them. ActionRank applies this to agent tool selection: given a trajectory so far and a catalog of available tools, score every tool in one pass instead of decoding a tool name token by token.

## Core pipeline

1. **Verbalizer** — serializes the task query, compressed trajectory history (prior tool calls + truncated results), and the full tool catalog (name + description) into a single text prompt.
2. **LLM backbone** — processes the prompt in a single forward pass (no decoding) to produce a pooled hidden state `h`.
3. **Catalog-aware scoring head** — each tool has a learned embedding; score = similarity between `h` and each tool embedding. Softmax over all candidate tools yields a ranked list.
4. **Output** — top-ranked tool(s) as the predicted next action, compared against ground truth.

## Data

- **Source**: ToolBench, starting with the G1 (single-domain) subset for a smaller catalog per task and faster iteration. Move to G3 (multi-domain) later if time allows.
- **Catalog file**: `tool_id → {name, description}`, built from the union of APIs across the chosen subset.
- **Step slicing**: each multi-tool trajectory is sliced into steps. At each step, the "context so far" is the query plus prior calls/results, and the label is the next tool actually called.
- **Split**: hold out ~15% of full trajectories (not individual steps, to avoid leakage between train/eval) for evaluation.

## Verbalization / context engineering

- Template: task query → compressed history of prior tool calls and truncated results → candidate catalog as `name: description` lines.
- Keep the last 2-3 steps of history in full; summarize or drop older steps.
- This context-budget tuning is the direct analog of GenRec's context engineering (retain high-signal, compress repetitive, drop low-signal).

## Model

- **Backbone**: `Qwen/Qwen2.5-1.5B-Instruct` via `transformers`, fp16.
- **Device**: `mps` locally on Apple Silicon (M3 Pro, 18GB unified memory); same script runs on Colab by switching to `device="cuda"` if a full run is too slow locally.
- **Pooling**: mean-pool (or last-token) final hidden states over the prompt to get a fixed vector `h`.
- **Tool embedding table**: `nn.Embedding(num_tools, hidden_dim)`.
- **Scoring head**: dot product of `h` against every tool embedding → logits over the catalog → softmax.

## Training tiers

### Tier 1 — frozen backbone (build and validate first)
- Run the backbone once per example in inference-only mode (no gradients through the LLM).
- Cache pooled vectors to disk.
- Train only the tool embedding table + a small MLP head on the cached vectors.
- Fast, low-memory, runs on CPU if needed. Proves the core thesis (catalog scoring beats generation) before touching LLM fine-tuning.

### Tier 2 — LoRA fine-tune (stretch goal, closer to GenRec's actual approach)
- Wrap the backbone with `peft`: target `q_proj`/`v_proj`, rank 8-16, alpha 16-32, dropout 0.05.
- Fine-tune LoRA adapters + scoring head jointly.
- AdamW on trainable params only, lr ~1e-4 to 2e-4, fp16, gradient accumulation to simulate a larger batch under memory constraints.
- Try small runs locally first (few hundred examples, 1-2 epochs) to confirm the pipeline works end to end; burst to Colab for full-scale training if needed, then bring the checkpoint back down for local eval.

## Loss

- Cross-entropy over the catalog. If the full catalog is too large per step, use the true label plus hard negatives (same-category tools) and in-batch negatives instead of scoring the entire catalog every step.

## Baseline

- Same backbone, prompted for standard function-calling-style generation (or constrained decoding to valid catalog names).
- Compared against ActionRank on:
  - Top-1 / top-5 accuracy against ground-truth next tool
  - Invalid/hallucinated tool rate
  - Wall-clock latency per decision

## File structure

```
data.py          # parse ToolBench, slice trajectories into steps
verbalize.py     # context + catalog -> prompt string
model.py         # backbone + pooling + scoring head
train_tier1.py   # cache embeddings, train head only
train_tier2.py   # LoRA fine-tune backbone + head
baseline.py      # function-calling baseline
eval.py          # accuracy, hallucination rate, latency -> results table
config.yaml
```


## Success criteria

- ActionRank matches or exceeds baseline top-1/top-5 accuracy.
- Near-zero hallucinated tool rate (structural guarantee from catalog-constrained scoring).
- Meaningful latency reduction per decision vs. autoregressive generation, attributable to prefill-only inference.
