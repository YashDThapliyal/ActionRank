# ActionRank: Does Netflix's "score, don't generate" idea work for agent tool selection?

Every agent framework picks its next tool the same way: the language model *writes out* a function call,
one token at a time. That is slow, it can name a tool that doesn't exist, and it gives you one guess rather
than a ranking. Netflix's recommendation team recently argued, in [GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3),
that when your choices come from a fixed catalog you shouldn't generate at all: run the LLM once over the
context, then score every catalog item from that single pass. I wanted to know whether the same trick
works when the "catalog" is an agent's toolbox.

The short answer, on ToolBench with a 1.5B-parameter Qwen backbone, is that once both approaches get the
same fine-tuning, **scoring the tool list in one forward pass is as accurate as generating the tool name
(66.6% vs. 66.8% top-1), never picks a tool that isn't offered (0% vs. 1.4%), gives a far better ranking
(98% vs. 91% top-5), and decides in less than half the time (250 ms vs. 639 ms).** Getting there took a
detour: the scorer's own fine-tuning did nothing until I changed one detail of how the prompt is pooled,
and that detail turned a 5-point loss into a tie. The remaining weakness is tools the scorer has never
seen in training, where the generator is still ahead (50% vs. 41%). The rest of this report is how I got
each of those numbers and what I think they mean.

## TL;DR

Same 500 held-out steps, same backbone, batch size 1 on a laptop GPU:

| system | trained? | top-1 | top-5 | picks a tool not on the list | latency / decision |
|---|---|---:|---:|---:|---:|
| generation (function-calling style), prompted | no | 31.8% | 53.4% | 1.8% | 726 ms |
| **ActionRank**, frozen backbone + span head | head only | 62.0% | 96.2% | 0.0% | ~230 ms |
| generation, LoRA fine-tuned | yes | **66.8%** | 91.4% | 1.4% | 639 ms |
| **ActionRank**, LoRA fine-tuned + span head | yes | 66.6% | **98.2%** | **0.0%** | **250 ms** |

## 1. Why I did this

GenRec's pitch is simple. Netflix has a fixed catalog of titles. An LLM understands a member's viewing
history far better than a classical recommender, but making it *generate* title identifiers with beam
search "introduces latency overhead that can be prohibitive at scale", and out-of-the-box LLMs
"hallucinate out-of-catalog titles". So GenRec verbalizes the history into text, runs a decoder-only LLM
over it once, takes "the hidden state at a pooling position" as a summary vector `h`, and scores every
catalog item with a *catalog-aware head*: each item has a learned embedding, and a small module combines
`h` with the embedding to give a score. Softmax over the catalog, rank, done. Because only existing
embeddings get scored, recommending a non-existent film is impossible by construction. Serving is
"prefill-only": the model consumes the context once and ranks the full candidate set in a single pass.

Agent tool selection has the same shape. There is a fixed set of tools. The agent has a history (what it
has called so far and what came back). It must pick the next action. Today that pick is generated, with
exactly the problems GenRec lists: decoding cost, hallucinated tool names, no distribution over the action
space. So the hypothesis was: **replace "generate a function call" with "score the toolbox", and you
should get zero hallucination and lower latency at no cost in accuracy.** The first two are almost
guaranteed by the design; the third is the empirical question.

## 2. The idea, mapped onto GenRec

| GenRec (Netflix) | ActionRank (this project) |
|---|---|
| verbalize viewing history + context + item metadata into one text sequence | verbalize task query + compressed call history + candidate tools into one prompt |
| context engineering: retain high-signal engagements, compress repetitive ones (binges), drop low-signal events | keep the last 3 tool calls in full (args + truncated results), collapse older calls to their names, truncate observations to 200 chars |
| decoder-only LLM backbone, shared with their foundation model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, unchanged |
| `h` = hidden state at a pooling position | `h` = mean over the prompt (v1) or the last token (v2); this choice turned out to matter a lot |
| catalog-aware head: learned embedding per item, score = f(h, e_i) | **table head**: learned embedding per tool (6,372 rows), cosine score with a residual MLP on `h` |
| for cold-start items, "include more detailed metadata" in the context | **span head**: no table at all; each candidate's vector is pooled from its own description line *inside the prompt*, so unseen tools get a representation for free |
| softmax over the catalog; reward-weighted ranking loss | softmax over the catalog, masked to the task's candidate list; plain cross-entropy (no reward weighting; every reference call counts the same) |
| prefill-only serving, one forward pass for the whole candidate set | prefill-only: one pass, candidates scored from the same hidden states, no decoding |
| Phase 1 domain adaptation, Phase 2 ranking fine-tuning of backbone + head | Tier 1: backbone frozen, head trained on cached vectors in a minute; Tier 2: LoRA on q/v projections fine-tuned jointly with the head |

The two places I deliberately departed from GenRec are the loss (no reward weighting, because ToolBench
has no reward signal beyond "this is what the reference agent did") and the span head, which has no
analogue in the post: it is my attempt at the cold-start problem, and it ended up being the best scorer.

## 3. Setup

**Data.** ToolBench G1 (single-tool tasks), from the `Adorg/ToolBench` mirror. Each answer file holds
the task's API list with descriptions and a DFS tree of the reference agent's calls. I keep the root-to-leaf
path that ends in `Finish` with an answer (3,394 of 5,000 files) and slice it into one example per call:
context = query + calls so far, label = the next call. `Finish` is a catalog tool, since deciding to stop
is part of choosing the next action. 15% of *trajectories* (not steps) are held out so no task leaks
between train and eval: 10,568 training steps from 2,885 trajectories, 1,855 held-out steps from 509.
The catalog is the union of every task's tools, 6,372 in all; each task lists 5.7 candidates on average
(max 11). One fact shaped everything that follows: **24% of held-out labels are tools that never appear
as a training label.** Only 3,597 of the 6,372 catalog tools have any training signal at all.

**Prompt.** `Task: … / History: … / Available tools: - name: description … / Next tool:`, about 410
tokens on average. Prompts are left-truncated at 1,280 tokens so the catalog and the suffix survive.

**Scoring heads.** The table head is GenRec's design: `score = cos(h + mlp(h), E[tool]) / τ`, with `E`
initialised from the backbone's own encoding of each tool's description. The span head drops `E`: for each
candidate it mean-pools the hidden states over that candidate's line in the prompt (character offsets from
the tokenizer), projects both sides with a residual MLP, and scores by cosine. Scores are scattered into a
catalog-width vector with `-inf` elsewhere, so evaluation code is identical for both heads. Both MLPs have
zero-initialised output layers, so an untrained head is plain cosine similarity.

**Training.** Tier 1 caches one backbone pass per prompt (about 80 minutes on the laptop for 12.4k prompts)
and trains only the head; the checkpoint epoch is chosen on a validation slice carved from *training*
trajectories, never on the held-out set. Tier 2 wraps the backbone with LoRA (rank 8, `q_proj`/`v_proj`,
lr 2e-4, effective batch 16) and trains adapter + head jointly on all 10,568 steps.

**Baselines.** The same backbone, chat-prompted with the same task/history/tool list and asked to reply
with exactly one tool name. Greedy decoding gives top-1 and latency; beam search (5) gives a top-5 list. A
prediction is a hallucination if the name isn't on the task's candidate list. The *prompted* baseline is
zero-shot; the *fine-tuned* baseline gets the same LoRA adapter, data and effective batch as Tier 2,
trained to emit the tool name with loss only on the answer tokens. Two cheap reference rows: uniform
random among candidates, and the globally most frequent label (always `Finish`).

**Metrics.** Top-1 and top-5 accuracy against the reference agent's next call, hallucination rate, and
wall-clock latency per decision measured identically for every system (prompt building, tokenization, and
the model call, at batch size 1; the beam pass is not timed).

## 4. Results

Everything below is on the first 500 held-out steps (same 500 for every system). Full tables, including
the reference rows, are in `results*/results*.md`; per-example predictions are next to them.

### 4.1 Without training, scoring wins easily

| system | top-1 | top-5 | hallucination | latency |
|---|---:|---:|---:|---:|
| prompted generation | 31.8% | 53.4% | 1.8% | 726 ms |
| ActionRank table head, frozen backbone | 49.4% | 91.4% | 0.0% | 235 ms |
| ActionRank span head, frozen backbone | 57.8% | 97.2% | 0.0% | 231 ms |
| random among candidates | 20.8% | 85.0% | – | – |

This is the result the hypothesis predicted, and it's also the least interesting one, because the scorer
has a trained head and the generator has nothing. The prediction dumps showed where the gap came from:
the prompted generator **never outputs `Finish`** (0 of 138 steps where stopping was the right call),
while the scorers learn it (81–89% recall). On non-`Finish` steps whose tool was seen in training, table
head and generator are level (47% vs. 46%). On the 121 steps whose tool was never seen in training, the
table head is at chance (17%) while the generator, which actually reads the description, gets 39%. A
learned embedding that was never trained cannot rank a tool, no matter how good `h` is. The span head,
which reads the description from inside the prompt, recovers most of that (33%).

### 4.2 A fair baseline erases the accuracy lead

Fine-tuning the generator with the same adapter and data (one epoch, 12 minutes on an A100) is the
strongest single intervention in the whole project:

| system | top-1 | top-5 | hallucination | latency | seen tools | unseen tools | `Finish` recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine-tuned generation | **66.8%** | 91.4% | 1.4% | 639 ms | 65.6% | **49.6%** | 84% |
| span head, frozen backbone | 57.8% | **97.2%** | **0.0%** | 231 ms | 52.3% | 33.1% | 89% |

The generator learns to stop, jumps 35 points, and leads on unseen tools by 16. At this point the honest
headline would have been "scoring buys hallucination-freedom and latency at the cost of accuracy".

### 4.3 The scorer couldn't be fine-tuned, until it could

The obvious response is to fine-tune the scorer's backbone too. The first attempt (LoRA + table head, all
10,568 steps, 2 epochs) was a clean null: 48.0% → 47.5% → 48.0% on a validation subset, loss flat. I
checked whether the run was broken. It wasn't: with the adapter enabled the pooled vector `h` changes by
about 10% relative to the frozen one, and the LoRA parameters receive a gradient norm of 0.51 against 1.18
for the head. The adapter was learning; the metric wasn't moving.

The cause was the pooling position. GenRec takes "the hidden state at a pooling position"; I had been
taking the *mean* over all ~400 prompt tokens. A small adapter has to shift hundreds of token states
coherently to move that mean, whereas the generator reads one sharp next-token distribution at the last
position. Switching the scorer to **last-token pooling** and re-running the identical recipe:

| Tier 2 (LoRA + table head), 2 epochs | validation subset top-1 by epoch |
|---|---|
| mean pooling | 48.0% → 47.5% → 48.0% |
| last-token pooling | 50.0% → 53.0% → 54.5% |

Same data, same adapter, same head; only the pooling position changed, and the scorer became trainable.
For a *frozen* backbone the pooling choice barely matters (span head 56.7% vs. 57.8% top-1 on the full
held-out set under mean vs. last), which is why it went unnoticed until Tier 2.

### 4.4 Matched fine-tuning: a tie on accuracy, a win on everything else

With last-token pooling, LoRA and the span head trained jointly (initialised from the frozen span head,
3 epochs over all steps):

| system | top-1 | top-5 | hallucination | latency | seen tools | unseen tools | `Finish` recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine-tuned generation | 66.8% | 91.4% | 1.4% | 639 ms | 65.6% | **49.6%** | 84% |
| **fine-tuned ActionRank (span head, last pooling)** | 66.6% | **98.2%** | **0.0%** | **250 ms** | **67.2%** | 41.3% | 88% |

The two pick the correct tool equally often. On tools seen in training the scorer is slightly ahead; on
unseen tools the generator keeps an 8-point lead. When the scorer is wrong, the right tool is in its top
five 98% of the time; the generator's beam-search alternatives are mostly respellings and comma-joined
variants of its first guess, so its top-5 is only 91%. The generator still names a tool that isn't on the
list once every ~70 decisions; the scorer cannot. And the scorer decides in 250 ms because it does one
prefill and no decoding: on the laptop the prefill is 229 ms of that, tokenization 2 ms, and the
per-candidate span pooling under 1 ms.

### 4.5 More epochs don't change the picture

Because the scorer's validation curve was still rising at epoch 3, I continued it to 6 epochs. Top-1 on
the 500 steps went 66.6% → 65.8% while the training loss kept falling (0.43 → 0.29) and the seen/unseen
split widened (71% / 38%): overfitting, not headroom. Continuing the generator from 1 to 6 epochs gave the
same shape on its training-time check (72 → 73 → 75 → 74 → 73%), though that run's final checkpoint was
lost to a reclaimed Colab session before it could be scored on the 500 steps, so the matched table above
compares the scorer at 3 epochs with the generator at 1. Both curves are flat, so I don't expect the row to
change, but it is a caveat and the first thing I'd tidy up.

## 5. What I take from this

**GenRec's argument transfers.** For choosing from a known action set, you don't need the LLM to write
the answer. Once both approaches are trained the same way, scoring the candidates in one pass matches
generation on accuracy and delivers the two properties the design promises: no out-of-catalog picks, and a
single forward pass instead of decoding. It also produces a usable ranking, which matters for an agent
that can retry.

**The pooling position is not a detail.** GenRec says "a pooling position" and moves on. In my setup it
was the difference between a scorer that could not be fine-tuned at all and one that reaches parity. My
reading is that mean pooling averages away whatever a low-rank adapter changes, while the last token is
where a decoder-only model concentrates its decision, which is exactly why generation reads from there.

**Reading descriptions beats remembering them.** The table head, GenRec's per-item embedding, is at
chance on tools with no training label; a quarter of the test set. Building each candidate's vector from
its description inside the prompt (the span head) is what made the scorer competitive, and it is the
part of the design closest to GenRec's own cold-start advice ("include more detailed metadata"). The
remaining gap to the generator lives entirely in that unseen-tool slice.

**The `Finish` confound is a warning about prompted baselines.** A large part of the scorer's apparent
advantage over the untrained generator was that the generator never stops. Any comparison against a
zero-shot function-calling baseline should check for this before claiming a win.

## 6. Limitations

- Candidate lists average 5.6 tools. This is shortlist ranking, not full-catalog retrieval, so the latency
  advantage of prefill-only scoring is *understated* relative to GenRec's setting (their candidate sets are
  large), and the accuracy numbers are easier than a full-catalog task would be.
- One dataset, one backbone size, one training run per system, 500 evaluation steps. A 0.2-point gap is
  noise; the 8-point unseen-tool gap (121 steps) is probably real but wide.
- The matched-budget comparison is 3 epochs vs. 1 (see 4.5).
- Labels are what one reference agent did, and ToolBench's G1 trajectories often call a tool's endpoints in
  an arbitrary order, so top-1 has a ceiling well below 100% for any system.
- Latencies are from Apple Silicon at batch size 1; the ratio should hold elsewhere, the absolute numbers
  won't.

## 7. What I'd do next

1. Score the 6-epoch generator so the matched-budget row is exact (both curves are flat; this is hygiene).
2. Attack the unseen-tool gap directly: a description-side objective so the span representation of a tool
   the model has never called still aligns with prompts that need it.
3. Move to ToolBench G3 (multi-tool tasks, larger candidate sets). That is where prefill-only scoring
   should pull away on latency, and where the "large catalog" motivation actually gets tested.
4. Three seeds and the full 1,855-step held-out set for every row.
5. A generator baseline with constrained decoding, so its hallucination rate is also zero and the
   comparison isolates ranking quality and latency.

## 8. Reproducing it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest                                      # unit tests, no model download
.venv/bin/python data.py                              # download + parse ToolBench G1 -> data/processed
.venv/bin/python train_tier1.py                       # cache backbone vectors, train the head (tier1.head: table | span)
.venv/bin/python eval.py --systems tier1,span,baseline
```

All scripts read `config.yaml`; `model.device` is `mps` / `cuda` / `cpu`, and `ACTIONRANK_DEVICE` or
`ACTIONRANK_CONFIG` override the device or the whole config (e.g. `config_last.yaml` for last-token
pooling, `config_last_span.yaml` for the fine-tuned span scorer). Fine-tuning runs (`train_tier2.py`,
`baseline_sft.py`) work locally but are meant for a CUDA GPU; `colab/` has a detached pipeline for the
Colab CLI (`colab exec` drops the connection after ~60 s of silence, so the pipeline logs to a file and is
polled with `colab/tail_remote.py`). `scripts/analyse_predictions.py` produces the seen/unseen/`Finish`
breakdowns from any results directory.

Files: `data.py` (parsing, catalog, step slicing, trajectory split), `verbalize.py` (prompt + candidate
spans), `model.py` (backbone, pooling, both heads), `train_tier1.py`, `train_tier2.py`, `baseline.py`
(prompted generator), `baseline_sft.py` (fine-tuned generator), `eval.py`, `config.yaml`.

## References

- Netflix Technology Blog, *GenRec: Towards LLM-Native Recommendation at Netflix* (2026).
  https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3
- *GenRec: An LLM-Backed Recommendation Ranker at Netflix*, arXiv:2608.10257.
- Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs* (ToolBench), arXiv:2307.16789.
- Qwen Team, *Qwen2.5 Technical Report*.
