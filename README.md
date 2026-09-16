# ActionRank

### Does Netflix's "score, don't generate" idea work for agent tool selection?

Every agent framework picks its next tool the same way: the language model *writes out* a function call, one token at a time. That is slow, it can name a tool that doesn't exist, and it gives you one guess rather than a ranking.

Netflix's recommendation team recently argued, in [GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3), that when your choices come from a fixed catalog you shouldn't generate at all: run the LLM once over the context, then score every catalog item from that single pass. I wanted to know whether the same trick works when the "catalog" is an agent's toolbox.

**The short answer**, on ToolBench with a 1.5B-parameter Qwen backbone, once both approaches get the same fine-tuning:

- **Same accuracy.** Scoring the tool list in one forward pass picks the right tool as often as generating its name: **66.6% vs. 66.0%** top-1.
- **Never an invented tool.** The scorer cannot pick a tool that isn't offered (**0%**); the fine-tuned generator still does, **0.6%** of the time.
- **A far better ranking.** When the scorer is wrong, the right tool is in its top five **98%** of the time, vs. **91%** for the generator.
- **Less than half the latency.** **250 ms vs. 575 ms** per decision, because there is one prefill and no decoding.
- **The detour.** The scorer's own fine-tuning did nothing until I changed one detail of how the prompt is pooled; that detail turned a 5-point loss into a tie.
- **The remaining weakness.** On tools the scorer has never seen in training, the generator is still ahead: **50% vs. 41%**.

The rest of this report is how I got each of those numbers and what I think they mean.

---

**Contents**

1. [TL;DR](#tldr)
2. [Why I did this](#1-why-i-did-this)
3. [The idea, mapped onto GenRec](#2-the-idea-mapped-onto-genrec)
4. [Setup](#3-setup)
5. [Results](#4-results)
6. [What I take from this](#5-what-i-take-from-this)
7. [Limitations](#6-limitations)
8. [What I'd do next](#7-what-id-do-next)
9. [Reproducing it](#8-reproducing-it)
10. [References](#references)

---

## TL;DR

**The test.** Every system gets the same 500 held-out ToolBench steps. A step is one decision point in an agent's run: the user's task, the tool calls made so far and what they returned, and a short list of candidate tools (about 6 per task, `Finish` included). The system has to name the tool the reference agent called next. All runs use the same Qwen2.5-1.5B backbone, batch size 1, on a laptop GPU.

**The metrics.**

- **Top-1** is how often the system's single best guess is the correct tool. This is the number that matters for an agent that simply executes its first choice.
- **Top-5** is how often the correct tool is anywhere in the system's five best guesses. It measures how good the *ranking* is, which matters if you retry after a failed call, re-rank with a second model, or show alternatives.
- **Hallucination rate** is how often the system names a tool that isn't on the task's candidate list at all. A generated name can be misspelled, made up, or a tool from a different task. A scorer can only choose from the list, so its rate is 0% by construction.
- **Latency** is wall-clock time for one decision, including prompt building and tokenization.

**The systems.** There are two ways to pick a tool, and each is tested untrained and fine-tuned.

- **Generation** is what agent frameworks do today: the LLM writes the tool's name token by token, function-calling style. *Prompted* is the stock model with a chat prompt and no training. *Fine-tuned* adds a LoRA adapter trained on ToolBench to emit the right name.
- **ActionRank** is the GenRec idea applied to tools: run the LLM over the prompt once, then score every candidate tool from that single pass, with no decoding. The score for each tool comes from the hidden states over that tool's own description line in the prompt (the "span head"). *Frozen backbone* means the LLM is untouched and only the small scoring head is trained, which takes a minute on cached vectors. *Fine-tuned* trains the same LoRA adapter the generator gets, jointly with the head.

**The results.**

| system | what is trained | top-1 | top-5 | hallucination rate | latency / decision |
|---|---|---:|---:|---:|---:|
| Generation, prompted (no training) | nothing | 31.8% | 53.4% | 1.8% | 726 ms |
| **ActionRank**, frozen backbone (last-token pooling) | scoring head only | 62.0% | 96.2% | 0.0% | ~260 ms |
| Generation, fine-tuned | LoRA adapter | 66.0% | 93.8% | 0.6% | 575 ms |
| **ActionRank**, fine-tuned | LoRA adapter + scoring head | 66.6% | **98.2%** | **0.0%** | **250 ms** |

Read top to bottom. With the backbone untouched, scoring beats generation by 30 points, but that is an unfair fight: the scorer's small head has seen ToolBench and the prompted generator hasn't. Give both the same LoRA adapter and the accuracy gap closes to a tie. What survives is everything else: the scorer keeps zero hallucinations, a much stronger ranking, and less than half the latency. The one place the generator still wins, tools never seen in training, is covered in section 4.4.

---

## 1. Why I did this

**GenRec's pitch is simple.** Netflix has a fixed catalog of titles. An LLM understands a member's viewing history far better than a classical recommender, but making it *generate* title identifiers with beam search "introduces latency overhead that can be prohibitive at scale", and out-of-the-box LLMs "hallucinate out-of-catalog titles".

So GenRec does this instead:

1. Verbalize the history into text.
2. Run a decoder-only LLM over it once; take "the hidden state at a pooling position" as a summary vector `h`.
3. Score every catalog item with a *catalog-aware head*: each item has a learned embedding, and a small module combines `h` with that embedding to give a score.
4. Softmax over the catalog, rank, done.

Because only existing embeddings get scored, recommending a non-existent film is impossible by construction. Serving is "prefill-only": the model consumes the context once and ranks the full candidate set in a single pass.

**Agent tool selection has the same shape.** There is a fixed set of tools. The agent has a history (what it has called so far and what came back). It must pick the next action. Today that pick is generated, with exactly the problems GenRec lists: decoding cost, hallucinated tool names, no distribution over the action space.

> **Hypothesis.** Replace "generate a function call" with "score the toolbox", and you should get zero hallucination and lower latency at no cost in accuracy. The first two are almost guaranteed by the design; the third is the empirical question.

---

## 2. The idea, mapped onto GenRec

| GenRec (Netflix) | ActionRank (this project) |
|---|---|
| Verbalize viewing history + context + item metadata into one text sequence | Verbalize task query + compressed call history + candidate tools into one prompt |
| Context engineering: retain high-signal engagements, compress repetitive ones (binges), drop low-signal events | Keep the last 3 tool calls in full (args + truncated results), collapse older calls to their names, truncate observations to 200 chars |
| Decoder-only LLM backbone, shared with their foundation model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, unchanged |
| `h` = hidden state at a pooling position | `h` = mean over the prompt (v1) or the last token (v2). This choice turned out to matter a lot |
| Catalog-aware head: learned embedding per item, score = f(h, e_i) | **Table head**: learned embedding per tool (6,372 rows), cosine score with a residual MLP on `h` |
| For cold-start items, "include more detailed metadata" in the context | **Span head**: no table at all. Each candidate's vector is pooled from its own description line *inside the prompt*, so unseen tools get a representation for free |
| Softmax over the catalog; reward-weighted ranking loss | Softmax over the catalog, masked to the task's candidate list; plain cross-entropy (no reward weighting) |
| Prefill-only serving, one forward pass for the whole candidate set | Prefill-only: one pass, candidates scored from the same hidden states, no decoding |
| Phase 1 domain adaptation, Phase 2 ranking fine-tuning of backbone + head | Tier 1: backbone frozen, head trained on cached vectors in a minute. Tier 2: LoRA on q/v projections fine-tuned jointly with the head |

Two deliberate departures from GenRec:

- **No reward weighting.** ToolBench has no reward signal beyond "this is what the reference agent did", so every call counts the same.
- **The span head.** It has no analogue in the post. It is my attempt at the cold-start problem, and it ended up being the best scorer.

---

## 3. Setup

### Data

- **Source.** ToolBench G1 (single-tool tasks), from the `Adorg/ToolBench` mirror. Each answer file holds the task's API list with descriptions and a DFS tree of the reference agent's calls.
- **Trajectories.** I keep the root-to-leaf path that ends in `Finish` with an answer: 3,394 of 5,000 files.
- **Steps.** Each trajectory is sliced into one example per call: context = query + calls so far, label = the next call. `Finish` is a catalog tool, since deciding to stop is part of choosing the next action.
- **Split.** 15% of *trajectories* (not steps) are held out so no task leaks between train and eval: 10,568 training steps from 2,885 trajectories, 1,855 held-out steps from 509.
- **Catalog.** The union of every task's tools: 6,372. Each task lists 5.7 candidates on average (max 11).

> **The fact that shaped everything:** 24% of held-out labels are tools that never appear as a training label. Only 3,597 of the 6,372 catalog tools have any training signal at all.

### Prompt

`Task: … / History: … / Available tools: - name: description … / Next tool:`, about 410 tokens on average. Prompts are left-truncated at 1,280 tokens so the catalog and the suffix survive.

### Scoring heads

- **Table head** (GenRec's design): `score = cos(h + mlp(h), E[tool]) / τ`, with `E` initialised from the backbone's own encoding of each tool's description.
- **Span head**: drops `E`. For each candidate it mean-pools the hidden states over that candidate's line in the prompt (character offsets from the tokenizer), projects both sides with a residual MLP, and scores by cosine.
- Scores are scattered into a catalog-width vector with `-inf` elsewhere, so evaluation code is identical for both heads. Both MLPs have zero-initialised output layers, so an untrained head is plain cosine similarity.

### Training

- **Tier 1.** Cache one backbone pass per prompt (about 80 minutes on the laptop for 12.4k prompts), train only the head. The checkpoint epoch is chosen on a validation slice carved from *training* trajectories, never on the held-out set.
- **Tier 2.** Wrap the backbone with LoRA (rank 8, `q_proj`/`v_proj`, lr 2e-4, effective batch 16) and train adapter + head jointly on all 10,568 steps.

### Baselines

- **Prompted generation.** The same backbone, chat-prompted with the same task/history/tool list and asked to reply with exactly one tool name. Greedy decoding gives top-1 and latency; beam search (5) gives a top-5 list. Zero-shot.
- **Fine-tuned generation.** Same LoRA adapter, data and effective batch as Tier 2, trained to emit the tool name with loss only on the answer tokens.
- **Reference rows.** Uniform random among candidates, and the globally most frequent label (always `Finish`).

A prediction is a *hallucination* if the name isn't on the task's candidate list.

### Metrics

Top-1 and top-5 accuracy against the reference agent's next call, hallucination rate, and wall-clock latency per decision measured identically for every system (prompt building, tokenization, and the model call, at batch size 1; the beam pass is not timed).

---

## 4. Results

Everything below is on the first 500 held-out steps (the same 500 for every system). Full tables, including the reference rows, are in `results*/results*.md`; per-example predictions are next to them.

### 4.1 Without training, scoring wins easily

| system | top-1 | top-5 | hallucination | latency |
|---|---:|---:|---:|---:|
| prompted generation | 31.8% | 53.4% | 1.8% | 726 ms |
| ActionRank table head, frozen backbone | 49.4% | 91.4% | 0.0% | 235 ms |
| ActionRank span head, frozen backbone | 57.8% | 97.2% | 0.0% | 231 ms |
| random among candidates | 20.8% | 85.0% | – | – |

This is the result the hypothesis predicted, and also the least interesting one: the scorer has a trained head and the generator has nothing. The prediction dumps show where the gap comes from.

- **Stopping.** The prompted generator **never outputs `Finish`** (0 of 138 steps where stopping was the right call). The scorers learn it (81–89% recall).
- **Seen tools.** On non-`Finish` steps whose tool was seen in training, the table head and the generator are level (47% vs. 46%).
- **Unseen tools.** On the 121 steps whose tool was never seen in training, the table head is at chance (17%) while the generator, which actually reads the description, gets 39%. A learned embedding that was never trained cannot rank a tool, no matter how good `h` is.
- **The span head**, which reads the description from inside the prompt, recovers most of that (33%).

### 4.2 A fair baseline erases the accuracy lead

Fine-tuning the generator with the same adapter and data (one epoch, 12 minutes on an A100) is the strongest single intervention in the whole project.

| system | top-1 | top-5 | hallucination | latency | seen tools | unseen tools | `Finish` recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine-tuned generation, 1 epoch | **66.8%** | 91.4% | 1.4% | 639 ms | 65.6% | **49.6%** | 84% |
| span head, frozen backbone | 57.8% | **97.2%** | **0.0%** | 231 ms | 52.3% | 33.1% | 89% |

The generator learns to stop, jumps 35 points, and leads on unseen tools by 16. At this point the honest headline would have been "scoring buys hallucination-freedom and latency at the cost of accuracy".

### 4.3 The scorer couldn't be fine-tuned, until it could

The obvious response is to fine-tune the scorer's backbone too. The first attempt (LoRA + table head, all 10,568 steps, 2 epochs) was a clean null: **48.0% → 47.5% → 48.0%** on a validation subset, loss flat.

I checked whether the run was broken. It wasn't:

- With the adapter enabled, the pooled vector `h` changes by about 10% relative to the frozen one.
- The LoRA parameters receive a gradient norm of 0.51, against 1.18 for the head.

The adapter was learning; the metric wasn't moving. **The cause was the pooling position.** GenRec takes "the hidden state at a pooling position"; I had been taking the *mean* over all ~400 prompt tokens. A small adapter has to shift hundreds of token states coherently to move that mean, whereas the generator reads one sharp next-token distribution at the last position.

Switching the scorer to **last-token pooling** and re-running the identical recipe:

| Tier 2 (LoRA + table head), 2 epochs | validation subset top-1 by epoch |
|---|---|
| mean pooling | 48.0% → 47.5% → 48.0% |
| last-token pooling | 50.0% → 53.0% → 54.5% |

Same data, same adapter, same head; only the pooling position changed, and the scorer became trainable. For a *frozen* backbone the pooling choice matters much less: on the full 1,855-step held-out set the span head scores 56.7% (mean) vs. 57.8% (last) top-1, and on the 500-step subset used in the tables above, 57.8% vs. 62.0%. Small enough that it went unnoticed until Tier 2.

### 4.4 Matched fine-tuning: a tie on accuracy, a win on everything else

With last-token pooling, LoRA and the span head trained jointly (initialised from the frozen span head, 3 epochs over all steps). The generator row is the same adapter budget trained for the same 3 epochs:

| system | top-1 | top-5 | hallucination | latency | seen tools | unseen tools | `Finish` recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine-tuned generation, 3 epochs | 66.0% | 93.8% | 0.6% | 575 ms | 68.0% | **48.8%** | 78% |
| **fine-tuned ActionRank** (span head, last pooling), 3 epochs | 66.6% | **98.2%** | **0.0%** | **250 ms** | 67.2% | 41.3% | **88%** |

- **Accuracy.** The two pick the correct tool equally often, overall and on tools seen in training (a 0.6-point and a 0.8-point gap in opposite directions, both noise at n=500). On unseen tools the generator keeps a 7.5-point lead.
- **Ranking.** When the scorer is wrong, the right tool is in its top five 98% of the time. The generator's beam-search alternatives are mostly respellings and comma-joined variants of its first guess, so its top-5 is only 94%.
- **Hallucination.** Three more epochs cut the generator's invented-tool rate from 1.4% to 0.6%, but it still names a tool that isn't on the list three times in 500 decisions; the scorer cannot.
- **Latency.** The scorer decides in 250 ms because it does one prefill and no decoding; the generator needs 575 ms (639 ms for the one-epoch checkpoint; same beam settings, so treat the difference as run-to-run laptop variance). On the laptop the prefill is 229 ms of that, tokenization 2 ms, and the per-candidate span pooling under 1 ms.

### 4.5 More epochs don't change the picture

- **Scorer, 3 → 6 epochs.** Top-1 on the 500 steps went 66.6% → 65.8% while the training loss kept falling (0.43 → 0.29) and the seen/unseen split widened (71% / 38%). Overfitting, not headroom.
- **Generator, 1 → 3 epochs.** Trained fresh for 3 epochs to match the scorer's budget, its training-time check went 31 → 73 → 75 → 77% and the 500-step numbers moved from 66.8% / 91.4% / 1.4% (1 epoch) to 66.0% / 93.8% / 0.6%. Top-1 is flat; the extra epochs buy a little top-5 and hallucination, not accuracy.
- **Generator, 6 epochs.** A continuation run plateaued on the same check (72 → 73 → 75 → 74 → 73%) and its final checkpoint was lost to a reclaimed Colab session, so the 6-epoch comparison exists only for the scorer. Given both 3 → 6 curves are flat, I did not rerun it.

---

## 5. What I take from this

**GenRec's argument transfers.** For choosing from a known action set, you don't need the LLM to write the answer. Once both approaches are trained the same way, scoring the candidates in one pass matches generation on accuracy and delivers the two properties the design promises: no out-of-catalog picks, and a single forward pass instead of decoding. It also produces a usable ranking, which matters for an agent that can retry.

**The pooling position is not a detail.** GenRec says "a pooling position" and moves on. In my setup it was the difference between a scorer that could not be fine-tuned at all and one that reaches parity. My reading: mean pooling averages away whatever a low-rank adapter changes, while the last token is where a decoder-only model concentrates its decision, which is exactly why generation reads from there.

**Reading descriptions beats remembering them.** The table head, GenRec's per-item embedding, is at chance on tools with no training label, a quarter of the test set. Building each candidate's vector from its description inside the prompt (the span head) is what made the scorer competitive, and it is the part of the design closest to GenRec's own cold-start advice. The remaining gap to the generator lives entirely in that unseen-tool slice.

**The `Finish` confound is a warning about prompted baselines.** A large part of the scorer's apparent advantage over the untrained generator was that the generator never stops. Any comparison against a zero-shot function-calling baseline should check for this before claiming a win.

---

## 6. Limitations

- **Small candidate lists.** They average 5.6 tools, so this is shortlist ranking, not full-catalog retrieval. The latency advantage of prefill-only scoring is *understated* relative to GenRec's setting, and the accuracy numbers are easier than a full-catalog task would be.
- **One of everything.** One dataset, one backbone size, one training run per system, 500 evaluation steps. A sub-1-point gap is noise; the 7.5-point unseen-tool gap (121 steps) is probably real but wide.
- **Label noise.** Labels are what one reference agent did, and ToolBench's G1 trajectories often call a tool's endpoints in an arbitrary order, so top-1 has a ceiling well below 100% for any system.
- **Hardware.** Latencies are from Apple Silicon at batch size 1; the ratio should hold elsewhere, the absolute numbers won't.

---

## 7. What I'd do next

1. **Attack the unseen-tool gap directly**: a description-side objective so the span representation of a tool the model has never called still aligns with prompts that need it.
2. **Move to ToolBench G3** (multi-tool tasks, larger candidate sets). That is where prefill-only scoring should pull away on latency, and where the "large catalog" motivation actually gets tested.
3. **Three seeds** and the full 1,855-step held-out set for every row.
4. **A constrained-decoding generator baseline**, so its hallucination rate is also zero and the comparison isolates ranking quality and latency.

---

## 8. Reproducing it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/pytest                                      # unit tests, no model download
.venv/bin/python data.py                              # download + parse ToolBench G1 -> data/processed
.venv/bin/python train_tier1.py                       # cache backbone vectors, train the head (tier1.head: table | span)
.venv/bin/python eval.py --systems tier1,span,baseline
```

- All scripts read `config.yaml`. `model.device` is `mps` / `cuda` / `cpu`.
- `ACTIONRANK_DEVICE` and `ACTIONRANK_CONFIG` override the device or the whole config. `config.yaml` is the frozen / mean-pooling setup of 4.1–4.3; the variants behind the later sections live in `configs/` (see the table below).
- Fine-tuning runs (`train_tier2.py`, `baseline_sft.py`) work locally but are meant for a CUDA GPU. `colab/pipelines/` holds one detached pipeline per experiment stage for the Colab CLI (`colab exec` drops the connection after ~60 s of silence, so each pipeline logs to a file, packages checkpoints as soon as a stage ends, and is polled with `colab/tail_remote.py`).
- `scripts/analyse_predictions.py` produces the seen/unseen/`Finish` breakdowns from any results directory.

| file | what it does |
|---|---|
| `data.py` | parsing, catalog, step slicing, trajectory split |
| `verbalize.py` | prompt + candidate spans |
| `model.py` | backbone, pooling, both scoring heads |
| `train_tier1.py` | frozen-backbone caching and head training |
| `train_tier2.py` | LoRA fine-tuning jointly with either head |
| `baseline.py` | prompted generator |
| `baseline_sft.py` | fine-tuned generator |
| `eval.py` | metrics, latency, results tables, prediction dumps |
| `config.yaml` | every knob |

Each experiment stage has its own config, Colab pipeline and results folder, numbered in the order the report tells the story. The results folders keep the markdown tables and training logs; metrics JSON and prediction dumps are regenerated by `eval.py` and not tracked.

| stage | report | config | Colab pipeline | results |
|---|---|---|---|---|
| frozen heads, prompted and fine-tuned (1 epoch) generator, Tier 2 with mean pooling | 4.1–4.3 | `config.yaml` | `colab/pipelines/01_mean_pooling.py` | `results/01-frozen-mean-pooling/` |
| last-token pooling: re-encode, heads, Tier 2 table head | 4.3 | `configs/last_pooling.yaml` | `colab/pipelines/02_last_token.py` | `results/02-last-token-pooling/` |
| fine-tuned span scorer, 3 epochs | 4.4 | `configs/last_span_3ep.yaml` | `colab/pipelines/03_span_lora_3ep.py` | `results/03-span-lora-3ep/` |
| span scorer continued to 6 epochs (generator continuation lost to a reclaimed VM) | 4.5 | `configs/last_span_6ep.yaml` | `colab/pipelines/04_six_epochs.py`, `04b_generator_6ep.py` | `results/04-span-lora-6ep/` |
| fine-tuned generator, 3 epochs (matched budget) | 4.4–4.5 | `configs/generator_3ep.yaml` | `colab/pipelines/05_generator_3ep.py` | `results/05-generator-lora-3ep/` |

---

## References

- Netflix Technology Blog, *GenRec: Towards LLM-Native Recommendation at Netflix* (2026). https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3
- *GenRec: An LLM-Backed Recommendation Ranker at Netflix*, arXiv:2608.10257.
- Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs* (ToolBench), arXiv:2307.16789.
- Qwen Team, *Qwen2.5 Technical Report*.
