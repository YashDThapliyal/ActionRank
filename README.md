# ActionRank

### Does Netflix's "score, don't generate" idea work for agent tool selection?

Every agent framework picks its next tool the same way: the language model *writes out* a function call, one token at a time. That is slow, it can name a tool that doesn't exist, and it gives you one guess rather than a ranking.

Netflix's recommendation team recently argued, in [GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3), that when your choices come from a fixed catalog you shouldn't generate at all: run the LLM once over the context, then score every catalog item from that single pass. I wanted to know whether the same trick works when the "catalog" is an agent's toolbox.

**The short answer**, on ToolBench with a 1.5B-parameter Qwen backbone, three training seeds per system, evaluated on 1,352 held-out steps that no training run ever looked at:

- **The generator is still more accurate, by less than it first looked.** Trained with GenRec's full Phase 2 objective (ranking loss plus a language-modeling loss), the scorer reaches **64.0%** top-1 against the generator's **66.6%**, a gap of about 2.6 points. With the ranking loss alone it was 62.5%, a gap of 4.1. On the steps where an actual tool is chosen, the gap is about 4 points.
- **The missing loss term explains part of the gap, and it helps where GenRec says it should.** On tools that were never a training label the scorer went from 44% to 49% (51% on two of three seeds); the generator is at 54%. On tools it trained on, the LM term barely moved it.
- **Never an invented tool.** The scorer cannot pick a tool that isn't offered (**0%** on every seed); the fine-tuned generator names a tool that doesn't exist about **0.9%** of the time.
- **A far better ranking, and level on GenRec's own metric.** The right tool is in the scorer's top five **97 to 98%** of the time, vs. **91%** for the generator. On Mean Reciprocal Rank, the offline metric GenRec reports, the two are level: **0.778 vs. 0.764** MRR@5, with no seed pairing favouring the generator and three of nine favouring the scorer.
- **Less than half the latency.** **247 ms vs. 632 ms** per decision on the laptop, because there is one prefill and no decoding.
- **More stable.** Scorer seeds land within 2 points of each other; the generator's span 5, almost entirely from how reliably each seed learns to stop.
- **The detour.** The scorer's fine-tuning did nothing until I changed one detail of how the prompt is pooled.
- **The correction.** My first write-up called top-1 a tie (66.6% vs. 66.0%). That was on a 500-step development set both training runs had been monitored against. A pre-registered rerun on untouched steps, two more seeds per system, and then a re-read of GenRec's recipe replaced it with the numbers above. Sections 4.6 and 4.7 have the story.

Both systems choose a tool *name*; neither generates arguments.

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

**The test.** A step is one decision point in an agent's run: the user's task, the tool calls made so far and what they returned, and a short list of candidate tools (2 to 11, about 6 on average, `Finish` included). The system has to name the tool the reference agent called next. It picks a name only; argument generation is out of scope. All runs use the same Qwen2.5-1.5B backbone. The headline numbers are on 1,352 held-out steps (371 trajectories) that no training run was monitored against, with three training seeds per system; the earlier tables in section 4 are on a 500-step development set.

**The metrics.**

- **Top-1** is how often the system's single best guess is the correct tool. This is the number that matters for an agent that simply executes its first choice.
- **Top-5** is how often the correct tool is anywhere in the system's five best guesses. For the generator that list is the greedy answer plus four distinct beam-search alternatives. It measures how good the *ranking* is, which matters if you retry after a failed call, re-rank with a second model, or show alternatives.
- **MRR@5** (Mean Reciprocal Rank) is the average of 1/rank of the correct tool in that five-item list, 0 if it is absent. It is the offline metric GenRec reports, and it credits a system for putting the right tool second or third rather than only first.
- **Hallucination rate** is how often the system names a tool that isn't on the task's candidate list at all. A generated name can be misspelled, made up, or a tool from a different task. A scorer can only choose from the list, so its rate is 0% by construction.
- **Latency** is wall-clock time for one decision, including prompt building and tokenization.

**The systems.** There are two ways to pick a tool, and each is tested untrained and fine-tuned.

- **Generation** is what agent frameworks do today: the LLM writes the tool's name token by token, function-calling style. *Prompted* is the stock model with a chat prompt and no training. *Fine-tuned* adds a LoRA adapter trained on ToolBench to emit the right name.
- **ActionRank** is the GenRec idea applied to tools: run the LLM over the prompt once, then score every candidate tool from that single pass, with no decoding. The score for each tool comes from the hidden states over that tool's own description line in the prompt (the "span head"). *Frozen backbone* means the LLM is untouched and only the small scoring head is trained, which takes a minute on cached vectors. *Fine-tuned* adds a LoRA adapter with the same configuration the generator gets and trains it jointly with the head, starting from the frozen-backbone head.

**The results**, every system fine-tuned with the same LoRA configuration for three epochs, three seeds each, on the 1,352 unmonitored held-out steps. Mean over seeds, with the seed range in brackets. Latency is measured on the laptop; seed replicates were evaluated on an A100 and are not timed.

| system | training objective | top-1 | top-1, tool-choice steps only | MRR@5 | top-5 | hallucination rate | latency / decision |
|---|---|---:|---:|---:|---:|---:|---:|
| Generation, fine-tuned | next-token (answer only) | **66.6%** [63.4, 68.3] | **62.7%** [59.7, 64.5] | 0.764 [0.736, 0.780] | 90.7% [89.5, 92.2] | 0.9% [0.9, 1.0] | 632 ms |
| ActionRank, fine-tuned | ranking only | 62.5% [61.8, 63.5] | 56.6% [54.5, 58.3] | 0.768 [0.764, 0.774] | 97.2% [97.0, 97.5] | **0.0%** | 251 ms |
| **ActionRank, fine-tuned, GenRec's full objective** | ranking + language modeling | 64.0% [63.0, 65.0] | 58.7% [55.7, 60.6] | **0.778** [0.775, 0.782] | **97.5%** [97.0, 98.0] | **0.0%** | 247 ms |

With GenRec's full Phase 2 objective the generator is more accurate by about 2.6 points overall and 4 points on steps where a tool (not `Finish`) is the answer; with the ranking loss alone the gaps were 4.1 and 6. Under the pre-registered rule (trajectory-clustered 95% CI inside ±3 points), no seed pairing of the full-objective scorer against the generator is decisive in either direction. On MRR@5 the two are level: against generator seeds 1 and 2 every difference is within ±0.006 with intervals straddling zero, and against generator seed 3 the scorer leads by about 0.04. The scorer never picks an off-list tool, ranks the alternatives far better, decides in less than half the time, and varies less across seeds. That is the trade.

**Development history.** The first version of this report used the 500-step development set below and called top-1 a tie. Both training scripts had logged accuracy on a prefix of those steps, and the scorer's edge there came from `Finish` recall that did not carry over. Section 4.6 has the rerun and the seeds.

| system, 500-step dev set | what is trained | top-1 | top-5 | hallucination rate | latency / decision |
|---|---|---:|---:|---:|---:|
| Generation, prompted (no training) | nothing | 31.8% | 52.8% | 1.8% | 726 ms |
| ActionRank, frozen backbone (last-token pooling) | scoring head only | 62.0% | 96.2% | 0.0% | ~260 ms |
| Generation, fine-tuned | LoRA adapter | 66.0% | 93.4% | 0.6% | 575 ms |
| ActionRank, fine-tuned | LoRA adapter + scoring head | 66.6% | 98.2% | 0.0% | 250 ms |

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

> **Hypothesis.** Replace "generate a function call" with "score the toolbox", and you should get zero hallucination and lower latency at no cost in accuracy. The first two are guaranteed by the design; whether the third holds, and whether such a scorer can be trained at all, is the empirical question.

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
| Phase 1: adapt the LLM on domain corpora, no labels. Phase 2: post-train backbone + head + item embeddings jointly on conversations (verbalized context → actual engagement) with a ranking loss **plus a language-modeling loss over the verbalized inputs and outputs**, reward-weighted | No Phase 1. Tier 1: backbone frozen, head trained on cached vectors. Tier 2: LoRA on q/v fine-tuned jointly with the head. Sections 4.4 to 4.6 use the **ranking loss only**; section 4.7 adds GenRec's LM loss over the verbalized prompt and answer, which is the objective the final numbers use |

Two deliberate departures from GenRec, and one omission I only recognised after the results were in:

- **No reward weighting.** ToolBench has no reward signal beyond "this is what the reference agent did", so every call counts the same.
- **The span head.** It has no analogue in the post. It is my attempt at the cold-start problem, and it ended up being the best scorer.
- **No Phase 1, and at first no language-modeling objective (the omission).** GenRec's Phase 2 loss is ranking *plus* next-token prediction over the verbalized conversation, and it starts from a domain-adapted backbone. Until section 4.7, ActionRank's Tier 2 was ranking only, from the stock backbone, while the generator baseline was trained with exactly the next-token objective the scorer lacked. Section 4.7 adds the LM term (pre-registered as Experiment A, Arm J) and it accounts for about a third of the gap. Phase 1 remains undone.

---

## 3. Setup

### Data

- **Source.** ToolBench G1 (single-tool tasks), from the `Adorg/ToolBench` mirror. Each answer file holds the task's API list with descriptions and a DFS tree of the reference agent's calls.
- **Trajectories.** I keep the root-to-leaf path that ends in `Finish` with an answer: 3,394 of 5,000 files.
- **Steps.** Each trajectory is sliced into one example per call: context = query + calls so far, label = the next call. `Finish` is a catalog tool, since deciding to stop is part of choosing the next action.
- **Split.** 15% of *trajectories* (not steps) are held out so no task leaks between train and eval: 10,568 training steps from 2,885 trajectories, 1,855 held-out steps from 509.
- **Catalog.** The union of every task's tools: 6,372. Each step lists between 2 and 11 candidates, about 6 on average.

> **The fact that shaped everything:** 24% of held-out labels are tools that never appear as a training label (12% never appear even as a candidate). Only 3,597 of the 6,372 catalog tools have any positive training signal at all. Throughout, "unseen" means *never a training label*, and "seen" means a training label at least once; both are computed over non-`Finish` steps (241 seen, 121 unseen in the 500).

### Prompt

`Task: … / History: … / Available tools: - name: description … / Next tool:`, about 410 tokens on average. Prompts are left-truncated at 1,280 tokens so the catalog and the suffix survive.

### Scoring heads

- **Table head** (GenRec's design): `score = cos(h + mlp(h), E[tool]) / τ`, with `E` initialised from the backbone's own encoding of each tool's description.
- **Span head**: drops `E`. For each candidate it mean-pools the hidden states over that candidate's line in the prompt (character offsets from the tokenizer), projects both sides with a residual MLP, and scores by cosine.
- Scores are scattered into a catalog-width vector with `-inf` elsewhere, so evaluation code is identical for both heads. Both MLPs have zero-initialised output layers, so an untrained head is plain cosine similarity.

### Training

- **Tier 1.** Cache one backbone pass per prompt (about 80 minutes on the laptop for 12.4k prompts), train only the head. The checkpoint epoch is chosen on a validation slice carved from *training* trajectories, never on the held-out set.
- **Tier 2.** Wrap the backbone with LoRA (rank 8, `q_proj`/`v_proj`, lr 2e-4, effective batch 16) and train adapter + head jointly on all 10,568 steps, with the head initialised from the Tier 1 checkpoint. After each epoch the script logs top-1 on the first 200 held-out steps. No checkpoint selection was done on that number, but those 200 steps sit inside the 500-step evaluation subset, which is why section 6 treats the 500 as a development set.

### Baselines

- **Prompted generation.** The same backbone, chat-prompted with the same task/history/tool list and asked to reply with exactly one tool name. Greedy decoding gives top-1 and latency; beam search (5) gives a top-5 list. Zero-shot.
- **Fine-tuned generation.** Same LoRA configuration, data and effective batch as Tier 2, trained from the base model to emit the tool name with loss only on the answer tokens. Its per-epoch log uses the first 100 held-out steps.
- **Reference rows.** Uniform random among candidates, and the globally most frequent label (always `Finish`).

A prediction is a *hallucination* if the name isn't on the task's candidate list.

### Metrics

Top-1 and top-5 accuracy against the reference agent's next call, hallucination rate, and wall-clock latency per decision measured identically for every system (prompt building, tokenization, and the model call, at batch size 1; the beam pass is not timed). Top-5 and MRR@5 are over a strict five-item list: the system's top-1 followed by its distinct ranked alternatives (for the generator, the greedy answer followed by distinct beam outputs). MRR@5 is the mean of 1/rank of the label in that list, 0 when absent; it is GenRec's offline metric and was added after the fact from the saved prediction files (`scripts/paired_test.py`), so it was not part of either pre-registration. The saved `results*.md` files were written by an earlier evaluator that credited the greedy answer *or* any of five beams (up to six guesses); the generator rows below are recomputed from the prediction files with the strict definition, which lowers the prompted generator from 53.4% to 52.8%, the 3-epoch generator from 93.8% to 93.4%, and random from 85.0% to 83.4%. Scorer rows are unchanged, since their list starts with their top-1.

---

## 4. Results

Everything below is on the first 500 held-out steps (the same 500 for every system). Full tables, including the reference rows, are in `results*/results*.md`; per-example predictions are next to them.

### 4.1 Without training, scoring wins easily

| system | top-1 | top-5 | hallucination | latency |
|---|---:|---:|---:|---:|
| prompted generation | 31.8% | 52.8% | 1.8% | 726 ms |
| ActionRank table head, frozen backbone | 49.4% | 91.4% | 0.0% | 235 ms |
| ActionRank span head, frozen backbone | 57.8% | 97.2% | 0.0% | 231 ms |
| random among candidates | 20.8% | 83.4% | – | – |

This is the result the hypothesis predicted, and also the least interesting one: the scorer has a trained head and the generator has nothing. The prediction dumps show where the gap comes from.

- **Stopping.** The prompted generator **never outputs `Finish`** (0 of 138 steps where stopping was the right call). The scorers learn it (81–89% recall).
- **Seen tools.** On non-`Finish` steps whose tool was a training label, the table head and the generator are level (47% vs. 46%).
- **Unseen tools.** On the 121 steps whose tool was never a training label, the table head is at chance (17%) while the generator, which actually reads the description, gets 39%. An embedding row that never received a positive example cannot rank a tool, no matter how good `h` is.
- **The span head**, which reads the description from inside the prompt, recovers most of that (33%).

### 4.2 A fair baseline erases the accuracy lead

Fine-tuning the generator with the same LoRA configuration and data (one epoch, 12 minutes on an A100) is the strongest single intervention in the whole project.

| system | top-1 | top-5 | hallucination | latency | seen tools | unseen tools | `Finish` recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine-tuned generation, 1 epoch | **66.8%** | 91.4% | 1.4% | 639 ms | 65.6% | **49.6%** | 84% |
| span head, frozen backbone | 57.8% | **97.2%** | **0.0%** | 231 ms | 52.3% | 33.1% | 89% |

The generator learns to stop, jumps 35 points, and leads on unseen tools by 16. At this point the honest headline would have been "scoring buys hallucination-freedom and latency at the cost of accuracy".

### 4.3 The scorer couldn't be fine-tuned, until it could

The obvious response is to fine-tune the scorer's backbone too. The first attempt (LoRA + table head, all 10,568 steps, 2 epochs) was a null: **48.0% → 47.5% → 48.0%** top-1 on the first 200 held-out steps. Training loss fell slightly (0.82 after epoch 1, about 0.79 after epoch 2); accuracy did not follow.

I checked whether the run was broken. It wasn't:

- With the adapter enabled, the pooled vector `h` changes by about 10% relative to the frozen one.
- The LoRA parameters receive a gradient norm of 0.51, against 1.18 for the head.

Both are one-off diagnostics from that run; the numbers were not saved to the repo.

The adapter was learning; the metric wasn't moving. **My best explanation is the pooling position.** GenRec takes "the hidden state at a pooling position"; I had been taking the *mean* over all ~400 prompt tokens. A small adapter has to shift hundreds of token states coherently to move that mean, whereas the generator reads one sharp next-token distribution at the last position.

Switching the scorer to **last-token pooling** and re-running the identical recipe:

| Tier 2 (LoRA + table head), 2 epochs | top-1 on the first 200 held-out steps, by epoch |
|---|---|
| mean pooling | 48.0% → 47.5% → 48.0% |
| last-token pooling | 50.0% → 53.0% → 54.5% |

Same data, same LoRA configuration, same head; only the pooling position changed, and the scorer became trainable. I have not isolated the mechanism beyond this one controlled swap. For a *frozen* backbone the pooling choice matters much less: on the full 1,855-step held-out set the span head scores 56.7% (mean) vs. 57.8% (last) top-1, and on the 500-step subset used in the tables above, 57.8% vs. 62.0%. Small enough that it went unnoticed until Tier 2.

### 4.4 Matched fine-tuning on the development set: a tie that did not survive

With last-token pooling, LoRA and the span head trained jointly (head initialised from the frozen Tier 1 span head, 3 epochs over all steps). Everything in this subsection is on the 500-step development set; section 4.6 is the evaluation that counts. The generator row uses the same LoRA configuration, data and 3 epochs, starting from the base model. Both get three epochs of backbone fine-tuning; the scorer additionally carries its one-minute head-only stage, so the budgets are matched on the backbone, not identical in total:

| system | top-1 | top-5 | hallucination | latency | seen tools | unseen tools | `Finish` recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine-tuned generation, 3 epochs | 66.0% | 93.4% | 0.6% | 575 ms | 68.0% | **48.8%** | 78% |
| **fine-tuned ActionRank** (span head, last pooling), 3 epochs | 66.6% | **98.2%** | **0.0%** | **250 ms** | 67.2% | 41.3% | **88%** |

- **Accuracy.** The two pick the correct tool equally often, overall and on training-label tools (a 0.6-point and a 0.8-point gap in opposite directions, both noise at n=500; paired, 57 steps only the scorer gets right vs. 54 only the generator). On tools that were never a training label the generator keeps a 7.5-point lead.
- **Ranking.** When the scorer is wrong, the right tool is in its top five 98% of the time. The generator's beam-search alternatives are mostly respellings and comma-joined variants of its first guess, so its top-5 is only 93%.
- **Hallucination.** Three more epochs cut the generator's invented-tool rate from 1.4% to 0.6%, but it still names a tool that isn't on the list three times in 500 decisions; the scorer cannot.
- **Latency.** The scorer decides in 250 ms because it does one prefill and no decoding; the generator needs 575 ms (639 ms for the one-epoch checkpoint; same beam settings, so treat the difference as run-to-run laptop variance). On the laptop the prefill is 229 ms of that, tokenization 2 ms, and the per-candidate span pooling under 1 ms (a one-off breakdown; `scripts/prototypes/span_breakdown.py` re-measures it).

### 4.5 More epochs don't change the picture

- **Scorer, 3 → 6 epochs.** Top-1 on the 500 steps went 66.6% → 65.8% while the training loss kept falling (0.54 at epoch 3, 0.29 at epoch 6) and the seen/unseen split widened (71% / 38%). Overfitting, not headroom.
- **Generator, 1 → 3 epochs.** Trained fresh for 3 epochs to match the scorer's budget, its training-time check went 31 → 73 → 75 → 77% and the 500-step numbers moved from 66.8% / 91.4% / 1.4% (1 epoch) to 66.0% / 93.4% / 0.6%. Top-1 is flat; the extra epochs buy a little top-5 and hallucination, not accuracy.
- **Generator, 6 epochs.** A continuation run plateaued on the same check (72 → 73 → 75 → 74% in the saved log, which stops at epoch 4) and its final checkpoint was lost to a reclaimed Colab session, so the 6-epoch comparison exists only for the scorer. Given both 3 → 6 curves are flat, I did not rerun it.

### 4.6 The tie did not survive a clean evaluation

A code review found that both training scripts log top-1 on a prefix of the held-out split after every epoch (`ds.eval[:200]` for the scorer, `ds.eval[:100]` for the generator), and those steps sit inside the 500 used above. No checkpoint was chosen on them, but they were visible while I made decisions, so the 500 is a development set. I wrote a pre-registered plan (`docs/experiments/2026-09-16-unmonitored-holdout-rerun.md`, committed before the results existed) to evaluate the two frozen final checkpoints on held-out steps 503 to 1,854: 1,352 steps from 371 trajectories, none monitored, none sharing a trajectory with the development prefix. The decision rule was a trajectory-clustered bootstrap CI on the top-1 difference with a ±3-point equivalence margin.

**The rerun.** Generator **68.3%**, scorer **63.5%**. Paired, the generator was right and the scorer wrong on 180 steps, the reverse on 115. The clustered 95% CI for the difference was [−7.3, −2.4] points. The tie was gone. Where did the development-set parity come from? `Finish` recall: on the 500 the scorer stopped correctly 88% of the time against the generator's 78%; on the 1,352 they were level (77% vs. 78%), and on non-`Finish` steps the generator led 64.5% to 58.3%.

**Seeds.** One run each cannot separate a 5-point effect from training noise, so I added a `train_seed` (shuffle order and LoRA initialisation; the LoRA init had been unseeded) and trained two more of each system with the identical recipe on an A100, evaluating each on the same 1,352 steps.

| top-1 on the 1,352 unmonitored steps | seed 1 | seed 2 | seed 3 | mean | `Finish` recall by seed | non-`Finish` top-1 by seed |
|---|---:|---:|---:|---:|---|---|
| generation, fine-tuned 3 epochs | 68.3% | 68.2% | 63.4% | 66.6% | 78 / 91 / 62% | 64.5 / 59.7 / 64.0% |
| ActionRank, fine-tuned 3 epochs | 63.5% | 62.4% | 61.8% | 62.5% | 77 / 83 / 74% | 58.3 / 54.5 / 57.1% |

Generator seed 3 is five points below its siblings, and the breakdown says why: its `Finish` recall is 62% where the others are 78% and 91%, while its accuracy on actual tool choices (64.0%) matches them. The generator's seed variance is almost entirely about how reliably it learns to stop. The scorer's overall number moves less than a point per seed.

All nine scorer-versus-generator pairings, scorer minus generator, trajectory-clustered 95% CI, ±3-point margin:

| | generator seed 1 | generator seed 2 | generator seed 3 |
|---|---|---|---|
| scorer seed 1 | −4.8 [−7.3, −2.4] | −4.7 [−7.3, −2.3] | +0.1 [−2.6, +2.8], equivalent |
| scorer seed 2 | −5.9 [−8.6, −3.2], generator better | −5.8 [−8.4, −3.2], generator better | −1.0 [−3.8, +1.8], inconclusive |
| scorer seed 3 | −6.5 [−9.0, −4.0], generator better | −6.4 [−9.1, −3.8], generator better | −1.6 [−4.2, +1.0], inconclusive |

Six of nine pairings favour the generator with intervals clear of zero; the three involving generator seed 3 are a tie or inconclusive. On non-`Finish` steps, the generator leads on every pairing. Everything the scorer was built for held on every seed: hallucination 0.0%, top-5 97.0 to 97.5% against 89.5 to 92.2%, and the latency ratio from the laptop run. On tools that were never a training label the generator's lead is 8 to 12 points seed for seed (mean 53.8% vs. 44.2%).

Per-seed tables are in `results/06-unmonitored-holdout/` and `results/07-seeds/`; `scripts/paired_test.py` reproduces the pairings and prints MRR@5 with its clustered interval.

### 4.7 Giving the scorer the rest of GenRec's recipe

Re-reading the GenRec post after the seeds, I found the departure I had not declared. GenRec's Phase 2 loss is the catalog-aware ranking objective *plus* "a language modeling objective over the verbalized inputs and outputs", trained jointly; the training data are conversations whose assistant turn is the actual engagement, and "during Phase-2 training, the LLM learns how assistant messages depend on user messages". ActionRank's Tier 2 had only the ranking loss. The generator baseline had only the LM loss. So the matched comparison in 4.6 was a half-recipe scorer against a generator that got the half the scorer was missing.

The pre-registered fix (`docs/experiments/2026-09-18-genrec-two-phase-training.md`, Arm J): the same Tier 2 run, with the loss changed to ranking cross-entropy plus next-token cross-entropy over the verbalized prompt followed by `Next tool: <label>` and EOS, from one forward pass, weight 1, fixed in advance. The pooled query and the span vectors are read from prompt positions only, so the answer cannot leak into the score (tested: the catalog logits are identical with and without it). Inference is untouched. Three seeds, evaluated once each on the same 1,352 steps.

| top-1 on the 1,352 unmonitored steps | seed 1 | seed 2 | seed 3 | mean | never-label tools | non-`Finish` | `Finish` recall |
|---|---:|---:|---:|---:|---|---|---|
| ActionRank, ranking loss only (4.6) | 63.5% | 62.4% | 61.8% | 62.5% | 46.8 / 42.2 / 43.5% | 58.3 / 54.5 / 57.1% | 77 / 83 / 74% |
| ActionRank, ranking + LM loss | 64.0% | 65.0% | 63.0% | 64.0% | 51.4 / 51.4 / 44.4% | 59.7 / 60.6 / 55.7% | 75 / 77 / 82% |
| generation, fine-tuned 3 epochs | 68.3% | 68.2% | 63.4% | 66.6% | 55.0 / 51.1 / 55.3% | 64.5 / 59.7 / 64.0% | 78 / 91 / 62% |

- **The LM term helps, and it helps where GenRec says it should.** Mean top-1 +1.5 points; on tools that were never a training label, two seeds gained 7 points (44 to 51%) and one did not move. That slice is the reading-the-description problem the LM objective targets, and on it the full-objective scorer is now within 5 points of the generator's mean (49.0% vs. 53.8%) instead of 10. On tools the model trained on, the gain is under a point.
- **The gap narrows from 4.1 to 2.6 points and stops being decisive.** All nine pairings of full-objective scorer seeds against generator seeds are *inconclusive* under the ±3 margin: six exclude zero in the generator's favour (worst −5.3, best −3.2), three straddle it (+1.6 to −0.4 against generator seed 3). Where the ranking-only scorer lost four of nine pairings outright, the full-objective scorer loses none outright and wins none.
- **On GenRec's metric the scorer is level with the generator.** MRR@5 is 0.778 for the full-objective scorer (0.775 to 0.782 by seed), 0.768 for the ranking-only scorer, and 0.764 for the generator (0.736 to 0.780). Across the nine full-objective pairings, none favours the generator, six straddle zero (every difference within ±0.006), and the three against generator seed 3 favour the scorer by about 0.04 with intervals clear of zero. The generator's first guess is right more often; when it is wrong, its alternatives are mostly respellings, while the scorer's second and third choices are real candidates, and MRR credits that.
- **The structural results are unchanged**: 0.0% hallucination on every seed, top-5 97.0 to 98.0%, and the same inference cost: 247 ms per decision on the laptop over the same 1,352 steps, one prefill, against 251 ms for the ranking-only scorer. The inference code is identical, so the 4 ms is run-to-run variance (`results/08-genrec-joint/latency_laptop_joint_s1/`).
- **What is left** is a 2 to 4 point deficit concentrated on tools the model trained on and on non-`Finish` steps. The LM loss does not touch that; GenRec's Phase 1, more data, or a larger backbone might. Those are future work.

Results are in `results/08-genrec-joint/`. The pre-registration document records the expectations (Arm J was expected to land within the margin on at least six of nine pairings; it landed inconclusive on nine of nine, with the never-label gain as predicted).

---

## 5. What I take from this

**Scoring is a trade, not a free win.** GenRec's structural promises transfer exactly: a scorer cannot pick an off-catalog tool, it ranks every candidate from one forward pass, and it decides in less than half the time. What did not fully transfer is accuracy parity. With GenRec's full Phase 2 objective and three seeds, generating the name is still about 2.6 points more accurate overall and about 4 on steps where a tool (not `Finish`) is the answer, though no seed pairing is decisive under the pre-registered margin, and on MRR, the ranking metric GenRec itself reports, the two are level. Whether that is a good trade depends on the agent: one that retries from a ranking, or that cannot afford an invalid call, gets a lot for 3 to 4 points; one that executes its first choice and can validate names cheaply gets less.

**Evaluate on data nobody looked at, replicate, then re-read the paper.** The tie I first reported was real on the development set and gone on untouched steps, because a `Finish`-recall edge specific to those 500 steps did not generalise. Seeds then showed that the generator's own number swings by 5 points depending on how well a run learns to stop. And going back to GenRec's text line by line turned up a loss term I had dropped, which recovered a third of the remaining gap. None of that was visible from one run on one slice.

**The pooling position is not a detail.** GenRec says "a pooling position" and moves on. In my setup it was the difference between a scorer that could not be fine-tuned at all and one within 3 points of the generator. My reading, which the one controlled swap supports but does not prove: mean pooling averages away whatever a low-rank adapter changes, while the last token is where a decoder-only model concentrates its decision, which is exactly why generation reads from there.

**Reading descriptions beats remembering them.** The table head, GenRec's per-item embedding, is at chance on tools with no training label, a quarter of the test set. Building each candidate's vector from its description inside the prompt (the span head) is what made the scorer competitive, and it is the part of the design closest to GenRec's own cold-start advice. With the ranking loss alone the scorer was about 10 points behind the generator on tools that were never a training label; GenRec's LM objective closed most of that (section 4.7). The remaining deficit sits on tools the model trained on.

**Stopping is the unstable part.** The prompted generator never stops; fine-tuned generator seeds stop correctly anywhere from 62% to 91% of the time; the scorer's `Finish` recall also moves 10 points across seeds. Any tool-selection comparison should report `Finish` recall separately, because it can swing the headline number by 5 points without the tool-choice accuracy changing at all.

---

## 6. Limitations

- **Small candidate lists.** They average about 6 tools, so this is shortlist ranking, not full-catalog retrieval. The latency advantage of prefill-only scoring is *understated* relative to GenRec's setting, and the accuracy numbers are easier than a full-catalog task would be.
- **The 1,352 steps are unmonitored, not pristine.** No training run logged them and they share no trajectory with the development prefix, but the frozen Tier 1 heads were evaluated on the full 1,855-step split during development (section 4.3 quotes those numbers), and those aggregates informed the choice of span head and pooling. A truly independent test needs a predeclared train/dev/test split and retraining. The 500-step tables in sections 4.1 to 4.5 are development results and should be read as such.
- **Tool-name selection only.** Neither system generates arguments. The latency and hallucination numbers cover choosing the tool, not producing a complete, valid call; a real agent still has to generate and validate arguments afterwards.
- **Head pre-training is not matched.** The scorer's 3-epoch run starts from a head that already had up to 60 head-only epochs on cached vectors. The generator starts from the base model. Backbone fine-tuning is matched; total task-specific training is not.
- **Beam search is a weak ranking baseline.** A constrained decoder restricted to candidate names, or scoring each candidate name's likelihood, would be a fairer zero-hallucination comparison for top-5 and latency.
- **Three seeds for the final pair, one for everything else.** The headline comparison has three training seeds per system on 1,352 steps. Every other row (prompted generator, frozen heads, pooling comparison, epoch sweeps) is one run on the 500-step development set. One dataset and one backbone size throughout.
- **Label noise.** Labels are what one reference agent did, and ToolBench's G1 trajectories often call a tool's endpoints in an arbitrary order, so top-1 has a ceiling well below 100% for any system.
- **Hardware.** Latencies are from Apple Silicon at batch size 1. I expect the ratio to hold elsewhere, but I have not measured it; the absolute numbers won't transfer.
- **Backbone size.** Everything is on a 1.5B model, and not every row would survive a scale-up the same way. The hallucination and latency gaps are structural: a scorer cannot name an off-list tool at any size, and the generator always pays for decoding on top of the same prefill. The untrained row is the most size-specific, since a strong model zero-shot would likely beat a frozen head outright. The unseen-tool gap is the one I'd expect to move: the span head scores a tool from the backbone's reading of its description line, and a bigger backbone reads descriptions better, so the generator's lead there (7.5 points on the development set, about 5 on the 1,352 steps with the full objective) might narrow. That is a hypothesis, not a result.

---

## 7. Future expansions

This first pass stops here: GenRec's Phase 2 recipe replicated on tool selection at 1.5B, compared against a matched generator on a clean held-out set with seeds and intervals. Each item below is something GenRec does, or calls out, that this project has not, with a note on feasibility.

1. **Phase 1 domain adaptation.** Continue pretraining the backbone with the next-token loss on ToolBench text with no labels (every tool description, plus the trajectories with the answers masked), then run Tier 2 on top. GenRec credits this with a 10 to 20% relative gain. Pre-registered as Arm P in `docs/experiments/2026-09-18-genrec-two-phase-training.md`; about 2 A100-hours; the most likely next win on never-label tools.
2. **More training data.** GenRec's headline ablation is that ranking quality improves monotonically with Phase 2 data. This project used 10,568 training steps from 2,885 trajectories, parsed from the 5,000 G1 answer files in the mirror; ToolBench's full G1 set is several times larger and was not downloaded. Keep the held-out split fixed, slice with the same verbalizer, retrain both systems. Cheap, and the 6-epoch overfitting in 4.5 says the models are data-limited.
3. **A larger backbone.** GenRec post-trained 1B to 10B backbones and found larger consistently better. Qwen2.5-7B fits LoRA training on a 40 GB A100; pre-registered in `docs/experiments/2026-09-18-larger-backbone.md`, about 6 A100-hours for one seed of everything. On hold.
4. **Score the full catalog, not the shortlist.** GenRec's real setting. The span head cannot do it (6,372 descriptions do not fit in a prompt); the table head can. A real design question, about a day of work.
5. **ToolBench G3** (multi-tool tasks, longer histories), where the context pooling gets stressed.
6. **A constrained-decoding generator baseline**, so its hallucination rate is also zero and the comparison isolates ranking quality and latency. Not GenRec, but the fairest generative comparison.
7. **A predeclared train/dev/test split with retraining**, so the final numbers come from data no stage of development touched. Section 4.6 is confirmatory, not independent.
8. **Reward weighting** cannot be done here: ToolBench has no engagement signal, and filtering to successful trajectories already makes every example's weight 1.

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
| both final checkpoints on held-out steps 503–1854 (pre-registered, laptop) | 4.6 | `configs/unmonitored_holdout.yaml` | local, `docs/experiments/2026-09-16-unmonitored-holdout-rerun.md` | `results/06-unmonitored-holdout/` |
| seed replicates 2 and 3 of both final systems, evaluated on the same 1,352 steps | 4.6 | `configs/seeds/*.yaml` | `colab/pipelines/07_seeds_span.py`, `07_seeds_generator.py` | `results/07-seeds/` |
| span scorer with GenRec's joint ranking + LM objective, seeds 1 to 3, same 1,352 steps; laptop latency of seed 1 | 4.7 | `configs/genrec/joint_s*.yaml`, `joint_s1_latency_laptop.yaml` | `colab/pipelines/08_joint_seeds.py` | `results/08-genrec-joint/` |

---

## References

- Netflix Technology Blog, *GenRec: Towards LLM-Native Recommendation at Netflix* (2026). https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3
- *GenRec: An LLM-Backed Recommendation Ranker at Netflix*, arXiv:2608.10257.
- Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs* (ToolBench), arXiv:2307.16789.
- Qwen Team, *Qwen2.5 Technical Report*.
