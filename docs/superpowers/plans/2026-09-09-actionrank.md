# ActionRank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ActionRank, a prefill-only catalog-scoring model for next-tool prediction on ToolBench G1, and show it matches/exceeds an autoregressive function-calling baseline on top-1/top-5 accuracy with ~zero hallucination and lower latency.

**Architecture:** ToolBench G1 answer files are parsed into trajectories (query, per-task function list, ordered tool calls with observations) and sliced into per-step examples. A verbalizer renders (query, compressed history, candidate catalog) into one prompt; Qwen2.5-1.5B-Instruct runs prefill only, hidden states are pooled to `h`, and a learned tool-embedding table scores every catalog tool by dot product (masked to the task's candidates). Tier 1 trains only the head on cached `h`; Tier 2 adds LoRA on q/v projections. A generation baseline (same backbone, greedy + beam-5) is compared on accuracy, hallucination and latency by `eval.py`.

**Tech Stack:** Python 3.12 (uv venv at `.venv`), torch 2.14 (MPS), transformers 5.17, peft 0.20, huggingface_hub, pyyaml, pytest.

**Spec:** `spec.md`

## Global Constraints

- Backbone: `Qwen/Qwen2.5-1.5B-Instruct` via `transformers`, fp16.
- Device: `mps` locally; same scripts run on Colab with `device: cuda`. Device is a config value, never hard-coded.
- Data: ToolBench G1 subset (Hugging Face mirror `Adorg/ToolBench`, path `answer/G1_answer/*.json`).
- Split: hold out ~15% of full trajectories (by query id, fixed seed), never individual steps.
- Verbalization: keep last 2-3 steps in full; summarize/drop older steps; catalog rendered as `name: description` lines.
- Tool embedding table: `nn.Embedding(num_tools, hidden_dim)`; score = dot product against pooled `h`; softmax over catalog.
- Loss: cross-entropy over the catalog, restricted to the task's candidate set (true label + candidate negatives); this is the "true label + hard negatives" option in the spec.
- Tier 2 LoRA: target `q_proj`/`v_proj`, rank 8-16, alpha 16-32, dropout 0.05, AdamW lr 1e-4..2e-4, gradient accumulation.
- Baseline: same backbone prompted to output a tool name; metrics top-1, top-5, invalid/hallucinated tool rate, wall-clock latency per decision.
- File layout per spec: `data.py`, `verbalize.py`, `model.py`, `train_tier1.py`, `train_tier2.py`, `baseline.py`, `eval.py`, `config.yaml` at repo root, plus `tests/`.
- Commit after every task; run Codex review (`codex-companion.mjs review --wait`) after each phase and fix findings. After a Codex review, run `git symbolic-ref -q --short HEAD` and `git checkout main` if it prints nothing (the review leaves a detached HEAD).
- Coding rules: immutable data (frozen dataclasses, new objects), files < 800 lines, functions < 50 lines, no silently swallowed errors, no hard-coded values (config.yaml).

---

## Data facts (verified 2026-09-09 on the mirror)

Each `answer/G1_answer/<qid>_ChatGPT_DFS_woFilter_w2.json` looks like:

```json
{
  "win": true,
  "tree": {"size": 7, "max_length": 7, "tree": {"node_type": "Action Input", "description": "", "children": [
      {"node_type": "Action", "description": "get_info_for_covid_19_india", "children": [
        {"node_type": "Action Input", "description": "{}", "observation": "{\"error\": \"\", \"response\": \"...\"}", "children": [
          {"node_type": "Action", "description": "Finish", "children": [
            {"node_type": "Action Input", "description": "{\"return_type\": \"give_answer\", \"final_answer\": \"...\"}", "children": []}]}]}]}]}},
  "forward_args": {...},
  "compare_candidates": [...],
  "answer_generation": {
    "valid_data": true, "query_count": 3, "total_tokens": 0, "final_answer": "...",
    "function": [{"name": "get_info_for_covid_19_india", "description": "This is the subfunction for tool \"covid_19_india\", you can use this tool.The description of this function is: \"...\"", "parameters": {...}}, ..., {"name": "Finish", "description": "If you think you get the result which can answer the task, call this function to give the final answer. ...", "parameters": {...}}],
    "chain": [], "query": "...", "finish_type": "give_answer"
  }
}
```

- `node_type` is one of `Action`, `Action Input`, `Thought`. A tool call is an `Action` node (description = function name) followed by its `Action Input` child (description = JSON args, `observation` = tool result string).
- Trees can branch (DFS with restarts). The trajectory we keep is the root-to-leaf path that ends in `Finish` with `give_answer`; if none exists the file is skipped when `require_win` is true.
- Every task's function list ends with `Finish`; `Finish` is treated as a catalog tool (predicting "stop" is a valid next action). Eval reports metrics both overall and excluding `Finish`-label steps.

---

### Task 1: Project scaffold and config

**Files:**
- Create: `config.yaml`, `config.py`, `requirements.txt`, `.gitignore`, `README.md`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `config.py::load_config(path: str | Path = "config.yaml") -> Config` where `Config` is a frozen dataclass with nested frozen dataclasses `DataConfig`, `VerbalizeConfig`, `ModelConfig`, `Tier1Config`, `Tier2Config`, `BaselineConfig`, `EvalConfig`. Every script calls `load_config()` and reads values from it.

- [x] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from config import load_config

def test_load_default_config_has_expected_sections():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.model.backbone == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.data.eval_fraction == 0.15
    assert cfg.verbalize.full_history_steps in (2, 3)
    assert cfg.tier2.lora_rank in range(8, 17)

def test_config_is_immutable():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.backbone = "x"
```

- [x] **Step 2: Run test to verify it fails** — `.venv/bin/pytest tests/test_config.py -v` → ImportError.

- [x] **Step 3: Write config.yaml and config.py**

```yaml
# config.yaml
data:
  hf_repo: Adorg/ToolBench
  subset: G1
  raw_dir: data/raw
  processed_dir: data/processed
  eval_fraction: 0.15
  split_seed: 13
  require_win: true
  max_files: null          # null = all; set small for smoke runs
verbalize:
  full_history_steps: 3
  observation_chars: 200
  args_chars: 120
  description_chars: 200
model:
  backbone: Qwen/Qwen2.5-1.5B-Instruct
  device: mps              # mps | cuda | cpu
  dtype: float16
  pooling: mean            # mean | last
  max_prompt_tokens: 1024
  head_hidden: 512
  score_temperature: 0.05
tier1:
  cache_dir: cache
  cache_batch_size: 8
  epochs: 20
  batch_size: 256
  lr: 0.001
  weight_decay: 0.01
  checkpoint: checkpoints/tier1_head.pt
tier2:
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lr: 0.0002
  epochs: 1
  batch_size: 2
  grad_accum: 8
  max_train_examples: 400
  checkpoint_dir: checkpoints/tier2
baseline:
  max_new_tokens: 32
  num_beams: 5
eval:
  max_eval_examples: 500
  results_dir: results
  latency_warmup: 3
```

```python
# config.py
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class DataConfig:
    hf_repo: str; subset: str; raw_dir: str; processed_dir: str
    eval_fraction: float; split_seed: int; require_win: bool; max_files: int | None

@dataclass(frozen=True)
class VerbalizeConfig:
    full_history_steps: int; observation_chars: int; args_chars: int; description_chars: int

@dataclass(frozen=True)
class ModelConfig:
    backbone: str; device: str; dtype: str; pooling: str
    max_prompt_tokens: int; head_hidden: int; score_temperature: float

@dataclass(frozen=True)
class Tier1Config:
    cache_dir: str; cache_batch_size: int; epochs: int; batch_size: int
    lr: float; weight_decay: float; checkpoint: str

@dataclass(frozen=True)
class Tier2Config:
    lora_rank: int; lora_alpha: int; lora_dropout: float; lr: float; epochs: int
    batch_size: int; grad_accum: int; max_train_examples: int; checkpoint_dir: str

@dataclass(frozen=True)
class BaselineConfig:
    max_new_tokens: int; num_beams: int

@dataclass(frozen=True)
class EvalConfig:
    max_eval_examples: int; results_dir: str; latency_warmup: int

@dataclass(frozen=True)
class Config:
    data: DataConfig; verbalize: VerbalizeConfig; model: ModelConfig
    tier1: Tier1Config; tier2: Tier2Config; baseline: BaselineConfig; eval: EvalConfig

_SECTIONS = {"data": DataConfig, "verbalize": VerbalizeConfig, "model": ModelConfig,
             "tier1": Tier1Config, "tier2": Tier2Config, "baseline": BaselineConfig, "eval": EvalConfig}

def _build(cls: type, raw: dict[str, Any], section: str):
    expected = {f.name for f in fields(cls)}
    missing, extra = expected - raw.keys(), raw.keys() - expected
    if missing or extra:
        raise ValueError(f"config section '{section}': missing={sorted(missing)} extra={sorted(extra)}")
    return cls(**raw)

def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    missing = _SECTIONS.keys() - raw.keys()
    if missing:
        raise ValueError(f"config missing sections: {sorted(missing)}")
    return Config(**{name: _build(cls, raw[name], name) for name, cls in _SECTIONS.items()})
```

`.gitignore`: `.venv/`, `data/raw/`, `data/processed/`, `cache/`, `checkpoints/`, `__pycache__/`, `.pytest_cache/`, `results/*.json` (keep `results/*.md`).

- [x] **Step 4: Run tests** — `.venv/bin/pytest tests/test_config.py -v` → PASS.
- [x] **Step 5: Commit** — `git add -A && git commit -m "chore: scaffold ActionRank project with config loader"`

---

### Task 2: Data parsing, catalog, step slicing, split (`data.py`)

**Files:**
- Create: `data.py`, `tests/fixtures/answer_win.json`, `tests/fixtures/answer_branching.json`, `tests/fixtures/answer_giveup.json`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ToolSpec(name: str, description: str)` — description already stripped of ToolBench boilerplate.
  - `@dataclass(frozen=True) Step(action: str, arguments: str, observation: str)`
  - `@dataclass(frozen=True) Trajectory(query_id: str, query: str, tools: tuple[ToolSpec, ...], steps: tuple[Step, ...])`
  - `@dataclass(frozen=True) Example(query_id: str, query: str, history: tuple[Step, ...], candidates: tuple[str, ...], label: str)`
  - `@dataclass(frozen=True) Catalog(tools: tuple[ToolSpec, ...])` with `.index(name) -> int`, `.names -> tuple[str,...]`, `__len__`, `describe(name) -> str`, `save(path)`, `Catalog.load(path)`.
  - `parse_answer_file(path: Path, require_win: bool) -> Trajectory | None`
  - `clean_description(raw: str) -> str`
  - `extract_main_path(tree_root: dict) -> tuple[Step, ...]` — the root-to-leaf path ending in Finish/give_answer, else the longest path.
  - `load_trajectories(raw_dir: Path, subset: str, require_win: bool, max_files: int | None) -> tuple[Trajectory, ...]`
  - `build_catalog(trajectories) -> Catalog`
  - `slice_steps(trajectory: Trajectory) -> tuple[Example, ...]`
  - `split_trajectories(trajectories, eval_fraction: float, seed: int) -> tuple[tuple[Trajectory,...], tuple[Trajectory,...]]` (train, eval)
  - `download_subset(hf_repo: str, subset: str, raw_dir: Path) -> Path`
  - `prepare_dataset(cfg: Config) -> Dataset` where `@dataclass(frozen=True) Dataset(catalog: Catalog, train: tuple[Example,...], eval: tuple[Example,...])`, also writes `data/processed/{catalog.json, train.jsonl, eval.jsonl}` and `load_dataset(cfg) -> Dataset` reads them back.
  - CLI: `python data.py` runs download + prepare and prints counts.

- [x] **Step 1: Write fixtures** — `answer_win.json` is the 1006 sample (3 calls + Finish give_answer) with observations truncated to ~300 chars; `answer_branching.json` has a first branch that ends in `give_up_and_restart` and a second branch that ends in `give_answer`; `answer_giveup.json` is the 10028 sample (win false).

- [x] **Step 2: Write the failing tests**

```python
# tests/test_data.py
from pathlib import Path
import json
from data import (parse_answer_file, clean_description, slice_steps, build_catalog,
                  split_trajectories, Trajectory, ToolSpec, Step, Catalog)
FIX = Path(__file__).parent / "fixtures"

def test_parse_win_file_yields_ordered_steps():
    t = parse_answer_file(FIX / "answer_win.json", require_win=True)
    assert t.query_id == "1006"
    assert [s.action for s in t.steps] == ["get_info_for_covid_19_india", "get_details_for_covid_19_india",
                                           "get_latest_updates_for_covid_19_india", "Finish"]
    assert t.steps[0].observation.startswith('{"error": ""')
    assert [x.name for x in t.tools][-1] == "Finish"

def test_parse_branching_file_takes_give_answer_path():
    t = parse_answer_file(FIX / "answer_branching.json", require_win=True)
    assert t.steps[-1].action == "Finish" and "give_answer" in t.steps[-1].arguments
    assert "give_up_and_restart" not in " ".join(s.arguments for s in t.steps)

def test_parse_giveup_file_skipped_when_require_win():
    assert parse_answer_file(FIX / "answer_giveup.json", require_win=True) is None
    assert parse_answer_file(FIX / "answer_giveup.json", require_win=False) is not None

def test_clean_description_strips_boilerplate():
    raw = 'This is the subfunction for tool "covid_19_india", you can use this tool.The description of this function is: "Get info on Covid 19 India."'
    assert clean_description(raw) == "Get info on Covid 19 India."

def test_slice_steps_produces_one_example_per_call_with_growing_history():
    t = parse_answer_file(FIX / "answer_win.json", require_win=True)
    ex = slice_steps(t)
    assert len(ex) == 4
    assert ex[0].history == () and ex[0].label == "get_info_for_covid_19_india"
    assert len(ex[3].history) == 3 and ex[3].label == "Finish"
    assert set(e.label for e in ex) <= set(ex[0].candidates)

def test_build_catalog_is_union_and_indexable():
    a = parse_answer_file(FIX / "answer_win.json", require_win=True)
    b = parse_answer_file(FIX / "answer_giveup.json", require_win=False)
    cat = build_catalog((a, b))
    assert "Finish" in cat.names and len(cat) == len(set(x.name for x in a.tools + b.tools))
    assert cat.names[cat.index("Finish")] == "Finish"

def test_split_is_by_trajectory_and_deterministic():
    trajs = tuple(Trajectory(query_id=str(i), query="q", tools=(ToolSpec("Finish", "d"),),
                             steps=(Step("Finish", "{}", ""),)) for i in range(100))
    tr1, ev1 = split_trajectories(trajs, 0.15, seed=13)
    tr2, ev2 = split_trajectories(trajs, 0.15, seed=13)
    assert len(ev1) == 15 and ev1 == ev2
    assert not {t.query_id for t in tr1} & {t.query_id for t in ev1}

def test_catalog_roundtrip(tmp_path):
    cat = Catalog((ToolSpec("a", "A"), ToolSpec("Finish", "F")))
    cat.save(tmp_path / "c.json")
    assert Catalog.load(tmp_path / "c.json") == cat
```

- [x] **Step 3: Run tests to verify they fail** — `.venv/bin/pytest tests/test_data.py -v` → ImportError.

- [x] **Step 4: Implement data.py**

Key implementation notes (full code written in the task):
- `extract_main_path`: DFS over `children`; collect every root-to-leaf path as a list of nodes; convert each path to steps by pairing `Action` nodes with their following `Action Input` node (skip `Thought`); prefer paths whose final step is `Finish` with `"give_answer"` in arguments; tie-break by fewest steps (the cleanest successful path); if none and `require_win` is False, take the longest path.
- `parse_answer_file`: returns None if `require_win` and not `data["win"]`; query id from filename prefix before `_`.
- `clean_description`: regex `The description of this function is: "(.*)"` with DOTALL, strip; if not matched return the raw string stripped.
- `slice_steps`: for k in range(len(steps)): `Example(history=steps[:k], candidates=tuple(t.name for t in tools), label=steps[k].action)`; drop examples whose label is not in candidates (log count).
- `split_trajectories`: `random.Random(seed).shuffle(list(sorted by query_id))`, first `round(n*eval_fraction)` are eval.
- `download_subset`: `huggingface_hub.snapshot_download(repo_type="dataset", allow_patterns=[f"answer/{subset}_answer/*"], local_dir=raw_dir)`.
- Serialization: `dataclasses.asdict` → JSON; jsonl for examples.

- [x] **Step 5: Run tests** → PASS. Then run `.venv/bin/python data.py` on the real download and record counts (files, win rate, trajectories, steps, catalog size, train/eval sizes) in the README's Data section.
- [x] **Step 6: Commit** — `git commit -m "feat: parse ToolBench G1 answers into trajectories, catalog, step examples"`

---

### Task 3: Verbalizer (`verbalize.py`)

**Files:**
- Create: `verbalize.py`, `tests/test_verbalize.py`

**Interfaces:**
- Consumes: `Example`, `Catalog`, `VerbalizeConfig`.
- Produces:
  - `render_history(history: tuple[Step,...], cfg: VerbalizeConfig) -> str` — last `full_history_steps` steps rendered as `Step i: called NAME(args[:args_chars]) -> observation[:observation_chars]`; older steps collapsed into one line `Earlier calls: a, b, c`.
  - `render_catalog(candidates: tuple[str,...], catalog: Catalog, cfg) -> str` — one `name: description[:description_chars]` line per candidate, in candidate order.
  - `build_prompt(example: Example, catalog: Catalog, cfg) -> str` — ActionRank prompt (ends with `Next tool:`).
  - `build_baseline_messages(example, catalog, cfg) -> list[dict]` — chat messages for the generation baseline (system: "reply with exactly one tool name from the list"; user: same body).
  - `truncate(text: str, limit: int) -> str` — appends `…` when cut.

- [x] **Step 1: Failing tests**

```python
# tests/test_verbalize.py
from data import Example, Step, Catalog, ToolSpec
from config import VerbalizeConfig
from verbalize import render_history, render_catalog, build_prompt, build_baseline_messages, truncate
CFG = VerbalizeConfig(full_history_steps=2, observation_chars=20, args_chars=10, description_chars=15)
CAT = Catalog((ToolSpec("a_tool", "Alpha does alpha things indeed"), ToolSpec("b_tool", "Beta"), ToolSpec("Finish", "Stop")))

def steps(n):
    return tuple(Step(f"tool{i}", '{"x": ' + str(i) + '}', "obs" * 20) for i in range(n))

def test_truncate_marks_cut():
    assert truncate("abcdef", 3) == "abc…" and truncate("ab", 3) == "ab"

def test_history_keeps_last_k_full_and_summarizes_older():
    text = render_history(steps(4), CFG)
    assert "Earlier calls: tool0, tool1" in text
    assert "tool2(" in text and "tool3(" in text
    assert text.count("->") == 2
    assert "obsobsobsobsobsobsob…" in text

def test_history_empty():
    assert render_history((), CFG) == "(no tool calls yet)"

def test_catalog_lines_follow_candidate_order_and_truncate():
    text = render_catalog(("b_tool", "a_tool"), CAT, CFG)
    assert text.splitlines() == ["b_tool: Beta", "a_tool: Alpha does alph…"]

def test_prompt_contains_all_sections():
    ex = Example("1", "Find alpha", steps(1), ("a_tool", "Finish"), "a_tool")
    p = build_prompt(ex, CAT, CFG)
    assert p.startswith("Task: Find alpha") and "Available tools:" in p and p.rstrip().endswith("Next tool:")

def test_baseline_messages_shape():
    ex = Example("1", "Find alpha", (), ("a_tool", "Finish"), "a_tool")
    msgs = build_baseline_messages(ex, CAT, CFG)
    assert [m["role"] for m in msgs] == ["system", "user"] and "a_tool" in msgs[1]["content"]
```

- [x] **Step 2: Run → fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run → pass.**
- [x] **Step 5: Commit** — `git commit -m "feat: verbalizer for context + catalog prompts"`

---

### Task 4: Backbone, pooling, scoring head (`model.py`)

**Files:**
- Create: `model.py`, `tests/test_model.py`

**Interfaces:**
- Produces:
  - `resolve_device(requested: str) -> torch.device` — falls back to cpu with a logged warning if mps/cuda unavailable.
  - `load_backbone(cfg: ModelConfig) -> tuple[PreTrainedTokenizer, PreTrainedModel]` — `AutoModelForCausalLM` in `cfg.dtype`, eval mode, tokenizer with left padding for generation, right padding for encoding handled per call.
  - `pool_hidden(hidden: Tensor[B,T,D], attention_mask: Tensor[B,T], pooling: str) -> Tensor[B,D]` (fp32 output).
  - `encode_prompts(tokenizer, model, prompts: list[str], cfg: ModelConfig) -> Tensor[B,D]` — prefill only via `model.model(...)`, last hidden layer, pooled.
  - `class ScoringHead(nn.Module)`: `__init__(num_tools, hidden_dim, head_hidden, temperature)`; `self.tool_embedding = nn.Embedding(num_tools, hidden_dim)`; `self.proj = Sequential(Linear(hidden_dim, head_hidden), GELU(), Linear(head_hidden, hidden_dim))`; `forward(h: Tensor[B,D], candidate_mask: Tensor[B,N] bool | None) -> logits Tensor[B,N]` where `logits = (normalize(proj(h)) @ normalize(E).T) / temperature`, masked positions set to `-inf`.
  - `build_candidate_mask(examples, catalog) -> BoolTensor[B,N]`, `labels_tensor(examples, catalog) -> LongTensor[B]`.
  - `class ActionRankModel(nn.Module)`: holds tokenizer, backbone, head, cfg; `forward(prompts, candidate_mask) -> logits`; `rank(prompts, candidate_mask, k) -> LongTensor[B,k]`.
  - `save_head(head, path)`, `load_head(path, num_tools, hidden_dim, cfg) -> ScoringHead`.

- [x] **Step 1: Failing tests (head + pooling only, no backbone download)**

```python
# tests/test_model.py
import torch
from model import pool_hidden, ScoringHead, build_candidate_mask, labels_tensor, resolve_device
from data import Example, Catalog, ToolSpec

def test_mean_pool_respects_mask():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(pool_hidden(h, m, "mean"), torch.tensor([[2.0, 2.0]]))

def test_last_pool_picks_last_real_token():
    h = torch.tensor([[[1.0], [3.0], [100.0]]]); m = torch.tensor([[1, 1, 0]])
    assert pool_hidden(h, m, "last").item() == 3.0

def test_head_masks_non_candidates_and_trains():
    head = ScoringHead(num_tools=5, hidden_dim=8, head_hidden=16, temperature=0.1)
    h = torch.randn(2, 8); mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
    logits = head(h, mask)
    assert logits.shape == (2, 5) and torch.isinf(logits[0, 2]) and torch.isfinite(logits[0, 0])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4]))
    loss.backward()
    assert head.tool_embedding.weight.grad is not None

def test_candidate_mask_and_labels():
    cat = Catalog((ToolSpec("a", ""), ToolSpec("b", ""), ToolSpec("Finish", "")))
    ex = [Example("1", "q", (), ("b", "Finish"), "Finish")]
    assert build_candidate_mask(ex, cat).tolist() == [[False, True, True]]
    assert labels_tensor(ex, cat).tolist() == [2]

def test_resolve_device_cpu_always_ok():
    assert resolve_device("cpu").type == "cpu"
```

- [x] **Step 2: Run → fail.** - [ ] **Step 3: Implement.** Add a `@pytest.mark.slow` test `test_encode_prompts_real_backbone` that loads the real backbone and checks `encode_prompts(...).shape == (2, 1536)`; it is skipped unless env `ACTIONRANK_SLOW=1`.
- [x] **Step 4: Run → pass**, then run the slow test once locally on MPS to confirm the backbone loads in fp16 and produces finite vectors.
- [x] **Step 5: Commit** — `git commit -m "feat: backbone encoder, pooling and catalog scoring head"`

---

### Task 5: Tier 1 — cache pooled vectors, train head (`train_tier1.py`)

**Files:**
- Create: `train_tier1.py`, `tests/test_train_tier1.py`

**Interfaces:**
- Consumes: `load_dataset`, `build_prompt`, `load_backbone`, `encode_prompts`, `ScoringHead`, `build_candidate_mask`, `labels_tensor`.
- Produces:
  - `@dataclass(frozen=True) CachedSplit(h: Tensor[N,D], labels: LongTensor[N], candidate_mask: BoolTensor[N,num_tools], query_ids: tuple[str,...])`
  - `cache_split(examples, catalog, tokenizer, backbone, cfg: Config, path: Path) -> CachedSplit` — batched, no grad, saves via `torch.save`, skips work if the file exists and `--refresh` not passed.
  - `load_cached(path) -> CachedSplit`
  - `train_head(train: CachedSplit, eval: CachedSplit, cfg, num_tools) -> tuple[ScoringHead, dict]` — AdamW, CE over masked logits, per-epoch eval top-1; returns best head (by eval top-1) and a history dict.
  - `topk_accuracy(logits, labels, k) -> float`
  - CLI `python train_tier1.py [--refresh] [--limit N]` → writes `checkpoints/tier1_head.pt` and `results/tier1_history.json`.

- [x] **Step 1: Failing tests** on a synthetic cache: 200 examples, 6 tools, `h` drawn from 6 gaussian clusters so the head must reach > 0.9 eval top-1 within 20 epochs; `topk_accuracy` unit test; `cache_split` test with a fake encoder (monkeypatch `encode_prompts`) verifying shapes and the skip-if-exists path.
- [x] **Step 2: Run → fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run → pass.**
- [x] **Step 5: Smoke run** `python train_tier1.py --limit 40` on real data, then full run. Record timing.
- [x] **Step 6: Commit** — `git commit -m "feat: tier 1 frozen-backbone caching and head training"`

---

### Task 6: Generation baseline (`baseline.py`)

**Files:**
- Create: `baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Produces:
  - `normalize_tool_name(text: str) -> str` — first line, strip quotes/backticks/trailing punctuation/`Action:` prefix, take the first whitespace token.
  - `@dataclass(frozen=True) BaselinePrediction(top1: str, topk: tuple[str,...], latency_s: float, in_candidates: bool, in_catalog: bool)`
  - `predict_one(tokenizer, model, messages, candidates, catalog, cfg) -> BaselinePrediction` — greedy generation for top-1 and latency (timed with `time.perf_counter`, `torch.mps.synchronize()` / `cuda.synchronize()` when applicable), then beam search (`num_beams=cfg.baseline.num_beams`, `num_return_sequences=num_beams`) for the top-k list (deduplicated, top-1 first).
  - `run_baseline(examples, catalog, cfg) -> tuple[BaselinePrediction,...]`
- [x] **Step 1: Failing tests** for `normalize_tool_name` (cases: `"get_info_for_x"`, `"`get_info_for_x`"`, `"Action: get_info_for_x\nAction Input: {}"`, `"\"Finish\"."`) and for a `dedupe_keep_order` helper.
- [x] **Step 2..4: fail → implement → pass.**
- [x] **Step 5: Commit** — `git commit -m "feat: function-calling generation baseline"`

---

### Task 7: Evaluation and results table (`eval.py`)

**Files:**
- Create: `eval.py`, `tests/test_eval.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Metrics(name: str, n: int, top1: float, top5: float, hallucination_rate: float, latency_mean_ms: float, latency_p50_ms: float, top1_no_finish: float, n_no_finish: int)`
  - `compute_metrics(name, labels: list[str], top1: list[str], topk: list[tuple[str,...]], valid: list[bool], latencies_s: list[float], finish_name="Finish") -> Metrics`
  - `evaluate_actionrank(model: ActionRankModel, examples, catalog, cfg, name) -> Metrics` — times each decision individually (prefill + head) for latency parity with the baseline.
  - `evaluate_baseline(examples, catalog, cfg) -> Metrics`
  - `reference_rows(train_examples, eval_examples) -> list[Metrics]` — `random-candidate` and `most-frequent-candidate` rows (no latency).
  - `render_table(rows: list[Metrics]) -> str` (markdown) and `write_results(rows, results_dir)` → `results/results.md` + `results/results.json`.
  - CLI `python eval.py --systems tier1,baseline[,tier2] [--limit N]`.
- [x] **Step 1: Failing tests** for `compute_metrics` on hand-built lists (top-1 2/4, top-5 3/4, hallucination 1/4, no-finish subset), `render_table` header/row count.
- [x] **Step 2..4: fail → implement → pass.**
- [x] **Step 5: Commit** — `git commit -m "feat: evaluation metrics and results table"`

---

### Task 8: Tier 2 — LoRA fine-tune (`train_tier2.py`)

**Files:**
- Create: `train_tier2.py`, `tests/test_train_tier2.py`

**Interfaces:**
- Produces:
  - `wrap_lora(backbone, cfg: Tier2Config) -> PeftModel` (`LoraConfig(r, lora_alpha, lora_dropout, target_modules=["q_proj","v_proj"], bias="none")`).
  - `train_tier2(dataset, cfg) -> Path` — joint AdamW over LoRA params + head params, CE over masked logits, grad accumulation, head initialised from the Tier 1 checkpoint if present; saves adapter (`peft save_pretrained`) + head to `checkpoints/tier2/`.
  - `load_tier2(cfg, catalog) -> ActionRankModel`.
- [x] **Step 1: Failing test** — `wrap_lora` on a tiny random `Qwen2ForCausalLM` (`Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=100)`) adds trainable params only in q/v projections and the trainable fraction is < 5%.
- [x] **Step 2..4: fail → implement → pass.**
- [x] **Step 5: Small local run** (`max_train_examples: 400`, 1 epoch) and evaluate with `eval.py --systems tier2`.
- [x] **Step 6: Commit** — `git commit -m "feat: tier 2 LoRA fine-tuning"`

---

### Task 9: Experiments, README, results

- [x] Run `python data.py` (full G1), `python train_tier1.py`, `python eval.py --systems tier1,baseline`, `python train_tier2.py`, `python eval.py --systems tier1,tier2,baseline`.
- [x] Write README: motivation, pipeline diagram (text), data stats, how to run, results table, discussion vs. success criteria, limitations, Colab switch (`device: cuda`).
- [x] Commit — `git commit -m "docs: results and README"`.

---

## Codex review gates

- Gate A after Tasks 1-4 (data + prompt + model).
- Gate B after Tasks 5-7 (training tier 1 + baseline + eval).
- Gate C after Tasks 8-9 (tier 2 + results).

Each gate: commit, `node /Users/yash/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs review --wait --base <sha-before-phase>`, fix CRITICAL/HIGH/MEDIUM findings with tests, re-run the full test suite, commit, `git symbolic-ref -q --short HEAD || git checkout main`.

## Self-review

- Spec coverage: verbalizer (T3), backbone/pooling/head (T4), tier 1 (T5), tier 2 (T8), loss (T4/T5), baseline (T6), metrics incl. latency and hallucination (T7), data/catalog/slicing/split (T2), config.yaml (T1), success criteria discussion (T9). Colab switch is a config value (T1/T4).
- Type consistency: `Example.candidates` is `tuple[str,...]` everywhere; `Catalog.index(name)`; `ScoringHead(num_tools, hidden_dim, head_hidden, temperature)`; `encode_prompts(tokenizer, model, prompts, cfg)` used by T5 and T7.
