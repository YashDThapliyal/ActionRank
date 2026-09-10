"""ToolBench G1 parsing: trajectories, catalog, per-step examples, train/eval split."""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from config import Config, load_config

log = logging.getLogger(__name__)

FINISH = "Finish"
GIVE_ANSWER = "give_answer"
_DESC_RE = re.compile(r'The description of this function is:\s*"(.*)"\s*$', re.DOTALL)
_TOOL_RE = re.compile(r'This is the subfunction for tool "([^"]+)", you can use this tool\.?')


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


@dataclass(frozen=True)
class Step:
    action: str
    arguments: str
    observation: str


@dataclass(frozen=True)
class Trajectory:
    query_id: str
    query: str
    tools: tuple[ToolSpec, ...]
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Example:
    query_id: str
    query: str
    history: tuple[Step, ...]
    candidates: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class Catalog:
    tools: tuple[ToolSpec, ...]

    def __post_init__(self) -> None:
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("catalog contains duplicate tool names")
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(names)})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def index(self, name: str) -> int:
        return self._index[name]  # type: ignore[attr-defined]

    def __contains__(self, name: object) -> bool:
        return name in self._index  # type: ignore[attr-defined]

    def describe(self, name: str) -> str:
        return self.tools[self.index(name)].description

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(t) for t in self.tools], indent=1))

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        return cls(tuple(ToolSpec(**t) for t in json.loads(path.read_text())))


@dataclass(frozen=True)
class Dataset:
    catalog: Catalog
    train: tuple[Example, ...]
    eval: tuple[Example, ...]


# --------------------------------------------------------------------------- parsing

def clean_description(raw: str) -> str:
    """Strip ToolBench's 'This is the subfunction for tool ...' boilerplate.

    Falls back to a short tool reference when the function has no description of its own.
    """
    match = _DESC_RE.search(raw)
    if match and match.group(1).strip():
        return match.group(1).strip()
    tool = _TOOL_RE.search(raw)
    if tool:
        return f'Subfunction of tool "{tool.group(1)}".'
    return raw.strip()


def _leaf_paths(node: dict) -> list[list[dict]]:
    children = node.get("children") or []
    if not children:
        return [[node]]
    return [[node, *path] for child in children for path in _leaf_paths(child)]


def _path_to_steps(path: list[dict]) -> tuple[Step, ...]:
    steps: list[Step] = []
    for node, nxt in zip(path, path[1:]):
        if node.get("node_type") != "Action" or nxt.get("node_type") != "Action Input":
            continue
        steps.append(Step(action=str(node.get("description", "")),
                          arguments=str(nxt.get("description", "")),
                          observation=str(nxt.get("observation", "") or "")))
    return tuple(steps)


def _is_success(steps: tuple[Step, ...]) -> bool:
    return bool(steps) and steps[-1].action == FINISH and GIVE_ANSWER in steps[-1].arguments


def extract_main_path(tree_root: dict) -> tuple[Step, ...]:
    """Root-to-leaf path ending in Finish/give_answer (shortest such); else the longest path."""
    candidates = [_path_to_steps(p) for p in _leaf_paths(tree_root)]
    successes = [s for s in candidates if _is_success(s)]
    if successes:
        return min(successes, key=len)
    return max(candidates, key=len) if candidates else ()


def query_id_from_path(path: Path) -> str:
    return path.stem.split("_ChatGPT")[0]


def parse_answer_file(path: Path, require_win: bool) -> Trajectory | None:
    with open(path) as fh:
        data = json.load(fh)
    if require_win and not data.get("win", False):
        return None
    gen = data["answer_generation"]
    tools = tuple(ToolSpec(name=f["name"], description=clean_description(f.get("description", "")))
                  for f in gen["function"])
    steps = extract_main_path(data["tree"]["tree"])
    if not steps:
        return None
    return Trajectory(query_id=query_id_from_path(path), query=gen["query"], tools=tools, steps=steps)


def load_trajectories(raw_dir: Path, subset: str, require_win: bool,
                      max_files: int | None) -> tuple[Trajectory, ...]:
    files = sorted((raw_dir / "answer" / f"{subset}_answer").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no answer files under {raw_dir}/answer/{subset}_answer")
    if max_files is not None:
        files = files[:max_files]
    kept: list[Trajectory] = []
    skipped = 0
    for path in files:
        traj = parse_answer_file(path, require_win)
        if traj is None:
            skipped += 1
            continue
        kept.append(traj)
    log.info("loaded %d trajectories from %d files (%d skipped)", len(kept), len(files), skipped)
    return tuple(kept)


# --------------------------------------------------------------------------- catalog / examples

def build_catalog(trajectories: Iterable[Trajectory]) -> Catalog:
    seen: dict[str, ToolSpec] = {}
    for traj in trajectories:
        for tool in traj.tools:
            seen.setdefault(tool.name, tool)
    return Catalog(tuple(seen.values()))


def slice_steps(trajectory: Trajectory) -> tuple[Example, ...]:
    candidates = tuple(t.name for t in trajectory.tools)
    allowed = set(candidates)
    examples: list[Example] = []
    for k, step in enumerate(trajectory.steps):
        if step.action not in allowed:
            log.debug("dropping step %d of %s: %r not in candidates", k, trajectory.query_id, step.action)
            continue
        examples.append(Example(query_id=trajectory.query_id, query=trajectory.query,
                                history=trajectory.steps[:k], candidates=candidates, label=step.action))
    return tuple(examples)


def split_trajectories(trajectories: tuple[Trajectory, ...], eval_fraction: float,
                       seed: int) -> tuple[tuple[Trajectory, ...], tuple[Trajectory, ...]]:
    if not 0 < eval_fraction < 1:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    ordered = sorted(trajectories, key=lambda t: t.query_id)
    random.Random(seed).shuffle(ordered)
    n_eval = round(len(ordered) * eval_fraction)
    return tuple(ordered[n_eval:]), tuple(ordered[:n_eval])


# --------------------------------------------------------------------------- I/O

def download_subset(hf_repo: str, subset: str, raw_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_repo, repo_type="dataset", local_dir=str(raw_dir),
                                  allow_patterns=[f"answer/{subset}_answer/*"]))


def _example_to_json(example: Example) -> str:
    return json.dumps(asdict(example))


def _example_from_json(line: str) -> Example:
    raw = json.loads(line)
    return Example(query_id=raw["query_id"], query=raw["query"],
                   history=tuple(Step(**s) for s in raw["history"]),
                   candidates=tuple(raw["candidates"]), label=raw["label"])


def _write_examples(path: Path, examples: tuple[Example, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(_example_to_json(ex) + "\n")


def _read_examples(path: Path) -> tuple[Example, ...]:
    with open(path) as fh:
        return tuple(_example_from_json(line) for line in fh if line.strip())


def prepare_dataset(cfg: Config) -> Dataset:
    """Parse raw answer files, build catalog + split, write data/processed, return Dataset."""
    d = cfg.data
    trajectories = load_trajectories(Path(d.raw_dir), d.subset, d.require_win, d.max_files)
    train_traj, eval_traj = split_trajectories(trajectories, d.eval_fraction, d.split_seed)
    catalog = build_catalog(trajectories)
    train = tuple(ex for t in train_traj for ex in slice_steps(t))
    evaluation = tuple(ex for t in eval_traj for ex in slice_steps(t))
    out = Path(d.processed_dir)
    catalog.save(out / "catalog.json")
    _write_examples(out / "train.jsonl", train)
    _write_examples(out / "eval.jsonl", evaluation)
    return Dataset(catalog=catalog, train=train, eval=evaluation)


def load_dataset(cfg: Config) -> Dataset:
    out = Path(cfg.data.processed_dir)
    return Dataset(catalog=Catalog.load(out / "catalog.json"),
                   train=_read_examples(out / "train.jsonl"),
                   eval=_read_examples(out / "eval.jsonl"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    download_subset(cfg.data.hf_repo, cfg.data.subset, Path(cfg.data.raw_dir))
    ds = prepare_dataset(cfg)
    finish_train = sum(ex.label == FINISH for ex in ds.train)
    print(f"catalog tools: {len(ds.catalog)}")
    print(f"train examples: {len(ds.train)} (Finish labels: {finish_train})")
    print(f"eval examples:  {len(ds.eval)}")


if __name__ == "__main__":
    main()
