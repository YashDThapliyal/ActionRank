"""Verbalizer: (query, compressed history, candidate catalog) -> prompt text."""
from __future__ import annotations

import re

from config import VerbalizeConfig
from data import Catalog, Example, Step

ELLIPSIS = "…"
NO_HISTORY = "(no tool calls yet)"
SUFFIX = "Next tool:"
BASELINE_SYSTEM = (
    "You are an agent choosing the next tool to call. Reply with exactly one tool name "
    "from the 'Available tools' list and nothing else."
)
_WS = re.compile(r"\s+")


def truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut to `limit` chars, marking the cut with an ellipsis."""
    flat = _WS.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[:limit] + ELLIPSIS


def _render_step(index: int, step: Step, cfg: VerbalizeConfig) -> str:
    args = truncate(step.arguments, cfg.args_chars)
    obs = truncate(step.observation, cfg.observation_chars) or "(empty)"
    return f"Step {index}: called {step.action}({args}) -> {obs}"


def render_history(history: tuple[Step, ...], cfg: VerbalizeConfig) -> str:
    """Last `full_history_steps` steps in full; older steps collapsed to their tool names."""
    if not history:
        return NO_HISTORY
    keep = max(cfg.full_history_steps, 0)
    older, recent = history[:-keep] if keep else history, history[-keep:] if keep else ()
    lines: list[str] = []
    if older:
        lines.append("Earlier calls: " + ", ".join(s.action for s in older))
    offset = len(older)
    lines.extend(_render_step(offset + i + 1, s, cfg) for i, s in enumerate(recent))
    return "\n".join(lines)


def render_catalog(candidates: tuple[str, ...], catalog: Catalog, cfg: VerbalizeConfig) -> str:
    """One `- name: description` line per candidate, in candidate order."""
    return "\n".join(
        f"- {name}: {truncate(catalog.describe(name), cfg.description_chars)}" for name in candidates
    )


def _body(example: Example, catalog: Catalog, cfg: VerbalizeConfig) -> str:
    return (
        f"Task: {truncate(example.query, 10_000)}\n\n"
        f"History:\n{render_history(example.history, cfg)}\n\n"
        f"Available tools:\n{render_catalog(example.candidates, catalog, cfg)}\n"
    )


def build_prompt(example: Example, catalog: Catalog, cfg: VerbalizeConfig) -> str:
    """ActionRank prompt: prefill-only, ends with 'Next tool:'."""
    return _body(example, catalog, cfg) + f"\n{SUFFIX}"


def build_baseline_messages(example: Example, catalog: Catalog, cfg: VerbalizeConfig) -> list[dict[str, str]]:
    """Chat messages for the generation baseline (same content, chat-templated)."""
    return [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": _body(example, catalog, cfg) + "\nWhich tool should be called next?"},
    ]
