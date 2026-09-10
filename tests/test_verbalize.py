from config import VerbalizeConfig
from data import Catalog, Example, Step, ToolSpec
from verbalize import build_baseline_messages, build_prompt, render_catalog, render_history, truncate

CFG = VerbalizeConfig(full_history_steps=2, observation_chars=20, args_chars=10, description_chars=15)
CAT = Catalog((ToolSpec("a_tool", "Alpha does alpha things indeed"), ToolSpec("b_tool", "Beta"), ToolSpec("Finish", "Stop")))


def steps(n):
    return tuple(Step(f"tool{i}", '{"x": ' + str(i) + '}', "obs" * 20) for i in range(n))


def test_truncate_marks_cut():
    assert truncate("abcdef", 3) == "abc…"
    assert truncate("ab", 3) == "ab"
    assert truncate("a\nb  c", 10) == "a b c"


def test_history_keeps_last_k_full_and_summarizes_older():
    text = render_history(steps(4), CFG)
    assert "Earlier calls: tool0, tool1" in text
    assert "tool2(" in text and "tool3(" in text
    assert text.count("->") == 2
    assert "obsobsobsobsobsobsob…" in text
    assert "tool0(" not in text


def test_history_no_summary_line_when_short():
    text = render_history(steps(2), CFG)
    assert "Earlier calls" not in text and text.count("->") == 2


def test_history_empty():
    assert render_history((), CFG) == "(no tool calls yet)"


def test_catalog_lines_follow_candidate_order_and_truncate():
    text = render_catalog(("b_tool", "a_tool"), CAT, CFG)
    assert text.splitlines() == ["- b_tool: Beta", "- a_tool: Alpha does alph…"]


def test_prompt_contains_all_sections():
    ex = Example("1", "Find alpha", steps(1), ("a_tool", "Finish"), "a_tool")
    p = build_prompt(ex, CAT, CFG)
    assert p.startswith("Task: Find alpha")
    assert "History:" in p and "Available tools:" in p
    assert p.rstrip().endswith("Next tool:")
    assert p.index("History:") < p.index("Available tools:")


def test_baseline_messages_shape():
    ex = Example("1", "Find alpha", (), ("a_tool", "Finish"), "a_tool")
    msgs = build_baseline_messages(ex, CAT, CFG)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "a_tool" in msgs[1]["content"] and "Task: Find alpha" in msgs[1]["content"]
    assert "exactly one tool name" in msgs[0]["content"]
