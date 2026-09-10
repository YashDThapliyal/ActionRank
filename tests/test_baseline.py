from baseline import dedupe_keep_order, normalize_tool_name


def test_normalize_plain_name():
    assert normalize_tool_name("get_info_for_x") == "get_info_for_x"


def test_normalize_strips_backticks_quotes_and_punctuation():
    assert normalize_tool_name("`get_info_for_x`") == "get_info_for_x"
    assert normalize_tool_name('"Finish".') == "Finish"
    assert normalize_tool_name("  Finish\n") == "Finish"


def test_normalize_takes_first_line_and_drops_action_prefix():
    assert normalize_tool_name("Action: get_info_for_x\nAction Input: {}") == "get_info_for_x"
    assert normalize_tool_name("Tool: get_info_for_x") == "get_info_for_x"


def test_normalize_drops_call_parentheses_and_trailing_text():
    assert normalize_tool_name("get_info_for_x({})") == "get_info_for_x"
    assert normalize_tool_name("get_info_for_x because it fits") == "get_info_for_x"


def test_normalize_empty():
    assert normalize_tool_name("") == ""


def test_dedupe_keep_order():
    assert dedupe_keep_order(["b", "a", "b", "c", "a"]) == ("b", "a", "c")


def test_normalize_splits_comma_joined_names():
    assert normalize_tool_name("daily_live_for_x,holidays_for_x") == "daily_live_for_x"
    assert normalize_tool_name("a_tool; b_tool") == "a_tool"


def test_predict_latency_includes_prompt_building_and_tokenization(monkeypatch):
    import time

    import torch

    import baseline
    from config import load_config
    from data import Catalog, Example, ToolSpec

    def slow_inputs(tokenizer, messages, device, max_tokens):
        time.sleep(0.05)
        return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    monkeypatch.setattr(baseline, "_chat_inputs", slow_inputs)
    monkeypatch.setattr(baseline, "generate_top1", lambda tok, m, inputs, cfg: "a_tool")
    monkeypatch.setattr(baseline, "generate_topk", lambda tok, m, inputs, cfg: ("a_tool", "b_tool"))
    cat = Catalog((ToolSpec("a_tool", "A"), ToolSpec("b_tool", "B")))
    pred = baseline.predict_one(None, torch.nn.Linear(1, 1), Example("1", "q", (), ("a_tool", "b_tool"), "a_tool"), cat, load_config())
    assert pred.latency_s >= 0.05
    assert pred.top1 == "a_tool" and pred.topk == ("a_tool", "b_tool")
