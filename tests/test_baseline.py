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
