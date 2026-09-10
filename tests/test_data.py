from pathlib import Path

from data import (
    Catalog,
    Step,
    ToolSpec,
    Trajectory,
    build_catalog,
    clean_description,
    parse_answer_file,
    slice_steps,
    split_trajectories,
)

FIX = Path(__file__).parent / "fixtures"


def test_parse_win_file_yields_ordered_steps():
    t = parse_answer_file(FIX / "answer_win.json", require_win=True)
    assert t is not None
    assert t.query_id == "answer_win"
    assert [s.action for s in t.steps] == [
        "get_info_for_covid_19_india",
        "get_details_for_covid_19_india",
        "get_latest_updates_for_covid_19_india",
        "Finish",
    ]
    assert t.steps[0].observation.startswith('{"error": ""')
    assert t.steps[0].arguments == "{}"
    assert [x.name for x in t.tools][-1] == "Finish"
    assert len(t.query) > 20


def test_parse_branching_file_takes_give_answer_path():
    t = parse_answer_file(FIX / "answer_branching.json", require_win=True)
    assert t is not None
    assert [s.action for s in t.steps] == ["media_sources_statistics_for_public_url_share", "Finish"]
    assert "give_answer" in t.steps[-1].arguments
    assert "give_up_and_restart" not in " ".join(s.arguments for s in t.steps)


def test_parse_giveup_file_skipped_when_require_win():
    assert parse_answer_file(FIX / "answer_giveup.json", require_win=True) is None
    t = parse_answer_file(FIX / "answer_giveup.json", require_win=False)
    assert t is not None and len(t.steps) == 2


def test_clean_description_strips_boilerplate():
    raw = (
        'This is the subfunction for tool "covid_19_india", you can use this tool.'
        'The description of this function is: "Get info on Covid 19 India."'
    )
    assert clean_description(raw) == "Get info on Covid 19 India."
    assert clean_description("  plain text  ") == "plain text"


def test_slice_steps_produces_one_example_per_call_with_growing_history():
    t = parse_answer_file(FIX / "answer_win.json", require_win=True)
    ex = slice_steps(t)
    assert len(ex) == 4
    assert ex[0].history == () and ex[0].label == "get_info_for_covid_19_india"
    assert len(ex[3].history) == 3 and ex[3].label == "Finish"
    assert ex[0].candidates == tuple(x.name for x in t.tools)
    assert set(e.label for e in ex) <= set(ex[0].candidates)


def test_slice_steps_drops_labels_outside_candidates():
    t = Trajectory("q", "query", (ToolSpec("a", "A"), ToolSpec("Finish", "F")),
                   (Step("a", "{}", "o"), Step("ghost", "{}", "o"), Step("Finish", "{}", "")))
    assert [e.label for e in slice_steps(t)] == ["a", "Finish"]


def test_build_catalog_is_union_and_indexable():
    a = parse_answer_file(FIX / "answer_win.json", require_win=True)
    b = parse_answer_file(FIX / "answer_giveup.json", require_win=False)
    cat = build_catalog((a, b))
    assert "Finish" in cat.names
    assert len(cat) == len({x.name for x in a.tools + b.tools})
    assert cat.names[cat.index("Finish")] == "Finish"
    assert cat.describe("Finish").startswith("If you think")


def test_split_is_by_trajectory_and_deterministic():
    trajs = tuple(
        Trajectory(query_id=str(i), query="q", tools=(ToolSpec("Finish", "d"),), steps=(Step("Finish", "{}", ""),))
        for i in range(100)
    )
    tr1, ev1 = split_trajectories(trajs, 0.15, seed=13)
    tr2, ev2 = split_trajectories(trajs, 0.15, seed=13)
    assert len(ev1) == 15 and len(tr1) == 85
    assert ev1 == ev2
    assert not {t.query_id for t in tr1} & {t.query_id for t in ev1}


def test_catalog_roundtrip(tmp_path):
    cat = Catalog((ToolSpec("a", "A"), ToolSpec("Finish", "F")))
    cat.save(tmp_path / "c.json")
    assert Catalog.load(tmp_path / "c.json") == cat


def test_clean_description_boilerplate_only_becomes_tool_reference():
    raw = 'This is the subfunction for tool "retrieve_dns_entries", you can use this tool.'
    assert clean_description(raw) == 'Subfunction of tool "retrieve_dns_entries".'


def test_clean_description_handles_multiline_and_trailing_junk():
    raw = ('This is the subfunction for tool "x", you can use this tool.The description of this function is: '
           '"Line one.\nLine two."')
    assert clean_description(raw) == "Line one.\nLine two."


def test_require_win_rejects_win_file_without_successful_path(tmp_path):
    import json

    raw = json.loads((FIX / "answer_giveup.json").read_text())
    raw["win"] = True
    path = tmp_path / "77_ChatGPT_DFS_woFilter_w2.json"
    path.write_text(json.dumps(raw))
    assert parse_answer_file(path, require_win=True) is None
    assert parse_answer_file(path, require_win=False) is not None
