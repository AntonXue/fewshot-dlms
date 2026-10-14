"""Unit tests for canvas: which records an episode uses, where they sit, and the token sequence.

The tests run on CPU with `FakeAdapter`, whose tokenizer maps every character to one token, so a
canvas can be compared with the text it should spell. List Functions-Easy families are read from
data/benchmarks/; the small hand-made family below is there to make the expected text readable.
"""
import pytest

from fewshot_dlms.canvas import BASE_DEMOS
from fewshot_dlms.canvas import active_episode
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.canvas import extend_layout
from fewshot_dlms.canvas import master_layout
from fewshot_dlms.data import load_families
from tests.fakes import FakeAdapter

LIST_REPORT = load_families("list_easy_fam", "report")
LIST_DEV = load_families("list_easy_fam", "dev")
SALT_REPORT = load_families("salt_rep_fam", "report")
FAM = LIST_REPORT[0]

# The first report family of the 24-slot layout used by the M x J grid, as the authors froze it
# (function c030; its 12-slot layout is FAM["meta"]["master_placement"]).
C030_LAYOUT_24 = [
    "D1", "D5", "D6", "D7", "D2", "T1", "T2", "D8", "D9", "T3", "D10", "D3",
    "T4", "T5", "D11", "T6", "T7", "D12", "D13", "D14", "D15", "D16", "D4", "T8",
]


def tiny_family():
    """A family that reverses lists, with two demonstrations and two targets laid out D1 T1 D2 T2."""
    return {
        "episode_id": "tiny:0",
        "source": "tiny",
        "instruction": "Reverse the list.",
        "demos": [
            {"id": "d1", "input": "[1, 2]", "output": "[2, 1]"},
            {"id": "d2", "input": "[3]", "output": "[3]"},
        ],
        "targets": [
            {"id": "t1", "input": "[4, 5]", "output": "[5, 4]"},
            {"id": "t2", "input": "[6, 7, 8]", "output": "[8, 7, 6]"},
        ],
        "meta": {"master_placement": ["D1", "T1", "D2", "T2"]},
    }


def slots(episode):
    """The record labels of an active episode, in prompt order."""
    return [entry["slot"] for entry in episode["entries"]]


# ---------------------------------------------------------------------------------------------
# build_canvas
# ---------------------------------------------------------------------------------------------


def test_sites_and_E_are_disjoint_and_masked():
    canvas = build_canvas(FakeAdapter(), FAM, M=4, J=2, lengths=[3, 5])
    flat = [position for site in canvas.sites for position in site]
    assert len(canvas.sites) == 2
    assert [len(site) for site in canvas.sites] == [3, 5]
    assert all(canvas.ids[position] == FakeAdapter.mask_id for position in flat)
    assert not set(flat) & set(canvas.E)
    assert len(canvas.E) > 0
    assert len(canvas.gold) == 2


def test_build_canvas_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_canvas(FakeAdapter(), FAM, M=4, J=2, lengths=[3])
    with pytest.raises(ValueError):
        build_canvas(FakeAdapter(), FAM, M=4, J=2, lengths=[3, 5, 7])


def test_build_canvas_rejects_empty_sites():
    with pytest.raises(ValueError):
        build_canvas(FakeAdapter(), FAM, M=4, J=2, lengths=[3, 0])


def test_canvas_spells_the_prompt():
    canvas = build_canvas(FakeAdapter(), tiny_family(), M=2, J=2, lengths=[2, 3])
    text = FakeAdapter().decode(canvas.ids)
    assert text == (
        "<h>Reverse the list."
        "Input: [1, 2]\nOutput: [2, 1]\n\n"
        "Input: [4, 5]\nOutput:__\n\n"
        "Input: [3]\nOutput: [3]\n\n"
        "Input: [6, 7, 8]\nOutput:___\n\n"
        "<e>"
    )


def test_E_is_exactly_the_demonstration_output_tokens():
    adapter = FakeAdapter()
    canvas = build_canvas(adapter, tiny_family(), M=2, J=2, lengths=[2, 3])
    assert canvas.E == sorted(canvas.E)
    assert adapter.decode([canvas.ids[position] for position in canvas.E]) == " [2, 1] [3]"


def test_sites_are_contiguous_and_follow_the_requested_lengths():
    canvas = build_canvas(FakeAdapter(), tiny_family(), M=2, J=2, lengths=[2, 3])
    assert [len(site) for site in canvas.sites] == [2, 3]
    for site in canvas.sites:
        assert site == list(range(site[0], site[0] + len(site)))


def test_gold_is_the_target_outputs_and_never_in_the_ids():
    adapter = FakeAdapter()
    canvas = build_canvas(adapter, tiny_family(), M=2, J=2, lengths=[2, 3])
    assert canvas.gold == ["[5, 4]", "[8, 7, 6]"]
    text = adapter.decode(canvas.ids)
    assert "[5, 4]" not in text
    assert "[8, 7, 6]" not in text


def test_canvas_meta_describes_the_episode():
    canvas = build_canvas(FakeAdapter(), FAM, M=4, J=2, lengths=[6, 21])
    assert canvas.meta["task"] == "list_easy_fam"
    assert canvas.meta["episode_id"] == FAM["episode_id"]
    assert canvas.meta["M"] == 4
    assert canvas.meta["J"] == 2
    assert canvas.meta["lengths"] == [6, 21]
    assert canvas.meta["placement"] == "interleaved"
    assert canvas.meta["model"] == "fake"
    assert canvas.meta["slots"] == ["D1", "D2", "T1", "T2", "D3", "D4"]


def test_oracle_lengths_give_one_mask_per_gold_token():
    adapter = FakeAdapter()
    gold_lengths = [len(adapter.encode(" " + target["output"])) for target in FAM["targets"][:2]]
    canvas = build_canvas(adapter, FAM, M=4, J=2, lengths=gold_lengths)
    assert [len(site) for site in canvas.sites] == gold_lengths


def test_changing_lengths_changes_only_the_target_sites():
    adapter = FakeAdapter()
    short = build_canvas(adapter, tiny_family(), M=2, J=2, lengths=[1, 1])
    long = build_canvas(adapter, tiny_family(), M=2, J=2, lengths=[1, 4])
    assert len(long.ids) - len(short.ids) == 3
    assert short.gold == long.gold
    assert adapter.decode(short.ids[: short.sites[1][0]]) == adapter.decode(long.ids[: long.sites[1][0]])


# ---------------------------------------------------------------------------------------------
# active_episode
# ---------------------------------------------------------------------------------------------


def test_active_episode_is_deterministic_and_nested():
    a = active_episode(FAM, 4, 2)
    b = active_episode(FAM, 4, 2)
    c = active_episode(FAM, 8, 2)
    assert a == b
    nested_ids = [demo["id"] for demo in c["demos"]][: len(a["demos"])]
    assert [demo["id"] for demo in a["demos"]] == nested_ids


def test_active_episode_uses_the_first_M_demos_and_first_J_targets():
    episode = active_episode(FAM, 4, 2)
    assert episode["demos"] == FAM["demos"][:4]
    assert episode["targets"] == FAM["targets"][:2]
    assert len(episode["entries"]) == 6


def test_entries_follow_the_master_layout():
    assert slots(active_episode(FAM, 4, 2)) == ["D1", "D2", "T1", "T2", "D3", "D4"]
    assert slots(active_episode(FAM, 4, 8)) == FAM["meta"]["master_placement"]


def test_a_smaller_cell_keeps_the_relative_order_of_a_larger_one():
    large = slots(active_episode(FAM, 16, 8))
    for M, J in [(1, 1), (2, 2), (4, 2), (8, 4), (16, 1)]:
        small = slots(active_episode(FAM, M, J))
        assert small == [slot for slot in large if slot in small]


def test_demos_first_and_demos_after_placements():
    assert slots(active_episode(FAM, 2, 2, "demos_first")) == ["D1", "D2", "T1", "T2"]
    assert slots(active_episode(FAM, 2, 2, "demos_after")) == ["T1", "T2", "D1", "D2"]


def test_entries_carry_role_and_record_text():
    entry = active_episode(FAM, 4, 2)["entries"][2]
    assert entry["slot"] == "T1"
    assert entry["role"] == "target"
    assert entry["input"] == FAM["targets"][0]["input"]
    assert entry["output"] == FAM["targets"][0]["output"]


def test_active_episode_rejects_bad_arguments():
    with pytest.raises(ValueError):
        active_episode(FAM, 0, 2)
    with pytest.raises(ValueError):
        active_episode(FAM, 4, 0)
    with pytest.raises(ValueError):
        active_episode(FAM, 17, 2)
    with pytest.raises(ValueError):
        active_episode(FAM, 4, 9)
    with pytest.raises(ValueError):
        active_episode(FAM, 4, 2, "sideways")


# ---------------------------------------------------------------------------------------------
# The master layout
# ---------------------------------------------------------------------------------------------


def test_up_to_four_demonstrations_use_the_published_twelve_slot_layout():
    assert BASE_DEMOS == 4
    assert master_layout(FAM, 4) == FAM["meta"]["master_placement"]


def test_extended_layout_matches_the_frozen_one_for_c030():
    assert master_layout(FAM, 16) == C030_LAYOUT_24


def test_extended_layout_is_a_permutation_that_restricts_to_the_published_one():
    for family in LIST_REPORT + LIST_DEV:
        base = family["meta"]["master_placement"]
        layout = extend_layout(base, family["meta"]["function_id"])
        expected = [f"D{i}" for i in range(1, 17)] + [f"T{j}" for j in range(1, 9)]
        assert sorted(layout) == sorted(expected)
        assert layout[0] == "D1"
        assert [slot for slot in layout if slot in base] == base
        targets = [slot for slot in layout if slot.startswith("T")]
        assert targets == [f"T{j}" for j in range(1, 9)]


def test_extended_layout_depends_on_the_function_id_only_through_the_seed():
    base = FAM["meta"]["master_placement"]
    assert extend_layout(base, "c030") == extend_layout(base, "c030")
    assert extend_layout(base, "c030") != extend_layout(base, "c031")


def test_every_frozen_layout_starts_with_a_demonstration():
    for family in LIST_REPORT + LIST_DEV + SALT_REPORT:
        assert family["meta"]["master_placement"][0] == "D1"


def test_more_than_four_demonstrations_need_the_list_functions_extension():
    with pytest.raises(ValueError):
        active_episode(SALT_REPORT[0], 8, 2)
