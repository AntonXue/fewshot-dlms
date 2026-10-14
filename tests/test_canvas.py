"""Unit tests for canvas: which records an episode uses, where they sit, and the token sequence.

The tests run on CPU with `FakeAdapter`, whose tokenizer maps every character to one token, so a
canvas can be compared with the text it should spell. List Functions-Easy families are read from
data/benchmarks/; the small hand-made family below is there to make the expected text readable.
"""
import hashlib
import os

import pytest

from fewshot_dlms.adapters.nemotron import Nemotron
from fewshot_dlms.canvas import AR_PERHOLE_NOTE
from fewshot_dlms.canvas import BASE_DEMOS
from fewshot_dlms.canvas import JAVA_INSTRUCTION
from fewshot_dlms.canvas import JAVA_ZERO_DEMO_INSTRUCTION
from fewshot_dlms.canvas import active_episode
from fewshot_dlms.canvas import ar_prompt_ids
from fewshot_dlms.canvas import build_ar_prompt
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.canvas import build_java_canvas
from fewshot_dlms.canvas import extend_layout
from fewshot_dlms.canvas import master_layout
from fewshot_dlms.data import load_families
from fewshot_dlms.data import read_jsonl
from tests.fakes import FakeAdapter

LIST_REPORT = load_families("list_easy_fam", "report")
LIST_DEV = load_families("list_easy_fam", "dev")
SALT_REPORT = load_families("salt_rep_fam", "report")
FAM = LIST_REPORT[0]

# The first report family of the 24-slot layout used by the M x J grid, as the authors froze it
# (function c030; its 12-slot layout is FAM["meta"]["master_placement"]).
C030_LAYOUT_24 = [
    "D1", "D5", "D6", "D7", "D2", "T1",
    "T2", "D8", "D9", "T3", "D10", "D3",
    "T4", "T5", "D11", "T6", "T7", "D12",
    "D13", "D14", "D15", "D16", "D4", "T8",
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


def test_fake_adapter_tokens_never_collide_with_the_mask():
    adapter = FakeAdapter()
    tokens = adapter.encode("[1, 2] \u00e9\u20ac")
    assert max(tokens) < adapter.mask_id
    assert adapter.decode(tokens + [adapter.mask_id] + tokens, mask="#") == "[1, 2] \u00e9\u20ac#[1, 2] \u00e9\u20ac"


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


# ---------------------------------------------------------------------------------------------
# build_java_canvas (Section 5: helper propagation)
# ---------------------------------------------------------------------------------------------


def java_site(name, old, new, n_head=1, n_tail=2):
    """A site whose enclosing method has `n_head` lines before the changed line and `n_tail` after it."""
    head = [f"    // head {i}" for i in range(n_head)]
    tail = [f"    // tail {i}" for i in range(n_tail)]
    return {
        "site_id": f"{name}@1",
        "method": name,
        "span_old": old,
        "span_new": new,
        "method_before": "\n".join(head + [old] + tail),
        "i1": n_head,
        "i2": n_head + 1,
    }


def tiny_java_episode():
    """One helper, two demonstrations and two targets, with one-line spans and short contexts."""
    return {
        "episode_id": "tiny_java",
        "repository": "toy",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "file": "src/main/java/Util.java",
        "helper": "twice",
        "helper_text": "    static int twice(int x) {\n        return 2 * x;\n    }\n",
        "demos": [
            java_site("a", "        y = x + x;", "        y = twice(x);"),
            java_site("b", "        z = w + w;", "        z = twice(w);"),
        ],
        "targets_visible": [
            java_site("c", "        u = v + v;", None),
            java_site("d", "        p = q + q;", None),
        ],
        "reference": {
            "targets": [
                {"site_id": "c@1", "span_new": "        u = twice(v);"},
                {"site_id": "d@1", "span_new": "        p = twice(q);"},
            ]
        },
        "M": 2,
        "J": 2,
    }


def test_java_canvas_spells_the_prompt():
    adapter = FakeAdapter()
    canvas = build_java_canvas(adapter, tiny_java_episode(), 2, 2, [3, 2])
    expected = (
        "<h>" + JAVA_INSTRUCTION
        + "// toy 0123456789 src/main/java/Util.java\n"
        + "// helper, already present in the file:\n"
        + "    static int twice(int x) {\n        return 2 * x;\n    }\n\n"
        + "// --- Util.java : a ---\n<old>\n    // head 0\n        y = x + x;\n    // tail 0\n    // tail 1\n</old>\n"
        + "<new>\n    // head 0\n        y = twice(x);\n    // tail 0\n    // tail 1\n</new>\n\n"
        + "// --- Util.java : b ---\n<old>\n    // head 0\n        z = w + w;\n    // tail 0\n    // tail 1\n</old>\n"
        + "<new>\n    // head 0\n        z = twice(w);\n    // tail 0\n    // tail 1\n</new>\n\n"
        + "// --- Util.java : c ---\n<old>\n    // head 0\n        u = v + v;\n    // tail 0\n    // tail 1\n</old>\n"
        + "<new>\n    // head 0\n___\n    // tail 0\n    // tail 1\n</new>\n\n"
        + "// --- Util.java : d ---\n<old>\n    // head 0\n        p = q + q;\n    // tail 0\n    // tail 1\n</old>\n"
        + "<new>\n    // head 0\n__\n    // tail 0\n    // tail 1\n</new>\n\n"
        + "<e>"
    )
    assert adapter.decode(canvas.ids) == expected


def test_java_canvas_E_is_the_accepted_edits_and_gold_the_references():
    adapter = FakeAdapter()
    canvas = build_java_canvas(adapter, tiny_java_episode(), 2, 2, [3, 2])
    assert canvas.E == sorted(canvas.E)
    assert adapter.decode([canvas.ids[position] for position in canvas.E]) == "        y = twice(x);        z = twice(w);"
    assert canvas.gold == ["        u = twice(v);", "        p = twice(q);"]
    assert [len(site) for site in canvas.sites] == [3, 2]
    assert canvas.sites[1][0] > canvas.sites[0][-1]
    assert canvas.meta["task"] == "java_helper"
    assert canvas.meta["episode_id"] == "tiny_java"
    assert (canvas.meta["M"], canvas.meta["J"]) == (2, 2)
    assert canvas.meta["lengths"] == [3, 2]
    assert canvas.meta["slots"] == ["D1", "D2", "T1", "T2"]
    assert canvas.meta["model"] == "fake"


def test_java_canvas_uses_the_first_M_demonstrations_and_first_J_targets():
    adapter = FakeAdapter()
    canvas = build_java_canvas(adapter, tiny_java_episode(), 1, 1, [4])
    text = adapter.decode(canvas.ids)
    assert "Util.java : a ---" in text
    assert "Util.java : b ---" not in text
    assert "Util.java : c ---" in text
    assert "Util.java : d ---" not in text
    assert canvas.gold == ["        u = twice(v);"]
    assert len(canvas.sites) == 1


def test_java_canvas_without_demonstrations_has_empty_E():
    adapter = FakeAdapter()
    canvas = build_java_canvas(adapter, tiny_java_episode(), 0, 2, [3, 2], instruction=JAVA_ZERO_DEMO_INSTRUCTION)
    text = adapter.decode(canvas.ids)
    assert canvas.E == []
    assert text.startswith("<h>" + JAVA_ZERO_DEMO_INSTRUCTION + "// toy")
    assert text.count("\n<old>\n") == 2
    assert text.count("</old>\n<new>\n") == 2
    assert canvas.meta["slots"] == ["T1", "T2"]


def test_java_canvas_trims_long_contexts_like_the_authors_renderer():
    adapter = FakeAdapter()
    episode = tiny_java_episode()
    episode["demos"] = []
    episode["targets_visible"] = [java_site("c", "        u = v + v;", None, n_head=25, n_tail=15)]
    episode["reference"]["targets"] = episode["reference"]["targets"][:1]
    canvas = build_java_canvas(adapter, episode, 0, 1, [1])
    text = adapter.decode(canvas.ids)
    # 20 lines before the changed span (behind a "// ..." marker), 12 after (before one); the old and
    # the new block of the target each show them.
    assert text.count("    // head") == 2 * 20
    assert "    // head 4\n" not in text
    assert "    // head 5\n" in text
    assert "    // tail 11\n    // ...\n" in text
    assert "    // tail 12" not in text
    assert "<old>\n    // ...\n    // head 5" in text


def test_java_canvas_rejects_bad_arguments():
    adapter = FakeAdapter()
    episode = tiny_java_episode()
    with pytest.raises(ValueError):
        build_java_canvas(adapter, episode, 2, 2, [3])
    with pytest.raises(ValueError):
        build_java_canvas(adapter, episode, 2, 2, [3, 0])
    with pytest.raises(ValueError):
        build_java_canvas(adapter, episode, 3, 2, [3, 2])
    with pytest.raises(ValueError):
        build_java_canvas(adapter, episode, -1, 2, [3, 2])
    with pytest.raises(ValueError):
        build_java_canvas(adapter, episode, 2, 3, [3, 2, 1])
    with pytest.raises(ValueError):
        build_java_canvas(adapter, episode, 2, 0, [])


# ---------------------------------------------------------------------------------------------
# The single-target prompt of the causal interfaces (Figure 4)
# ---------------------------------------------------------------------------------------------


class RecordingAdapter(FakeAdapter):
    """The fake adapter, keeping every piece of text it is asked to tokenize, in order."""

    def __init__(self):
        self.pieces = []

    def encode(self, text):
        self.pieces.append(text)
        return super().encode(text)


class NemotronText(FakeAdapter):
    """The byte-level fake with Nemotron's chat strings, so a prompt reads back as Nemotron's prompt text."""

    chat_header = Nemotron.chat_header
    chat_user_open = Nemotron.chat_user_open
    chat_user_close_assistant_open = Nemotron.chat_user_close_assistant_open
    chat_end = Nemotron.chat_end


def sst2_report_families():
    """The SST-2 report families; they need the rebuilt text: skip locally, fail under CI when it is missing."""
    try:
        return {family["episode_id"]: family for family in load_families("sst2_fam", "report")}
    except FileNotFoundError:
        message = 'SST-2 text not built: run `pip install -e ".[data]"` and `python data/benchmarks/build.py`'
        if os.environ.get("CI", "").lower() not in ("", "0", "false"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)


def test_ar_prompt_contains_demos_and_placeholder():
    family = tiny_family()
    text, gold = build_ar_prompt(FakeAdapter(), family, M=2, j=1)
    assert text.endswith("Input: [4, 5]\nOutput:")
    assert all(" " + demo["output"] + "\n\n" in text for demo in family["demos"])
    assert gold == ["[5, 4]"]

    text, gold = build_ar_prompt(FakeAdapter(), family, M=2, j=1, placeholder_note=True)
    assert AR_PERHOLE_NOTE % (1, 1) in text
    assert text.endswith("Input: [4, 5]\nOutput: [[1]]<a>[[1]]:")
    assert all(" " + demo["output"] + "\n\n" in text for demo in family["demos"])
    assert gold == ["[5, 4]"]


def test_ar_prompt_spells_the_causal_suffix_prompt():
    text, _ = build_ar_prompt(FakeAdapter(), tiny_family(), M=2, j=2)
    assert text == (
        "<h>Reverse the list."
        "Input: [1, 2]\nOutput: [2, 1]\n\n"
        "Input: [3]\nOutput: [3]\n\n"
        "Input: [6, 7, 8]\nOutput:"
    )


def test_ar_prompt_spells_the_per_hole_prompt():
    text, _ = build_ar_prompt(FakeAdapter(), tiny_family(), M=1, j=1, placeholder_note=True)
    assert text == (
        "<u>Reverse the list."
        " Some records have their output replaced by a numbered placeholder [[k]]."
        " Write the output for placeholder [[1]] only, as '[[1]]: <output>'.\n\n"
        "Input: [1, 2]\nOutput: [2, 1]\n\n"
        "Input: [4, 5]\nOutput: [[1]]"
        "<a>[[1]]:"
    )


def test_ar_prompt_ids_spell_the_prompt_text():
    adapter = FakeAdapter()
    for placeholder_note in (False, True):
        text, _ = build_ar_prompt(adapter, FAM, M=4, j=3, placeholder_note=placeholder_note)
        ids = ar_prompt_ids(adapter, FAM, M=4, j=3, placeholder_note=placeholder_note)
        assert adapter.decode(ids) == text


def test_ar_prompt_ids_are_tokenized_piece_by_piece():
    # Real tokenizers merge across piece boundaries (Qwen3 reads "]\n\n" as one token), so every piece is
    # tokenized on its own, as the authors' compiler did; the demonstration outputs are pieces of their own.
    adapter = RecordingAdapter()
    ar_prompt_ids(adapter, tiny_family(), M=2, j=1)
    assert adapter.pieces == [
        "<h>Reverse the list.",
        "Input: [1, 2]\nOutput:", " [2, 1]", "\n\n",
        "Input: [3]\nOutput:", " [3]", "\n\n",
        "Input: [4, 5]\nOutput:",
    ]


def test_P0_prompt_deletes_the_demonstration_outputs_and_keeps_the_labels():
    adapter = FakeAdapter()
    p1 = ar_prompt_ids(adapter, tiny_family(), M=2, j=1)
    p0 = ar_prompt_ids(adapter, tiny_family(), M=2, j=1, hide_E=True)
    assert adapter.decode(p0) == (
        "<h>Reverse the list."
        "Input: [1, 2]\nOutput:\n\n"
        "Input: [3]\nOutput:\n\n"
        "Input: [4, 5]\nOutput:"
    )
    assert len(p1) - len(p0) == len(adapter.encode(" [2, 1]")) + len(adapter.encode(" [3]"))


def test_ar_prompt_uses_the_first_M_demos_and_target_j():
    text, gold = build_ar_prompt(FakeAdapter(), FAM, M=3, j=5)
    for demo in FAM["demos"][:3]:
        assert f"Input: {demo['input']}\nOutput: {demo['output']}\n\n" in text
    for demo in FAM["demos"][3:]:
        assert f"Input: {demo['input']}\n" not in text
    assert text.endswith(f"Input: {FAM['targets'][4]['input']}\nOutput:")
    assert gold == [FAM["targets"][4]["output"]]


def test_ar_prompt_rejects_bad_arguments():
    adapter = FakeAdapter()
    for M, j in ((0, 1), (17, 1), (4, 0), (4, 9)):
        with pytest.raises(ValueError):
            build_ar_prompt(adapter, FAM, M=M, j=j)
        with pytest.raises(ValueError):
            ar_prompt_ids(adapter, FAM, M=M, j=j)


def sha256_of_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_ar_prompt_is_the_frozen_nemotron_prompt_text(root):
    # The first two records of each frozen Figure 4 pool store the decoded P1 and P0 prompts of their episode; the
    # fixtures keep their SHA-256 (SST-2 text is not redistributed).
    families = sst2_report_families()
    adapter = NemotronText()
    for record in read_jsonl(root / "data" / "fixtures" / "fig4" / "nemotron_ar_sst2_M4.jsonl"):
        family = families[record["episode_id"]]
        text, gold = build_ar_prompt(adapter, family, record["M"], 1)
        p0_text = adapter.decode(ar_prompt_ids(adapter, family, record["M"], 1, hide_E=True))
        assert sha256_of_text(text) == record["checks"]["p1_prompt_text_sha256"]
        assert sha256_of_text(p0_text) == record["checks"]["p0_prompt_text_sha256"]
        assert record["target_inputs_sha256"] == [sha256_of_text(family["targets"][0]["input"])]
        assert gold == record["gold"]
