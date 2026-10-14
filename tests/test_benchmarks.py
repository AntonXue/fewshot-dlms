"""Frozen benchmark families: the loader, the checksums, and the SST-2 / MIT Movie rebuild."""

import json
import os

import pytest

from data.benchmarks import build
from fewshot_dlms.data import (
    BANKS,
    BENCHMARK_DIR,
    BENCHMARKS,
    TEXT_FREE_BENCHMARKS,
    load_families,
)

FAMILY_KEYS = {
    "episode_id",
    "bank",
    "source",
    "rule_id",
    "category",
    "split",
    "instruction",
    "demos",
    "targets",
    "max_M",
    "max_J",
    "grader",
    "meta",
}


@pytest.fixture(scope="module")
def rebuilt_text():
    """Need the rebuilt SST-2 / MIT Movie text: skip locally, fail under CI when it is missing."""
    try:
        for task in TEXT_FREE_BENCHMARKS:
            for bank in BANKS:
                load_families(task, bank)
    except FileNotFoundError:
        pass
    else:
        return

    message = (
        "SST-2 / MIT Movie text not built: "
        'run `pip install -e ".[data]"` and `python data/benchmarks/build.py`'
    )
    if os.environ.get("CI", "").lower() not in ("", "0", "false"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def test_four_tasks_64_families(rebuilt_text):
    for task in BENCHMARKS:
        families = load_families(task, "report")
        assert len(families) == 64, task
        assert set(families[0]) == FAMILY_KEYS, task
        assert set(families[0]["demos"][0]) == {"id", "input", "output"}, task
        assert load_families(task) == families, task


def test_sha_file_covers_every_family(rebuilt_text):
    shas = json.loads((BENCHMARK_DIR / "sha256.json").read_text(encoding="utf-8"))
    recorded = {
        f"{bank}/{task}/{family['episode_id']}"
        for task in BENCHMARKS
        for bank in BANKS
        for family in load_families(task, bank)
    }
    assert set(shas) == recorded
    assert all(len(sha) == 64 for sha in shas.values())


def test_builder_detects_tampered_text():
    family = load_families("list_easy_fam", "report")[0]
    build.verify([family], "report", "list_easy_fam")

    tampered = dict(family)
    tampered["demos"] = [dict(family["demos"][0], input="tampered")] + family["demos"][1:]
    with pytest.raises(build.IntegrityError, match=family["episode_id"]):
        build.verify([tampered], "report", "list_easy_fam")


def test_text_free_files_ship_ids_only():
    for task in TEXT_FREE_BENCHMARKS:
        for bank in BANKS:
            path = BENCHMARK_DIR / bank / f"{task}.ids.jsonl"
            with path.open(encoding="utf-8") as handle:
                families = [json.loads(line) for line in handle]
            assert families, path
            for family in families:
                for entry in family["demos"] + family["targets"]:
                    assert entry["input"] is None, family["episode_id"]
                    assert entry["output"] is None, family["episode_id"]
                    assert entry["source_ref"], family["episode_id"]


def test_missing_rebuilt_text_points_to_the_builder(tmp_path, monkeypatch):
    monkeypatch.setattr("fewshot_dlms.data.BENCHMARK_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="run python data/benchmarks/build.py"):
        load_families("sst2_fam", "report")


def test_rebuild_restores_text_with_the_original_normalisation():
    family = {
        "meta": {"classes": ["negative", "positive"], "codes": {"negative": "H", "positive": "C"}},
        "demos": [{"id": "d00", "input": None, "output": None, "source_ref": "train:1"}],
        "targets": [{"id": "t0", "input": None, "output": None, "source_ref": "train:0"}],
    }
    train = [("a dull film \n", 0), (" a fine film ", 1)]
    rebuilt = build.rebuild_sst2(family, train)
    assert rebuilt["demos"] == [{"id": "d00", "input": "a fine film", "output": "C"}]
    assert rebuilt["targets"] == [{"id": "t0", "input": "a dull film", "output": "H"}]

    line = "what:O 1984:B-Year film:O by:O steven:B-Director spielberg:I-Director <=> <NULL>"
    utterance, answer = build.parse_movie_line(line)
    assert utterance == "what 1984 film by steven spielberg"
    assert answer == "steven spielberg"
