"""The fixtures do not redistribute the text of SST-2 or MIT Movie (data/benchmarks/README.md), nor run metadata.

Fixture records cut from SST-2 or MIT Movie pools keep a SHA-256 digest of every field that held benchmark text
(tools/extract_fixtures.py); the tests that need the text rebuild it from the families that
data/benchmarks/build.py writes locally, and compare digests. No fixture file, of any figure, may hold a machine
path, a run-metadata key that names a machine, or the vocabulary of the authors' internal notes.
"""
import json
import os
import re

import pytest

from fewshot_dlms.data import BANKS
from fewshot_dlms.data import TEXT_FREE_BENCHMARKS
from fewshot_dlms.data import load_families
from fewshot_dlms.leakcheck import find_private

FIXTURE_DIRS = ("fig4",)  # fixture directories cut from SST-2 / MIT Movie pools

# This site's patterns (lower-cased text), next to leakcheck's generic ones (machine paths, run metadata that names a
# machine): the vocabulary of the authors' internal notes.
LOCAL_PATTERNS = [
    r"hand-?\s?off",
    r"\.claude",
]


def fixture_paths(root):
    return [path for name in FIXTURE_DIRS for path in sorted((root / "data" / "fixtures" / name).glob("*.jsonl"))]


def string_values(value):
    """Every string value inside a JSON value (dictionary keys excluded), depth first."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


def benchmark_sentences():
    """The input text of every SST-2 and MIT Movie record, both banks; skip locally, fail under CI without it."""
    try:
        families = [family for task in TEXT_FREE_BENCHMARKS for bank in BANKS for family in load_families(task, bank)]
    except FileNotFoundError:
        message = (
            "SST-2 / MIT Movie text not built: "
            'run `pip install -e ".[data]"` and `python data/benchmarks/build.py`'
        )
        if os.environ.get("CI", "").lower() not in ("", "0", "false"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    return {record["input"] for family in families for record in family["demos"] + family["targets"]}


def test_fixtures_exist(root):
    assert fixture_paths(root)


def test_fixtures_hold_no_prompt_text(root):
    for path in fixture_paths(root):
        for line in path.read_text(encoding="utf-8").splitlines():
            for value in string_values(json.loads(line)):
                assert "Input:" not in value and "Output:" not in value, (path.name, value[:80])


def test_fixtures_hold_no_benchmark_sentences(root):
    sentences = benchmark_sentences()
    for path in fixture_paths(root):
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            values.extend(string_values(json.loads(line)))
        leaked = sorted({sentence for sentence in sentences for value in values if sentence in value})
        assert not leaked, (path.name, leaked[:5])


def test_no_fixture_holds_private_strings(root):
    """Every file under data/fixtures (fig4, fig5, fig6, the walkthrough pool and any later fixture directory)."""
    paths = sorted(path for path in (root / "data" / "fixtures").rglob("*") if path.is_file())
    names = {path.relative_to(root / "data" / "fixtures").as_posix() for path in paths}
    assert {"walkthrough.jsonl", "fig6/head_predictions_llada.json"} <= names
    for path in paths:
        text = path.read_bytes().decode("utf-8", errors="replace").lower()
        assert not find_private(text), f"{find_private(text)} in {path.name}"
        for pattern in LOCAL_PATTERNS:
            assert re.search(pattern, text) is None, f"{pattern!r} matches {path.name}"
