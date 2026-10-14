"""Reproduce two frozen Figure 5 families with this repository's own code.

data/fixtures/fig5/llada_M4_J2.jsonl holds the first two report families of the authors' LLaDA-1.5 run
at M = 4 demonstrations and J = 2 targets: K = 16 candidate infills each, with token ids, texts, and
the log-likelihoods ell1 (demonstration outputs visible) and ell0 (masked).
expected_llada_M4_J2.json holds the authors' own numbers for these families. The tests recompute them
from the pool records with grading.py and scoring.py and require agreement to 1e-12.
"""
import json

import numpy as np
import pytest

from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.data import read_jsonl
from fewshot_dlms.grading import answer
from fewshot_dlms.grading import grade
from fewshot_dlms.scoring import family_metrics
from fewshot_dlms.scoring import pool_weights
from tests.fakes import FakeAdapter

TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def records(root):
    return read_jsonl(root / "data" / "fixtures" / "fig5" / "llada_M4_J2.jsonl")


@pytest.fixture(scope="module")
def expected(root):
    path = root / "data" / "fixtures" / "fig5" / "expected_llada_M4_J2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def correctness(record):
    """The K x J matrix of 0/1 per-target correctness of a record's candidates."""
    correct = np.zeros((len(record["candidates"]), record["J"]))
    for k, candidate in enumerate(record["candidates"]):
        for j, (text, gold) in enumerate(zip(candidate["texts"], record["gold"])):
            is_correct, _ = grade(record["grader"], answer(text, record["model"]), gold)
            correct[k, j] = is_correct
    return correct


def metrics_at(record, beta):
    """family_metrics of a record under the weights softmax(beta * (ell1 - ell0))."""
    ell1 = [candidate["ell1"] for candidate in record["candidates"]]
    ell0 = [candidate["ell0"] for candidate in record["candidates"]]
    weights = pool_weights(ell1, ell0, beta)
    return family_metrics(correctness(record), weights)


def test_fixture_holds_two_complete_families(records, expected):
    assert [record["episode_id"] for record in records] == list(expected["families"])
    assert len(records) == 2
    for record in records:
        assert len(record["candidates"]) == record["K"] == 16
        assert (record["M"], record["J"]) == (expected["M"], expected["J"]) == (4, 2)


def test_fixture_carries_no_run_metadata(records):
    for record in records:
        assert not {"host", "gpu_env", "wall_time"} & set(record)


def test_metrics_match_the_authors_run(records, expected):
    for record in records:
        for beta, authors in expected["families"][record["episode_id"]].items():
            ours = metrics_at(record, float(beta))
            for name, value in authors.items():
                assert abs(ours[name] - value) < TOLERANCE, (record["episode_id"], beta, name)


def test_example_guidance_changes_A_all_of_the_first_family(records):
    # Family 000 has 14 of 16 candidates fully correct; weighting by Q_0.5 moves A_all from 14/16 to 0.9347.
    record = records[0]
    assert metrics_at(record, 0.0)["A_all"] == pytest.approx(14 / 16, abs=TOLERANCE)
    assert metrics_at(record, 0.5)["A_all"] == pytest.approx(0.9346902702655528, abs=TOLERANCE)


def test_canvas_has_the_recorded_layout(records):
    families = {family["episode_id"]: family for family in load_families("list_easy_fam", "report")}
    for record in records:
        canvas = build_canvas(
            FakeAdapter(), families[record["episode_id"]], record["M"], record["J"], record["oracle_L"]
        )
        assert canvas.meta["slots"] == record["record_labels"]
        assert [len(site) for site in canvas.sites] == record["oracle_L"]
        assert canvas.gold == record["gold"]
        assert len(canvas.E) > 0
