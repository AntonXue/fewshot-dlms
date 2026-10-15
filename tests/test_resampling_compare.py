"""Resampling check (fidelity level 3): `analyze.py --compare` judges a rerun by its 95 % CI.

GPU resampling gives fresh draws, so a rerun is not compared with the paper's numbers for equality: the
question is whether the frozen mean lies inside the rerun's 95 % family-bootstrap interval. This test feeds
experiments/multi_target/analyze.py two synthetic reruns of the LLaDA-1.5 cell M = 4, J = 2 (64 report
families, K = 16) and checks the verdict it prints against data/tables/multi_target/cells.csv:

  - a pool whose unguided A_all matches the frozen one (0.516) must be reported "inside";
  - a pool whose unguided A_all is far above it must be reported "outside".

Every candidate of a synthetic family is either all-correct (both targets equal the reference answers) or
all-wrong, so a family's A_all is 0 or 1 at every beta and the cell mean is the share of all-correct families. The
verdict at beta = 0.5 (frozen A_all 0.564) is printed on its own line and must agree: "inside" for the matching pool,
"outside" for the far one.
The records are the first fixture record of data/fixtures/multi_target/llada_M4_J2.jsonl with the episode id and the
candidate texts replaced. test_parity_multi_target.test_compare_reports_inside_ci checks the printed numbers on the
two-family fixture; this test pins both verdicts.
"""
import copy
import json

import pandas as pd
import pytest

from fewshot_dlms.data import load_families
from fewshot_dlms.data import read_jsonl
from tests.test_parity_multi_target import COMPARE_LINE
from tests.test_parity_multi_target import COMPARE_LINE_GUIDED
from tests.test_parity_multi_target import run_analyze

WRONG = " [99]"  # a well-formed integer list that is no reference answer of the fixture


def frozen_A_all(root, beta=0.0):
    cells = pd.read_csv(root / "data" / "tables" / "multi_target" / "cells.csv", float_precision="round_trip")
    row = cells[(cells.model == "llada") & (cells.M == 4) & (cells.J == 2) & (cells.beta == beta)]
    assert len(row) == 1
    return float(row.A_all.iloc[0])


def write_synthetic_rerun(root, pools, n_correct):
    """A LLaDA-1.5 (4, 2) pool file over the 64 report families; the first `n_correct` are all-correct."""
    template = read_jsonl(root / "data" / "fixtures" / "multi_target" / "llada_M4_J2.jsonl")[0]
    family_ids = [family["episode_id"] for family in load_families("list_easy_fam", "report")]
    assert len(family_ids) == 64
    path = pools / "llada" / "list_easy_fam_M4_J2.jsonl"
    path.parent.mkdir(parents=True)
    with open(path, "w", encoding="utf-8") as handle:
        for i, family_id in enumerate(family_ids):
            record = copy.deepcopy(template)
            record["episode_id"] = family_id
            texts = [" " + gold for gold in record["gold"]] if i < n_correct else [WRONG] * record["J"]
            for candidate in record["candidates"]:
                candidate["texts"] = list(texts)
            handle.write(json.dumps(record) + "\n")


@pytest.mark.parametrize("offset, expected", [(0.0, "inside"), (0.42, "outside")], ids=["matching", "far"])
def test_compare_reports_ci_verdict(root, tmp_path, offset, expected):
    frozen = frozen_A_all(root)
    n_correct = round((frozen + offset) * 64)
    write_synthetic_rerun(root, tmp_path / "pools", n_correct)

    result = run_analyze(root, tmp_path / "pools", tmp_path / "out", "--compare", str(root / "data" / "tables" / "multi_target"))
    assert result.returncode == 0, result.stderr
    matches = [match for match in map(COMPARE_LINE.match, result.stdout.splitlines()) if match]
    assert len(matches) == 1
    model, M, J, printed_frozen, lo, hi, verdict = matches[0].groups()
    assert (model, M, J) == ("llada", "4", "2")

    rerun = pd.read_csv(tmp_path / "out" / "cells.csv", float_precision="round_trip")
    cell = rerun[rerun.beta == 0.0].iloc[0]
    assert cell.families == 64
    assert cell.A_all == pytest.approx(n_correct / 64, abs=1e-12)
    assert float(printed_frozen) == pytest.approx(frozen, abs=5e-4)
    assert (float(lo), float(hi)) == pytest.approx((cell.A_all_lo, cell.A_all_hi), abs=5e-4)
    assert verdict == expected
    assert (cell.A_all_lo <= frozen <= cell.A_all_hi) == (expected == "inside")

    guided = [match for match in map(COMPARE_LINE_GUIDED.match, result.stdout.splitlines()) if match]
    assert len(guided) == 1
    model, M, J, printed_frozen, lo, hi, verdict = guided[0].groups()
    assert (model, M, J) == ("llada", "4", "2")
    frozen = frozen_A_all(root, 0.5)
    cell = rerun[rerun.beta == 0.5].iloc[0]
    assert cell.A_all == pytest.approx(n_correct / 64, abs=1e-12)
    assert float(printed_frozen) == pytest.approx(frozen, abs=5e-4)
    assert (float(lo), float(hi)) == pytest.approx((cell.A_all_lo, cell.A_all_hi), abs=5e-4)
    assert verdict == expected
    assert (cell.A_all_lo <= frozen <= cell.A_all_hi) == (expected == "inside")
