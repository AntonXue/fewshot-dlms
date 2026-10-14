"""Analysis parity (fidelity level 2 of README.md) for Figure 5 and its appendix mechanism and M x J
sharpening figures.

experiments/fig5_multi_target/analyze.py recomputes the paper's tables from the frozen pools and must
reproduce data/tables/fig5/ to 1e-9 in every numeric column (and exactly in every other column):

  cells.csv        Figure 5 and Section 4.2: A_all, A_target and the paired gains per (model, M, J, beta)
  gold.csv         gold-answer support per (model, M, J)
  diagnostics.csv  pool coverage, reward spread, ESS per (model, M, J)
  sharpening.csv   M x J sharpening figure (Fig. 10 ACL / Fig. 11 arXiv): sharpening control vs example guidance
                   per (model, M, J)
  mechanism.csv    mechanism figure (Fig. 8, both builds): gold NLL by reveal position, cross-benchmark study, J = 8

The parity tests need the fetched pools (`python -m fewshot_dlms.fetch pools java`). The other tests run
on every `make test`: the --compare report of a resampled cell (fidelity level 3) on the two-family
fixture pool, the message for missing pools, plot.py on the committed tables, and run.py's seed rule, protocol hash
and resume rule.
Pixel identity of the figures with the paper's PDFs is checked by hand (README of the experiment).
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fewshot_dlms.data import read_jsonl

ANALYZE = "experiments/fig5_multi_target/analyze.py"
TOLERANCE = 1e-9
KEYS = {
    "cells": ["model", "M", "J", "beta"],
    "gold": ["model", "M", "J"],
    "diagnostics": ["model", "M", "J"],
    "sharpening": ["model", "M", "J"],
    "mechanism": ["model", "source", "J", "position"],
}
COMPARE_LINE = re.compile(
    r"^(\S+) M=(\d+) J=(\d+): frozen A_all ([0-9.]+), rerun CI \[([0-9.]+), ([0-9.]+)\] -> frozen mean (inside|outside)"
)
# The verdict of the same cell under guidance at beta = 0.5, printed after the ordinary one.
COMPARE_LINE_GUIDED = re.compile(
    r"^(\S+) M=(\d+) J=(\d+) beta=0\.5: frozen A_all ([0-9.]+), rerun CI \[([0-9.]+), ([0-9.]+)\] -> frozen mean (inside|outside)"
)


def run_analyze(root, pools, out, *extra):
    command = [sys.executable, str(root / ANALYZE), "--pools", str(pools), "--out", str(out), *extra]
    return subprocess.run(command, capture_output=True, text=True, cwd=root)


@pytest.fixture(scope="module")
def analysis(root, artifacts, tmp_path_factory):
    """Run analyze.py once on the frozen pools; returns (output directory, stdout of --compare)."""
    pools = artifacts / "pools" / "fig5"
    if not pools.exists():
        pytest.skip("Figure 5 pools not fetched: run `python -m fewshot_dlms.fetch pools java`")
    out = tmp_path_factory.mktemp("fig5")
    result = run_analyze(root, pools, out, "--compare", str(root / "data" / "tables" / "fig5"))
    assert result.returncode == 0, result.stderr
    return out, result.stdout


def max_abs_diff(ours, paper, keys):
    """Largest |ours - paper| over the numeric columns after aligning rows on `keys`.

    Columns must agree in name and order. Non-numeric columns must be equal; a NaN must sit where the
    paper has one.
    """
    assert list(ours.columns) == list(paper.columns)
    ours = ours.sort_values(keys).reset_index(drop=True)
    paper = paper.sort_values(keys).reset_index(drop=True)
    assert ours[keys].equals(paper[keys])
    worst = 0.0
    for column in paper.columns:
        if paper[column].dtype.kind != "f":
            assert ours[column].equals(paper[column]), column
            continue
        a = ours[column].to_numpy(float)
        b = paper[column].to_numpy(float)
        assert (np.isnan(a) == np.isnan(b)).all(), column
        finite = ~np.isnan(b)
        if finite.any():
            worst = max(worst, float(np.abs(a[finite] - b[finite]).max()))
    return worst


@pytest.mark.parity
@pytest.mark.parametrize("name", list(KEYS))
def test_table_matches_paper(root, analysis, name):
    out, _ = analysis
    ours = pd.read_csv(out / f"{name}.csv", float_precision="round_trip")
    paper = pd.read_csv(root / "data" / "tables" / "fig5" / f"{name}.csv", float_precision="round_trip")
    assert max_abs_diff(ours, paper, KEYS[name]) < TOLERANCE, name


@pytest.mark.parity
def test_frozen_pools_lie_inside_their_own_intervals(analysis):
    """Compared with itself, every cell reports the frozen mean inside the 95 % CI, at beta = 0 and 0.5."""
    _, stdout = analysis
    for pattern in (COMPARE_LINE, COMPARE_LINE_GUIDED):
        lines = [pattern.match(line) for line in stdout.splitlines()]
        verdicts = [match.group(7) for match in lines if match]
        assert len(verdicts) == 48
        assert set(verdicts) == {"inside"}


def test_compare_reports_inside_ci(root, tmp_path):
    """A resampled cell is judged by whether the frozen mean lies inside the rerun's 95 % CI.

    The two-family fixture pool plays the rerun of the LLaDA-1.5 cell M = 4, J = 2; --compare prints one verdict
    for ordinary sampling (beta = 0) and one for guidance at beta = 0.5.
    """
    pools = tmp_path / "pools"
    (pools / "llada").mkdir(parents=True)
    shutil.copy(root / "data" / "fixtures" / "fig5" / "llada_M4_J2.jsonl", pools / "llada" / "list_easy_fam_M4_J2.jsonl")
    result = run_analyze(root, pools, tmp_path / "out", "--compare", str(root / "data" / "tables" / "fig5"))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    paper = pd.read_csv(root / "data" / "tables" / "fig5" / "cells.csv")
    rerun = pd.read_csv(tmp_path / "out" / "cells.csv")
    for beta, pattern in ((0.0, COMPARE_LINE), (0.5, COMPARE_LINE_GUIDED)):
        matches = [match for match in map(pattern.match, lines) if match]
        assert len(matches) == 1, beta
        model, M, J, frozen, lo, hi, verdict = matches[0].groups()
        assert (model, M, J) == ("llada", "4", "2")

        paper_cell = paper[(paper.model == "llada") & (paper.M == 4) & (paper.J == 2) & (paper.beta == beta)].iloc[0]
        rerun_cell = rerun[rerun.beta == beta].iloc[0]
        assert rerun_cell.families == 2
        assert float(frozen) == pytest.approx(paper_cell.A_all, abs=5e-4)
        assert float(lo) == pytest.approx(rerun_cell.A_all_lo, abs=5e-4)
        assert float(hi) == pytest.approx(rerun_cell.A_all_hi, abs=5e-4)
        inside = rerun_cell.A_all_lo <= paper_cell.A_all <= rerun_cell.A_all_hi
        assert verdict == ("inside" if inside else "outside")
    guided = next(index for index, line in enumerate(lines) if COMPARE_LINE_GUIDED.match(line))
    assert COMPARE_LINE.match(lines[guided - 1])  # each cell's guided verdict follows its ordinary one


def test_missing_pools_name_the_fetch_command(root, tmp_path):
    result = run_analyze(root, tmp_path / "no_pools", tmp_path / "out")
    assert result.returncode == 2
    assert "run: python -m fewshot_dlms.fetch pools java" in result.stderr + result.stdout


def test_plot_writes_the_four_figures(root, tmp_path):
    """plot.py draws Figure 5 and the three appendix figures from the committed tables alone (no pools, no GPU)."""
    command = [sys.executable, str(root / "experiments/fig5_multi_target/plot.py"), "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    for name in ("fig5", "fig8_mechanism", "fig9_crossbench", "fig10_sharpening"):
        assert (tmp_path / f"{name}.pdf").stat().st_size > 10_000, name


def load_run(root):
    """experiments/fig5_multi_target/run.py as a module (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("fig5_run", root / "experiments/fig5_multi_target/run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


def test_run_seed_rule_matches_the_frozen_records(root):
    """run.py's proposal seed of every fixture record is the seed the authors' run stored."""
    run = load_run(root)
    for model in ("llada", "nemotron_pt", "dqwen"):
        for record in read_jsonl(root / "data" / "fixtures" / "fig5" / f"{model}_M4_J2.jsonl"):
            assert run.proposal_seed(model, record["episode_id"], record["M"], record["J"]) == record["seed"]


LLADA = SimpleNamespace(name="llada", model_id="GSAI-ML/LLaDA-1.5", revision="rev")


def run_config_for(run, M, J, K):
    """run.py's configuration of a LLaDA-1.5 run of the cells M x J with K candidates (no GPU, no model)."""
    return run.config_of(SimpleNamespace(bank="report", M=M, J=J, K=K, chunk=run.CHUNK), LLADA)


def test_pool_protocol_depends_on_what_decides_the_records(root):
    """A pool's protocol hash changes with K (and revision, chunk, ...), not with the other cells a run also samples;
    it is the hash that a run of that cell alone writes."""
    run = load_run(root)
    alone = run.protocol_hash(run_config_for(run, [4], [2], 16))
    wide = run.protocol_hash(run.pool_config(run_config_for(run, [2, 4, 8, 16], [1, 2, 4, 8], 16), 4, 2))
    assert wide == alone == run.protocol_hash(run.pool_config(run_config_for(run, [4], [2], 16), 4, 2))
    assert run.protocol_hash(run.pool_config(run_config_for(run, [4], [2], 8), 4, 2)) != alone
    assert run.protocol_hash(run.pool_config(run_config_for(run, [4], [2, 4], 16), 4, 4)) != alone


def test_resume_skips_only_records_of_the_same_protocol(root, tmp_path):
    """done_episodes returns the families a run of the same protocol may skip, and refuses a file of another protocol
    (the frozen pools carry the authors' hash, so a rerun never appends to them)."""
    run = load_run(root)
    digest = run.protocol_hash(run_config_for(run, [4], [2], 16))
    path = tmp_path / "pool.jsonl"
    assert run.done_episodes(path, digest) == set()
    good = [json.dumps({"episode_id": name, "protocol_hash": digest}) for name in ("a", "b")]
    path.write_text("\n".join(good) + '\n{"episode_id": "c", "protoc', encoding="utf-8")  # an interrupted write
    assert run.done_episodes(path, digest) == {"a", "b"}
    path.write_text(good[0] + "\n" + json.dumps({"episode_id": "b", "protocol_hash": "98fe06a36f6b6503"}) + "\n")
    with pytest.raises(run.ProtocolMismatch, match="98fe06a36f6b6503"):
        run.done_episodes(path, digest)


def test_run_refuses_a_pool_of_another_protocol(root, tmp_path):
    """run.py exits with status 2, naming the file, before it loads a model, when one of its pools holds another
    protocol; the other pools of the run are not touched."""
    pool = tmp_path / "llada" / "list_easy_fam_M4_J2.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({"episode_id": "list_easy_fam:report:000", "protocol_hash": "0123456789abcdef"}) + "\n")
    command = [sys.executable, str(root / "experiments/fig5_multi_target/run.py"), "--model", "llada"]
    command += ["--M", "4", "--J", "1,2", "--K", "4", "--families", "2", "--out", str(tmp_path), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2, result.stderr
    assert str(pool) in result.stderr
    assert "loaded in" not in result.stdout
    assert sorted(path.name for path in pool.parent.iterdir()) == [pool.name]


def test_run_counts_a_failed_family_and_exits_non_zero(root, tmp_path, monkeypatch, capsys):
    """A family that raises is reported and counted, the others are written, and the run ends with RUN_FAILED and exit
    status 1 (no model: load and run_canvas are replaced); the next run writes only the failed family."""
    run = load_run(root)
    monkeypatch.setattr(run, "load", lambda name, device: SimpleNamespace(model_id="GSAI-ML/LLaDA-1.5", revision="rev"))
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run.torch.cuda, "empty_cache", lambda: None)
    body = {"n_tokens": 1, "oracle_L": 1, "n_unique": 1, "seconds_gen": 0.0, "seconds_score": 0.0, "checks": None,
            "gold_ell1": 0.0, "gold_ell0": 0.0}

    def flaky(adapter, family, *arguments, **options):
        if family["episode_id"] == "list_easy_fam:report:001":
            raise RuntimeError("injected failure")
        return body

    monkeypatch.setattr(run, "run_canvas", flaky)
    argv = ["run.py", "--model", "llada", "--M", "4", "--J", "2", "--K", "4", "--families", "3", "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", [*argv, "--device", "cpu"])
    with pytest.raises(SystemExit) as exit:
        run.main()
    stdout = capsys.readouterr().out
    assert exit.value.code == 1
    assert "FAILED list_easy_fam:report:001 M=4 J=2" in stdout
    assert "RUN_FAILED llada" in stdout and "failed families: 1" in stdout
    assert "RUN_DONE" not in stdout
    pool = tmp_path / "llada" / "list_easy_fam_M4_J2.jsonl"
    assert [record["episode_id"] for record in read_jsonl(pool)] == ["list_easy_fam:report:000", "list_easy_fam:report:002"]

    monkeypatch.setattr(run, "run_canvas", lambda adapter, family, *arguments, **options: body)
    run.main()
    stdout = capsys.readouterr().out
    assert "CELL_DONE llada M=4 J=2 new=1" in stdout
    assert "RUN_DONE llada" in stdout
