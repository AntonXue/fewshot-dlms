"""Analysis parity (fidelity level 2 of README.md) for Figure 4, its development-bank beta selection and the
appendix tables that share its pools (Figures 9 and 11).

experiments/fig4_single_target/analyze.py recomputes the paper's tables from the frozen pools and must
reproduce them to 1e-9 in every numeric column (and exactly in every other column):

  sec41_accuracy.csv     Figure 4 and Section 4.1: accuracy and paired gains per (model, benchmark, M, beta)
  gold_reward_cells.csv  gold example reward per target token per (model, benchmark, M) (Figure 4's gold row)
  dev_sec41.csv          the same cells on the development bank (beta selection)
  dev_sec42.csv          the cross-benchmark cells on the development bank (beta selection)
  sec44_power.csv        sharpening residuals: guidance against the matched-exponent power control (LLaDA-1.5)
  cells_crossbench.csv   Figure 9 (Appendix B.3), in data/tables/fig5/: the cross-benchmark study, M = 4, K = 64
  beta_sweep.csv         Figure 11 (Appendix B.6): the guidance-strength sweep on List Functions-Easy
  beta_locks.json        the development-only beta of each model, recomputed from the development bank

The parity tests need the fetched pools (`python -m fewshot_dlms.fetch pools java`). The other tests run on
every `make test`: the --compare report of a resampled cell (fidelity level 3) on a two-family fixture, the
message for missing pools, plot.py on the committed tables, and run.py's seed rule and protocol hash.
Pixel identity of the figures with the paper's PDFs is checked by hand (README of the experiment).
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ANALYZE = "experiments/fig4_single_target/analyze.py"
TOLERANCE = 1e-9
# table -> (reference table under data/tables, the columns that identify a row)
TABLES = {
    "sec41_accuracy": ("fig4/sec41_accuracy.csv", ["bank", "model", "source", "M", "beta"]),
    "gold_reward_cells": ("fig4/gold_reward_cells.csv", ["model", "source", "M"]),
    "dev_sec41": ("fig4/dev_sec41.csv", ["bank", "model", "source", "M", "beta"]),
    "dev_sec42": ("fig4/dev_sec42.csv", ["bank", "model", "source", "J", "beta"]),
    "sec44_power": ("fig4/sec44_power.csv", ["bank", "source", "J", "tau", "beta"]),
    "cells_crossbench": ("fig5/cells_crossbench.csv", ["bank", "model", "source", "J", "beta"]),
    "beta_sweep": ("fig4/beta_sweep.csv", ["model", "J", "beta"]),
}
# The fields of a frozen lock file that the development bank determines.
LOCK_FIELDS = ["beta", "n_cells", "cells", "best_positive_below_ordinary", "partial_cells_excluded", "sections"]
COMPARE_LINE = re.compile(
    r"^(\S+) (\S+) M=(\d+): frozen accuracy ([0-9.]+), rerun CI \[([0-9.]+), ([0-9.]+)\] -> frozen mean (inside|outside)"
)


def run_analyze(root, pools, out, *extra):
    command = [sys.executable, str(root / ANALYZE), "--pools", str(pools), "--out", str(out), *extra]
    return subprocess.run(command, capture_output=True, text=True, cwd=root)


def load_module(root, path, name):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis(root, artifacts, tmp_path_factory):
    """Run analyze.py once on the frozen pools with every option; returns (output directory, stdout)."""
    pools = artifacts / "pools" / "fig4"
    if not pools.exists():
        pytest.skip("Figure 4 pools not fetched: run `python -m fewshot_dlms.fetch pools java`")
    out = tmp_path_factory.mktemp("fig4")
    reference = root / "data" / "tables" / "fig4"
    result = run_analyze(root, pools, out, "--crossbench", "--beta-sweep", "--compare", str(reference))
    assert result.returncode == 0, result.stderr
    return out, result.stdout


def max_abs_diff(ours, paper, keys):
    """Largest |ours - paper| over the numeric columns after aligning rows on `keys`.

    Columns must agree in name and order. Non-numeric columns must be equal; a NaN must sit where the
    paper has one.
    """
    assert list(ours.columns) == list(paper.columns)
    assert len(ours) == len(paper)
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
@pytest.mark.parametrize("name", list(TABLES))
def test_table_matches_paper(root, analysis, name):
    out, _ = analysis
    reference, keys = TABLES[name]
    ours = pd.read_csv(out / f"{name}.csv", float_precision="round_trip")
    paper = pd.read_csv(root / "data" / "tables" / reference, float_precision="round_trip")
    assert max_abs_diff(ours, paper, keys) < TOLERANCE, name


@pytest.mark.parity
def test_beta_locks_match_frozen(root, analysis):
    """The development-only beta of every model, recomputed from the development bank, is the frozen one.

    Also the development statistic behind it (mean A_target per beta over the development cells) to 1e-9. LLaDA-1.5's
    lock was frozen before the last development cells were complete; analyze.py reproduces it from the cells it used and
    checks that the rule applied to all complete development cells selects the same beta.
    """
    out, _ = analysis
    ours = json.loads((out / "beta_locks.json").read_text())
    frozen = json.loads((root / "data" / "tables" / "fig4" / "beta_locks.json").read_text())
    assert ours["rule"] == frozen["rule"]
    assert ours["betas"] == frozen["betas"]
    assert sorted(ours["models"]) == sorted(frozen["models"])
    for model, lock in frozen["models"].items():
        mine = ours["models"][model]
        for field in LOCK_FIELDS:
            assert mine[field] == lock[field], (model, field)
        assert sorted(mine["dev_mean_by_beta"]) == sorted(lock["dev_mean_by_beta"])
        for beta, value in lock["dev_mean_by_beta"].items():
            assert abs(mine["dev_mean_by_beta"][beta] - value) < TOLERANCE, (model, beta)
        assert abs(mine["dev_gain_vs_ordinary_pp"] - lock["dev_gain_vs_ordinary_pp"]) < 100 * TOLERANCE, model
        assert mine["all_complete_cells"]["beta"] == lock["beta"], model


@pytest.mark.parity
def test_frozen_pools_lie_inside_their_own_intervals(analysis):
    """Compared with itself, every Section 4.1 cell reports the frozen mean inside the 95 % CI."""
    _, stdout = analysis
    lines = [COMPARE_LINE.match(line) for line in stdout.splitlines()]
    verdicts = [match.group(7) for match in lines if match]
    assert len(verdicts) == 80
    assert set(verdicts) == {"inside"}


def test_compare_reports_inside_ci(root, tmp_path):
    """A resampled cell is judged by whether the frozen mean lies inside the rerun's 95 % CI.

    The two-family fixture pool of Qwen3-8B (SST-2, M = 4, K = 64) plays the rerun of that cell.
    """
    pools = tmp_path / "pools" / "report" / "qwen"
    pools.mkdir(parents=True)
    shutil.copy(root / "data" / "fixtures" / "fig4" / "qwen_sst2_M4.jsonl", pools / "sst2_fam_M4_J1.jsonl")
    result = run_analyze(root, tmp_path / "pools", tmp_path / "out", "--compare", str(root / "data" / "tables" / "fig4"))
    assert result.returncode == 0, result.stderr
    matches = [COMPARE_LINE.match(line) for line in result.stdout.splitlines()]
    matches = [match for match in matches if match]
    assert len(matches) == 1
    model, source, M, frozen, lo, hi, verdict = matches[0].groups()
    assert (model, source, M) == ("qwen", "sst2_fam", "4")

    paper = pd.read_csv(root / "data" / "tables" / "fig4" / "sec41_accuracy.csv")
    paper_cell = paper[(paper.model == "qwen") & (paper.source == "sst2_fam") & (paper.M == 4) & (paper.beta == 0.0)].iloc[0]
    rerun = pd.read_csv(tmp_path / "out" / "sec41_accuracy.csv")
    rerun_cell = rerun[rerun.beta == 0.0].iloc[0]
    assert rerun_cell.families == 2
    assert float(frozen) == pytest.approx(paper_cell.accuracy, abs=5e-4)
    assert float(lo) == pytest.approx(rerun_cell.ci_lo, abs=5e-4)
    assert float(hi) == pytest.approx(rerun_cell.ci_hi, abs=5e-4)
    inside = rerun_cell.ci_lo <= paper_cell.accuracy <= rerun_cell.ci_hi
    assert verdict == ("inside" if inside else "outside")


def test_missing_pools_name_the_fetch_command(root, tmp_path):
    result = run_analyze(root, tmp_path / "no_pools", tmp_path / "out")
    assert result.returncode == 2
    assert "run: python -m fewshot_dlms.fetch pools java" in result.stderr + result.stdout


def test_plot_writes_the_figures(root, tmp_path):
    """plot.py draws Figure 4, its gold-reward row and Figure 11 from the committed tables alone (no pools, no GPU)."""
    command = [sys.executable, str(root / "experiments/fig4_single_target/plot.py"), "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    for name in ("fig4", "gold_reward_row", "fig11_beta_sweep"):
        assert (tmp_path / f"{name}.pdf").stat().st_size > 10_000, name


def test_run_seed_rule_is_the_replayed_one(root):
    """run.py's proposal seed of a single-target record is the seed with which tests/test_gpu_replay.py redraws the
    frozen continuations token for token; the cross-benchmark cells add the placement to the key."""
    run = load_module(root, "experiments/fig4_single_target/run.py", "fig4_run")
    replay = load_module(root, "tests/test_gpu_replay.py", "gpu_replay")
    from fewshot_dlms.data import read_jsonl

    for name in ("nemotron_ar", "qwen"):
        for record in read_jsonl(root / "data" / "fixtures" / "fig4" / f"{name}_sst2_M4.jsonl"):
            assert run.proposal_seed(name, record["episode_id"], record["M"], record["J"], False) == replay.ar_seed(name, record)
    single = run.proposal_seed("llada", "list_easy_fam:report:000", 4, 1, False)
    master = run.proposal_seed("llada", "list_easy_fam:report:000", 4, 1, True)
    assert single != master


def test_run_protocol_hash_does_not_depend_on_the_other_cells(root):
    """A cell's protocol hash is a function of that cell alone: requesting more cells or tasks does not change it."""
    run = load_module(root, "experiments/fig4_single_target/run.py", "fig4_run")
    def config(K=64, master=False):
        return run.cell_config("llada", "GSAI-ML/LLaDA-1.5", "rev", "report", "list_easy_fam", 4, 1, master, K, 32, False)

    assert run.protocol_hash(config()) == run.protocol_hash(config())
    assert run.protocol_hash(config()) != run.protocol_hash(config(K=4))
    assert run.protocol_hash(config()) != run.protocol_hash(config(master=True))
    assert not any(isinstance(value, list) for value in config().values())  # no list of requested cells or tasks


class ToyText:
    """A byte-level toy model for run.py on CPU: token = one byte of UTF-8 text, mask id 256, deterministic logits that
    favour the byte after the previous token. `kind` "dlm" fills canvases, "ar" continues prompts (newline stops)."""

    name = "toy"
    model_id = "toy"
    revision = "0"
    mask_id = 256
    V = 257
    tok = range(256)
    end_of_turn_id = 255
    default_support = "ordinary"
    sample_chunk = 32
    p0_chunk = 16

    def __init__(self, kind):
        self.kind = kind

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(token for token in ids if token < 256).decode("utf-8", errors="replace")

    def chat_header(self, instruction):
        return "<h>" + instruction

    def chat_user_open(self):
        return "<u>"

    def chat_user_close_assistant_open(self):
        return "<a>"

    def chat_end(self):
        return "<e>"

    def begin_reveal(self, order):
        return None

    def support(self, kind, codes=None):
        import torch

        mask = torch.zeros(self.V, dtype=torch.bool)
        mask[32:127] = True
        if kind == "free":
            mask[10] = True
            mask[self.end_of_turn_id] = True
        return mask

    def stop_ids(self):
        return {10, self.end_of_turn_id}

    def counted_logits(self, batch_ids, positions):
        import torch

        logits = torch.zeros(len(batch_ids), len(positions[0]), self.V)
        for b, (ids, where) in enumerate(zip(batch_ids.tolist(), positions)):
            for i, p in enumerate(where):
                previous = ids[p - 1] if p > 0 else 0
                logits[b, i, 32 + (previous + 1) % 95] += 2.0
                logits[b, i, 10] += 1.0
        return logits


def test_run_writes_complete_records_on_toy_models(root):
    """run.py's four run functions on a byte-level toy: K answers, P0 scores for each, the reference scored, the seed."""
    run = load_module(root, "experiments/fig4_single_target/run.py", "fig4_run")
    from fewshot_dlms.data import load_families

    family = load_families("list_easy_fam", "report")[0]
    gold_text = " " + family["targets"][0]["output"]
    for master, J in ((False, 1), (True, 2)):
        body = run.run_canvas(ToyText("dlm"), family, 2, J, master, 4, 32, True, True)
        assert len(body["candidates"]) == 4
        assert all(len(c["texts"]) == J and np.isfinite(c["ell0"]) for c in body["candidates"])
        assert body["oracle_L"][0] == len(gold_text.encode())
        assert body["gold_T"] == sum(body["oracle_L"])
        assert body["checks"]["replay_P1_max_abs_seq"] < 1e-9
        assert body["seed"] == run.proposal_seed("toy", family["episode_id"], 2, J, master)
    for model in ("nemotron_ar", "qwen", "qwen_perhole"):
        toy = ToyText("ar")
        body = run.run_causal(toy, model, family, 2, 4, 32, True, model != "qwen_perhole")
        assert len(body["candidates"]) == 4
        assert body["checks"]["replay_P1_max_abs_seq"] < 1e-9
        if model == "qwen":
            assert all(c["toks"][-1] in toy.stop_ids() or len(c["toks"]) == run.FREE_MAX_NEW for c in body["candidates"])
            assert body["gold_T"] == len(gold_text.encode()) + 1
        else:
            assert all(len(c["toks"]) == len(gold_text.encode()) for c in body["candidates"])
        if model == "qwen_perhole":
            assert "[[1]]" in body["checks"]["p1_prompt_text"]
            assert body["seed"] == run.proposal_seed(model, family["episode_id"], 2, 1, False, (("hole", 0),))
            with pytest.raises(ValueError):
                run.run_causal(toy, model, family, 2, 4, 32, True, True)
