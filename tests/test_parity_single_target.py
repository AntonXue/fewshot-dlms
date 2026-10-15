"""Analysis parity (fidelity level 2 of README.md) for Figure 4, its development-bank beta selection and the
appendix tables that share its pools (cross-benchmark and guidance-strength sweep figures).

experiments/single_target/analyze.py recomputes the paper's tables from the frozen pools and must
reproduce them to 1e-9 in every numeric column (and exactly in every other column):

  sec41_accuracy.csv     Figure 4 and Section 4.1: accuracy and paired gains per (model, benchmark, M, beta)
  gold_reward_cells.csv  gold example reward per target token per (model, benchmark, M) (Figure 4's gold row)
  dev_sec41.csv          the same cells on the development bank (beta selection)
  dev_sec42.csv          the cross-benchmark cells on the development bank (beta selection)
  sec44_power.csv        sharpening residuals: guidance against the matched-exponent power control (LLaDA-1.5)
  cells_crossbench.csv   cross-benchmark figure (Fig. 9, both builds), in data/tables/multi_target/: the cross-benchmark
                         study, M = 4, K = 64
  beta_sweep.csv         guidance-strength sweep figure (Fig. 10, arXiv version only): the guidance-strength sweep
                         on List Functions-Easy
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

ANALYZE = "experiments/single_target/analyze.py"
TOLERANCE = 1e-9
# table -> (reference table under data/tables, the columns that identify a row)
TABLES = {
    "sec41_accuracy": ("single_target/sec41_accuracy.csv", ["bank", "model", "source", "M", "beta"]),
    "gold_reward_cells": ("single_target/gold_reward_cells.csv", ["model", "source", "M"]),
    "dev_sec41": ("single_target/dev_sec41.csv", ["bank", "model", "source", "M", "beta"]),
    "dev_sec42": ("single_target/dev_sec42.csv", ["bank", "model", "source", "J", "beta"]),
    "sec44_power": ("single_target/sec44_power.csv", ["bank", "source", "J", "tau", "beta"]),
    "cells_crossbench": ("multi_target/cells_crossbench.csv", ["bank", "model", "source", "J", "beta"]),
    "beta_sweep": ("single_target/beta_sweep.csv", ["model", "J", "beta"]),
}
# The fields of a frozen lock file that the development bank determines.
LOCK_FIELDS = ["beta", "n_cells", "cells", "cell_set", "best_positive_below_ordinary", "partial_cells_excluded", "sections"]
COMPARE_LINE = re.compile(
    r"^(\S+) (\S+) M=(\d+): frozen accuracy ([0-9.]+), rerun CI \[([0-9.]+), ([0-9.]+)\] -> frozen mean (inside|outside)"
)
# The verdict of the same cell under guidance at beta = 0.5 (Figure 4's strength), printed after the ordinary one.
COMPARE_LINE_GUIDED = re.compile(
    r"^(\S+) (\S+) M=(\d+) beta=0\.5: frozen accuracy ([0-9.]+), rerun CI \[([0-9.]+), ([0-9.]+)\] -> frozen mean (inside|outside)"
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
    pools = artifacts / "pools" / "single_target"
    if not pools.exists():
        pytest.skip("Figure 4 pools not fetched: run `python -m fewshot_dlms.fetch pools java`")
    out = tmp_path_factory.mktemp("single_target")
    reference = root / "data" / "tables" / "single_target"
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
    frozen = json.loads((root / "data" / "tables" / "single_target" / "beta_locks.json").read_text())
    assert ours["rule"] == frozen["rule"]
    assert ours["betas"] == frozen["betas"]
    assert sorted(ours["models"]) == sorted(frozen["models"])
    for model, lock in frozen["models"].items():
        mine = ours["models"][model]
        for field in LOCK_FIELDS:
            assert mine[field] == lock[field], (model, field)
        assert mine.get("partial_cells_excluded_note") == lock.get("partial_cells_excluded_note"), model
        assert sorted(mine["dev_mean_by_beta"]) == sorted(lock["dev_mean_by_beta"])
        for beta, value in lock["dev_mean_by_beta"].items():
            assert abs(mine["dev_mean_by_beta"][beta] - value) < TOLERANCE, (model, beta)
        assert abs(mine["dev_gain_vs_ordinary_pp"] - lock["dev_gain_vs_ordinary_pp"]) < 100 * TOLERANCE, model
        assert mine["all_complete_cells"]["beta"] == lock["beta"], model


@pytest.mark.parity
def test_frozen_pools_lie_inside_their_own_intervals(analysis):
    """Compared with itself, every Section 4.1 cell reports the frozen mean inside the 95 % CI, at beta = 0 and 0.5."""
    _, stdout = analysis
    for pattern in (COMPARE_LINE, COMPARE_LINE_GUIDED):
        lines = [pattern.match(line) for line in stdout.splitlines()]
        verdicts = [match.group(7) for match in lines if match]
        assert len(verdicts) == 80
        assert set(verdicts) == {"inside"}


def test_compare_reports_inside_ci(root, tmp_path):
    """A resampled cell is judged by whether the frozen mean lies inside the rerun's 95 % CI.

    The two-family fixture pool of Qwen3-8B (SST-2, M = 4, K = 64) plays the rerun of that cell; --compare prints one
    verdict for ordinary sampling (beta = 0) and one for guidance at beta = 0.5.
    """
    pools = tmp_path / "pools" / "report" / "qwen"
    pools.mkdir(parents=True)
    shutil.copy(root / "data" / "fixtures" / "single_target" / "qwen_sst2_M4.jsonl", pools / "sst2_fam_M4_J1.jsonl")
    result = run_analyze(root, tmp_path / "pools", tmp_path / "out", "--compare", str(root / "data" / "tables" / "single_target"))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    paper = pd.read_csv(root / "data" / "tables" / "single_target" / "sec41_accuracy.csv")
    rerun = pd.read_csv(tmp_path / "out" / "sec41_accuracy.csv")
    for beta, pattern in ((0.0, COMPARE_LINE), (0.5, COMPARE_LINE_GUIDED)):
        matches = [match for match in map(pattern.match, lines) if match]
        assert len(matches) == 1, beta
        model, source, M, frozen, lo, hi, verdict = matches[0].groups()
        assert (model, source, M) == ("qwen", "sst2_fam", "4")

        paper_cell = paper[(paper.model == "qwen") & (paper.source == "sst2_fam") & (paper.M == 4) & (paper.beta == beta)].iloc[0]
        rerun_cell = rerun[rerun.beta == beta].iloc[0]
        assert rerun_cell.families == 2
        assert float(frozen) == pytest.approx(paper_cell.accuracy, abs=5e-4)
        assert float(lo) == pytest.approx(rerun_cell.ci_lo, abs=5e-4)
        assert float(hi) == pytest.approx(rerun_cell.ci_hi, abs=5e-4)
        inside = rerun_cell.ci_lo <= paper_cell.accuracy <= rerun_cell.ci_hi
        assert verdict == ("inside" if inside else "outside")
    guided = next(index for index, line in enumerate(lines) if COMPARE_LINE_GUIDED.match(line))
    assert COMPARE_LINE.match(lines[guided - 1])  # each cell's guided verdict follows its ordinary one


def test_missing_pools_name_the_fetch_command(root, tmp_path):
    result = run_analyze(root, tmp_path / "no_pools", tmp_path / "out")
    assert result.returncode == 2
    assert "run: python -m fewshot_dlms.fetch pools java" in result.stderr + result.stdout


@pytest.mark.parametrize("option", ["--crossbench", "--beta-sweep"])
def test_missing_optional_pools_name_the_input_and_the_fetch_command(root, tmp_path, option):
    """--crossbench and --beta-sweep need the cross-benchmark pools; without them analyze.py says which and exits 2."""
    pools = tmp_path / "pools" / "report" / "qwen"
    pools.mkdir(parents=True)
    shutil.copy(root / "data" / "fixtures" / "single_target" / "qwen_sst2_M4.jsonl", pools / "sst2_fam_M4_J1.jsonl")
    result = run_analyze(root, tmp_path / "pools", tmp_path / "out", option)
    assert result.returncode == 2
    assert option in result.stderr
    assert "_il.jsonl" in result.stderr
    assert "run: python -m fewshot_dlms.fetch pools java" in result.stderr
    assert not (tmp_path / "out").exists()


def test_plot_writes_the_figures(root, tmp_path):
    """plot.py draws Figure 4, its gold-reward row and the guidance-strength sweep figure from the committed tables
    alone (no pools, no GPU)."""
    command = [sys.executable, str(root / "experiments/single_target/plot.py"), "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    for name in ("single_target", "reference_reward_row", "guidance_strength"):
        assert (tmp_path / f"{name}.pdf").stat().st_size > 10_000, name


def test_run_seed_rule_is_the_replayed_one(root):
    """run.py's proposal seed of a single-target record is the seed with which tests/test_gpu_replay.py redraws the
    frozen continuations token for token; the cross-benchmark cells add the placement to the key."""
    run = load_module(root, "experiments/single_target/run.py", "single_target_run")
    replay = load_module(root, "tests/test_gpu_replay.py", "gpu_replay")
    from fewshot_dlms.data import read_jsonl

    for name in ("nemotron_ar", "qwen"):
        for record in read_jsonl(root / "data" / "fixtures" / "single_target" / f"{name}_sst2_M4.jsonl"):
            assert run.proposal_seed(name, record["episode_id"], record["M"], record["J"], False) == replay.ar_seed(name, record)
    single = run.proposal_seed("llada", "list_easy_fam:report:000", 4, 1, False)
    master = run.proposal_seed("llada", "list_easy_fam:report:000", 4, 1, True)
    assert single != master


def test_run_protocol_hash_does_not_depend_on_the_other_cells(root):
    """A cell's protocol hash is a function of that cell alone: requesting more cells or tasks does not change it."""
    run = load_module(root, "experiments/single_target/run.py", "single_target_run")
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
    """run.py's run functions on a byte-level toy: K answers, P0 scores for each, the reference scored, the seed."""
    run = load_module(root, "experiments/single_target/run.py", "single_target_run")
    from fewshot_dlms.data import load_families

    family = load_families("list_easy_fam", "report")[0]
    gold_text = " " + family["targets"][0]["output"]
    for master, J in ((False, 1), (True, 2)):
        body = run.run_canvas(ToyText("dlm"), "llada", family, 2, J, master, 4, 32, True, True)
        assert len(body["candidates"]) == 4
        assert all(len(c["texts"]) == J and np.isfinite(c["ell0"]) for c in body["candidates"])
        assert body["oracle_L"][0] == len(gold_text.encode())
        assert body["gold_T"] == sum(body["oracle_L"])
        assert body["checks"]["replay_P1_max_abs_seq"] < 1e-9
        assert body["seed"] == run.proposal_seed("llada", family["episode_id"], 2, J, master)
    for model in ("nemotron_ar", "qwen"):
        toy = ToyText("ar")
        body = run.run_causal(toy, model, family, 2, 4, True, True)
        assert len(body["candidates"]) == 4
        assert body["checks"]["replay_P1_max_abs_seq"] < 1e-9
        assert body["seed"] == run.proposal_seed(model, family["episode_id"], 2, 1, False)
        if model == "qwen":
            assert all(c["toks"][-1] in toy.stop_ids() or len(c["toks"]) == run.FREE_MAX_NEW for c in body["candidates"])
            assert body["gold_T"] == len(gold_text.encode()) + 1
        else:
            assert all(len(c["toks"]) == len(gold_text.encode()) for c in body["candidates"])
            assert body["gold_T"] == len(gold_text.encode())


# ---------------------------------------------------------------------------------------------
# run.py main(): resume, refusal, failures, manifest (toy model in place of the checkpoint)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def toy_run(root, monkeypatch, capsys):
    """run.py with the toy canvas model in place of a checkpoint and no GPU; returns (module, call) where
    call(out, *arguments) runs main() for LLaDA-1.5's List Functions-Easy cells and returns (exit code, stdout), or
    (exit code, stdout, stderr) with stderr=True. Every model load is counted in run.loads."""
    run = load_module(root, "experiments/single_target/run.py", "single_target_run_main")
    run.loads = []

    def toy_load(name, device):
        run.loads.append(name)
        return ToyText("dlm")

    monkeypatch.setattr(run, "load", toy_load)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)

    def call(out, *arguments, stderr=False):
        argv = ["run.py", "--model", "llada", "--tasks", "list_easy_fam", "--device", "cpu", "--out", str(out), *arguments]
        monkeypatch.setattr(sys, "argv", argv)
        code = 0
        try:
            run.main()
        except SystemExit as exit:
            code = exit.code
        captured = capsys.readouterr()
        if stderr:
            return code, captured.out, captured.err
        return code, captured.out

    return run, call


def pool_lines(out, name):
    return (out / "report" / "llada" / f"{name}.jsonl").read_text().splitlines()


def test_run_resumes_and_keeps_each_cells_hash(toy_run, tmp_path):
    """A second run into the same --out writes nothing; a wider run (more M, more families) resumes the M = 2 cell where
    it stopped, because a cell's protocol hash does not depend on the other cells requested."""
    run, call = toy_run
    out = tmp_path / "pools"
    code, stdout = call(out, "--M", "2", "--K", "4", "--families", "2")
    assert code == 0
    assert "CELL_DONE llada list_easy_fam_M2_J1 new=2" in stdout
    first = pool_lines(out, "list_easy_fam_M2_J1")
    assert len(first) == 2

    (path,) = (out / "report" / "llada").glob("manifest_*.json")
    session = json.loads(path.read_text())["sessions"][0]
    code, stdout = call(out, "--M", "2", "--K", "4", "--families", "2")
    assert code == 0
    assert "CELL_DONE llada list_easy_fam_M2_J1 new=0" in stdout
    assert pool_lines(out, "list_easy_fam_M2_J1") == first
    sessions = json.loads(path.read_text())["sessions"]  # the resumed run keeps the first session's record
    assert len(sessions) == 2 and sessions[0] == session

    code, stdout = call(out, "--M", "2,4", "--K", "4", "--families", "5")
    assert code == 0
    assert "REFUSED" not in stdout
    assert "CELL_DONE llada list_easy_fam_M2_J1 new=3" in stdout
    assert "CELL_DONE llada list_easy_fam_M4_J1 new=5" in stdout
    lines = pool_lines(out, "list_easy_fam_M2_J1")
    assert lines[:2] == first
    records = [json.loads(line) for line in lines]
    assert len({record["protocol_hash"] for record in records}) == 1
    assert [record["episode_id"] for record in records] == [f"list_easy_fam:report:{i:03d}" for i in range(5)]

    manifests = [json.loads(path.read_text()) for path in (out / "report" / "llada").glob("manifest_*.json")]
    assert len(manifests) == 2  # one per distinct set of requested cells
    manifest = next(m for m in manifests if "list_easy_fam_M4_J1" in m["cells"])
    assert set(manifest) == {"cells", "seeds", "sessions"}
    assert set(manifest["sessions"][0]) == {"versions", "command"}
    assert set(manifest["cells"]) == {"list_easy_fam_M2_J1", "list_easy_fam_M4_J1"}
    assert manifest["cells"]["list_easy_fam_M2_J1"]["protocol_hash"] == records[0]["protocol_hash"]
    assert manifest["seeds"]["list_easy_fam_M2_J1"]["list_easy_fam:report:000"] == records[0]["seed"]


def test_run_refuses_a_pool_of_another_protocol(toy_run, tmp_path):
    """Changing K changes the cell's protocol: the existing file is refused before the model is loaded, nothing is
    appended to any cell, the exit code is 2 and the message names the file."""
    run, call = toy_run
    out = tmp_path / "pools"
    assert call(out, "--M", "2", "--K", "4", "--families", "2")[0] == 0
    assert run.loads == ["llada"]
    before = pool_lines(out, "list_easy_fam_M2_J1")
    code, stdout, stderr = call(out, "--M", "4,2", "--K", "3", "--families", "2", stderr=True)
    assert code == 2
    assert f"refusing to append: {out / 'report' / 'llada' / 'list_easy_fam_M2_J1.jsonl'}" in stderr
    assert run.loads == ["llada"]
    assert "CELL_DONE" not in stdout
    assert pool_lines(out, "list_easy_fam_M2_J1") == before
    assert not (out / "report" / "llada" / "list_easy_fam_M4_J1.jsonl").exists()


def test_done_episodes_resumes_only_records_of_the_same_protocol(root, tmp_path):
    """done_episodes returns the families a run of the same protocol may skip (a cut-off last line is sampled again),
    and refuses a file that holds a record of another protocol, or one without a protocol hash (the frozen pools)."""
    run = load_module(root, "experiments/single_target/run.py", "single_target_run")
    config = run.cell_config("llada", "GSAI-ML/LLaDA-1.5", "rev", "report", "list_easy_fam", 2, 1, False, 4, 32, False)
    digest = run.protocol_hash(config)
    path = tmp_path / "pool.jsonl"
    assert run.done_episodes(path, digest) == set()
    good = [json.dumps({"episode_id": name, "protocol_hash": digest}) for name in ("a", "b")]
    path.write_text("\n".join(good) + '\n{"episode_id": "c", "protoc', encoding="utf-8")  # an interrupted write
    assert run.done_episodes(path, digest) == {"a", "b"}
    for other in ({"episode_id": "d", "protocol_hash": "0123456789abcdef"}, {"episode_id": "d"}):
        path.write_text(good[0] + "\n" + json.dumps(other) + "\n", encoding="utf-8")
        with pytest.raises(run.ProtocolMismatch, match=str(other.get("protocol_hash"))):
            run.done_episodes(path, digest)


def test_run_script_refuses_a_pool_of_another_protocol(root, tmp_path):
    """run.py, as a script, exits with status 2 naming the file before it loads a model when a cell's pool holds
    another protocol (here a hash of no possible configuration)."""
    pool = tmp_path / "report" / "llada" / "list_easy_fam_M4_J1.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({"episode_id": "list_easy_fam:report:000", "protocol_hash": "0123456789abcdef"}) + "\n")
    command = [sys.executable, str(root / "experiments/single_target/run.py"), "--model", "llada"]
    command += ["--tasks", "list_easy_fam", "--M", "4", "--K", "4", "--families", "2"]
    command += ["--out", str(tmp_path), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2, result.stderr
    assert str(pool) in result.stderr
    assert "loaded in" not in result.stdout


def test_run_counts_a_failed_family_and_retries_it(toy_run, tmp_path, monkeypatch):
    """A family that raises is reported and counted, the others are written, the exit code is 1; the next run writes
    only the failed family."""
    run, call = toy_run
    out = tmp_path / "pools"
    real = run.run_canvas

    def flaky(adapter, model, family, *arguments):
        if family["episode_id"] == "list_easy_fam:report:001":
            raise RuntimeError("injected failure")
        return real(adapter, model, family, *arguments)

    monkeypatch.setattr(run, "run_canvas", flaky)
    code, stdout = call(out, "--M", "2", "--K", "4", "--families", "3")
    assert code == 1
    assert "FAILED list_easy_fam:report:001" in stdout
    assert "RUN_FAILED llada: 1 families or cells failed" in stdout
    assert len(pool_lines(out, "list_easy_fam_M2_J1")) == 2

    monkeypatch.setattr(run, "run_canvas", real)
    code, stdout = call(out, "--M", "2", "--K", "4", "--families", "3")
    assert code == 0
    assert "CELL_DONE llada list_easy_fam_M2_J1 new=1" in stdout
    episodes = [json.loads(line)["episode_id"] for line in pool_lines(out, "list_easy_fam_M2_J1")]
    assert sorted(episodes) == [f"list_easy_fam:report:{i:03d}" for i in range(3)]
