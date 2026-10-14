"""Analysis parity (fidelity level 2 of README.md) for Figure 6 (Section 4.3) and its appendix fidelity row.

experiments/fig6_lengths/analyze.py recomputes the paper's Section 4.3 tables from the frozen pools and the frozen
head's distributions, and must reproduce data/tables/fig6/ to 1e-9 in every numeric column (and exactly in every
other column):

  sec43_length_metrics.csv       per backbone and J: the learned head's mass at |L - L*| <= 0, 1, 2 (P0, P1, P2)
                                 with family-bootstrap intervals, NLL, E|L - L*|, mode exact, ECE; and the
                                 visible-demonstration length prior (appendix fidelity row)
  sec43_downstream.csv           Figure 6, top row: A_avg (A_target) with oracle vs sampled learned lengths against J
  sec43_length_perturbation.csv  Figure 6, bottom row: A_avg with L = L* + delta at every target, J = 1, 2

The parity tests need the fetched pools (`python -m fewshot_dlms.fetch pools java`). The other tests run on every
`make test`: analyze.py on a small synthetic pool, the message for missing pools, plot.py on the committed tables,
run.py's seed and offset rules, and the frozen configuration. Pixel identity of the figures with the paper's PDFs is
checked by hand (README of the experiment).
"""
import importlib.util
import json
import re
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from fewshot_dlms.data import load_families
from fewshot_dlms.data import read_jsonl
from fewshot_dlms.lengths import length_mass
from fewshot_dlms.lengths import site_metrics

ANALYZE = "experiments/fig6_lengths/analyze.py"
TOLERANCE = 1e-9
KEYS = {
    "sec43_length_metrics": ["backbone", "J", "run"],
    "sec43_downstream": ["backbone", "condition", "J"],
    "sec43_length_perturbation": ["backbone", "J", "delta", "family_set"],
}
COMPARE_LINE = re.compile(
    r"^(\S+) (\S+) J=(\d+): frozen A_avg ([0-9.]+), rerun CI \[([0-9.]+), ([0-9.]+)\] -> frozen mean (inside|outside)"
)
MODELS = ["llada", "nemotron_pt", "dqwen"]


def run_analyze(root, pools, out, *extra):
    command = [sys.executable, str(root / ANALYZE), "--pools", str(pools), "--out", str(out), *extra]
    return subprocess.run(command, capture_output=True, text=True, cwd=root)


def load_module(root, name):
    """An experiment script imported as a module (run.py and analyze.py are scripts, not package modules)."""
    spec = importlib.util.spec_from_file_location(f"fig6_{name}", root / "experiments" / "fig6_lengths" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analysis(root, artifacts, tmp_path_factory):
    """Run analyze.py once on the frozen pools; returns (output directory, stdout of --compare)."""
    pools = artifacts / "pools" / "fig6"
    if not pools.exists():
        pytest.skip("Figure 6 pools not fetched: run `python -m fewshot_dlms.fetch pools java`")
    out = tmp_path_factory.mktemp("fig6")
    result = run_analyze(root, pools, out, "--compare", str(root / "data" / "tables" / "fig6"))
    assert result.returncode == 0, result.stderr
    return out, result.stdout


def max_abs_diff(ours, paper, keys):
    """Largest |ours - paper| over the numeric columns after aligning rows on `keys`.

    Columns must agree in name and order. Non-numeric columns must be equal; a NaN must sit where the paper has one.
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
    paper = pd.read_csv(root / "data" / "tables" / "fig6" / f"{name}.csv", float_precision="round_trip")
    assert max_abs_diff(ours, paper, KEYS[name]) < TOLERANCE, name


@pytest.mark.parity
def test_frozen_pools_lie_inside_their_own_intervals(analysis):
    """Compared with itself, every downstream cell reports the frozen mean inside the 95 % CI."""
    _, stdout = analysis
    lines = [COMPARE_LINE.match(line) for line in stdout.splitlines()]
    verdicts = [match.group(7) for match in lines if match]
    assert len(verdicts) == 24
    assert set(verdicts) == {"inside"}


@pytest.mark.parity
@pytest.mark.parametrize("model", MODELS)
def test_offset_pools_hold_exactly_the_families_run_keeps(root, artifacts, model):
    """The forced-offset pools skip a family when some L*_j + delta < 1 (never clipped); run.py keeps the same families."""
    run = load_module(root, "run")
    pools = artifacts / "pools" / "fig6" / model
    if not (pools / "length_head_report.jsonl").exists():
        pytest.skip(f"Figure 6 pools of {model} not fetched: run `python -m fewshot_dlms.fetch pools java`")
    oracle = {(row["episode_id"], row["J"]): row["oracle"] for row in read_jsonl(pools / "length_head_report.jsonl")}
    family_ids = [family["episode_id"] for family in load_families("list_easy_fam", "report")]
    for J in (1, 2):
        for delta in (-2, -1, 1, 2):
            path = pools / f"list_easy_fam_M4_J{J}_{run.condition_name('offset', delta)}.jsonl"
            stored = {record["episode_id"] for record in read_jsonl(path)}
            kept = {fid for fid in family_ids if run.offset_lengths(oracle[(fid, J)], delta) is not None}
            assert stored == kept, (model, J, delta)
            for record in read_jsonl(path):
                expected = [length + delta for length in record["oracle_L"]]
                assert all(candidate["alloc_L"] == expected for candidate in record["candidates"])


def write_pool(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def synthetic_record(family, J, texts, allocations):
    """A pool record in run.py's format: K candidates with the texts of their J targets and their lengths."""
    oracle_L = [len(target["output"]) for target in family["targets"][:J]]
    candidates = []
    for candidate_texts, alloc in zip(texts, allocations):
        candidates.append({"toks": [], "texts": candidate_texts, "ell1": 0.0, "ell0": float("nan"), "alloc_L": alloc})
    return {
        "episode_id": family["episode_id"],
        "model": "llada",
        "M": 4,
        "J": J,
        "K": len(candidates),
        "grader": family["grader"],
        "gold": [target["output"] for target in family["targets"][:J]],
        "oracle_L": oracle_L,
        "n_tokens": 100,
        "candidates": candidates,
    }


def test_analyze_on_a_synthetic_pool(root, tmp_path):
    """The downstream metrics of two families, oracle vs learned lengths at J = 1, computed by hand."""
    families = load_families("list_easy_fam", "report")[:2]
    pools = tmp_path / "pools"
    oracle = []
    learned = []
    for family in families:
        gold = " " + family["targets"][0]["output"]
        L = len(family["targets"][0]["output"])
        oracle.append(synthetic_record(family, 1, [[gold], [gold], [" [99]"], [gold]], [[L]] * 4))
        learned.append(synthetic_record(family, 1, [[gold], [" ["], [gold], [gold + "]"]], [[L], [L - 1], [L], [L + 2]]))
    write_pool(pools / "llada" / "list_easy_fam_M4_J1_oracle.jsonl", oracle)
    write_pool(pools / "llada" / "list_easy_fam_M4_J1_learned.jsonl", learned)
    result = run_analyze(root, pools, tmp_path / "out", "--compare", str(root / "data" / "tables" / "fig6"))
    assert result.returncode == 0, result.stderr

    table = pd.read_csv(tmp_path / "out" / "sec43_downstream.csv").set_index("condition")
    assert list(table.index) == ["oracle", "learned"]
    assert table.loc["oracle", "A_target"] == 0.75
    assert table.loc["learned", "A_target"] == 0.5
    assert table.loc["learned", "length_exact"] == 0.5
    assert table.loc["learned", "diff_vs_oracle"] == -0.25
    assert table.loc["learned", "format_failure"] == 0.5
    assert table.loc["learned", "pr_L_ne_Lstar_given_correct"] == 0.0
    assert table.loc["oracle", "families"] == 2
    assert table.loc["learned", "K"] == 4
    lengths = [len(family["targets"][0]["output"]) for family in families]
    assert table.loc["learned", "generated_tokens_per_family"] == np.mean([4 * L + 1 for L in lengths])
    assert table.loc["learned", "oracle_tokens_per_family_per_candidate"] == np.mean(lengths)

    verdicts = [COMPARE_LINE.match(line) for line in result.stdout.splitlines()]
    verdicts = [match.groups() for match in verdicts if match]
    assert [(model, condition, J) for model, condition, J, *_ in verdicts] == [("llada", "oracle", "1"), ("llada", "learned", "1")]


def test_missing_pools_name_the_fetch_command(root, tmp_path):
    result = run_analyze(root, tmp_path / "no_pools", tmp_path / "out")
    assert result.returncode == 2
    assert "run: python -m fewshot_dlms.fetch pools java" in result.stderr + result.stdout


def test_plot_writes_the_figures(root, tmp_path):
    """plot.py draws Figure 6 and the appendix fidelity row from the committed tables alone (no pools, no GPU)."""
    command = [sys.executable, str(root / "experiments/fig6_lengths/plot.py"), "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    for name in ("fig6", "fidelity_row"):
        assert (tmp_path / f"{name}.pdf").stat().st_size > 10_000, name


def test_run_seed_rule_matches_the_authors(root):
    """run.py's content seed is the authors' (dr_worker.seed_of, master placement): values computed with their code."""
    run = load_module(root, "run")
    assert run.proposal_seed("llada", "list_easy_fam:report:000", 4, 1) == 1904909867
    assert run.proposal_seed("nemotron_pt", "list_easy_fam:report:005", 4, 2) == 1190557220
    assert run.proposal_seed("dqwen", "list_easy_fam:report:063", 4, 8) == 804057039


def test_offset_rule_skips_instead_of_clipping(root):
    """A forced offset that would leave a target with fewer than one token skips the family (the paper's rule)."""
    run = load_module(root, "run")
    assert run.offset_lengths([3, 1], -1) is None
    assert run.offset_lengths([3, 2], -1) == [2, 1]
    assert run.offset_lengths([3, 2], 2) == [5, 4]
    assert run.offset_lengths([3, 2], 0) == [3, 2]
    assert run.condition_name("offset", -2) == "delta-2"
    assert run.condition_name("offset", 0) == "delta0"
    assert run.condition_name("offset", 1) == "delta+1"
    assert run.condition_name("learned", 0) == "learned"


def test_length_mass_and_site_metrics():
    """P_r is the mass within r tokens of L*, cut at the ends of the support; site_metrics summarises it."""
    probs = np.array([[0.1, 0.2, 0.3, 0.4, 0.0], [0.5, 0.25, 0.0, 0.0, 0.25]])
    lengths = np.array([3, 1])
    assert np.allclose(length_mass(probs, lengths, 0), [0.3, 0.5])
    assert np.allclose(length_mass(probs, lengths, 1), [0.9, 0.75])
    assert np.allclose(length_mass(probs, lengths, 2), [1.0, 0.75])
    metrics = site_metrics(probs, lengths, lmax=5)
    assert metrics["sites"] == 2
    assert metrics["P0"] == pytest.approx(0.4)
    assert metrics["P1"] == pytest.approx(0.825)
    assert metrics["mode_exact"] == 0.5  # site 1's mode is 4, site 2's is 1
    assert metrics["nll_site"] == pytest.approx(-(np.log(0.3) + np.log(0.5)) / 2)
    assert metrics["eae"] == pytest.approx(((0.2 + 0.2 + 0.4) + (0.25 + 4 * 0.25)) / 2)
    out_of_support = site_metrics(probs, np.array([3, 9]), lmax=5)
    assert out_of_support["out_of_support"] == 0.5
    assert out_of_support["P0"] == pytest.approx(0.15)


def run_config_for(run, K):
    """run.py's configuration of an oracle run of LLaDA-1.5 at J = 1, 2 with K candidates (no GPU, no model)."""
    args = run.argparse.Namespace(model="llada", mode="oracle", delta=[0], J=[1, 2], K=K, chunk=run.CHUNK)
    return run.run_config(args, "GSAI-ML/LLaDA-1.5", "rev", None)


def test_pool_protocol_depends_on_what_decides_the_records(root):
    """A pool's protocol hash changes with K (and mode, head, ...), not with the other pools a run also samples."""
    run = load_module(root, "run")
    small = run.protocol_hash(run.pool_config(run_config_for(run, 4), "oracle", 1, 4))
    paper = run.protocol_hash(run.pool_config(run_config_for(run, 64), "oracle", 1, 64))
    assert small != paper
    args = run.argparse.Namespace(model="llada", mode="oracle", delta=[0], J=[1], K=64, chunk=run.CHUNK)
    only_J1 = run.protocol_hash(run.pool_config(run.run_config(args, "GSAI-ML/LLaDA-1.5", "rev", None), "oracle", 1, 64))
    assert only_J1 == paper


def test_resume_skips_only_records_of_the_same_protocol(root, tmp_path):
    """done_episodes returns the families a run of the same protocol may skip, and refuses a file of another protocol."""
    run = load_module(root, "run")
    digest = run.protocol_hash(run.pool_config(run_config_for(run, 4), "oracle", 1, 4))
    path = tmp_path / "pool.jsonl"
    assert run.done_episodes(path, digest) == set()
    write_pool(path, [{"episode_id": "a", "protocol_hash": digest}, {"episode_id": "b", "protocol_hash": digest}])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"episode_id": "c", "protoc')  # an interrupted write: that family is sampled again
    assert run.done_episodes(path, digest) == {"a", "b"}
    write_pool(path, [{"episode_id": "a", "protocol_hash": digest}, {"episode_id": "b", "protocol_hash": "0123456789abcdef"}])
    with pytest.raises(run.ProtocolMismatch, match="0123456789abcdef"):
        run.done_episodes(path, digest)


def test_run_refuses_a_pool_of_another_protocol(root, tmp_path):
    """run.py exits with status 2, naming the file, before it loads a model, when its pool holds another protocol."""
    pool = tmp_path / "llada" / "list_easy_fam_M4_J1_oracle.jsonl"
    write_pool(pool, [{"episode_id": "list_easy_fam:report:000", "protocol_hash": "0123456789abcdef"}])
    command = [sys.executable, str(root / "experiments/fig6_lengths/run.py"), "--model", "llada", "--mode", "oracle"]
    command += ["--J", "1", "--K", "4", "--families", "2", "--out", str(tmp_path), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2, result.stderr
    assert str(pool) in result.stderr
    assert "loaded in" not in result.stdout


def test_learned_run_without_its_head_names_the_fetch_command(root, tmp_path):
    """run.py --mode learned exits with status 2 and the fetch command, before it loads a model, when the head file is
    missing."""
    command = [sys.executable, str(root / "experiments/fig6_lengths/run.py"), "--model", "llada", "--mode", "learned"]
    command += ["--head", str(tmp_path / "missing.pt"), "--J", "1", "--families", "1", "--out", str(tmp_path), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2, result.stderr
    assert "python -m fewshot_dlms.fetch heads" in result.stderr
    assert "Traceback" not in result.stderr
    assert "loaded in" not in result.stdout


def test_run_counts_a_family_whose_lengths_fail(root, tmp_path, monkeypatch, capsys):
    """A family whose lengths or canvas raise (here oracle_lengths) is reported and counted like a failed draw: the
    others are written and the run exits with status 1 (no model: load and the per-family steps are replaced)."""
    run = load_module(root, "run")
    adapter = types.SimpleNamespace(model_id="GSAI-ML/LLaDA-1.5", revision="rev")
    monkeypatch.setattr(run, "load", lambda name, device: adapter)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run.torch.cuda, "empty_cache", lambda: None)

    def lengths(adapter, family, J):
        if family["episode_id"] == "list_easy_fam:report:001":
            raise RuntimeError("injected failure")
        return [1] * J

    body = {"n_unique_alloc": 1, "n_unique": 1, "seconds_gen": 0.0, "checks": None}
    monkeypatch.setattr(run, "oracle_lengths", lengths)
    monkeypatch.setattr(run, "build_canvas", lambda *arguments: None)
    monkeypatch.setattr(run, "run_family", lambda *arguments, **options: body)
    argv = ["run.py", "--model", "llada", "--mode", "oracle", "--J", "1", "--K", "2", "--families", "3"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(tmp_path), "--device", "cpu"])
    with pytest.raises(SystemExit) as exit:
        run.main()
    stdout = capsys.readouterr().out
    assert exit.value.code == 1
    assert "FAILED list_easy_fam:report:001" in stdout and "failed families: 1" in stdout
    pool = tmp_path / "llada" / "list_easy_fam_M4_J1_oracle.jsonl"
    assert [record["episode_id"] for record in read_jsonl(pool)] == ["list_easy_fam:report:000", "list_easy_fam:report:002"]


def test_frozen_config_has_no_absolute_paths(root):
    """experiments/fig6_lengths/frozen_config.json is the authors' freeze record with machine paths removed."""
    text = (root / "experiments" / "fig6_lengths" / "frozen_config.json").read_text(encoding="utf-8")
    config = json.loads(text)
    assert not re.search(r'"/[A-Za-z]', text), "a JSON string starts with an absolute path"
    assert not re.search(r"[\s(]/[A-Za-z]+/", text), "a text field names an absolute path"
    assert {name: entry["run"] for name, entry in config["backbones"].items()} == {
        "llada": "fin_llada_m70_100k",
        "nemotron": "fin_nemotron_m70_100k",
        "dqwen": "fin_dqwen_m70_100k_g2",
    }
    assert config["protocol"]["K_by_J"] == {"1": 64, "2": 32, "4": 16, "8": 8}
