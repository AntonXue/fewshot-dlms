"""Analysis parity (fidelity level 2 of README.md) for Figure 7 (Section 5) and its Appendix C tables.

experiments/fig7_java/analyze.py recomputes the paper's Java tables from the frozen pools, the judge's verdicts
and the length head's predictions, and must reproduce data/tables/fig7/ to 1e-9 in every numeric column (and
exactly in every other column):

  summary_by_cell.csv          Figure 7 (right) and Table 1 (Appendix C): joint judged success per cell, regime, rule
  paired_intervals.csv         guidance minus ordinary, learned minus oracle, paired commit bootstrap
  metrics_by_commit.csv        commit means per cell and regime
  strata.csv                   the same as summary_by_cell.csv in the strata of Appendix C
  beta_sensitivity.csv         the guidance-strength grid of Appendix C
  length_fidelity_by_bin.csv   Figure 7 (left): length mass within +-2 tokens by reference length, head C vs prior
  length_fidelity.csv          the same over all 80 unique target sites
  zero_demo_main_table.csv     the zero-demonstration ablation of Appendix C (M = 0 vs M = 1)
  pool_taxonomy_estimates.csv  binary pool classes and their estimates (Appendix C)
  checkstyle_case.json         the Checkstyle pool of Section 5 (numbers to 1e-6)

The parity tests need the fetched Java artifacts (`python -m fewshot_dlms.fetch pools java`). They also check
that judge.py rebuilds every frozen judge packet byte for byte and that run.py's seed and length-sampling rules
give the stored seeds and allocations. The other tests run on every `make test`.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fewshot_dlms.data import load_java_episodes
from fewshot_dlms.data import read_jsonl

ANALYZE = "experiments/fig7_java/analyze.py"
TOLERANCE = 1e-9
JSON_TOLERANCE = 1e-6
KEYS = {
    "summary_by_cell": ["stratum", "cell", "regime", "rule"],
    "paired_intervals": ["stratum", "cell", "contrast"],
    "metrics_by_commit": ["cell", "regime", "commit10"],
    "strata": ["stratum", "cell", "regime", "rule"],
    "beta_sensitivity": ["regime", "beta"],
    "length_fidelity_by_bin": ["population", "head", "bin"],
    "length_fidelity": ["population", "head"],
    "zero_demo_main_table": ["stratum", "block"],
    "pool_taxonomy_estimates": ["regime", "subset"],
}
TEXT_COLUMNS = {"commit10": str}  # commit ids such as 08174e2019 must not be read as numbers


def load_script(root, name):
    """An experiment script of experiments/fig7_java as a module (they are scripts, not a package)."""
    path = root / "experiments" / "fig7_java" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"fig7_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_analyze(root, pools, out, *extra):
    command = [sys.executable, str(root / ANALYZE), "--pools", str(pools), "--out", str(out), *extra]
    return subprocess.run(command, capture_output=True, text=True, cwd=root)


@pytest.fixture(scope="module")
def java(artifacts):
    """The fetched Java artifacts: pools/, judging/, length/, zero_demo/."""
    path = artifacts / "java"
    if not (path / "pools").exists():
        pytest.skip("Java artifacts not fetched: run `python -m fewshot_dlms.fetch pools java`")
    return path


@pytest.fixture(scope="module")
def analysis(root, java, tmp_path_factory):
    """Run analyze.py once on the frozen artifacts; returns its output directory."""
    out = tmp_path_factory.mktemp("fig7")
    result = run_analyze(root, java / "pools", out, "--compare", str(root / "data" / "tables" / "fig7"))
    assert result.returncode == 0, result.stderr
    return out


def read_table(path):
    return pd.read_csv(path, float_precision="round_trip", dtype=TEXT_COLUMNS)


def max_abs_diff(ours, paper, keys):
    """Largest |ours - paper| over the numeric columns after aligning rows on `keys`.

    Columns must agree in name and order. Non-numeric columns must be equal; a NaN must sit where the paper
    has one.
    """
    assert list(ours.columns) == list(paper.columns)
    assert not paper.duplicated(keys).any()
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


def assert_json_close(ours, paper, where="checkstyle_case"):
    """Numbers within JSON_TOLERANCE, everything else equal, recursively."""
    if isinstance(paper, dict):
        assert sorted(ours) == sorted(paper), where
        for key in paper:
            assert_json_close(ours[key], paper[key], f"{where}.{key}")
    elif isinstance(paper, list):
        assert len(ours) == len(paper), where
        for i, (a, b) in enumerate(zip(ours, paper)):
            assert_json_close(a, b, f"{where}[{i}]")
    elif isinstance(paper, bool) or paper is None or isinstance(paper, str):
        assert ours == paper, where
    else:
        assert abs(ours - paper) <= JSON_TOLERANCE, (where, ours, paper)


@pytest.mark.parity
@pytest.mark.parametrize("name", list(KEYS))
def test_table_matches_paper(root, analysis, name):
    ours = read_table(analysis / f"{name}.csv")
    paper = read_table(root / "data" / "tables" / "fig7" / f"{name}.csv")
    assert max_abs_diff(ours, paper, KEYS[name]) < TOLERANCE, name


@pytest.mark.parity
def test_checkstyle_case_matches_paper(root, analysis):
    ours = json.loads((analysis / "checkstyle_case.json").read_text())
    paper = json.loads((root / "data" / "tables" / "fig7" / "checkstyle_case.json").read_text())
    assert_json_close(ours, paper)


@pytest.mark.parity
def test_judge_rebuilds_every_frozen_packet(root, java):
    """judge.py's packets are byte for byte the packets the paper's judge saw (2,089 report packets)."""
    judge = load_script(root, "judge")
    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    index = {record["key"]: record for record in read_jsonl(java / "judging" / "packet_index.jsonl")}
    packets = judge.report_packets(judge.read_pools(java / "pools"), episodes, judge.rubric_sha(judge.RUBRIC))
    assert len(packets) == len(index) == 2089
    assert {packet["key"] for packet in packets} == set(index)
    for packet in packets:
        frozen = index[packet["key"]]
        assert judge.sha256_text(packet["text"]) == frozen["packet_sha256"], packet["key"]
        assert packet["label_map"] == frozen["label_map"]
        assert packet["episode_id"] == frozen["episode_id"]


@pytest.mark.parity
def test_judge_rebuilds_the_calibration_packets(root, java):
    """The 46 calibration controls of Appendix C: the same packet bytes as the paper's calibration."""
    judge = load_script(root, "judge")
    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    controls = read_jsonl(java / "judging" / "calibration" / "controls.jsonl")
    assert len(controls) == 46
    for control in controls:
        text, labels = judge.calibration_packet(episodes[control["episode_id"]], control)
        assert judge.sha256_text(text) == control["packet_sha256"], control["n"]
        assert labels == control["label_map"]


@pytest.mark.parity
def test_judge_rebuilds_the_zero_demo_packets(root, java):
    """The zero-demonstration ablation judged every arm with the anchor's first demonstration (698 packets)."""
    judge = load_script(root, "judge")
    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    index = read_jsonl(java / "zero_demo" / "judging" / "packet_index.jsonl")
    assert len(index) == 698
    for frozen in index:
        anchor = episodes[frozen["anchor_episode_id"]]
        key = judge.packet_key(judge.rubric_sha(judge.RUBRIC), anchor, frozen["candidate_texts"])
        assert key == frozen["key"]
        text, labels = judge.packet(anchor, frozen["candidate_texts"], judge.packet_seed(key))
        assert judge.sha256_text(text) == frozen["packet_sha256"], frozen["n"]
        assert labels == frozen["label_map"]


@pytest.mark.parity
def test_learned_allocations_are_redrawn_from_the_stored_head_predictions(root, java):
    """run.py's head-C sampler (Section 5): from the stored predictions, every stored allocation of the learned pools."""
    run = load_script(root, "run")
    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    probs = run.read_head_probs(java / "length" / run.HEAD_PROBS_FILE)
    pools = load_script(root, "analyze").read_main_pools(java / "pools")
    learned = [record for (episode_id, regime), record in sorted(pools.items()) if regime == "learned"]
    assert len(learned) == 127
    for record in learned:
        episode = episodes[record["episode_id"]]
        allocations, rejections = run.learned_allocations(episode, probs, record["K"])
        stored = {candidate["draw"]: candidate["allocation"] for candidate in record["candidates"]}
        assert [list(allocation) for allocation in allocations] == [stored[k] for k in range(record["K"])]
        assert rejections == record["allocation_rejections"]
        assert len(set(allocations)) == record["distinct_allocations"]


@pytest.mark.parity
def test_zero_demo_units_and_seeds_match_the_stored_pools(root, java):
    """run.py's zero-demonstration units (80 anchors x M in {0, 1}, 47 x M = 2) and their seeds are the stored ones."""
    run = load_script(root, "run")
    units = {unit["unit_id"]: unit for unit in run.zero_demo_units(load_java_episodes(), [0, 1, 2], [1, 2])}
    stored = {}
    for path in sorted((java / "zero_demo" / "pools").glob("pools_*.jsonl")):
        for record in read_jsonl(path):
            stored.setdefault(record["unit_id"], record)
    assert sorted(units) == sorted(stored)
    assert len(units) == 207
    for unit_id, record in stored.items():
        unit = units[unit_id]
        assert unit["arm_source_episode"] == record["arm_source_episode"]
        assert unit["target_sites"] == record["target_sites"]
        assert run.zero_demo_seed(unit) == record["seed"]


@pytest.mark.parity
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="calls the Anthropic API: needs ANTHROPIC_API_KEY")
def test_judge_api_on_two_calibration_controls(root, java):
    """judge.py through the Anthropic API on the first two calibration controls (a reference rewrite, expected
    correct, and a no-op, expected incorrect), as the paper's judge called them. Costs two API calls."""
    anthropic = pytest.importorskip("anthropic")
    judge = load_script(root, "judge")
    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    controls = read_jsonl(java / "judging" / "calibration" / "controls.jsonl")[:2]
    client = judge.make_client()
    for control in controls:
        text, labels = judge.calibration_packet(episodes[control["episode_id"]], control)
        try:
            reply = judge.call_judge(client, judge.MODEL, judge.RUBRIC.read_text(), text)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as error:
            pytest.skip(f"the API key was rejected: {error}")
        except anthropic.BadRequestError as error:
            if "credit balance" in str(error):
                pytest.skip(f"the API account has no credit: {error}")
            raise
        verdict = judge.parse_verdict(reply, labels)
        assert verdict["joint"] == control["expected"], (control["control"], verdict)


def test_missing_pools_name_the_fetch_command(root, tmp_path):
    result = run_analyze(root, tmp_path / "no_pools", tmp_path / "out")
    assert result.returncode == 2
    assert "run: python -m fewshot_dlms.fetch pools java" in result.stderr + result.stdout


def test_plot_writes_figure_7(root, tmp_path):
    """plot.py draws Figure 7 from the committed tables alone (no pools, no GPU)."""
    command = [sys.executable, str(root / "experiments/fig7_java/plot.py"), "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "fig7.pdf").stat().st_size > 10_000


def test_verdict_parsing_follows_the_paper(root):
    """judge.parse_verdict reads the JSON object out of a reply and maps the shuffled labels back to t1, t2."""
    judge = load_script(root, "judge")
    reply = (
        "Here is my verdict.\n"
        '{"targets": [{"label": "Target 1", "verdict": "correct", "failure_tag": "none", "reason": "ok"},'
        ' {"label": "Target 2", "verdict": "incorrect", "failure_tag": "wrong_argument", "reason": "swapped"}],'
        ' "joint": "incorrect", "instance_ill_posed": false, "ill_posed_reason": ""}\nDone.'
    )
    verdict = judge.parse_verdict(reply, {"Target 1": "t2", "Target 2": "t1"})
    assert verdict["joint"] == "incorrect"
    assert verdict["per_target"] == {"t2": "correct", "t1": "incorrect"}
    assert verdict["failure_tags"] == {"t2": "none", "t1": "wrong_argument"}
    assert verdict["instance_ill_posed"] is False
    with pytest.raises(ValueError):
        judge.parse_verdict('{"targets": [], "joint": "maybe"}', {"Target 1": "t1"})


def test_packet_key_and_seed_rules(root):
    """The judge's cache key and target-order seed (fr_packets.py), and the rubric's hash prefix."""
    judge = load_script(root, "judge")
    episode = {"episode_id": "E", "demo_sites": ["a@1", "b@2"]}
    texts = {"t2": "y", "t1": "x"}
    text = '7a893ac4a9d60950|E|a@1|b@2|{"t1": "x", "t2": "y"}'
    key = hashlib.sha256(text.encode()).hexdigest()[:20]
    assert judge.packet_key("7a893ac4a9d60950", episode, texts) == key
    assert judge.packet_seed(key) == int(hashlib.sha256(f"s5final-judge|{key}".encode()).hexdigest()[:12], 16)
    assert judge.rubric_sha(judge.RUBRIC) == "7a893ac4a9d60950"


def test_run_seed_rules(root):
    """run.py's generation seeds: SHA-256 of the '|'-joined parts, first 12 hex digits, mod 2**31."""
    run = load_script(root, "run")
    expected = int(hashlib.sha256(b"0e1d3c9040b09598|s5final|oracle|E|(13,)").hexdigest()[:12], 16) % 2**31
    assert run.generation_seed("oracle", "E", [13]) == expected
    expected = int(hashlib.sha256(b"0e1d3c9040b09598|s5final|learned|E|(9, 31)").hexdigest()[:12], 16) % 2**31
    assert run.generation_seed("learned", "E", [9, 31]) == expected
    unit = {"anchor_episode_id": "A", "target_sites": ["f@1", "g@2"], "M": 0}
    expected = int(hashlib.sha256(b"c2a07e4a701f0a41|demo_ablation|A|f@1+g@2|0").hexdigest()[:12], 16) % 2**31
    assert run.zero_demo_seed(unit) == expected


def test_length_draw_rule(root):
    """Head C's allocation draw: u = SHA-256('s5-final-alloc-v1' | protocol hash | episode | J | k | j | redraw)."""
    run = load_script(root, "run")
    text = "s5-final-alloc-v1|0e1d3c9040b09598|E|2|3|1|0"
    u = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) / 2**64
    assert run.allocation_uniform("E", 2, 3, 1, 0) == u
    probs = np.zeros((1, 256))
    probs[0, 4] = 1.0  # all mass on length 5
    episode = {"episode_id": "E", "J": 1, "oracle_lengths": [7], "oracle_canvas_tokens": 100}
    allocations, rejections = run.learned_allocations(episode, {"E|1|0": probs[0]}, 3)
    assert allocations == [(5,), (5,), (5,)]
    assert rejections == 0
    episode = {"episode_id": "E", "J": 1, "oracle_lengths": [7], "oracle_canvas_tokens": 2048}
    uniform = np.ones(256)
    allocations, rejections = run.learned_allocations(episode, {"E|1|0": uniform}, 4)
    assert all(2048 - 7 + a[0] <= run.CTX_CAP for a in allocations)  # draws above the canvas cap are redrawn
    assert rejections > 0
