"""Analysis parity (fidelity level 2 of README.md) for Figure 7 (Section 5) and its Appendix C tables.

experiments/java_refactoring/analyze.py recomputes the paper's Java tables from the frozen pools, the judge's verdicts
and the length head's predictions, and must reproduce data/tables/java_refactoring/ to 1e-9 in every numeric column (and
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
give the stored seeds and allocations. One test calls the Anthropic API (marker `api`, `make test-api`); the
others run on every `make test`.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from fewshot_dlms.data import load_java_episodes
from fewshot_dlms.data import read_jsonl

ANALYZE = "experiments/java_refactoring/analyze.py"
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
    """An experiment script of experiments/java_refactoring as a module (they are scripts, not a package)."""
    path = root / "experiments" / "java_refactoring" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"java_refactoring_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_analyze(root, pools, out, *extra):
    command = [sys.executable, str(root / ANALYZE), "--pools", str(pools), "--out", str(out), *extra]
    return subprocess.run(command, capture_output=True, text=True, cwd=root)


@pytest.fixture(scope="module")
def java(artifact_root):
    """The fetched Java artifacts: pools/, judging/, length/, zero_demo/ (the java group alone is enough)."""
    path = artifact_root / "java"
    if not (path / "pools").exists():
        pytest.skip("Java artifacts not fetched: run `python -m fewshot_dlms.fetch java`")
    return path


@pytest.fixture(scope="module")
def analysis(root, java, tmp_path_factory):
    """Run analyze.py once on the frozen artifacts; returns its output directory."""
    out = tmp_path_factory.mktemp("java_refactoring")
    result = run_analyze(root, java / "pools", out, "--compare", str(root / "data" / "tables" / "java_refactoring"))
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
    paper = read_table(root / "data" / "tables" / "java_refactoring" / f"{name}.csv")
    assert max_abs_diff(ours, paper, KEYS[name]) < TOLERANCE, name


@pytest.mark.parity
def test_checkstyle_case_matches_paper(root, analysis):
    ours = json.loads((analysis / "checkstyle_case.json").read_text())
    paper = json.loads((root / "data" / "tables" / "java_refactoring" / "checkstyle_case.json").read_text())
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


@pytest.mark.api
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="calls the Anthropic API: needs ANTHROPIC_API_KEY")
def test_judge_api_on_two_calibration_controls(root, java):
    """judge.py through the Anthropic API on the first two calibration controls (a reference rewrite, expected
    correct, and a no-op, expected incorrect), as the paper's judge called them.

    Costs two API calls, so it carries its own marker and runs only on request (`make test-api`, i.e.
    `pytest -m api`); `make test` and `make test-parity` never run it. It also needs the fetched artifacts.
    """
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
    command = [sys.executable, str(root / "experiments/java_refactoring/plot.py"), "--out", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "java_refactoring.pdf").stat().st_size > 10_000


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


@pytest.mark.parametrize("reply", [
    '{"targets": {"label": "Target 1"}, "joint": "correct"}',  # targets not a list
    '{"targets": ["Target 1"], "joint": "correct"}',  # a target that is not an object
    '{"targets": [{"label": ["Target 1"], "verdict": "correct"}], "joint": "correct"}',  # a label that is not a string
])
def test_malformed_verdicts_are_rejected_not_raised(root, reply):
    """A reply that is JSON but not the rubric's schema raises ValueError, so the packet stays unjudged."""
    judge = load_script(root, "judge")
    with pytest.raises(ValueError):
        judge.parse_verdict(reply, {"Target 1": "t1"})


class FakeMessages:
    """Stands in for client.messages: returns one fixed response object."""

    def __init__(self, response):
        self.response = response

    def create(self, **kwargs):
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def test_judge_leaves_a_packet_unjudged_on_refusal_or_bad_reply(root):
    """judge_one returns None for a declined request (with or without stop_details) and for a malformed reply."""
    judge = load_script(root, "judge")
    args = argparse.Namespace(model=judge.MODEL)
    labels = {"Target 1": "t1"}
    refusal = types.SimpleNamespace(stop_reason="refusal", content=[])  # no stop_details attribute at all
    assert judge.judge_one(FakeClient(refusal), args, "rubric", "packet", labels) is None
    block = types.SimpleNamespace(type="text", text='{"targets": [7], "joint": "correct"}')
    malformed = types.SimpleNamespace(stop_reason="end_turn", content=[block])
    assert judge.judge_one(FakeClient(malformed), args, "rubric", "packet", labels) is None
    block = types.SimpleNamespace(type="text", text='{"targets": [{"label": "Target 1", "verdict": "correct"}], "joint": "correct"}')
    valid = types.SimpleNamespace(stop_reason="end_turn", content=[block])
    assert judge.judge_one(FakeClient(valid), args, "rubric", "packet", labels)["per_target"] == {"t1": "correct"}


def run_judge(root, *arguments, without_key=True):
    environment = dict(os.environ)
    if without_key:
        environment.pop("ANTHROPIC_API_KEY", None)
        environment.pop("ANTHROPIC_AUTH_TOKEN", None)
    command = [sys.executable, str(root / "experiments/java_refactoring/judge.py"), *arguments]
    return subprocess.run(command, capture_output=True, text=True, cwd=root, env=environment)


def test_judge_refuses_zero_demo_pools(root, tmp_path):
    """The ablation's pools are keyed by unit: judge.py says it ships the paper's verdicts and exits 2."""
    record = {"unit_id": "A|M0", "anchor_episode_id": "A", "M": 0, "J": 1, "candidates": []}
    (tmp_path / "pools_zero_demo.jsonl").write_text(json.dumps(record) + "\n")
    result = run_judge(root, "--pools", str(tmp_path))
    assert result.returncode == 2
    assert "the zero-demo ablation ships the paper's verdicts; re-judging it is not supported" in result.stderr


def test_judge_without_an_api_key_exits_before_judging(root, tmp_path):
    """Without ANTHROPIC_API_KEY judge.py names the variable and exits 2, before any banner (pools and --calibrate)."""
    episode = load_java_episodes()[0]
    record = {"episode_id": episode["episode_id"], "regime": "oracle", "candidates": [{"text_by_target": {"t1": "x"}, "text_sha": "s"}]}
    (tmp_path / "pools_oracle.jsonl").write_text(json.dumps(record) + "\n")
    result = run_judge(root, "--pools", str(tmp_path), "--out", str(tmp_path / "verdicts.jsonl"))
    assert result.returncode == 2
    assert "ANTHROPIC_API_KEY" in result.stderr
    assert result.stdout == ""
    controls = tmp_path / "calibration" / "controls.jsonl"
    controls.parent.mkdir()
    controls.write_text("{}\n")
    result = run_judge(root, "--calibrate", "--controls", str(controls), "--out", str(tmp_path / "calibration.jsonl"))
    assert result.returncode == 2
    assert "ANTHROPIC_API_KEY" in result.stderr
    assert result.stdout == ""


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


LLADA = types.SimpleNamespace(name="llada", model_id="GSAI-ML/LLaDA-1.5", revision="rev")


def pool_hash(run, study="main", regime="oracle", M=(1, 2), J=(1, 2), K=16):
    """The config_hash of the records of a run of run.py (no GPU, no model)."""
    args = argparse.Namespace(study=study, regime=regime, M=list(M), J=list(J), K=K, chunk=run.CHUNK, head_probs=None)
    return run.config_hash(run.pool_config(run.config_of(args, LLADA, "")))


def test_pool_config_hash_depends_on_what_decides_the_records(root):
    """A pool's config hash changes with K, the regime or the study, not with the cells (M, J) a run selects."""
    run = load_script(root, "run")
    paper = pool_hash(run)
    assert pool_hash(run, M=[1], J=[2]) == paper
    assert pool_hash(run, K=4) != paper
    assert pool_hash(run, regime="learned") != paper
    assert pool_hash(run, study="zero_demo", M=[0, 1, 2]) != paper


def test_resume_skips_only_records_of_the_same_configuration(root, tmp_path):
    """done_units returns the units a run of the same pool configuration may skip, and refuses a file of another one
    (the frozen pools carry the paper's protocol_hash but no config_hash, so a rerun never appends to them)."""
    run = load_script(root, "run")
    digest = pool_hash(run)
    path = tmp_path / "pools_oracle.jsonl"
    assert run.done_units(path, "episode_id", digest) == set()
    good = [json.dumps({"episode_id": name, "config_hash": digest}) for name in ("a", "b")]
    path.write_text("\n".join(good) + '\n{"episode_id": "c", "conf', encoding="utf-8")  # an interrupted write
    assert run.done_units(path, "episode_id", digest) == {"a", "b"}
    path.write_text(good[0] + "\n" + json.dumps({"episode_id": "b", "protocol_hash": run.PROTOCOL_HASH}) + "\n")
    with pytest.raises(run.ProtocolMismatch, match="None"):
        run.done_units(path, "episode_id", digest)


@pytest.mark.parametrize("study, pool", [("main", "pools_oracle.jsonl"), ("zero_demo", "pools_zero_demo.jsonl")])
def test_run_refuses_a_pool_of_another_configuration(root, tmp_path, study, pool):
    """run.py exits with status 2, naming the file, before it loads a model, when its pool holds records of another
    configuration (here a record as the frozen pools hold them, without config_hash)."""
    path = tmp_path / pool
    path.write_text(json.dumps({"episode_id": "E", "unit_id": "E|M0", "protocol_hash": "0e1d3c9040b09598"}) + "\n")
    command = [sys.executable, str(root / "experiments/java_refactoring/run.py"), "--study", study, "--K", "4"]
    command += ["--families", "2", "--out", str(tmp_path), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2, result.stderr
    assert str(path) in result.stderr
    assert "loaded in" not in result.stdout


@pytest.mark.parametrize("study", ["main", "zero_demo"])
def test_run_counts_a_failed_unit_and_exits_non_zero(root, tmp_path, monkeypatch, capsys, study):
    """An episode (ablation unit) that raises is reported and counted, the others are written, and the run ends with
    RUN_FAILED and exit status 1 (no model: load and the per-unit function are replaced); the next run writes only the
    failed unit."""
    run = load_script(root, "run")
    adapter = types.SimpleNamespace(model_id="GSAI-ML/LLaDA-1.5", revision="rev")
    monkeypatch.setattr(run, "load", lambda name, device: adapter)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run.torch.cuda, "empty_cache", lambda: None)
    field = "episode_id" if study == "main" else "unit_id"
    seen = []

    def body_of(key):
        return {field: key, "distinct_allocations": 1, "allocation_rejections": 0, "n_unique_text": 1,
                "max_p1_replay_absdiff": 0.0, "seconds_gen": 0.0, "seconds_score": 0.0}

    def flaky(adapter, unit, *arguments):
        seen.append(unit[field])
        if len(seen) == 2:
            raise RuntimeError("injected failure")
        return body_of(unit[field])

    name = "run_main_episode" if study == "main" else "run_zero_demo_unit"
    monkeypatch.setattr(run, name, flaky)
    argv = ["run.py", "--study", study, "--K", "4", "--families", "2", "--out", str(tmp_path), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit:
        run.main()
    stdout = capsys.readouterr().out
    assert exit.value.code == 1
    assert f"FAILED {seen[1]}" in stdout
    assert f"RUN_FAILED {study}" in stdout and "failed families: 1" in stdout
    assert "RUN_DONE" not in stdout
    pool = tmp_path / ("pools_oracle.jsonl" if study == "main" else "pools_zero_demo.jsonl")
    written = [record[field] for record in read_jsonl(pool)]
    assert seen[1] not in written and len(written) == len(seen) - 1

    monkeypatch.setattr(run, name, lambda adapter, unit, *arguments: body_of(unit[field]))
    run.main()
    stdout = capsys.readouterr().out
    assert "RUN_DONE" in stdout
    assert [record[field] for record in read_jsonl(pool)] == written + [seen[1]]


@pytest.mark.parametrize("option, group", [("--head", "heads"), ("--head-probs", "java")])
def test_learned_run_without_its_head_names_the_fetch_command(root, tmp_path, option, group):
    """run.py --regime learned exits with status 2 and the fetch command, before it loads a model, when the head
    checkpoint (or the stored predictions) is missing."""
    command = [sys.executable, str(root / "experiments/java_refactoring/run.py"), "--regime", "learned"]
    command += [option, str(tmp_path / "missing"), "--families", "1", "--out", str(tmp_path), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2, result.stderr
    assert f"python -m fewshot_dlms.fetch {group}" in result.stderr
    assert "Traceback" not in result.stderr
    assert "loaded in" not in result.stdout
