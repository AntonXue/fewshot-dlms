"""examples/walkthrough.py --cached prints the stored pool without a GPU and without importing torch; the stored pool's
hash and seed are Figure 5's, and the live run's option checks and order of output hold without a model."""
import importlib.util
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

SCRIPT = "examples/walkthrough.py"


def run_cached(root, code=None):
    """Run the script (or `code`, which runs it in-process) in a fresh interpreter from the repository root."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    command = [sys.executable, SCRIPT, "--cached"] if code is None else [sys.executable, "-c", code]
    return subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)


def test_walkthrough_cached_runs(root):
    result = run_cached(root)
    assert result.returncode == 0, result.stderr
    assert "Q_0.5" in result.stdout


def test_walkthrough_cached_table_has_every_candidate(root):
    record = json.loads((root / "data" / "fixtures" / "walkthrough.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert (record["M"], record["J"], record["K"]) == (2, 2, 8)
    assert len(record["candidates"]) == 8
    result = run_cached(root)
    assert result.returncode == 0, result.stderr
    for column in ("ell1", "ell0", "w_0.5", "joint", "A_all"):
        assert column in result.stdout
    assert "<<T1" in result.stdout and "<<T2" in result.stdout
    assert record["provenance"]["revision"][:8] in result.stdout


def test_walkthrough_cached_does_not_import_torch(root):
    code = (
        "import runpy, sys\n"
        "sys.argv = ['walkthrough.py', '--cached']\n"
        f"runpy.run_path({SCRIPT!r}, run_name='__main__')\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
    )
    result = run_cached(root, code)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------------------
# The fixture's provenance, the printed table, and the live run's checks (CPU, no model)
# ---------------------------------------------------------------------------------------------

CANDIDATE_ROW = re.compile(r"^\s*(\d+)\s+-?\d+\.\d{2}\s+-?\d+\.\d{2}\s+-?\d+\.\d{2}\s+\d\.\d{4}(\s+[01]){3}\s+\S")


def load_script(root, path, name):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_record(root):
    return json.loads((root / "data" / "fixtures" / "walkthrough.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_fixture_hash_and_seed_are_figure_5s(root):
    """The fixture's protocol_hash and seed are those of Figure 5's run.py for the cell (M, J) = (2, 2) at K = 8."""
    run = load_script(root, "experiments/multi_target/run.py", "multi_target_run_walkthrough")
    walkthrough = load_script(root, SCRIPT, "walkthrough")
    record = fixture_record(root)
    adapter = SimpleNamespace(name=record["model"], model_id=record["model_name"], revision=record["revision"])
    settings = SimpleNamespace(bank="report", M=[record["M"]], J=[record["J"]], K=record["K"], chunk=run.CHUNK)
    config = run.config_of(settings, adapter)
    assert run.protocol_hash(config) == record["protocol_hash"] == "d35aee98e185001a"
    assert run.protocol_hash(run.pool_config(config, record["M"], record["J"])) == record["protocol_hash"]
    assert walkthrough.pool_protocol_hash(run, adapter) == record["protocol_hash"]
    seed = run.proposal_seed(record["model"], record["episode_id"], record["M"], record["J"])
    assert seed == record["seed"] == record["provenance"]["seed"] == 2019691244


def test_walkthrough_cached_prints_eight_candidate_rows(root):
    result = run_cached(root)
    assert result.returncode == 0, result.stderr
    rows = [match for match in map(CANDIDATE_ROW.match, result.stdout.splitlines()) if match]
    assert [int(match.group(1)) for match in rows] == list(range(8))
    assert "Identical picks are expected" in result.stdout


def run_script(root, *arguments):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    command = [sys.executable, SCRIPT, *arguments]
    return subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)


def test_live_run_options_fail_before_loading_the_model(root, tmp_path):
    """--fixture belongs to --cached; --save-fixture needs an existing directory: both are argparse errors (exit 2)."""
    result = run_script(root, "--fixture", str(tmp_path / "pool.jsonl"))
    assert result.returncode == 2
    assert "--fixture names the stored pool that --cached reads" in result.stderr
    result = run_script(root, "--save-fixture", str(tmp_path / "missing" / "pool.jsonl"), "--device", "cpu")
    assert result.returncode == 2
    assert f"the directory {tmp_path / 'missing'} does not exist" in result.stderr
    assert "loaded" not in result.stderr


def test_live_run_prints_the_report_before_it_saves_the_fixture(root, tmp_path, monkeypatch, capsys):
    """A live run (here the stored record in place of the GPU) prints the table, then writes the fixture."""
    walkthrough = load_script(root, SCRIPT, "walkthrough_live")
    record = fixture_record(root)
    events = []
    print_report = walkthrough.print_report
    save_fixture = walkthrough.save_fixture
    monkeypatch.setattr(walkthrough, "live_record", lambda device: record)
    monkeypatch.setattr(walkthrough, "print_report", lambda r: events.append("report") or print_report(r))
    monkeypatch.setattr(walkthrough, "save_fixture", lambda path, r: events.append("save") or save_fixture(path, r))
    out = tmp_path / "pool.jsonl"
    monkeypatch.setattr(sys, "argv", ["walkthrough.py", "--save-fixture", str(out), "--device", "cpu"])
    walkthrough.main()
    assert events == ["report", "save"]
    assert json.loads(out.read_text(encoding="utf-8")) == record
    assert "Q_0.5" in capsys.readouterr().out


def test_provenance_names_the_cpu_device(root):
    walkthrough = load_script(root, SCRIPT, "walkthrough_device")
    assert walkthrough.device_name("cpu") == "cpu"
