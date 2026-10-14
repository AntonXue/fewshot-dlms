"""examples/walkthrough.py --cached prints the stored pool without a GPU and without importing torch."""
import json
import os
import subprocess
import sys

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
