"""fewshot_dlms.fetch: the pinned dataset revision, the download call, the SHA-256 check, the fetch hint, the one-line
remedies when the Hub refuses the login or cannot be reached, and --revision's skipped pin check.

CPU only, no network: huggingface_hub.snapshot_download is replaced by a stub in every test that could call it.
"""
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import pytest

from fewshot_dlms import fetch


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def refuse_download(*args, **kwargs):
    raise AssertionError("snapshot_download must not be called")


def test_fetch_revision_is_pinned(root, monkeypatch, tmp_path):
    """PINNED_REVISION is the revision of data/artifacts_revision.txt and ARTIFACTS.md, and download() passes it."""
    assert fetch.PINNED_REVISION == (root / "data" / "artifacts_revision.txt").read_text(encoding="utf-8").strip()
    artifacts_md = (root / "ARTIFACTS.md").read_text(encoding="utf-8")
    match = re.search(r"\| pinned revision \| `([0-9a-f]{40})`", artifacts_md)
    assert match, "ARTIFACTS.md names no pinned revision"
    assert match.group(1) == fetch.PINNED_REVISION

    calls = []

    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", stub)
    dest = tmp_path / "artifacts"
    assert fetch.download(["pools", "java"], dest=dest) == dest
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs == {
        "repo_id": "UT-IFML/fewshot-dlms",
        "repo_type": "dataset",
        "revision": fetch.PINNED_REVISION,
        "allow_patterns": ["pools/*", "java/*", "SHA256SUMS", "README.md"],
        "local_dir": dest,
    }


def test_download_defaults_and_unknown_group(monkeypatch):
    """The default destination is artifacts/ (relative to the working directory); an unknown group downloads nothing."""
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    fetch.download(["heads"])
    assert calls[0]["local_dir"] == Path("artifacts")
    assert calls[0]["revision"] == fetch.PINNED_REVISION
    with pytest.raises(ValueError, match="unknown group"):
        fetch.download(["pools/fig5"])
    assert len(calls) == 1


def test_sums_hash_is_pinned(root):
    """SUMS_SHA256 is the SHA-256 of the committed SHA256SUMS, as recorded in ARTIFACTS.md."""
    assert fetch.SUMS_SHA256 == sha256((root / "SHA256SUMS").read_bytes())
    assert fetch.SUMS_SHA256 in (root / "ARTIFACTS.md").read_text(encoding="utf-8")


def test_fetch_message_without_artifacts(root, tmp_path):
    """analyze.py on a missing pools directory exits 2 and names the fetch command (fig4, 6, 7: test_parity_fig*)."""
    command = [
        sys.executable,
        str(root / "experiments/fig5_multi_target/analyze.py"),
        "--pools",
        str(tmp_path / "no_pools"),
        "--out",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 2
    assert "python -m fewshot_dlms.fetch" in result.stderr


def test_verify_reports_tampered_files_only(tmp_path):
    """A good file passes, a tampered one is a mismatch, a missing one is not (missing() lists it)."""
    good = b'{"k": 1}\n'
    original = b'{"k": 2}\n'
    absent = b'{"k": 4}\n'
    dest = tmp_path / "artifacts"
    (dest / "pools").mkdir(parents=True)
    (dest / "pools" / "good.jsonl").write_bytes(good)
    (dest / "pools" / "tampered.jsonl").write_bytes(b'{"k": 3}\n')
    sums = [
        f"{sha256(good)}  pools/good.jsonl",
        f"{sha256(original)}  pools/tampered.jsonl",
        f"{sha256(absent)}  pools/missing.jsonl",
    ]
    (dest / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    assert fetch.verify(dest) == ["pools/tampered.jsonl"]
    assert fetch.missing(dest, ["pools"]) == ["pools/missing.jsonl"]
    assert fetch.missing(dest, ["java"]) == []


def test_verify_falls_back_to_the_committed_sums(root, tmp_path):
    """Without a downloaded SHA256SUMS, verify() reads the repository's root copy."""
    dest = tmp_path / "artifacts"
    dest.mkdir()
    shutil.copyfile(root / "artifacts_upload" / "README.md", dest / "README.md")
    assert fetch.sums_file(dest) == root / "SHA256SUMS"
    assert fetch.verify(dest) == []
    with open(dest / "README.md", "ab") as handle:
        handle.write(b"\n")
    assert fetch.verify(dest) == ["README.md"]


def test_main_without_groups_prints_the_table(monkeypatch, capsys):
    """No group named: usage and the group table, exit 2, nothing downloaded."""
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse_download)
    assert fetch.main([]) == 2
    out = capsys.readouterr().out
    assert "usage: python -m fewshot_dlms.fetch" in out
    assert fetch.PINNED_REVISION in out
    for group in ("pools", "java", "heads", "head_training"):
        assert re.search(rf"^{group} ", out, re.MULTILINE), group


def test_main_checks_the_download(root, monkeypatch, tmp_path, capsys):
    """After the download, a tampered or missing file of a requested group fails the run; a clean check passes."""
    dest = tmp_path / "artifacts"
    sums = fetch.read_sums(root / "SHA256SUMS")
    java = sorted(name for name in sums if name.startswith("java/"))

    def stub(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "SHA256SUMS", local_dir / "SHA256SUMS")
        shutil.copyfile(root / "artifacts_upload" / "README.md", local_dir / "README.md")
        (local_dir / java[0]).parent.mkdir(parents=True, exist_ok=True)
        (local_dir / java[0]).write_bytes(b"not the frozen file\n")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", stub)
    assert fetch.main(["java", "--dest", str(dest)]) == 1
    err = capsys.readouterr().err
    assert f"SHA-256 mismatch: {dest / java[0]}" in err
    assert f"missing: {dest / java[1]}" in err
    assert "delete the listed files and run the fetch again" in err

    (dest / java[0]).unlink()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse_download)
    assert fetch.main(["--check", "--dest", str(dest)]) == 0
    assert f"OK: 1 of the {len(sums)} files listed in {dest / 'SHA256SUMS'}" in capsys.readouterr().out
    (dest / "SHA256SUMS").write_text("", encoding="utf-8")
    assert fetch.main(["--check", "--dest", str(dest)]) == 1
    assert "is not the SHA256SUMS of the pinned revision" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# The Hub refuses the login or cannot be reached: one line with the cause and the remedy, exit 1, no token
# ---------------------------------------------------------------------------------------------

TOKEN = "hf_" + "NotARealTokenOnlyATestSentinel"


def hub_error(cls, status):
    """An HTTP error of huggingface_hub with the given status; its message holds the sentinel token, which the fetch
    must not echo."""
    message = f"{status} Client Error (Authorization: Bearer {TOKEN})"
    try:
        import httpx

        request = httpx.Request("GET", f"https://huggingface.co/api/datasets/{fetch.REPO_ID}")
        response = httpx.Response(status, request=request)
    except ImportError:  # huggingface_hub < 1.0 speaks requests
        import requests

        response = requests.Response()
        response.status_code = status
    return cls(message, response=response)


def network_error():
    try:
        import httpx

        return httpx.ConnectError(f"connection refused ({TOKEN})")
    except ImportError:
        import requests

        return requests.exceptions.ConnectionError(f"connection refused ({TOKEN})")


HUB = huggingface_hub.errors
# name -> (the error the stubbed download raises, phrases the one printed line must hold)
FAILURES = {
    "private": (lambda: hub_error(HUB.RepositoryNotFoundError, 401), ["private", "hf auth login", "HF_TOKEN"]),
    "gated": (lambda: hub_error(HUB.GatedRepoError, 403), ["gated", "hf auth login", "HF_TOKEN"]),
    "http_401": (lambda: hub_error(HUB.HfHubHTTPError, 401), ["private", "HTTP 401", "HF_TOKEN"]),
    "http_403": (lambda: hub_error(HUB.HfHubHTTPError, 403), ["private", "HTTP 403", "HF_TOKEN"]),
    "revision": (lambda: hub_error(HUB.RevisionNotFoundError, 404), ["no revision", "--revision"]),
    "offline": (lambda: HUB.LocalEntryNotFoundError(f"cannot find ({TOKEN})"), ["no network", "--check"]),
    "connection": (network_error, ["no network", "HF_HUB_OFFLINE"]),
    "builtin_connection": (lambda: ConnectionError(TOKEN), ["no network"]),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_download_failure_prints_one_line_with_the_remedy(monkeypatch, tmp_path, capsys, name):
    make_error, phrases = FAILURES[name]

    def stub(**kwargs):
        raise make_error()

    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", stub)
    assert fetch.main(["java", "--dest", str(tmp_path / "artifacts")]) == 1
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert len(lines) == 1, lines
    for phrase in phrases:
        assert phrase in lines[0], (phrase, lines[0])
    assert TOKEN not in captured.out + captured.err


def test_download_failure_leaves_other_errors_alone(monkeypatch, tmp_path):
    """A server error the fetch cannot explain (here HTTP 500 raised by the stub) propagates unchanged."""
    def stub(**kwargs):
        raise hub_error(HUB.HfHubHTTPError, 500)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", stub)
    with pytest.raises(HUB.HfHubHTTPError):
        fetch.main(["java", "--dest", str(tmp_path / "artifacts")])


def test_another_revision_skips_the_pin_check_with_a_warning(root, monkeypatch, tmp_path, capsys):
    """--revision passes the revision to the Hub and, with or without --check, warns that the SHA256SUMS pin check is
    skipped: a SHA256SUMS that is not the pinned one is then accepted, and the files are checked against it."""
    revision = "0" * 40
    calls = []

    def stub(**kwargs):
        calls.append(kwargs["revision"])
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "java").mkdir(parents=True, exist_ok=True)
        content = b"{}\n"
        (local_dir / "java" / "a.json").write_bytes(content)
        (local_dir / "SHA256SUMS").write_text(f"{sha256(content)}  java/a.json\n", encoding="utf-8")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", stub)
    dest = tmp_path / "artifacts"
    assert fetch.main(["java", "--dest", str(dest), "--revision", revision]) == 0
    assert calls == [revision]
    captured = capsys.readouterr()
    assert f"revision {revision} is not the pinned" in captured.err and "is skipped" in captured.err
    assert "OK: 1 of the 1 files" in captured.out

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse_download)
    assert fetch.main(["--check", "--dest", str(dest), "--revision", revision]) == 0
    assert "is skipped" in capsys.readouterr().err
    assert fetch.main(["--check", "--dest", str(dest)]) == 1  # the pinned revision's check rejects that SHA256SUMS
    assert "is not the SHA256SUMS of the pinned revision" in capsys.readouterr().err
