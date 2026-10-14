"""fewshot_dlms.fetch: the pinned dataset revision, the download call, the SHA-256 check and the fetch hint.

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

    (dest / java[0]).unlink()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse_download)
    assert fetch.main(["--check", "--dest", str(dest)]) == 0
    assert f"OK: 1 of the {len(sums)} files listed in {dest / 'SHA256SUMS'}" in capsys.readouterr().out
    (dest / "SHA256SUMS").write_text("", encoding="utf-8")
    assert fetch.main(["--check", "--dest", str(dest)]) == 1
    assert "is not the SHA256SUMS of the pinned revision" in capsys.readouterr().err
