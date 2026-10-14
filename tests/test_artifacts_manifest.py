"""The manifest of the frozen artifacts: SHA256SUMS, ARTIFACTS.md, the pinned revision and the data card.

The artifacts live on the Hugging Face dataset UT-IFML/fewshot-dlms (tools/assemble_artifacts.py builds it).
SHA256SUMS at the repository root lists every file of that dataset as "<sha256>  <path>" (the format of
sha256sum), data/artifacts_revision.txt holds the dataset revision that fewshot_dlms.fetch pins, and
artifacts_upload/README.md is the data card uploaded as the dataset's README.md. CPU only; no download.
"""
import hashlib
import json
import re

import pytest

from fewshot_dlms.leakcheck import find_private

SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")
GROUP_PATH = re.compile(r"^(pools|java|heads|head_training)/[^/].*$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
# Internal working notes, which no shipped text may hold, next to leakcheck's machine paths and metadata keys
# (tools/assemble_artifacts.py checks the same).
LOCAL_PATTERN = re.compile(r"hand-off|\.claude")
CARD_KEYS = ("license: other", "license_name:", "pretty_name:", "size_categories:")


@pytest.fixture(scope="module")
def sums(root):
    """{path: sha256} read from SHA256SUMS; fails on a malformed or repeated line."""
    entries = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = SUMS_LINE.match(line)
        assert match, f"malformed SHA256SUMS line: {line!r}"
        digest, path = match.groups()
        assert path not in entries, f"{path} listed twice"
        entries[path] = digest
    return entries


@pytest.fixture(scope="module")
def revision(root):
    lines = (root / "data" / "artifacts_revision.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and REVISION.match(lines[0]), lines
    return lines[0]


def test_sums_list_relative_paths_of_the_four_groups(sums):
    assert sums
    for path in sums:
        assert not path.startswith("/") and "\\" not in path, path
        assert ".." not in path.split("/") and "." not in path.split("/"), path
        assert path == "README.md" or GROUP_PATH.match(path), path


def test_sums_are_sorted_and_cover_every_group(sums):
    paths = list(sums)
    assert paths == sorted(paths)
    assert "README.md" in sums
    for group in ("pools/fig4/", "pools/fig5/", "pools/fig6/", "java/", "heads/", "head_training/"):
        assert any(path.startswith(group) for path in paths), group


def test_artifacts_md_names_the_pinned_revision_and_the_sums(root, revision):
    text = (root / "ARTIFACTS.md").read_text(encoding="utf-8")
    assert revision in text
    sums_sha = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    assert sums_sha in text


def test_data_card_is_the_uploaded_readme(root, sums):
    card = (root / "artifacts_upload" / "README.md").read_bytes()
    assert hashlib.sha256(card).hexdigest() == sums["README.md"]
    text = card.decode("utf-8")
    front_matter = text.split("---")[1]
    for key in CARD_KEYS:
        assert key in front_matter, key


def test_manifest_files_hold_no_machine_paths(root):
    for name in ("ARTIFACTS.md", "SHA256SUMS", "artifacts_upload/README.md", "data/artifacts_revision.txt"):
        text = (root / name).read_text(encoding="utf-8")
        assert not find_private(text), (name, find_private(text))
        assert not LOCAL_PATTERN.search(text), name


@pytest.fixture(scope="module")
def assembler(root):
    """tools/assemble_artifacts.py as a module (it imports its sibling extract_fixtures from tools/)."""
    import sys

    sys.path.insert(0, str(root / "tools"))
    import assemble_artifacts

    return assemble_artifacts


def test_check_fails_on_a_private_path_inside_a_binary_file(assembler, tmp_path):
    """A machine path in a binary file (as once pickled in the heads' configs) fails like one in a text file."""
    machine_path = "/" + "hdd9" + "/someone/run"
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads" / "x.pt").write_bytes(b"PK\x03\x04\x00\x00" + machine_path.encode() + b"\x00\xff")
    (tmp_path / "heads" / "clean.pt").write_bytes(b"PK\x03\x04\x00\x00\x01\x02\xff")
    failures = assembler.check(tmp_path, ["heads/clean.pt", "heads/x.pt"])
    assert len(failures) == 1 and failures[0].startswith("heads/x.pt")


def test_check_allows_an_invented_path_only_in_sampled_texts(assembler, tmp_path):
    """The documented exception: a model's sampled text may name a path; any other field may not."""
    invented = "file: /home/someone/in.py"
    sampled = {"episode_id": "e", "candidates": [{"toks": [1], "texts": [invented]}]}
    elsewhere = {"episode_id": "e", "note": invented, "candidates": [{"toks": [1], "texts": ["a"]}]}
    (tmp_path / "sampled.jsonl").write_text(json.dumps(sampled) + "\n")
    (tmp_path / "elsewhere.jsonl").write_text(json.dumps(elsewhere) + "\n")
    assert assembler.check(tmp_path, ["sampled.jsonl"]) == []
    assert len(assembler.check(tmp_path, ["elsewhere.jsonl"])) == 1
