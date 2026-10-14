"""The manifest of the frozen artifacts: SHA256SUMS, ARTIFACTS.md, the pinned revision and the data card.

The artifacts live on the Hugging Face dataset UT-IFML/fewshot-dlms (tools/assemble_artifacts.py builds it).
SHA256SUMS at the repository root lists every file of that dataset as "<sha256>  <path>" (the format of
sha256sum), data/artifacts_revision.txt holds the dataset revision that fewshot_dlms.fetch pins, and
artifacts_upload/README.md is the data card uploaded as the dataset's README.md. CPU only; no download.
"""
import hashlib
import re

import pytest

SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")
GROUP_PATH = re.compile(r"^(pools|java|heads|head_training)/[^/].*$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
# Machine paths and internal working notes, which no shipped text may hold (tools/assemble_artifacts.py).
PRIVATE_PATTERN = re.compile(r"/(?:hdd|ssd)\d|/home/|/mnt/|hand-off|\.claude")
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
        assert not PRIVATE_PATTERN.search(text), name
