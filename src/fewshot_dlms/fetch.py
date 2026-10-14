"""Download the frozen artifacts from the Hugging Face dataset UT-IFML/fewshot-dlms and check their SHA-256.

Run from the repository root (the tests and every analyze.py read artifacts/ there):

    python -m fewshot_dlms.fetch pools java       # what make test-parity and the analyze.py scripts need (386 MB)
    python -m fewshot_dlms.fetch heads            # the four length heads (3.47 GB), for the GPU runs
    python -m fewshot_dlms.fetch head_training    # the heads' training data (1.28 GB)
    python -m fewshot_dlms.fetch --check          # check the files already under artifacts/; download nothing
    python -m fewshot_dlms.fetch                  # print this usage and the group table; download nothing

A group is a top-level directory of the dataset. Every download is at PINNED_REVISION, the dataset commit
recorded in data/artifacts_revision.txt and ARTIFACTS.md (--revision overrides it), and lands under artifacts/
(--dest) with the dataset's layout, e.g. artifacts/pools/fig5/llada/list_easy_fam_M4_J2.jsonl. SHA256SUMS and
README.md (the data card) come with every group, so the check below also works offline later.

After the download, every file present under artifacts/ is checked against SHA256SUMS ("<sha256>  <path>" per
line, the format of sha256sum): the downloaded copy, whose own SHA-256 must be SUMS_SHA256 at the pinned
revision, or the repository's root copy when none was downloaded. A file of a requested group that is absent
also fails the run; files of groups not requested are not needed. huggingface_hub keeps its download metadata
in artifacts/.cache/huggingface/, so a second run skips the files it already has.

While the dataset is private, downloading needs a Hugging Face login with access to it; huggingface_hub finds
the token (`hf auth login` or HF_TOKEN) by itself.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import huggingface_hub

REPO_ID = "UT-IFML/fewshot-dlms"
REPO_TYPE = "dataset"
# The dataset commit of the frozen artifacts. tests/test_fetch.py checks it against data/artifacts_revision.txt
# and ARTIFACTS.md, so a stale pin fails make test.
PINNED_REVISION = "5a79ed36648816ffbc27e3c11d7938692835d227"
# SHA-256 of SHA256SUMS at PINNED_REVISION (ARTIFACTS.md), checked against the downloaded copy.
SUMS_SHA256 = "bdead7d377a47cf4835985e11ec53f6ed4b056c10374e77fa82b31d4c6d11fe9"
SUMS_NAME = "SHA256SUMS"
CARD_NAME = "README.md"
# The repository root (src/fewshot_dlms/fetch.py -> ../..), which holds the committed SHA256SUMS.
ROOT = Path(__file__).resolve().parents[2]
# group -> (download size, what reads it); the sizes are those of ARTIFACTS.md.
GROUPS = {
    "pools": ("376 MB", "Figures 4-6 and the Appendix B figures: analyze.py of fig4_single_target, fig5_multi_target, fig6_lengths"),
    "java": ("10 MB", "Figure 7: analyze.py and judge.py of fig7_java"),
    "heads": ("3.47 GB", "the four length heads: run.py --mode learned (fig6), --regime learned (fig7), make test-gpu"),
    "head_training": ("1.28 GB", "the heads' training prompts and configs: fig6_lengths/train_head.py"),
}
SUMS_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
BLOCK_BYTES = 1 << 20


def allow_patterns(groups):
    """The snapshot_download patterns of the groups, plus SHA256SUMS and the data card README.md."""
    patterns = []
    for group in groups:
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}; the groups are {', '.join(GROUPS)}")
        patterns.append(f"{group}/*")
    patterns.append(SUMS_NAME)
    patterns.append(CARD_NAME)
    return patterns


def download(groups, revision=PINNED_REVISION, dest=Path("artifacts")):
    """Download the groups of the dataset at `revision` into `dest` (with SHA256SUMS and README.md); returns dest."""
    patterns = allow_patterns(groups)
    huggingface_hub.snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        allow_patterns=patterns,
        local_dir=dest,
    )
    return Path(dest)


def sha256_file(path):
    """SHA-256 of a file, read in blocks of 1 MiB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        block = handle.read(BLOCK_BYTES)
        while block:
            digest.update(block)
            block = handle.read(BLOCK_BYTES)
    return digest.hexdigest()


def read_sums(path):
    """{relative path: sha256} of a SHA256SUMS file; raises ValueError on a malformed line."""
    sums = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = SUMS_LINE.match(line)
        if match is None:
            raise ValueError(f"malformed line in {path}: {line!r}")
        sums[match.group(2)] = match.group(1)
    return sums


def sums_file(dest):
    """The SHA256SUMS that verify() reads: the downloaded copy in dest, else the repository's root copy."""
    downloaded = Path(dest) / SUMS_NAME
    if downloaded.is_file():
        return downloaded
    return ROOT / SUMS_NAME


def verify(dest):
    """Files present under dest whose SHA-256 differs from SHA256SUMS (paths relative to dest, sorted).

    A file listed in SHA256SUMS but absent from dest is not a mismatch: fetching some groups only is normal.
    missing() lists the absent files of given groups.
    """
    sums = read_sums(sums_file(dest))
    mismatched = []
    for name in sorted(sums):
        path = Path(dest) / name
        if path.is_file() and sha256_file(path) != sums[name]:
            mismatched.append(name)
    return mismatched


def missing(dest, groups):
    """Files of the groups listed in SHA256SUMS but absent from dest (sorted)."""
    sums = read_sums(sums_file(dest))
    prefixes = tuple(f"{group}/" for group in groups)
    absent = []
    for name in sorted(sums):
        if name.startswith(prefixes) and not (Path(dest) / name).is_file():
            absent.append(name)
    return absent


def group_table():
    """The groups with their download sizes and readers, printed when no group is named."""
    lines = [f"dataset {REPO_ID}, pinned revision {PINNED_REVISION}", ""]
    lines.append(f"{'group':<14} {'size':>8}  what reads it")
    for group, (size, readers) in GROUPS.items():
        lines.append(f"{group:<14} {size:>8}  {readers}")
    lines.append("")
    lines.append("make test-parity, make reproduce and every analyze.py need: python -m fewshot_dlms.fetch pools java")
    return "\n".join(lines)


def main(argv=None):
    """Download the named groups (or, with --check, nothing) and check the files under --dest; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m fewshot_dlms.fetch",
        description=f"Download groups of the frozen artifacts ({REPO_ID}) and check them against SHA256SUMS.",
    )
    parser.add_argument("groups", nargs="*", metavar="group", help=f"one or more of: {', '.join(GROUPS)}")
    parser.add_argument("--revision", default=PINNED_REVISION, help="dataset revision (default: the pinned one)")
    parser.add_argument("--dest", default="artifacts", help="download directory (default: artifacts)")
    parser.add_argument("--check", action="store_true", help="only check the files already under --dest")
    args = parser.parse_args(argv)
    unknown = [group for group in args.groups if group not in GROUPS]
    if unknown:
        parser.error(f"unknown group {', '.join(unknown)}; the groups are {', '.join(GROUPS)}")
    dest = Path(args.dest)

    if not args.groups and not args.check:
        parser.print_usage()
        print()
        print(group_table())
        return 2

    if args.check and not dest.is_dir():
        print(f"no artifacts in {dest}; run: python -m fewshot_dlms.fetch pools java", file=sys.stderr)
        return 1
    if not args.check:
        if args.revision != PINNED_REVISION:
            print(f"warning: revision {args.revision} is not the pinned {PINNED_REVISION}", file=sys.stderr)
        print(f"downloading {', '.join(args.groups)} from {REPO_ID}@{args.revision} into {dest}/")
        download(args.groups, revision=args.revision, dest=dest)

    downloaded_sums = dest / SUMS_NAME
    if downloaded_sums.is_file() and args.revision == PINNED_REVISION and sha256_file(downloaded_sums) != SUMS_SHA256:
        print(f"{downloaded_sums} is not the SHA256SUMS of the pinned revision {PINNED_REVISION}", file=sys.stderr)
        return 1
    sums_path = sums_file(dest)
    sums = read_sums(sums_path)
    present = [name for name in sums if (dest / name).is_file()]
    mismatched = verify(dest)
    absent = missing(dest, args.groups)
    for name in mismatched:
        print(f"SHA-256 mismatch: {dest / name}", file=sys.stderr)
    for name in absent:
        print(f"missing: {dest / name}", file=sys.stderr)
    if mismatched or absent:
        print(f"{len(mismatched)} mismatched, {len(absent)} missing of the files in {sums_path}", file=sys.stderr)
        return 1
    print(f"OK: {len(present)} of the {len(sums)} files listed in {sums_path} are under {dest}/ and match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
