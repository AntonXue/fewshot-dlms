"""Download the frozen artifacts from the Hugging Face dataset UT-IFML/fewshot-dlms and check their SHA-256.

Run from the repository root (the tests and every analyze.py read artifacts/ there):

    python -m fewshot_dlms.fetch pools java       # what make test-parity and the analyze.py scripts need (386 MB)
    python -m fewshot_dlms.fetch heads            # the four length heads (3.47 GB), for the GPU runs
    python -m fewshot_dlms.fetch head_training    # the heads' training data (1.03 GB)
    python -m fewshot_dlms.fetch --check          # check the files already under artifacts/; download nothing
    python -m fewshot_dlms.fetch                  # print this usage and the group table; download nothing

A group is a top-level directory of the dataset. Every download is at PINNED_REVISION, the dataset commit
recorded in data/artifacts_revision.txt and ARTIFACTS.md, and lands under artifacts/ (--dest) with the dataset's
layout, e.g. artifacts/pools/multi_target/llada/list_easy_fam_M4_J2.jsonl. SHA256SUMS and README.md (the data card) come
with every group, so the check below also works offline later.

After the download, every file present under artifacts/ is checked against SHA256SUMS ("<sha256>  <path>" per
line, the format of sha256sum): the downloaded copy, whose own SHA-256 must be SUMS_SHA256 at the pinned
revision, or the repository's root copy when none was downloaded. A file of a requested group that is absent
also fails the run; files of groups not requested are not needed. A file that fails the check is listed; delete
the listed files and run the fetch again. huggingface_hub keeps its download metadata in
artifacts/.cache/huggingface/, so a second run skips the files it already has.

--revision downloads (or, with --check, checks) another dataset revision. Its SHA256SUMS cannot be the pinned
one, so the check of SHA256SUMS against SUMS_SHA256 is skipped, with a warning, and the files are checked against
the SHA256SUMS of that revision only: nothing ties them to the paper's artifacts.

While the dataset is private, downloading needs a Hugging Face login with access to it; huggingface_hub finds
the token (`hf auth login` or HF_TOKEN) by itself. When the Hub refuses the login or cannot be reached, the
fetch prints one line with the cause and the remedy (download_failure) and exits with status 1; it never prints
the token or the Hub's own message.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import huggingface_hub
from huggingface_hub import errors as hub_errors

REPO_ID = "UT-IFML/fewshot-dlms"
REPO_TYPE = "dataset"
# The dataset commit of the frozen artifacts. tests/test_fetch.py checks it against data/artifacts_revision.txt
# and ARTIFACTS.md, so a stale pin fails make test.
PINNED_REVISION = "f65650d34e80892b94a69083371fd939ed64edcc"
# SHA-256 of SHA256SUMS at PINNED_REVISION (ARTIFACTS.md), checked against the downloaded copy.
SUMS_SHA256 = "2fe80a4bc7887064976881bfdb9b285e1c4d75d99991caab2ffe28483c6c1304"
SUMS_NAME = "SHA256SUMS"
CARD_NAME = "README.md"
# The repository root (src/fewshot_dlms/fetch.py -> ../..), which holds the committed SHA256SUMS.
ROOT = Path(__file__).resolve().parents[2]
# group -> (download size, what reads it); the sizes are those of ARTIFACTS.md, figure numbers those of arXiv v1.
GROUPS = {
    "pools": ("376 MB", "pools/{single_target,multi_target,length_allocation}: Figures 5-7, 9-12, their analyze.py"),
    "java": ("10 MB", "Figure 8, Appendix C: analyze.py and judge.py of java_refactoring"),
    "heads": ("3.47 GB", "the four length heads (predictors): run.py --mode learned (length_allocation), --regime learned (java_refactoring), make test-gpu"),
    "head_training": ("1.03 GB", "the heads' training prompts and configs: length_allocation/train_head.py"),
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


def network_errors():
    """The exception classes of an unreachable Hub: huggingface_hub's LocalEntryNotFoundError (raised when it cannot
    reach the Hub and has no local copy, or HF_HUB_OFFLINE=1), the built-in connection and timeout errors, and the
    transport errors of the HTTP library of the installed huggingface_hub (httpx from 1.0, requests before)."""
    errors = [hub_errors.LocalEntryNotFoundError, ConnectionError, TimeoutError]
    try:
        import httpx

        errors.append(httpx.TransportError)
    except ImportError:
        pass
    try:
        import requests

        errors.extend([requests.exceptions.ConnectionError, requests.exceptions.Timeout])
    except ImportError:
        pass
    return tuple(errors)


def download_failure(error, revision, dest):
    """One line naming why the download failed and what to do, or None for an error this module does not explain.

    Built from the error's class and HTTP status only, never from its message, so it cannot carry the token.
    """
    cannot = f"cannot download {REPO_ID}:"
    login = "run `hf auth login` or set HF_TOKEN to a token with access to it, then run the fetch again"
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(error, hub_errors.GatedRepoError):
        return f"{cannot} the dataset is gated and this login has no access: {login}"
    if isinstance(error, hub_errors.RevisionNotFoundError):
        return f"{cannot} it has no revision {revision} (the pinned one is {PINNED_REVISION}): check --revision"
    if isinstance(error, hub_errors.RepositoryNotFoundError):
        return f"{cannot} the dataset is private (or not visible to this login): {login}"
    if isinstance(error, hub_errors.HfHubHTTPError) and status in (401, 403):
        return f"{cannot} the dataset is private and the Hub refused this login (HTTP {status}): {login}"
    if isinstance(error, network_errors()):
        return (
            f"{cannot} no network (or HF_HUB_OFFLINE=1, or the Hub is down) and {dest}/ holds no copy: connect and run "
            "the fetch again, or copy a fetched artifacts/ here and run python -m fewshot_dlms.fetch --check"
        )
    return None


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
    revision_help = "dataset revision (default: the pinned one; another skips the pin check of SHA256SUMS)"
    parser.add_argument("--revision", default=PINNED_REVISION, help=revision_help)
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
    pinned = args.revision == PINNED_REVISION
    if not pinned:
        print(
            f"warning: revision {args.revision} is not the pinned {PINNED_REVISION}: the check of {SUMS_NAME} against "
            f"the pinned SHA-256 is skipped, so the files are checked only against the {SUMS_NAME} of that revision",
            file=sys.stderr,
        )
    if not args.check:
        print(f"downloading {', '.join(args.groups)} from {REPO_ID}@{args.revision} into {dest}/")
        try:
            download(args.groups, revision=args.revision, dest=dest)
        except Exception as error:
            message = download_failure(error, args.revision, dest)
            if message is None:
                raise
            print(message, file=sys.stderr)
            return 1

    downloaded_sums = dest / SUMS_NAME
    if downloaded_sums.is_file() and pinned and sha256_file(downloaded_sums) != SUMS_SHA256:
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
        remedy = "delete the listed files and run the fetch again" if mismatched else "run the fetch again"
        counts = f"{len(mismatched)} mismatched, {len(absent)} missing of the files in {sums_path}"
        print(f"{counts}: {remedy}", file=sys.stderr)
        return 1
    print(f"OK: {len(present)} of the {len(sums)} files listed in {sums_path} are under {dest}/ and match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
