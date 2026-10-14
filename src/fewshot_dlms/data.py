"""Loader for the frozen benchmark families shipped in data/benchmarks/.

There are four benchmarks and two banks of families each: "dev" (used to select beta) and "report"
(used for every reported number). A family is one episode specification: an instruction, 16 ordered
demonstrations and 8 ordered targets.

List Functions-Easy and SALT ship verbatim. SST-2 and MIT Movie ship without text (ids only); run
`python data/benchmarks/build.py` once to rebuild their text from the public corpora. The rebuilt
families are cached in data/benchmarks/_built/ and are checked against data/benchmarks/sha256.json.

The 127 Java helper-propagation episodes of Section 5 ship in data/java/ (see PROVENANCE.md there);
`load_java_episodes` reads them.
"""

import hashlib
import json
from pathlib import Path

BENCHMARKS = ["sst2_fam", "list_easy_fam", "salt_rep_fam", "movie_dir_fam"]
# The benchmarks that ship as ids only; data/benchmarks/build.py rebuilds their text.
TEXT_FREE_BENCHMARKS = ["sst2_fam", "movie_dir_fam"]
BANKS = ("dev", "report")

# The directory is found relative to this file, so the package must be installed from a checkout
# (`pip install -e .`).
BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"
JAVA_DIR = Path(__file__).resolve().parents[2] / "data" / "java"


def read_jsonl(path: Path) -> list[dict]:
    """Read a newline-delimited JSON file into a list of records."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def family_sha(family: dict) -> str:
    """SHA-256 of one full-text family: the unit recorded in data/benchmarks/sha256.json."""
    canonical = json.dumps(family, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_families(task: str, bank: str = "report") -> list[dict]:
    """Return the frozen families of one benchmark and bank, in manifest order.

    Each record has exactly the keys of the authors' frozen manifest: episode_id, bank, source,
    rule_id, category, split, instruction, demos, targets, max_M, max_J, grader and meta, where
    demos and targets are lists of {id, input, output}.

    Raises FileNotFoundError when the text of SST-2 / MIT Movie has not been rebuilt yet.
    """
    if task not in BENCHMARKS:
        raise ValueError(f"unknown task {task!r}; expected one of {BENCHMARKS}")
    if bank not in BANKS:
        raise ValueError(f"unknown bank {bank!r}; expected one of {BANKS}")

    verbatim = BENCHMARK_DIR / bank / f"{task}.jsonl"
    rebuilt = BENCHMARK_DIR / "_built" / bank / f"{task}.jsonl"
    for path in (verbatim, rebuilt):
        if path.exists():
            return read_jsonl(path)
    raise FileNotFoundError("run python data/benchmarks/build.py")


def load_java_episodes() -> list[dict]:
    """Return the 127 Java helper-propagation episodes of Section 5, in manifest order.

    Each record has the fields of the authors' frozen manifest (episode_id, cell, M, J, repository,
    commit, file, helper, demo_sites, target_sites, oracle_lengths, the flags of Appendix C, ...), and
    the text the canvas is built from: `helper_text`, and for every site in `demos` and
    `targets_visible` the enclosing method before the commit (`method_before`) with the line range
    [i1, i2) of the changed span. The reference answers are in `reference["targets"]`. The first
    argument of `canvas.build_java_canvas` is such a record.
    """
    return read_jsonl(JAVA_DIR / "episodes.jsonl")
