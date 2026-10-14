"""Provenance of data/benchmarks/: how the frozen families were exported from the authors' run.

One-off script, kept for the record; the repository works without it. It reads the manifests
<src>/{dev,report}/<task>.jsonl that the authors froze before running any model and writes

  <dest>/<bank>/<task>.jsonl      List Functions-Easy and SALT, copied byte for byte
  <dest>/<bank>/<task>.ids.jsonl  SST-2 and MIT Movie, with the `input` and `output` of every
                                  demonstration and target set to null and a `source_ref` naming the
                                  upstream record (rebuilt by data/benchmarks/build.py)
  <dest>/sha256.json              SHA-256 of every full-text family,
                                  keyed "<bank>/<task>/<episode_id>"

The hashes are computed on the full-text families, before the text of SST-2 / MIT Movie is removed.

Usage:
    python tools/export_benchmarks.py --src <frozen run>/manifests
"""

import argparse
import json
import shutil
from pathlib import Path

from fewshot_dlms.data import BANKS, BENCHMARK_DIR, BENCHMARKS, family_sha, read_jsonl

TEXT_FREE_TASKS = ["sst2_fam", "movie_dir_fam"]

# The upstream record of each demonstration / target is listed in the family's meta, in entry order.
# (The entry ids of SST-2 are positional, d00..d15 and t0..t7, so they cannot be the reference.)
RECORD_ID_KEYS = {"demos": "demo_record_ids", "targets": "target_record_ids"}


def strip_text(family: dict) -> dict:
    """Return the family with the text of every demonstration and target removed."""
    stripped = dict(family)
    for role, meta_key in RECORD_ID_KEYS.items():
        record_ids = family["meta"][meta_key]
        stripped[role] = [
            {"id": entry["id"], "input": None, "output": None, "source_ref": record_id}
            for entry, record_id in zip(family[role], record_ids, strict=True)
        ]
    return stripped


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export(src: Path, dest: Path) -> None:
    """Export every bank and benchmark from `src` into `dest` and write sha256.json."""
    shas = {}
    for bank in BANKS:
        for task in BENCHMARKS:
            source_file = src / bank / f"{task}.jsonl"
            families = read_jsonl(source_file)
            for family in families:
                shas[f"{bank}/{task}/{family['episode_id']}"] = family_sha(family)

            if task in TEXT_FREE_TASKS:
                write_jsonl(dest / bank / f"{task}.ids.jsonl", [strip_text(f) for f in families])
            else:
                (dest / bank).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, dest / bank / f"{task}.jsonl")
            print(f"{bank}/{task}: {len(families)} families")

    sha_text = json.dumps(shas, indent=1, sort_keys=True) + "\n"
    (dest / "sha256.json").write_text(sha_text, encoding="utf-8")
    print(f"sha256.json: {len(shas)} families")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="manifests directory of the authors' frozen run (has dev/ and report/); not shipped",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=BENCHMARK_DIR,
        help="output directory (default: data/benchmarks)",
    )
    args = parser.parse_args()
    if args.src is None:
        parser.error("--src is required: the manifests directory of the authors' frozen run")
    export(args.src, args.dest)


if __name__ == "__main__":
    main()
