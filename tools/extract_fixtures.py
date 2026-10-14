"""Provenance of data/fixtures/: how the test fixtures were cut from the authors' run.

One-off script, kept for the record; the repository works without it. The fixtures let the CPU tests
reproduce a frozen result with this repository's own code, without downloading any artifacts.

Figure 5 (multi-target, M = 4 demonstrations, J = 2 targets) writes, per model,

  fig5/<model>_M4_J2.jsonl           the first two complete pool records of the report families,
                                     copied unchanged (K = 16 candidates each, with token ids and
                                     both log-likelihoods ell1 and ell0)
  fig5/expected_<model>_M4_J2.json   the authors' own per-family numbers (A_target, A_all, ess,
                                     max_w at beta = 0 and beta = 0.5), copied from the families.csv
                                     of the same run

tests/test_fixture_fig5.py recomputes those numbers from the pool records with scoring.py and
grading.py and requires agreement to 1e-12.

Usage:
    python tools/extract_fixtures.py --src <run directory of the Section 4.2 grid>

The run directory holds pools/<model>/list_easy_fam_M4_J2.jsonl and analysis/families.csv.
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "fixtures"

N_FAMILIES = 2  # families per fixture
K = 16  # candidates per complete pool record
M = 4
J = 2
BETAS = ("0.0", "0.5")  # as written in families.csv: ordinary sampling and the common strength
METRICS = ("A_target", "A_all", "ess", "max_w")

# One entry per fixture: its file stem, the model name in families.csv, and the pool file in the run.
FIG5_FIXTURES = [
    {"stem": "llada_M4_J2", "model": "llada", "pool": "pools/llada/list_easy_fam_M4_J2.jsonl"},
]


def first_complete_lines(pool_path: Path, n: int) -> list[str]:
    """The stored lines of the first complete record of each family, for the first `n` families.

    A record is complete when it has all K candidates. Pool files are appended to by parallel
    workers, so lines are not in manifest order; the episode ids (list_easy_fam:report:000, ...)
    sort in manifest order. Lines are returned exactly as stored.
    """
    line_of_family = {}
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if len(record["candidates"]) == K and record["episode_id"] not in line_of_family:
                line_of_family[record["episode_id"]] = line.rstrip("\n")
    return [line_of_family[episode_id] for episode_id in sorted(line_of_family)[:n]]


def authors_numbers(families_csv: Path, model: str, episode_ids: list[str]) -> dict:
    """{episode_id: {beta: {metric: value}}} read from the authors' families.csv."""
    numbers = {episode_id: {} for episode_id in episode_ids}
    with families_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model or int(row["M"]) != M or int(row["J"]) != J:
                continue
            if row["family"] in numbers and row["beta"] in BETAS:
                numbers[row["family"]][row["beta"]] = {name: float(row[name]) for name in METRICS}
    for episode_id, by_beta in numbers.items():
        if sorted(by_beta) != sorted(BETAS):
            raise ValueError(f"families.csv has no rows at beta {BETAS} for {model} {episode_id}")
    return numbers


def extract_fig5(src: Path, dest: Path) -> None:
    """Write the Figure 5 pool fixture and its expected numbers for every model in FIG5_FIXTURES."""
    out_dir = dest / "fig5"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fixture in FIG5_FIXTURES:
        lines = first_complete_lines(src / fixture["pool"], N_FAMILIES)
        records = [json.loads(line) for line in lines]
        episode_ids = [record["episode_id"] for record in records]
        expected = {
            "model": fixture["model"],
            "M": M,
            "J": J,
            "source": "families.csv of the authors' Section 4.2 grid run",
            "protocol_hash": records[0]["protocol_hash"],
            "families": authors_numbers(src / "analysis" / "families.csv", fixture["model"], episode_ids),
        }
        (out_dir / f"{fixture['stem']}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (out_dir / f"expected_{fixture['stem']}.json").open("w", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
            handle.write("\n")
        print(f"fig5/{fixture['stem']}: {len(records)} families {episode_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="run directory of the Section 4.2 grid")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="default: data/fixtures of this repository")
    args = parser.parse_args()
    extract_fig5(args.src, args.dest)


if __name__ == "__main__":
    main()
