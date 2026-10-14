"""Provenance of data/fixtures/: how the test fixtures were cut from the authors' run.

One-off script, kept for the record; the repository works without it. The fixtures let the CPU tests
reproduce a frozen result with this repository's own code, without downloading any artifacts.

Figure 5 (multi-target, M = 4 demonstrations, J = 2 targets) writes, for each of the three models
(llada = LLaDA-1.5, nemotron_pt = Nemotron prefix/tail, dqwen = dQwen3.5),

  fig5/<model>_M4_J2.jsonl           the first two complete pool records of the report families
                                     (K = 16 candidates each, with token ids and both
                                     log-likelihoods ell1 and ell0), copied with the run metadata
                                     (host, gpu_env, wall_time) removed and nothing else changed
  fig5/expected_<model>_M4_J2.json   the authors' own per-family numbers (A_target, A_all, ess,
                                     max_w at beta = 0 and beta = 0.5), copied from the families.csv
                                     of the same run

tests/test_fixture_fig5.py recomputes those numbers from the pool records of each model with scoring.py
and grading.py and requires agreement to 1e-12. tests/test_gpu_replay.py rebuilds the canvas of every
record with the real tokenizer and re-scores the stored tokens of each model on a GPU.

Usage:
    python tools/extract_fixtures.py --src <run directory of the Section 4.2 grid>

The run directory holds pools/<model>/list_easy_fam_M4_J2.jsonl and analysis/families.csv.

Figure 6 (target lengths, Section 4.3) writes

  fig6/head_predictions_llada.json   for the first ten report families of List Functions-Easy at M = 4
                                     demonstrations and J = 2 targets: the frozen LLaDA-1.5 length head's
                                     distribution over the lengths 1..256 at both sites, the SHA-256 of the
                                     compact prompt it read, and the first 64 learned allocations the authors
                                     drew from it

The distributions are the authors' own predictions, copied from the frozen evaluation of the Section 4.3
run (float32 values, written in their shortest round-trip form). That evaluation is the output of the
private length-project code (lpx_model.Branch on the lp_model packed LLaDA forward pass), run once after
the configuration freeze with

    python experiments/length_project/lpx_alloc.py <length project> predict --gpu 0 --backbone llada \
        --run fin_llada_m70_100k --data listbench_report

and the allocations were drawn from them with

    python experiments/length_project/lpx_alloc.py <length project> alloc --backbone llada \
        --run fin_llada_m70_100k --bank report --K 64 --conds learned --Js 1,2,4,8

tests/test_lengths.py redraws the allocations with lengths.sample_allocations; tests/test_gpu_head.py runs
the ported head (lengths.LengthHead) on the same canvases and compares the distributions.

Usage:
    python tools/extract_fixtures.py --fixture fig6_head --src <run directory of the Section 4.3 length project>

The run directory holds data/listbench_report_llada.jsonl (the compact prompts), eval/<run>_listbench_report.npz
(the predictions) and downstream/report/llada/alloc_learned_<run>_list_easy_fam_J<J>.json (the allocations).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "fixtures"

N_FAMILIES = 2  # families per fixture
K = 16  # candidates per complete pool record
M = 4
J = 2
BETAS = ("0.0", "0.5")  # as written in families.csv: ordinary sampling and the common strength
METRICS = ("A_target", "A_all", "ess", "max_w")
# Keys of a pool record that describe the machine and the moment of the run, not the result.
RUN_METADATA = ("host", "gpu_env", "wall_time")

# One entry per fixture: its file stem, the model name in families.csv, and the pool file in the run.
FIG5_FIXTURES = [
    {"stem": "llada_M4_J2", "model": "llada", "pool": "pools/llada/list_easy_fam_M4_J2.jsonl"},
    {"stem": "nemotron_pt_M4_J2", "model": "nemotron_pt", "pool": "pools/nemotron_pt/list_easy_fam_M4_J2.jsonl"},
    {"stem": "dqwen_M4_J2", "model": "dqwen", "pool": "pools/dqwen/list_easy_fam_M4_J2.jsonl"},
]


def first_complete_records(pool_path: Path, n: int) -> list[dict]:
    """The first complete record of each family, for the first `n` families, without run metadata.

    A record is complete when it has all K candidates. Pool files are appended to by parallel
    workers, so lines are not in manifest order; the episode ids (list_easy_fam:report:000, ...)
    sort in manifest order. Apart from the keys in RUN_METADATA, every record is returned as stored
    (same keys in the same order).
    """
    record_of_family = {}
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if len(record["candidates"]) == K and record["episode_id"] not in record_of_family:
                record_of_family[record["episode_id"]] = record
    records = [record_of_family[episode_id] for episode_id in sorted(record_of_family)[:n]]
    return [{key: value for key, value in record.items() if key not in RUN_METADATA} for record in records]


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
        records = first_complete_records(src / fixture["pool"], N_FAMILIES)
        episode_ids = [record["episode_id"] for record in records]
        expected = {
            "model": fixture["model"],
            "M": M,
            "J": J,
            "source": "families.csv of the authors' Section 4.2 grid run",
            "protocol_hash": records[0]["protocol_hash"],
            "families": authors_numbers(src / "analysis" / "families.csv", fixture["model"], episode_ids),
        }
        with (out_dir / f"{fixture['stem']}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        with (out_dir / f"expected_{fixture['stem']}.json").open("w", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
            handle.write("\n")
        print(f"fig5/{fixture['stem']}: {len(records)} families {episode_ids}")


# The Figure 6 head fixture: the frozen LLaDA-1.5 head of Section 4.3 on the first report families.
FIG6_HEAD = {
    "stem": "head_predictions_llada",
    "run": "fin_llada_m70_100k",
    "backbone": "llada",
    "task": "list_easy_fam",
    "bank": "report",
    "M": 4,
    "J": 2,
    "n_families": 10,
    "K": 64,  # allocations stored per family (the run drew 64 per family and used the first 64 / J = 32)
}


def shortest_float32(value) -> float:
    """A float32 value as the shortest decimal that reads back as the same float32."""
    return float(str(np.float32(value)))


def extract_fig6_head(src: Path, dest: Path) -> None:
    """Write fig6/head_predictions_llada.json from the frozen Section 4.3 evaluation (see the module docstring)."""
    spec = FIG6_HEAD
    run = spec["run"]
    J = spec["J"]
    rows = [json.loads(line) for line in (src / "data" / f"listbench_{spec['bank']}_{spec['backbone']}.jsonl").open()]
    rows = [row for row in rows if row["J"] == J]
    episode_ids = sorted(row["episode_id"] for row in rows)[: spec["n_families"]]
    row_of = {row["episode_id"]: row for row in rows}

    predictions = np.load(src / "eval" / f"{run}_listbench_{spec['bank']}.npz")
    probs_of = dict(zip(predictions["ids"].tolist(), predictions["P"]))
    lmax = int(predictions["lmax"])

    allocation_file = f"alloc_learned_{run}_{spec['task']}_J{J}.json"
    allocations = json.loads((src / "downstream" / spec["bank"] / spec["backbone"] / allocation_file).read_text())

    canvases = []
    for episode_id in episode_ids:
        row = row_of[episode_id]
        if row["labels"] != row["oracle"] or len(row["mask_positions"]) != J:
            raise ValueError(f"unexpected compact prompt for {episode_id}")
        probs = [[shortest_float32(p) for p in probs_of[f"{episode_id}|{J}|{j}"]] for j in range(J)]
        canvases.append(
            {
                "episode_id": episode_id,
                "oracle_L": row["oracle"],
                "compact_sha256": hashlib.sha256(json.dumps(row["input_ids"]).encode()).hexdigest(),
                "compact_tokens": len(row["input_ids"]),
                "mask_positions": row["mask_positions"],
                "probs": probs,
                "allocations": allocations[episode_id][: spec["K"]],
            }
        )

    fixture = {
        "description": (
            "Frozen Section 4.3 length head (LLaDA-1.5) on the first report families of List Functions-Easy, "
            "M = 4, J = 2: per-site probabilities over the lengths 1..lmax (probs[j][c] is length c + 1), the "
            "SHA-256 of the compact prompt (JSON list of token ids), and the authors' first K learned allocations, "
            "drawn with seed (bank, backbone, family, J)"
        ),
        "source": "the authors' frozen Section 4.3 evaluation (lpx_alloc.py predict / alloc); see tools/extract_fixtures.py",
        "head": {
            "run": run,
            "backbone": spec["backbone"],
            "lmax": lmax,
            "checkpoint_sha256": str(predictions["ckpt_sha"]),
        },
        "task": spec["task"],
        "bank": spec["bank"],
        "M": spec["M"],
        "J": J,
        "canvases": canvases,
    }
    out_dir = dest / "fig6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{spec['stem']}.json").open("w", encoding="utf-8") as handle:
        json.dump(fixture, handle)
        handle.write("\n")
    print(f"fig6/{spec['stem']}: {len(canvases)} families {episode_ids[0]} .. {episode_ids[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="run directory of the Section 4.2 grid (fig5) or of the Section 4.3 length project (fig6_head)",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="default: data/fixtures of this repository")
    parser.add_argument("--fixture", choices=["fig5", "fig6_head"], default="fig5", help="which fixture to write (default: fig5)")
    args = parser.parse_args()
    if args.fixture == "fig6_head":
        extract_fig6_head(args.src, args.dest)
    else:
        extract_fig5(args.src, args.dest)


if __name__ == "__main__":
    main()
