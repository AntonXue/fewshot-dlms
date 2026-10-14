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

Figure 4 (single target, SST-2, M = 4 demonstrations) writes, for the two causal interfaces (nemotron_ar =
Nemotron's causal suffix, qwen = Qwen3-8B's free-length continuation),

  fig4/<model>_sst2_M4.jsonl         the first two complete pool records of the report families (K = 64
                                     continuations each, with token ids, ell1 and ell0), run metadata removed
                                     as above. These two records carry the frozen run's prompt checks (the
                                     decoded P1 and P0 prompts).

The text of SST-2 and MIT Movie is not redistributed (data/benchmarks/README.md). In records of those
benchmarks every field that holds their text (the target inputs and the decoded prompts) is replaced by its
SHA-256 digest, under the same key with "_sha256" appended; the answers stay (SST-2's are letter codes).
tests/test_canvas.py compares the digest of the Nemotron prompt text it builds with the stored one;
tests/test_gpu_replay.py does the same for both models and re-scores their stored continuations on a GPU;
tests/test_fixture_leaks.py checks that no benchmark sentence is left.

Usage:
    python tools/extract_fixtures.py --src <run directory of the Section 4.2 grid>
    python tools/extract_fixtures.py --fig4-src <run directory of the single-target sweep>

The Section 4.2 run directory holds pools/<model>/list_easy_fam_M4_J2.jsonl and analysis/families.csv; the
single-target one holds pools/report/<model>/sst2_fam_M4_J1.jsonl.

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
from fewshot_dlms.data import TEXT_FREE_BENCHMARKS

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


def first_complete_records(pool_path: Path, n: int, k: int = K) -> list[dict]:
    """The first complete record of each family, for the first `n` families, without run metadata.

    A record is complete when it has all `k` candidates (K by default). Pool files are appended to by parallel
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
            if len(record["candidates"]) == k and record["episode_id"] not in record_of_family:
                record_of_family[record["episode_id"]] = record
    records = [record_of_family[episode_id] for episode_id in sorted(record_of_family)[:n]]
    return [{key: value for key, value in record.items() if key not in RUN_METADATA} for record in records]


# Figure 4: one entry per fixture, as for Figure 5; K = 64 continuations per complete record.
FIG4_K = 64
FIG4_FIXTURES = [
    {"stem": "nemotron_ar_sst2_M4", "pool": "pools/report/nemotron_ar/sst2_fam_M4_J1.jsonl"},
    {"stem": "qwen_sst2_M4", "pool": "pools/report/qwen/sst2_fam_M4_J1.jsonl"},
]


# Fields of a pool record that hold benchmark text: a list of target inputs, and the decoded prompts in "checks".
TEXT_FIELDS = ("target_inputs",)
CHECK_TEXT_FIELDS = ("p1_prompt_text", "p0_prompt_text")


def sha256_text(text: str) -> str:
    """SHA-256 of a text in UTF-8, as hex."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def without_benchmark_text(record: dict) -> dict:
    """The record with each field in TEXT_FIELDS / CHECK_TEXT_FIELDS replaced by its digest(s), key + "_sha256".

    Applies to SST-2 and MIT Movie records (data.TEXT_FREE_BENCHMARKS); other records are returned unchanged.
    Keys keep their order. Raises ValueError if prompt text ("Input:") is left anywhere in the result.
    """
    if record["source"] not in TEXT_FREE_BENCHMARKS:
        return record
    hashed = {}
    for key, value in record.items():
        if key in TEXT_FIELDS:
            hashed[key + "_sha256"] = [sha256_text(text) for text in value]
        elif key == "checks" and value:
            checks = {}
            for check, result in value.items():
                if check in CHECK_TEXT_FIELDS:
                    checks[check + "_sha256"] = sha256_text(result)
                else:
                    checks[check] = result
            hashed[key] = checks
        else:
            hashed[key] = value
    if "Input:" in json.dumps(hashed):
        raise ValueError(f"{record['episode_id']}: prompt text left after hashing")
    return hashed


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
    # SHA-256 of the head as published (artifacts/heads/llada.pt): the authors' final.pt re-saved without the run
    # directory pickled in its training config (tools/assemble_artifacts.py), tensors identical. The evaluation
    # records the original file's SHA-256, written as trained_checkpoint_sha256.
    "published_checkpoint_sha256": "a49355771567e3b091e8a5d79b09aadd460b677af0f850af6d4afe27360e1452",
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
            "checkpoint_sha256": spec["published_checkpoint_sha256"],
            "trained_checkpoint_sha256": str(predictions["ckpt_sha"]),
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


def extract_fig4(src: Path, dest: Path) -> None:
    """Write the Figure 4 pool fixture of every causal interface in FIG4_FIXTURES."""
    out_dir = dest / "fig4"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fixture in FIG4_FIXTURES:
        records = first_complete_records(src / fixture["pool"], N_FAMILIES, FIG4_K)
        missing = [record["episode_id"] for record in records if not record.get("checks")]
        records = [without_benchmark_text(record) for record in records]
        if missing:
            raise ValueError(f"{fixture['stem']}: records without the prompt checks: {missing}")
        with (out_dir / f"{fixture['stem']}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"fig4/{fixture['stem']}: {len(records)} families {[record['episode_id'] for record in records]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src",
        type=Path,
        help="run directory of the Section 4.2 grid (fig5) or of the Section 4.3 length project (fig6_head)",
    )
    parser.add_argument("--fig4-src", type=Path, help="run directory of the single-target sweep (Figure 4 fixtures)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="default: data/fixtures of this repository")
    parser.add_argument("--fixture", choices=["fig5", "fig6_head"], default="fig5", help="which fixture to write (default: fig5)")
    args = parser.parse_args()
    if args.fixture == "fig6_head":
        if args.src is None:
            parser.error("--fixture fig6_head needs --src")
        extract_fig6_head(args.src, args.dest)
        return
    if args.src is None and args.fig4_src is None:
        parser.error("give --src, --fig4-src or both")
    if args.src is not None:
        extract_fig5(args.src, args.dest)
    if args.fig4_src is not None:
        extract_fig4(args.fig4_src, args.dest)


if __name__ == "__main__":
    main()
