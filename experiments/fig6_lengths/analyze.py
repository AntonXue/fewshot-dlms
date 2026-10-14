# ported from editdiff/experiments/length_project/lpx_ds.py, experiments/length_project/lpx_tables.py and experiments/algo_investigation/fs_analyze_lib.py::{load, cell} @ d276fa6
"""Figure 6 (Section 4.3) and the appendix fidelity row (Appendix B.5), from the frozen pools (CPU only, no model).

Section 4.3 asks how many tokens each masked target should get. List Functions-Easy, 64 report families, M = 4
demonstrations, J in {1, 2, 4, 8} targets, ordinary sampling (beta = 0: every candidate has weight 1/K). A pool file
holds one record per family (see run.py), one file per model, J and allocation condition:

    <pools>/<model>/list_easy_fam_M4_J{J}_<condition>.jsonl

    oracle    every target gets its reference length L*_j; K = 64 (the Section 4.1/4.2 pools of the same families)
    learned   every candidate draws its own lengths from the frozen length head (lengths.LengthHead.sample); K = 64 / J
    delta<d>  every candidate gets L*_j + d at every target, d in {-2, -1, +1, +2} (and 0, see below), J in {1, 2},
              K = 64 / J; a family with some L*_j + d < 1 is not in the pool (skipped, never clipped)

Per family, with c_kj = 1 if target j of candidate k is correct (grading.py), and per cell over families:

    A_target = mean_k mean_j c_kj    per-target accuracy (A_avg in the paper; Figure 6 plots it)
    A_all    = mean_k prod_j c_kj    every target correct
    learned - oracle                 the per-family difference of A_target, with a paired family bootstrap
    length_exact                     share of allocated lengths L_kj equal to L*_j
    Pr(L != L* | correct)            among correct targets, the share whose length was not the reference's

The length head's own accuracy (sec43_length_metrics.csv, the appendix fidelity row) comes from its distributions
q_j over the lengths 1..256 at every target site of the report prompts, as the authors' evaluation computed them:

    <pools>/<model>/length_head_report.npz     P [sites, 256] (float32), ids "<family>|<J>|<j>", y = L*_j
    <pools>/<model>/length_head_report.jsonl   one row per (family, J): episode_id, J, oracle, demo_L (the lengths
                                               of the M demonstration outputs), and the compact prompt

    P_r = sum_{|l - L*| <= r} q(l), r = 0, 1, 2; NLL = -log q(L*); E|L - L*|; mode exact; ECE of the mode

averaged within a family, then over families. The visible-demonstration prior (the empirical distribution of the M
demonstration output lengths, no learning) is the reference row.

Intervals: 10,000 family resamples, percentile 95 %, numpy default_rng seeded from the first 8 hex digits of
SHA-256("lpx-bootstrap-v1") = BOOTSTRAP_SEED plus J (A_target), 100 + J (learned - oracle) or 200 + J (P_r); the
common-family view of the offsets uses seed 1000 + J (lpx_tables.py). Families are resampled in pool-file order
(A_target, offsets) or in sorted order (paired differences, P_r), as in the authors' analysis.

Outputs in --out: sec43_downstream.csv (Figure 6, top row), sec43_length_perturbation.csv (Figure 6, bottom row),
sec43_length_metrics.csv (appendix fidelity row), families.csv (per-family values), coverage.json.

--compare DIR prints, for every table also in DIR, the largest |difference| per numeric column, and for every
downstream cell a resampling verdict (README.md, fidelity level 3): is the frozen A_avg inside the 95 % CI of the
pools analysed here?

    python experiments/fig6_lengths/analyze.py --pools artifacts/pools/fig6 --compare data/tables/fig6
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fewshot_dlms.grading import answer
from fewshot_dlms.grading import grade
from fewshot_dlms.lengths import length_mass

HERE = Path(__file__).resolve().parent
FETCH_HINT = "run: python -m fewshot_dlms.fetch pools java"
FROZEN_CONFIG = HERE / "frozen_config.json"

# The protocol of Section 4.3, fixed before the report run (frozen_config.json, "protocol").
TASK = "list_easy_fam"
BANK = "report"
M = 4
J_GRID = [1, 2, 4, 8]
OFFSET_J = [1, 2]
DELTAS = [-2, -1, 1, 2]
K_MAX = 64  # the analysis keeps at most the first 64 candidates of a record
NBOOT = 10000
# (model as in the pool directories and fewshot_dlms.adapters, backbone as in the tables and the head checkpoints)
MODELS = [("llada", "llada"), ("nemotron_pt", "nemotron"), ("dqwen", "dqwen")]
BACKBONE = dict(MODELS)
MODEL_OF_BACKBONE = {backbone: model for model, backbone in MODELS}
# The delta = 0 arm of the offset study. For dQwen3.5 it is a rerun of the oracle allocation in the same session as its
# offsets (its kernels drift between sessions, tests/gpu_util.py), so its offsets are not paired with the oracle pool.
ZERO_ARM = {"llada": "oracle", "nemotron_pt": "oracle", "dqwen": "delta0"}

BOOTSTRAP_SEED = int(hashlib.sha256(b"lpx-bootstrap-v1").hexdigest()[:8], 16)
COMMON_SEED = 1000  # the common-family view of the offsets: default_rng(COMMON_SEED + J)
COMMON = "common to all deltas"
ALL_VALID = "all valid for this delta"

# The tables of --compare and the columns that identify a row.
KEYS = {
    "sec43_length_metrics": ["backbone", "J", "run"],
    "sec43_downstream": ["backbone", "condition", "J"],
    "sec43_length_perturbation": ["backbone", "J", "delta", "family_set"],
}


def condition_name(delta):
    """The condition of a forced offset: 'delta-2', 'delta-1', 'delta0', 'delta+1', 'delta+2'."""
    return "delta0" if delta == 0 else f"delta{delta:+d}"


def delta_of(condition):
    """The offset of a condition: 0 for the oracle and for 'delta0', d for 'delta<d>'."""
    return 0 if condition == "oracle" else int(condition.replace("delta", ""))


# ---------------------------------------------------------------------------------------------
# Reading pools
# ---------------------------------------------------------------------------------------------


def read_records(path):
    """The records of a pool file in file order, the first record of each family only (fs_analyze_lib.load).

    Blank lines and lines that do not parse (an interrupted write) are skipped. A family can appear twice in a file
    (the LLaDA-1.5 oracle pool at J = 8 holds one family twice); the first record counts.
    """
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.setdefault(record["episode_id"], record)
    return list(records.values())


def load_pool(pools, model, J, condition):
    """The records of one pool, each cut to its first K_MAX candidates; [] if the file does not exist."""
    path = Path(pools) / model / f"{TASK}_M{M}_J{J}_{condition}.jsonl"
    if not path.exists():
        return []
    return [dict(record, candidates=record["candidates"][:K_MAX]) for record in read_records(path)]


def candidate_lengths(record, candidate):
    """The J allocated lengths of one candidate: its own alloc_L, else the record's, else the oracle lengths."""
    if "alloc_L" in candidate:
        return candidate["alloc_L"]
    if record.get("alloc_L"):
        return record["alloc_L"]
    return record["oracle_L"]


def family_cell(model, record):
    """K x J matrices of correctness and well-formedness, the allocations and the oracle lengths of one family."""
    J = record["J"]
    candidates = record["candidates"]
    correct = np.zeros((len(candidates), J))
    parsed = np.zeros((len(candidates), J), bool)
    for k, candidate in enumerate(candidates):
        for j in range(J):
            is_correct, parse_ok = grade(record["grader"], answer(candidate["texts"][j], model), record["gold"][j])
            correct[k, j] = is_correct
            parsed[k, j] = parse_ok
    allocations = [candidate_lengths(record, candidate) for candidate in candidates]
    oracle = record["oracle_L"]
    exact = np.array([[allocation[j] == oracle[j] for j in range(J)] for allocation in allocations])
    return {
        "family": record["episode_id"],
        "correct": correct,
        "parsed": parsed,
        "allocations": allocations,
        "oracle": oracle,
        "exact": exact,
    }


# ---------------------------------------------------------------------------------------------
# Downstream accuracy (Figure 6)
# ---------------------------------------------------------------------------------------------


def percentile_ci(resampled_means):
    """The 2.5th and 97.5th percentiles of the resampled means."""
    return np.percentile(resampled_means, 2.5), np.percentile(resampled_means, 97.5)


def downstream_row(backbone, condition, J, records, cells):
    """One condition x J cell at beta = 0 (lpx_ds.py): accuracies with a family bootstrap and the allocation facts."""
    A_target = np.array([cell["correct"].mean() for cell in cells])
    A_all = np.array([cell["correct"].all(1).mean() for cell in cells])
    correct = [cell["correct"] > 0 for cell in cells]
    correct_but_not_exact = [(cell["correct"] > 0) & ~cell["exact"] for cell in cells]
    n_correct = sum(x.sum() for x in correct)
    if n_correct:
        not_exact_given_correct = float(sum(x.sum() for x in correct_but_not_exact) / n_correct)
    else:
        not_exact_given_correct = float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED + J)
    idx = rng.integers(0, len(cells), size=(NBOOT, len(cells)))
    lo, hi = percentile_ci(A_target[idx].mean(1))
    return {
        "bank": BANK,
        "backbone": backbone,
        "condition": condition,
        "J": J,
        "K": int(min(len(record["candidates"]) for record in records)),
        "families": len(cells),
        "A_target": A_target.mean(),
        "A_target_lo": lo,
        "A_target_hi": hi,
        "A_all": A_all.mean(),
        "length_exact": float(np.mean([cell["exact"].mean() for cell in cells])),
        "generated_tokens_per_family": float(np.mean([sum(sum(a) for a in cell["allocations"]) for cell in cells])),
        "oracle_tokens_per_family_per_candidate": float(np.mean([sum(cell["oracle"]) for cell in cells])),
        "pr_L_ne_Lstar_given_correct": not_exact_given_correct,
        "format_failure": float(np.mean([1 - cell["parsed"].mean() for cell in cells])),
        "diff_vs_oracle": np.nan,
        "diff_lo": np.nan,
        "diff_hi": np.nan,
    }


def paired_difference(row, values, oracle_values, J):
    """Fill in A_target minus the oracle's over the families of both pools (sorted ids), with a paired bootstrap."""
    ids = sorted(set(values) & set(oracle_values))
    d = np.array([values[i] - oracle_values[i] for i in ids])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 100 + J)
    resampled = d[rng.integers(0, len(d), size=(NBOOT, len(d)))].mean(1)
    lo, hi = percentile_ci(resampled)
    row["diff_vs_oracle"] = d.mean()
    row["diff_lo"] = lo
    row["diff_hi"] = hi


def condition_cells(pools, model, conditions, Js, family_rows):
    """Rows of every (J, condition) with a pool, in that order, and the per-family A_target of each."""
    backbone = BACKBONE[model]
    rows = []
    per_family = {}
    for J in Js:
        for condition in conditions:
            records = load_pool(pools, model, J, condition)
            if not records:
                continue
            cells = [family_cell(model, record) for record in records]
            rows.append(downstream_row(backbone, condition, J, records, cells))
            per_family[(condition, J)] = {cell["family"]: cell["correct"].mean() for cell in cells}
            for cell in cells:
                family_rows.append({
                    "model": model,
                    "backbone": backbone,
                    "condition": condition,
                    "J": J,
                    "family": cell["family"],
                    "K": len(cell["allocations"]),
                    "A_target": cell["correct"].mean(),
                    "A_all": cell["correct"].all(1).mean(),
                    "length_exact": cell["exact"].mean(),
                    "format_failure": 1 - cell["parsed"].mean(),
                })
    for row in rows:
        if row["condition"] == "oracle" or ("oracle", row["J"]) not in per_family:
            continue
        paired_difference(row, per_family[(row["condition"], row["J"])], per_family[("oracle", row["J"])], row["J"])
    return rows, per_family


def common_family_rows(backbone, per_family):
    """The offsets on the families valid at every delta (lpx_tables.py), so that each curve compares the same families.

    The delta = 0 arm is labelled 'oracle' here whichever pool it comes from (ZERO_ARM).
    """
    rows = []
    for J in sorted({J for _, J in per_family}):
        arms = {delta_of(condition): values for (condition, j), values in per_family.items() if j == J}
        common = set.intersection(*(set(values) for values in arms.values()))
        for delta in sorted(arms):
            v = np.array([value for family, value in arms[delta].items() if family in common])
            rng = np.random.default_rng(COMMON_SEED + J)
            resampled = v[rng.integers(0, len(v), size=(NBOOT, len(v)))].mean(1)
            lo, hi = percentile_ci(resampled)
            rows.append({
                "backbone": backbone,
                "condition": "oracle" if delta == 0 else f"delta{delta:+d}",
                "J": J,
                "delta": delta,
                "families": len(v),
                "A_target": v.mean(),
                "A_target_lo": lo,
                "A_target_hi": hi,
                "family_set": COMMON,
            })
    return rows


# ---------------------------------------------------------------------------------------------
# Length-head fidelity (appendix fidelity row)
# ---------------------------------------------------------------------------------------------


def family_mean(values, in_cell, family_of_site, families):
    """The mean over families of the within-family mean of a per-site value, over the sites of one J."""
    return float(np.mean([values[in_cell][family_of_site == family].mean() for family in families]))


def length_metric_rows(backbone, run, npz_path, head_rows_path):
    """Per J: the learned head's P_0, P_1, P_2 (+ family bootstrap), NLL, E|L - L*|, mode exact, ECE; then the
    visible-demonstration prior's P_r, E|L - L*| and mode exact (lpx_ds.py --run)."""
    z = np.load(npz_path)
    P = z["P"].astype(np.float64)
    P /= P.sum(1, keepdims=True)
    ids = z["ids"].tolist()
    y = z["y"]
    lmax = P.shape[1]
    lengths = np.arange(1, lmax + 1)
    family_of_site = np.array([i.split("|")[0] for i in ids])
    J_of_site = np.array([int(i.split("|")[1]) for i in ids])

    p0 = length_mass(P, y, 0)
    p1 = length_mass(P, y, 1)
    p2 = length_mass(P, y, 2)
    nll = -np.log(np.clip(p0, 1e-300, None))
    expected_abs_error = (P * np.abs(lengths[None] - y[:, None])).sum(1)
    mode = P.argmax(1) + 1
    confidence = P.max(1)

    rows = []
    for J in sorted(set(J_of_site)):
        s = J_of_site == J
        f = family_of_site[s]
        families = sorted(set(f))
        bins = np.minimum((confidence[s] * 10).astype(int), 9)
        ece = sum(
            (bins == b).mean() * abs(confidence[s][bins == b].mean() - (mode[s] == y[s])[bins == b].mean())
            for b in range(10)
            if (bins == b).any()
        )
        row = {
            "bank": BANK,
            "backbone": backbone,
            "run": run,
            "J": int(J),
            "families": len(families),
            "sites": int(s.sum()),
            "P0": family_mean(p0, s, f, families),
            "P1": family_mean(p1, s, f, families),
            "P2": family_mean(p2, s, f, families),
            "nll": family_mean(nll, s, f, families),
            "expected_abs_error": family_mean(expected_abs_error, s, f, families),
            "mode_exact": family_mean((mode == y).astype(float), s, f, families),
            "ece_mode": float(ece),
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED + 200 + int(J))
        idx = rng.integers(0, len(families), size=(NBOOT, len(families)))
        for name, v in (("P0", p0), ("P1", p1), ("P2", p2)):
            per_family = np.array([v[s][f == family].mean() for family in families])
            lo, hi = percentile_ci(per_family[idx].mean(1))
            row[f"{name}_lo"] = float(lo)
            row[f"{name}_hi"] = float(hi)
        rows.append(row)

    # The visible-demonstration prior: q_demo(l) = share of the M demonstration outputs with l tokens.
    demo_lengths = {}
    with open(head_rows_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                head_row = json.loads(line)
                demo_lengths[(head_row["episode_id"], head_row["J"])] = head_row["demo_L"]
    prior = np.zeros_like(P)
    for k, site in enumerate(ids):
        family, J, _ = site.split("|")
        counts = np.bincount(np.clip(np.asarray(demo_lengths[(family, int(J))], int), 1, lmax), minlength=lmax + 1)
        counts = counts[1:].astype(float)
        prior[k] = counts / counts.sum()
    q0 = length_mass(prior, y, 0)
    q1 = length_mass(prior, y, 1)
    q2 = length_mass(prior, y, 2)
    prior_abs_error = (prior * np.abs(lengths[None] - y[:, None])).sum(1)
    prior_mode_exact = ((prior.argmax(1) + 1) == y).astype(float)
    for J in sorted(set(J_of_site)):
        s = J_of_site == J
        f = family_of_site[s]
        families = sorted(set(f))
        rows.append({
            "bank": BANK,
            "backbone": backbone,
            "run": "visible_demo_prior",
            "J": int(J),
            "families": len(families),
            "sites": int(s.sum()),
            "P0": family_mean(q0, s, f, families),
            "P1": family_mean(q1, s, f, families),
            "P2": family_mean(q2, s, f, families),
            "expected_abs_error": family_mean(prior_abs_error, s, f, families),
            "mode_exact": family_mean(prior_mode_exact, s, f, families),
        })
    return rows


# ---------------------------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------------------------


def head_runs():
    """The run name of each backbone's frozen head (frozen_config.json): the `run` column of the tables."""
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    return {backbone: entry["run"] for backbone, entry in config["backbones"].items()}


def analyse(pools, models):
    """The three Section 4.3 tables (lists of DataFrames to concatenate, per backbone) and the per-family rows."""
    runs = head_runs()
    metrics = []
    downstream = []
    perturbation = []
    family_rows = []
    coverage = {}
    for model in models:
        backbone = BACKBONE[model]
        run = runs[backbone]

        npz_path = Path(pools) / model / "length_head_report.npz"
        head_rows_path = Path(pools) / model / "length_head_report.jsonl"
        if npz_path.exists() and head_rows_path.exists():
            metrics.append(pd.DataFrame(length_metric_rows(backbone, run, npz_path, head_rows_path)))

        rows, per_family = condition_cells(pools, model, ["oracle", "learned"], J_GRID, family_rows)
        for (condition, J), values in per_family.items():
            coverage[f"{model}|{condition}|J{J}"] = len(values)
        if rows:
            frame = pd.DataFrame(rows)
            frame["run"] = run
            downstream.append(frame)
            for row in rows:
                print(
                    f"{model:11s} {row['condition']:7s} J={row['J']} fam {row['families']:2d} K {row['K']:2d}"
                    f" | A_avg {row['A_target']:.3f} [{row['A_target_lo']:.3f}, {row['A_target_hi']:.3f}]"
                    f" | exact L {row['length_exact']:.3f} | learned - oracle {row['diff_vs_oracle']:+.3f}",
                    flush=True,
                )

        offsets = [ZERO_ARM[model]] + [condition_name(delta) for delta in DELTAS]
        offset_family_rows = []
        rows, per_family = condition_cells(pools, model, offsets, OFFSET_J, offset_family_rows)
        family_rows.extend(row for row in offset_family_rows if row["condition"] != "oracle")
        for (condition, J), values in per_family.items():
            coverage[f"{model}|{condition}|J{J}"] = len(values)
        if rows:
            frame = pd.DataFrame(rows)
            frame["delta"] = frame.condition.map(delta_of)
            frame["family_set"] = ALL_VALID
            frame = pd.concat([frame, pd.DataFrame(common_family_rows(backbone, per_family))], ignore_index=True)
            perturbation.append(frame)
    return {"metrics": metrics, "downstream": downstream, "perturbation": perturbation}, family_rows, coverage


def write_tables(out, parts, family_rows, coverage):
    """Write the tables to `out` (CSV with pandas' defaults, as in the paper's analysis); returns them by name."""
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, key in (("sec43_length_metrics", "metrics"), ("sec43_downstream", "downstream"), ("sec43_length_perturbation", "perturbation")):
        if not parts[key]:
            print(f"no inputs for {name}.csv")
            continue
        frames[name] = pd.concat(parts[key], ignore_index=True)
        frames[name].to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(family_rows).to_csv(out / "families.csv", index=False)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=1))
    print(f"wrote {', '.join(sorted(frames))} to {out}")
    return frames


# ---------------------------------------------------------------------------------------------
# --compare
# ---------------------------------------------------------------------------------------------


def max_abs_diff(a, b):
    """Largest |a - b| over two aligned float arrays; NaN in both counts as equal, NaN in one as infinite."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    diff[np.isnan(diff)] = np.inf
    return float(diff.max()) if len(diff) else 0.0


def compare(frames, reference):
    """Print the largest |difference| per numeric column against the tables in `reference`, then the resampling
    verdict of every downstream cell: is the frozen A_avg inside the 95 % CI computed here?

    CSVs are parsed with float_precision="round_trip", so a value written by pandas reads back exactly.
    """
    for name, keys in KEYS.items():
        path = Path(reference) / f"{name}.csv"
        if name not in frames or not path.exists():
            continue
        theirs = pd.read_csv(path, float_precision="round_trip")
        merged = frames[name].merge(theirs, on=keys, suffixes=("", "_ref"))
        columns = [c for c in theirs.columns if c not in keys and theirs[c].dtype.kind == "f" and c in frames[name]]
        worst = max(((max_abs_diff(merged[c], merged[c + "_ref"]), c) for c in columns), default=(0.0, "-"))
        where = f" ({worst[1]})" if worst[0] > 0 else ""
        print(f"{name}: {len(merged)} rows in common; largest |diff| {worst[0]:.3e}{where}")
        for column in columns:
            print(f"  {column:40s} max |diff| {max_abs_diff(merged[column], merged[column + '_ref']):.3e}")

    path = Path(reference) / "sec43_downstream.csv"
    if "sec43_downstream" not in frames or not path.exists():
        return
    frozen = pd.read_csv(path, float_precision="round_trip").set_index(["backbone", "condition", "J"])
    for _, cell in frames["sec43_downstream"].iterrows():
        key = (cell.backbone, cell.condition, cell.J)
        if key not in frozen.index:
            continue
        value = frozen.loc[key, "A_target"]
        inside = cell.A_target_lo <= value <= cell.A_target_hi
        verdict = "inside" if inside else "outside"
        print(
            f"{MODEL_OF_BACKBONE[cell.backbone]} {cell.condition} J={cell.J}: frozen A_avg {value:.3f},"
            f" rerun CI [{cell.A_target_lo:.3f}, {cell.A_target_hi:.3f}] -> frozen mean {verdict} the rerun's 95 % CI"
            f" ({cell.families} families)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pools", required=True, help="directory with <model>/list_easy_fam_M4_J{J}_<condition>.jsonl")
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--models", default=",".join(model for model, _ in MODELS))
    parser.add_argument("--compare", default=None, help="directory of reference tables, e.g. data/tables/fig6")
    args = parser.parse_args()

    pools = Path(args.pools)
    models = args.models.split(",")
    if not pools.is_dir() or not any((pools / model).is_dir() for model in models):
        print(f"no Figure 6 pools in {pools}; {FETCH_HINT}", file=sys.stderr)
        sys.exit(2)

    parts, family_rows, coverage = analyse(pools, models)
    frames = write_tables(Path(args.out), parts, family_rows, coverage)
    if args.compare:
        compare(frames, args.compare)


if __name__ == "__main__":
    main()
