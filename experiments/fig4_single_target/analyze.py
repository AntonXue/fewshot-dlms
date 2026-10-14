# ported from editdiff/experiments/algo_investigation/fs_analyze.py and fs_analyze_lib.py::{load, cell, weights, fam_metrics}, experiments/paper_integration/agent_a/fig5_compute.py (gold rows) and experiments/gold_reward/gr_analyze.py (gold reward per family) @ d276fa6; experiments/beta_sweep/bs_sweep.py (Appendix B.6) @ 1aeba87
"""Figure 4 (Section 4.1), the development-bank choice of beta, the sharpening residuals of Section 4.4 and the two
appendix studies that reuse these pools (Figure 9, Appendix B.3; Figure 11, Appendix B.6). CPU only, no model.

The single-target study of Section 4.1: four benchmarks (SST-2 with opaque output codes, List Functions-Easy, SALT,
Movie Director), four model interfaces (LLaDA-1.5 and dQwen3.5 canvases, Nemotron's causal suffix, Qwen3-8B free-length
continuation), M in {1, 2, 4, 8, 16} demonstrations followed by one target (J = 1), K = 64 answers per family sampled from
P1, oracle target length (Qwen3-8B: free length). A pool file holds one record per family (see run.py):

    <pools>/<bank>/<model>/<task>_M{M}_J1.jsonl           Section 4.1 (demonstrations first, target last)
    <pools>/<bank>/<model>/<task>_M4_J{J}_il.jsonl        cross-benchmark study, J in {1, 2, 4, 8} targets placed in the
                                                          family's 12-slot master layout (llada, nemotron_pt, dqwen)
    <pools>/gold/<model>/<task>_M{M}_J1.jsonl             the reference answers scored under P1 and P0 (report bank)

Per family, with weights w = softmax(beta * (ell1 - ell0)) over its K answers (scoring.pool_weights),

    A_target = sum_k w_k mean_j c_kj      accuracy (with J = 1, the accuracy of Figure 4)
    A_all    = sum_k w_k prod_j c_kj      every target correct (the cross-benchmark study)

where c_kj is 1 if target j of answer k is correct (grading.py). beta = 0 is ordinary sampling; beta = 0.5 is the common
strength of Figure 4. Cells are resampled over families (bootstrap.py): seed string mtfsp-20261003-bootstrap-v1, cell
key "<model>|<task>", one index matrix shared by every beta (paired gains) and by every M or J of that model and task.
Families are taken in the order of the pool file, as the paper's analysis took them.

Outputs in --out:
    sec41_accuracy.csv      Figure 4: accuracy and the paired gain over beta = 0 at every beta, per (model, task, M)
    gold_reward_cells.csv   Figure 4's gold row: gold example reward per target token (ell1* - ell0*) / T per cell
    gold_families.csv       the same per family (T counts Qwen3-8B's stop token)
    dev_sec41.csv           the Section 4.1 cells on the development bank (A_target, A_all and both gains)
    dev_sec42.csv           the cross-benchmark cells on the development bank
    beta_locks.json         the development-only beta of each model (Appendix B.2), recomputed from the two tables above
    sec44_power.csv         sharpening residuals (LLaDA-1.5): guidance at (beta, tau) against ordinary power sampling at
                            the matched exponent tau_eff = 1 / (1 / tau + beta), J in {1, 4, 8}
    diagnostics.csv         per cell: coverage, malformed rate, reward spread, prompt sizes, ESS at the locked beta
    cells_crossbench.csv    with --crossbench: Figure 9, A_all and its paired gain per (model, task, J), M = 4
    beta_sweep.csv          with --beta-sweep: Figure 11, A_all gain and effective support exp(H) / K against beta,
                            List Functions-Easy, M = 4, J in {1, 4}

--compare DIR [DIR ...] prints, for every table also in a DIR, the largest |difference| per numeric column, and for
every Section 4.1 cell a resampling verdict (README.md, fidelity level 3): is the frozen ordinary accuracy inside the
95 % CI of the pools analysed here?

    python experiments/fig4_single_target/analyze.py --pools artifacts/pools/fig4 --crossbench --beta-sweep \\
        --compare data/tables/fig4 data/tables/fig5
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fewshot_dlms.bootstrap import boot_sd
from fewshot_dlms.bootstrap import cell_indices
from fewshot_dlms.bootstrap import ci
from fewshot_dlms.data import BENCHMARK_DIR
from fewshot_dlms.data import read_jsonl
from fewshot_dlms.grading import answer
from fewshot_dlms.grading import grade
from fewshot_dlms.scoring import pool_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FETCH_HINT = "run: python -m fewshot_dlms.fetch pools java"

# The grid of Section 4.1 and of the cross-benchmark study, fixed before the runs. The orders are the row orders of
# the paper's tables.
TASKS = [
    ("sst2_fam", "SST-2 codes"),
    ("list_easy_fam", "List Functions-Easy"),
    ("salt_rep_fam", "SALT repetition"),
    ("movie_dir_fam", "Movie Director"),
]
MODELS41 = ["llada", "nemotron_ar", "dqwen", "qwen"]
MODELS42 = ["llada", "nemotron_pt", "dqwen"]
M_GRID = [1, 2, 4, 8, 16]
J_GRID = [1, 2, 4, 8]
CROSSBENCH_M = 4
BETAS = [0, 0.025, 0.05, 0.125, 0.25, 0.5, 1, 2]  # beta = 0 is ordinary sampling (the paper's beta grid, as frozen)
TAUS = [1, 0.85, 0.7, 0.5, 0.35, 0.25, 0.1]  # Section 4.4: temperatures of power sampling, P1^(1/tau)
FIGURE_BETA = 0.5  # the common guidance strength of Figure 4
NBOOT = 10000
BOOTSTRAP_SEED = "mtfsp-20261003-bootstrap-v1"  # cell key f"{model}|{task}"

# Appendix B.2: the development-only choice of beta. Per model, the mean A_target over its development cells (Section
# 4.1: M in {2, 4, 8, 16}; cross-benchmark study: J in {1, 2, 4, 8}), equal cell weights; the best positive beta, ties
# within 0.25 pp going to the smallest. A cell counts once at least MIN_DEV_FAMILIES of its families are complete.
LOCK_RULE = (
    "per model: mean A_target over cells (benchmark x grid point: 4.1 M in {2,4,8,16}; 4.2 J in {1,2,4,8}; Qwen3: 4.1 "
    "only), equal cell weights; best positive beta; ties within 0.25 pp -> smallest beta"
)
LOCK_SECTIONS = {"llada": ["41", "42"], "dqwen": ["41", "42"], "nemotron_ar": ["41"], "nemotron_pt": ["42"], "qwen": ["41"]}
LOCK_TIE = 0.0025
MIN_DEV_FAMILIES = 16
# LLaDA-1.5's lock was frozen first (09:56 UTC, 3 October), before the last development cells existed. It averages the
# 28 cells complete at that time: no J = 8 cell (List Functions-Easy had 13 of its 21 families, the others none), and
# the J = 1 point of the cross-benchmark study was then the Section 4.1 M = 4 pool (target last), before that point was
# moved to the master layout. analyze.py reproduces the lock from these cells and also applies the rule to every
# complete development cell (all_complete_cells in beta_locks.json), which selects the same beta.
# Its partial_cells_excluded is historical: what the lock file recorded when it was written. Every development cell of
# the frozen pools is complete, so the same field derived today is empty (partial_cells_excluded_derived).
ALL_COMPLETE_CELL_SET = "all complete development cells of the frozen pools (at least 16 families each)"
FROZEN_LOCK_CELLS = {
    "llada": {
        "without_J": 8,
        "J1_from_section41_M4": True,
        "cell_set": (
            "frozen 3 Oct development set (lock written 09:56 UTC): the 28 cells complete at that time; J = 8 excluded"
            " (List Functions-Easy J = 8 had 13 of 21 families, the other benchmarks none); the J = 1 point read from"
            " the Section 4.1 M = 4 pool (before the J = 1 master-layout correction)"
        ),
        "historical_partial_cells_excluded": ["list_easy_fam:8:13fam"],
    },
}

# Section 4.4: the sharpening residuals use LLaDA-1.5's report pools at M = 4: J = 1 is the Section 4.1 pool, J = 4 and
# J = 8 the cross-benchmark pools. Their own family resamples: cell key f"llada|{task}|44|J{J}".
POWER_MODEL = "llada"
POWER_J = [1, 4, 8]

# Appendix B.6, Figure 11: the guidance-strength sweep over the stored cross-benchmark pools of List Functions-Easy.
SWEEP_TASK = "list_easy_fam"
SWEEP_MODELS = [
    ("llada", "LLaDA-1.5"),
    ("nemotron_pt", "Nemotron-Labs-Diffusion-8B prefix/tail"),
    ("dqwen", "dQwen3.5-9B-Base"),
]
SWEEP_J = [1, 4]
SWEEP_BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]

# Per-family gold table: the benchmark names of the gold-reward probe.
GOLD_LABELS = {"sst2_fam": "SST-2", "list_easy_fam": "List Functions", "salt_rep_fam": "SALT", "movie_dir_fam": "Movie Director"}

# The tables of --compare and the columns that identify a row.
KEYS = {
    "sec41_accuracy": ["bank", "model", "source", "M", "beta"],
    "gold_reward_cells": ["model", "source", "M"],
    "dev_sec41": ["bank", "model", "source", "M", "beta"],
    "dev_sec42": ["bank", "model", "source", "J", "beta"],
    "sec44_power": ["bank", "source", "J", "tau", "beta"],
    "cells_crossbench": ["bank", "model", "source", "J", "beta"],
    "beta_sweep": ["model", "J", "beta"],
}


# ---------------------------------------------------------------------------------------------
# Reading pools
# ---------------------------------------------------------------------------------------------


def read_records(path):
    """The JSON records of a pool file, skipping blank lines and lines that do not parse (an interrupted write)."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def pool_path(pools, bank, model, task, M, J, crossbench):
    """Section 4.1 pools end in _J1.jsonl; the cross-benchmark (master layout) pools in _J{J}_il.jsonl."""
    suffix = "_il" if crossbench else ""
    return Path(pools) / bank / model / f"{task}_M{M}_J{J}{suffix}.jsonl"


def load_pool(path):
    """The pool records of one cell in file order: the first complete record (all K answers) of each family.

    One of the paper's pool files holds a family twice; the first complete record counts.
    """
    if not path.exists():
        return []
    records = {}
    for record in read_records(path):
        if len(record["candidates"]) != record["K"]:
            continue
        records.setdefault(record["episode_id"], record)
    return list(records.values())


def output_codes(bank):
    """SST-2: the set of output codes of each family ({episode_id: set}), for the well-formedness flag.

    Read from the frozen id file of the bank, which keeps every family's meta, so no rebuilt text is needed.
    """
    codes = {}
    for family in read_jsonl(BENCHMARK_DIR / bank / "sst2_fam.ids.jsonl"):
        codes[family["episode_id"]] = set(family["meta"]["codes"].values())
    return codes


def grading_name(model):
    """The model name `grading.answer` gets: as in the paper's analysis, every name starting with "qwen" is graded as "qwen"."""
    return "qwen" if model.startswith("qwen") else model


def family_summary(record, model, codes):
    """Everything the tables need from one family's pool record.

    correct[k, j] and parsed[k, j] are the correctness and well-formedness of target j of answer k; for SST-2 an answer
    is well formed when it is one of the family's output codes (`codes`), otherwise as `grading.grade` says.
    """
    candidates = record["candidates"]
    J = record["J"]
    name = grading_name(model)
    correct = np.zeros((len(candidates), J))
    parsed = np.zeros((len(candidates), J), bool)
    for k, candidate in enumerate(candidates):
        for j in range(J):
            text = answer(candidate["texts"][j], name)
            is_correct, parse_ok = grade(record["grader"], text, record["gold"][j])
            correct[k, j] = is_correct
            parsed[k, j] = (text.strip() in codes) if codes is not None else parse_ok
    oracle_lengths = record.get("oracle_L")
    return {
        "family": record["episode_id"],
        "correct": correct,
        "parsed": parsed,
        "ell1": np.array([candidate["ell1"] for candidate in candidates], float),
        "ell0": np.array([candidate["ell0"] for candidate in candidates], float),
        "prompt_tokens": record["n_tokens"],
        "target_tokens": int(sum(oracle_lengths)) if oracle_lengths else None,
        "seconds": float(record.get("seconds_gen", 0) + record.get("seconds_score", 0)),
        "n_unique": len({tuple(candidate["toks"]) for candidate in candidates}),
        "K": len(candidates),
    }


def cell_families(pools, bank, model, task, M, J, crossbench, codes):
    """family_summary of every family of one cell, in file order ([] when the pool is missing)."""
    records = load_pool(pool_path(pools, bank, model, task, M, J, crossbench))
    family_codes = codes if task == "sst2_fam" else {}
    return [family_summary(record, model, family_codes.get(record["episode_id"])) for record in records]


# ---------------------------------------------------------------------------------------------
# Cell tables (Section 4.1, cross-benchmark study, development bank)
# ---------------------------------------------------------------------------------------------


def cell_rows(bank, model, task, label, F, grid_name, grid_value, idx_key, lock_beta, nboot):
    """One row per beta for one cell, and one diagnostics row.

    Per beta: A_target and A_all with 95 % intervals, the paired gains over beta = 0 (gain_target, gain_all), the mean
    and median effective sample size 1 / sum w^2, the mean largest weight, and the mean accuracy of each target position.
    """
    idx = cell_indices(BOOTSTRAP_SEED, idx_key, len(F), nboot)
    weights = {beta: [pool_weights(f["ell1"], f["ell0"], beta) for f in F] for beta in BETAS}
    A_target = {beta: np.array([w @ f["correct"].mean(1) for w, f in zip(weights[beta], F)]) for beta in BETAS}
    A_all = {beta: np.array([w @ f["correct"].all(1) for w, f in zip(weights[beta], F)]) for beta in BETAS}
    J = F[0]["correct"].shape[1]
    rows = []
    for beta in BETAS:
        at, at_lo, at_hi = ci(A_target[beta], idx)
        aa, aa_lo, aa_hi = ci(A_all[beta], idx)
        gt, gt_lo, gt_hi = ci(A_target[beta] - A_target[0], idx)
        ga, ga_lo, ga_hi = ci(A_all[beta] - A_all[0], idx)
        ess = np.array([1.0 / (w ** 2).sum() for w in weights[beta]])
        max_w = np.array([w.max() for w in weights[beta]])
        row = {
            "bank": bank,
            "model": model,
            "benchmark": label,
            "source": task,
            grid_name: grid_value,
            "beta": beta,
            "families": len(F),
            "A_target": at,
            "A_target_lo": at_lo,
            "A_target_hi": at_hi,
            "A_all": aa,
            "A_all_lo": aa_lo,
            "A_all_hi": aa_hi,
            "gain_target": gt,
            "gain_target_lo": gt_lo,
            "gain_target_hi": gt_hi,
            "gain_all": ga,
            "gain_all_lo": ga_lo,
            "gain_all_hi": ga_hi,
            "ess_mean": float(ess.mean()),
            "ess_median": float(np.median(ess)),
            "max_weight_mean": float(max_w.mean()),
        }
        for j in range(J):
            row[f"acc_target_{j + 1}"] = float(np.mean([w @ f["correct"][:, j] for w, f in zip(weights[beta], F)]))
        rows.append(row)
    return rows, diagnostics_row(bank, model, task, label, F, grid_name, grid_value, lock_beta)


def diagnostics_row(bank, model, task, label, F, grid_name, grid_value, lock_beta):
    """Pool-level facts of one cell: coverage (some answer correct), malformed rate, the spread of the example reward
    R = ell1 - ell0 over all answers, prompt and target sizes, runtime, and ESS / largest weight at the locked beta."""
    rewards = np.concatenate([f["ell1"] - f["ell0"] for f in F])
    target_tokens = [f["target_tokens"] for f in F if f["target_tokens"] is not None]
    row = {
        "bank": bank,
        "model": model,
        "benchmark": label,
        "source": task,
        grid_name: grid_value,
        "families": len(F),
        "K": F[0]["K"],
        "coverage_any_correct_target": float(np.mean([f["correct"].mean(1).max() > 0 for f in F])),
        "coverage_all_targets_correct": float(np.mean([f["correct"].all(1).any() for f in F])),
        "coverage_target1": float(np.mean([f["correct"][:, 0].max() > 0 for f in F])),
        "mean_correct_fraction_unweighted": float(np.mean([f["correct"].mean() for f in F])),
        "malformed_rate": float(np.mean([1 - f["parsed"].mean() for f in F])),
        "unique_candidates_mean": float(np.mean([f["n_unique"] for f in F])),
        "reward_mean": float(rewards.mean()),
        "reward_sd": float(rewards.std()),
        "reward_q05": float(np.percentile(rewards, 5)),
        "reward_q50": float(np.percentile(rewards, 50)),
        "reward_q95": float(np.percentile(rewards, 95)),
        "prompt_tokens_median": float(np.median([f["prompt_tokens"] for f in F])),
        "prompt_tokens_p90": float(np.percentile([f["prompt_tokens"] for f in F], 90)),
        "prompt_tokens_max": int(max(f["prompt_tokens"] for f in F)),
        "target_tokens_mean": float(np.mean(target_tokens)) if target_tokens else None,
        "seconds_per_family": float(np.mean([f["seconds"] for f in F])),
    }
    if lock_beta is not None:
        weights = [pool_weights(f["ell1"], f["ell0"], lock_beta) for f in F]
        row["ess_mean_locked_beta"] = float(np.mean([1.0 / (w ** 2).sum() for w in weights]))
        row["max_weight_mean_locked_beta"] = float(np.mean([w.max() for w in weights]))
    return row


def section_tables(pools, bank, codes, locks, crossbench, nboot):
    """The Section 4.1 cells and (with crossbench) the cross-benchmark cells of one bank, as two lists of rows, and
    their diagnostics rows."""
    rows41 = []
    rows42 = []
    diagnostics = []
    for model in MODELS41:
        for task, label in TASKS:
            for M in M_GRID:
                F = cell_families(pools, bank, model, task, M, 1, False, codes)
                if not F:
                    continue
                check_K(F, model, task, M, 1)
                rows, diagnostic = cell_rows(bank, model, task, label, F, "M", M, f"{model}|{task}", locks.get(model), nboot)
                rows41 += rows
                diagnostic["section"] = "4.1"
                diagnostic["J"] = 1
                diagnostics.append(diagnostic)
    if crossbench:
        for model in MODELS42:
            for task, label in TASKS:
                for J in J_GRID:
                    F = cell_families(pools, bank, model, task, CROSSBENCH_M, J, True, codes)
                    if not F:
                        continue
                    check_K(F, model, task, CROSSBENCH_M, J)
                    rows, diagnostic = cell_rows(bank, model, task, label, F, "J", J, f"{model}|{task}", locks.get(model), nboot)
                    rows42 += rows
                    diagnostic["section"] = "4.2"
                    diagnostic["M"] = CROSSBENCH_M
                    diagnostics.append(diagnostic)
    return rows41, rows42, diagnostics


def check_K(F, model, task, M, J):
    """Every family of a cell must have the same number K of answers."""
    Ks = {f["K"] for f in F}
    if len(Ks) != 1:
        raise ValueError(f"{model} {task} M={M} J={J}: records with different K {sorted(Ks)}")


def section41_table(rows):
    """sec41_accuracy.csv: the Section 4.1 report cells with A_target named accuracy and the A_all columns dropped (with
    J = 1 they are the same numbers)."""
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("A_all") or c.startswith("gain_all")])
    return frame.rename(columns={
        "A_target": "accuracy",
        "A_target_lo": "ci_lo",
        "A_target_hi": "ci_hi",
        "gain_target": "paired_gain",
        "gain_target_lo": "paired_gain_lo",
        "gain_target_hi": "paired_gain_hi",
    })


def crossbench_table(rows):
    """cells_crossbench.csv (the paper's sec42_all.csv): the cross-benchmark report cells, A_all and its paired gain."""
    frame = pd.DataFrame(rows)
    keep = [c for c in frame.columns if not c.startswith("A_target") and not c.startswith("gain_target") and not c.startswith("acc_target_")]
    return frame[keep].rename(columns={
        "A_all_lo": "ci_lo",
        "A_all_hi": "ci_hi",
        "gain_all": "paired_gain",
        "gain_all_lo": "paired_gain_lo",
        "gain_all_hi": "paired_gain_hi",
    })


# ---------------------------------------------------------------------------------------------
# Development-only beta (Appendix B.2)
# ---------------------------------------------------------------------------------------------


def lock_from_cells(cells):
    """Apply the rule to a list of (label, {beta: A_target}) cells: the beta and the development statistic behind it."""
    means = {beta: float(np.mean([values[beta] for _, values in cells])) for beta in BETAS}
    positive = {beta: mean for beta, mean in means.items() if beta > 0}
    best = max(positive.values())
    candidates = sorted(beta for beta, mean in positive.items() if mean >= best - LOCK_TIE)
    beta = candidates[0]
    return {
        "beta": beta,
        "n_cells": len(cells),
        "cells": [label for label, _ in cells],
        "dev_mean_by_beta": {str(b): means[b] for b in BETAS},
        "dev_gain_vs_ordinary_pp": 100 * (means[beta] - means[0]),
        "best_positive_below_ordinary": best < means[0],
    }


def dev_cells(dev41, dev42, model):
    """The development cells of a model under the rule, in the order of the frozen lock files (task, then M or J), and the
    cells left out because fewer than MIN_DEV_FAMILIES families are complete."""
    cells = []
    partial = []
    sections = LOCK_SECTIONS[model]
    if "41" in sections and len(dev41):
        frame = dev41[(dev41.model == model) & (dev41.M >= 2)]
        for (task, M), group in frame.groupby(["source", "M"]):
            values = dict(zip(group.beta, group.A_target))
            if group.families.iloc[0] >= MIN_DEV_FAMILIES:
                cells.append((f"{task}:M={M}", values))
            else:
                partial.append(f"{task}:{M}:{group.families.iloc[0]}fam")
    if "42" in sections and len(dev42):
        frame = dev42[dev42.model == model]
        for (task, J), group in frame.groupby(["source", "J"]):
            values = dict(zip(group.beta, group.A_target))
            if group.families.iloc[0] >= MIN_DEV_FAMILIES:
                cells.append((f"{task}:J={J}", values))
            else:
                partial.append(f"{task}:{J}:{group.families.iloc[0]}fam")
    return cells, partial


def frozen_cells(dev41, cells, frozen):
    """The cells a lock frozen before the last development cells used (FROZEN_LOCK_CELLS)."""
    kept = []
    for label, values in cells:
        if label.endswith(f":J={frozen['without_J']}"):
            continue
        if frozen["J1_from_section41_M4"] and label.endswith(":J=1"):
            task = label.split(":")[0]
            group = dev41[(dev41.model == "llada") & (dev41.source == task) & (dev41.M == 4)]
            values = dict(zip(group.beta, group.A_target))
        kept.append((label, values))
    return kept


def beta_locks(dev41, dev42):
    """beta_locks.json: per model, the development-only beta and the statistic behind it (Appendix B.2)."""
    locks = {"rule": LOCK_RULE, "bank": "dev", "betas": BETAS, "models": {}}
    for model in LOCK_SECTIONS:
        cells, partial = dev_cells(dev41, dev42, model)
        if not cells:
            continue
        everything = lock_from_cells(cells)
        if model in FROZEN_LOCK_CELLS:
            frozen = FROZEN_LOCK_CELLS[model]
            lock = lock_from_cells(frozen_cells(dev41, cells, frozen))
            lock["cell_set"] = frozen["cell_set"]
            lock["partial_cells_excluded"] = frozen["historical_partial_cells_excluded"]
            lock["partial_cells_excluded_note"] = "historical: as recorded in the frozen lock file when it was written"
            lock["partial_cells_excluded_derived"] = partial
        else:
            lock = dict(everything)
            lock["cell_set"] = ALL_COMPLETE_CELL_SET
            lock["partial_cells_excluded"] = partial
        lock["sections"] = LOCK_SECTIONS[model]
        lock["all_complete_cells"] = {key: everything[key] for key in ("beta", "n_cells", "dev_mean_by_beta")}
        locks["models"][model] = lock
        print(
            f"lock {model:11s} beta {lock['beta']} ({lock['n_cells']} cells, +{lock['dev_gain_vs_ordinary_pp']:.2f} pp over"
            f" ordinary); rule on all {everything['n_cells']} complete cells: beta {everything['beta']}",
            flush=True,
        )
    return locks


# ---------------------------------------------------------------------------------------------
# Gold example reward (Figure 4's gold row)
# ---------------------------------------------------------------------------------------------


def gold_records(pools, model, task, M):
    """The reference answers of a Section 4.1 report cell scored under P1 and P0, one dict per family in file order.

    run.py --score-gold stores them in the pool records (gold_ell1, gold_ell0, gold_T); the frozen gold-reward probe
    stores them in <pools>/gold/<model>/<task>_M{M}_J1.jsonl (ell1, ell0, T). T is the number of target tokens of the
    reference answer; for Qwen3-8B it counts the stop token, whose probability is part of ell1 and ell0.
    """
    pool = load_pool(pool_path(pools, "report", model, task, M, 1, False))
    if pool and all("gold_ell1" in record for record in pool):
        return [
            {
                "family": record["episode_id"],
                "T": record["gold_T"],
                "ell1": record["gold_ell1"],
                "ell0": record["gold_ell0"],
                "logp_stop1": record.get("gold_logp_stop1"),
                "logp_stop0": record.get("gold_logp_stop0"),
            }
            for record in pool
        ]
    path = Path(pools) / "gold" / model / f"{task}_M{M}_J1.jsonl"
    if not path.exists():
        return []
    first = {}
    for record in read_records(path):
        first.setdefault(record["episode_id"], record)
    return [
        {
            "family": record["episode_id"],
            "T": record["T"],
            "ell1": record["ell1"],
            "ell0": record["ell0"],
            "logp_stop1": record.get("logp_stop1"),
            "logp_stop0": record.get("logp_stop0"),
        }
        for record in first.values()
    ]


def gold_family_rows(model, task, M, gold):
    """Per family: the gold reward R* = ell1* - ell0*, per target token, and the gold perplexities under P1 and P0."""
    rows = []
    for g in gold:
        T = g["T"]
        reward = g["ell1"] - g["ell0"]
        row = {
            "section": "4.1",
            "benchmark": GOLD_LABELS[task],
            "source": task,
            "model": model,
            "family_id": g["family"],
            "M": M,
            "J": 1,
            "target_tokens": T,
            "ell1_gold": g["ell1"],
            "ell0_gold": g["ell0"],
            "reward_gold": reward,
            "reward_per_token": reward / T,
            "ppl1_gold": float(np.exp(-g["ell1"] / T)),
            "ppl0_gold": float(np.exp(-g["ell0"] / T)),
            "ppl_ratio_gold": float(np.exp(reward / T)),
            "nll1_per_token": -g["ell1"] / T,
            "nll0_per_token": -g["ell0"] / T,
        }
        if g["logp_stop1"] is not None:
            row["logp_stop1"] = g["logp_stop1"]
            row["logp_stop0"] = g["logp_stop0"]
        rows.append(row)
    return rows


def gold_cell_row(model, task, label, M, family_rows, nboot):
    """The gold row of Figure 4 for one cell: mean gold reward per target token with a 95 % family-bootstrap interval
    (bootstrap.boot_sd, seed string f"4.1|{model}|{task}|{M}|rpt"), and the mean gold NLL per token under P1 and P0."""
    reward = np.array([row["reward_per_token"] for row in family_rows])
    mean, lo, hi = boot_sd(reward, f"4.1|{model}|{task}|{M}|rpt", nboot)
    return {
        "model": model,
        "benchmark": label,
        "source": task,
        "M": M,
        "J": 1,
        "families": len(family_rows),
        "gold_reward_per_token": mean,
        "gold_rpt_lo": lo,
        "gold_rpt_hi": hi,
        "gold_nll1_per_token": float(np.mean([row["nll1_per_token"] for row in family_rows])),
        "gold_nll0_per_token": float(np.mean([row["nll0_per_token"] for row in family_rows])),
        "gold_T_mean": float(np.mean([row["target_tokens"] for row in family_rows])),
    }


def gold_tables(pools, nboot):
    """gold_families.csv and gold_reward_cells.csv for every Section 4.1 report cell with scored reference answers."""
    families = []
    cells = []
    for model in MODELS41:
        for task, label in TASKS:
            for M in M_GRID:
                gold = gold_records(pools, model, task, M)
                if not gold:
                    continue
                rows = gold_family_rows(model, task, M, gold)
                families += rows
                cells.append(gold_cell_row(model, task, label, M, rows, nboot))
    return families, cells


# ---------------------------------------------------------------------------------------------
# Section 4.4: sharpening residuals
# ---------------------------------------------------------------------------------------------


def power_rows(pools, codes, lock_beta, nboot):
    """sec44_power.csv: LLaDA-1.5, M = 4, J in {1, 4, 8}; every (tau, beta) of the grid.

    Guidance at (beta, tau) targets P1^(1/tau + beta) P0^(-beta), pool weights pool_weights(ell1, ell0, beta, tau). Its
    total exponent on P1 equals that of ordinary power sampling at tau_eff = 1 / (1 / tau + beta) (weights
    pool_weights(ell1, ell0, 0, tau_eff)). gain_vs_matched_power = A_target(beta, tau) - A_target(0, tau_eff), with
    paired family-bootstrap intervals (cell key f"llada|{task}|44|J{J}"). is_locked_beta marks LLaDA-1.5's locked beta.
    """
    rows = []
    for task, label in TASKS:
        for J in POWER_J:
            F = cell_families(pools, "report", POWER_MODEL, task, CROSSBENCH_M, J, J > 1, codes)
            if not F:
                continue
            idx = cell_indices(BOOTSTRAP_SEED, f"{POWER_MODEL}|{task}|44|J{J}", len(F), nboot)
            for tau in TAUS:
                for beta in BETAS:
                    weights = [pool_weights(f["ell1"], f["ell0"], beta, tau) for f in F]
                    A_target = np.array([w @ f["correct"].mean(1) for w, f in zip(weights, F)])
                    A_all = np.array([w @ f["correct"].all(1) for w, f in zip(weights, F)])
                    ess = np.array([1.0 / (w ** 2).sum() for w in weights])
                    tau_eff = 1.0 / (1.0 / tau + beta)
                    matched = [pool_weights(f["ell1"], f["ell0"], 0.0, tau_eff) for f in F]
                    A_target_matched = np.array([w @ f["correct"].mean(1) for w, f in zip(matched, F)])
                    A_all_matched = np.array([w @ f["correct"].all(1) for w, f in zip(matched, F)])
                    at, lo, hi = ci(A_target, idx)
                    d, d_lo, d_hi = ci(A_target - A_target_matched, idx)
                    rows.append({
                        "bank": "report",
                        "benchmark": label,
                        "source": task,
                        "J": J,
                        "tau": tau,
                        "beta": beta,
                        "families": len(F),
                        "A_target": at,
                        "A_target_lo": lo,
                        "A_target_hi": hi,
                        "A_all": float(A_all.mean()),
                        "ess_mean": float(ess.mean()),
                        "tau_eff": tau_eff,
                        "A_target_matched_power_only": float(A_target_matched.mean()),
                        "A_all_matched_power_only": float(A_all_matched.mean()),
                        "gain_vs_matched_power": d,
                        "gain_vs_matched_lo": d_lo,
                        "gain_vs_matched_hi": d_hi,
                        "is_locked_beta": beta == lock_beta,
                    })
    return rows


# ---------------------------------------------------------------------------------------------
# Appendix B.6: guidance-strength sweep
# ---------------------------------------------------------------------------------------------


def weights_and_entropy(ell1, ell0, beta):
    """w = softmax(beta * R) over the K draws (duplicates kept) and its entropy H = -sum_k w_k log w_k.

    log w is taken from the max-shifted logits (log w = beta R - max - log Z), so a weight that underflows to 0 adds 0.
    """
    logits = beta * (np.asarray(ell1, float) - np.asarray(ell0, float))
    logits = logits - logits.max()
    unnormalised = np.exp(logits)
    Z = unnormalised.sum()
    w = unnormalised / Z
    log_w = logits - np.log(Z)
    return w, float(-(w * log_w).sum())


def sweep_rows(pools, nboot):
    """beta_sweep.csv: the guidance-strength sweep of Appendix B.6 (Figure 11).

    Cells: List Functions-Easy, cross-benchmark pools (M = 4, K = 64, oracle lengths), J in {1, 4}, three diffusion models.
    Per family and beta: w = softmax(beta R), A_all = sum_k w_k c_k (c_k = every target of draw k correct),
    H = -sum_k w_k log w_k, effective support D_eff = exp(H) / K (1 at beta = 0), ESS = 1 / sum w^2. Cell values are family
    means with 95 % family-bootstrap intervals (the cell's index matrix, key f"{model}|list_easy_fam", shared across beta,
    so the gains over beta = 0 are paired). A_all and the gains in percent / percentage points.
    """
    rows = []
    for model, model_label in SWEEP_MODELS:
        for J in SWEEP_J:
            records = load_pool(pool_path(pools, "report", model, SWEEP_TASK, CROSSBENCH_M, J, True))
            if not records:
                continue
            F = [family_summary(record, model, None) for record in records]
            K = F[0]["K"]
            idx = cell_indices(BOOTSTRAP_SEED, f"{model}|{SWEEP_TASK}", len(F), nboot)
            values = {}
            for beta in SWEEP_BETAS:
                A_all = []
                H = []
                ess = []
                max_w = []
                for f in F:
                    w, entropy = weights_and_entropy(f["ell1"], f["ell0"], beta)
                    A_all.append(float(w @ f["correct"].all(1).astype(float)))
                    H.append(entropy)
                    ess.append(float(1.0 / (w ** 2).sum()))
                    max_w.append(float(w.max()))
                values[beta] = {"A_all": np.array(A_all), "H": np.array(H), "ess": np.array(ess), "max_w": np.array(max_w)}
            ordinary = values[0.0]["A_all"]
            for beta in SWEEP_BETAS:
                v = values[beta]
                support = np.exp(v["H"]) / K
                a, a_lo, a_hi = ci(v["A_all"], idx)
                g, g_lo, g_hi = ci(v["A_all"] - ordinary, idx)
                d, d_lo, d_hi = ci(support, idx)
                rows.append({
                    "model": model,
                    "model_label": model_label,
                    "benchmark": "List Functions-Easy",
                    "M": CROSSBENCH_M,
                    "J": J,
                    "K": K,
                    "families": len(F),
                    "beta": beta,
                    "A_all_pct": 100 * a,
                    "A_all_lo_pct": 100 * a_lo,
                    "A_all_hi_pct": 100 * a_hi,
                    "gain_pp": 100 * g,
                    "gain_lo_pp": 100 * g_lo,
                    "gain_hi_pp": 100 * g_hi,
                    "D_eff": d,
                    "D_eff_lo": d_lo,
                    "D_eff_hi": d_hi,
                    "H_mean_nats": float(v["H"].mean()),
                    "ESS_mean": float(v["ess"].mean()),
                    "max_w_mean": float(v["max_w"].mean()),
                    "families_gain_pos": int((v["A_all"] - ordinary > 1e-12).sum()),
                    "families_gain_neg": int((v["A_all"] - ordinary < -1e-12).sum()),
                })
            gains = [row["gain_pp"] for row in rows[-len(SWEEP_BETAS):]]
            print(f"sweep {model:11s} J={J} gain over beta = 0 (pp): " + " ".join(f"{gain:+.2f}" for gain in gains), flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------------------------


def frozen_lock_beta(model):
    """The locked beta of `model` in data/tables/fig4/beta_locks.json (used when the development pools are absent)."""
    path = ROOT / "data" / "tables" / "fig4" / "beta_locks.json"
    return json.loads(path.read_text())["models"][model]["beta"]


def analyse(pools, crossbench, beta_sweep, nboot):
    """Every table the pools allow; returns {name: DataFrame} (beta_locks as a dict under "beta_locks")."""
    frames = {}
    codes = {bank: output_codes(bank) for bank in ("dev", "report")}

    dev41, dev42, _ = section_tables(pools, "dev", codes["dev"], {}, True, nboot)
    locks = None
    if dev41 or dev42:
        frames["dev_sec41"] = pd.DataFrame(dev41)
        frames["dev_sec42"] = pd.DataFrame(dev42)
        locks = beta_locks(frames["dev_sec41"], frames["dev_sec42"])
        frames["beta_locks"] = locks
    if locks is not None:
        locked = {model: lock["beta"] for model, lock in locks["models"].items()}
    else:
        locked = {}

    report41, report42, diagnostics = section_tables(pools, "report", codes["report"], locked, crossbench, nboot)
    if report41:
        frames["sec41_accuracy"] = section41_table(report41)
        for row in report41:
            if row["beta"] == FIGURE_BETA:
                print(
                    f"{row['model']:11s} {row['source']:13s} M={row['M']:2d} accuracy gain at beta = 0.5:"
                    f" {100 * row['gain_target']:+6.2f} pp [{100 * row['gain_target_lo']:+6.2f}, {100 * row['gain_target_hi']:+6.2f}]",
                    flush=True,
                )
    if report42:
        frames["cells_crossbench"] = crossbench_table(report42)
    if diagnostics:
        frames["diagnostics"] = pd.DataFrame(diagnostics)

    gold_families, gold_cells = gold_tables(pools, nboot)
    if gold_cells:
        frames["gold_families"] = pd.DataFrame(gold_families)
        frames["gold_reward_cells"] = pd.DataFrame(gold_cells)

    lock_beta = locked.get(POWER_MODEL)
    if lock_beta is None:
        lock_beta = frozen_lock_beta(POWER_MODEL)
    power = power_rows(pools, codes["report"], lock_beta, nboot)
    if power:
        frames["sec44_power"] = pd.DataFrame(power)

    if beta_sweep:
        sweep = sweep_rows(pools, nboot)
        if sweep:
            frames["beta_sweep"] = pd.DataFrame(sweep)
    return frames


def write_tables(out, frames):
    """Write every table to `out` (CSV with pandas' defaults, as in the paper's analysis; beta_locks as JSON)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if name == "beta_locks":
            (out / "beta_locks.json").write_text(json.dumps(frame, indent=1) + "\n")
            continue
        frame.to_csv(out / f"{name}.csv", index=False)
    print(f"wrote {', '.join(sorted(frames))} to {out}")


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


def compare(frames, references):
    """Print the largest |difference| per numeric column against the tables in the reference directories, then the
    resampling verdict of every Section 4.1 cell: is the frozen ordinary accuracy inside the 95 % CI computed here?

    CSVs are parsed with float_precision="round_trip", so a value written by pandas reads back exactly.
    """
    for name, keys in KEYS.items():
        paths = [Path(reference) / f"{name}.csv" for reference in references]
        paths = [path for path in paths if path.exists()]
        if name not in frames or frames[name].empty or not paths:
            continue
        ours = pd.read_csv(io_csv(frames[name]), float_precision="round_trip")
        theirs = pd.read_csv(paths[0], float_precision="round_trip")
        merged = ours.merge(theirs, on=keys, suffixes=("", "_ref"))
        columns = [c for c in theirs.columns if c not in keys and theirs[c].dtype.kind == "f" and c in ours]
        worst = max(((max_abs_diff(merged[c], merged[c + "_ref"]), c) for c in columns), default=(0.0, "-"))
        where = f" ({worst[1]})" if worst[0] > 0 else ""
        print(f"{name}: {len(merged)} rows in common with {paths[0]}; largest |diff| {worst[0]:.3e}{where}")

    paths = [Path(reference) / "sec41_accuracy.csv" for reference in references]
    paths = [path for path in paths if path.exists()]
    if "sec41_accuracy" not in frames or not paths:
        return
    frozen = pd.read_csv(paths[0], float_precision="round_trip")
    frozen = frozen[frozen.beta == 0.0].set_index(["model", "source", "M"])
    rerun = frames["sec41_accuracy"]
    rerun = rerun[rerun.beta == 0.0]
    for _, cell in rerun.iterrows():
        key = (cell.model, cell.source, cell.M)
        if key not in frozen.index:
            continue
        value = frozen.loc[key, "accuracy"]
        inside = cell.ci_lo <= value <= cell.ci_hi
        verdict = "inside" if inside else "outside"
        print(
            f"{cell.model} {cell.source} M={cell.M}: frozen accuracy {value:.3f}, rerun CI [{cell.ci_lo:.3f}, {cell.ci_hi:.3f}]"
            f" -> frozen mean {verdict} the rerun's 95 % CI ({cell.families} families)"
        )


def io_csv(frame):
    """The CSV text of a frame as pandas writes it, as a readable buffer (so --compare sees the written values)."""
    return io.StringIO(frame.to_csv(index=False))


def has_pools(pools):
    """True if the pools directory holds at least one pool file of this pipeline."""
    for bank in ("dev", "report"):
        if any((pools / bank).glob("*/*.jsonl")):
            return True
    return False


def missing_inputs(pools, crossbench, beta_sweep):
    """For each requested optional table whose pools are all absent, a line naming the files it needs."""
    missing = []
    if crossbench:
        paths = [pool_path(pools, "report", model, task, CROSSBENCH_M, J, True) for model in MODELS42 for task, _ in TASKS for J in J_GRID]
        if not any(path.exists() for path in paths):
            missing.append(
                f"--crossbench: no cross-benchmark pools {pools}/report/<model>/<task>_M{CROSSBENCH_M}_J<J>_il.jsonl"
                f" (models {', '.join(MODELS42)}; J in {J_GRID})"
            )
    if beta_sweep:
        paths = [pool_path(pools, "report", model, SWEEP_TASK, CROSSBENCH_M, J, True) for model, _ in SWEEP_MODELS for J in SWEEP_J]
        if not any(path.exists() for path in paths):
            missing.append(
                f"--beta-sweep: no List Functions-Easy cross-benchmark pools {pools}/report/<model>/{SWEEP_TASK}_M{CROSSBENCH_M}_J<J>_il.jsonl"
                f" (models {', '.join(model for model, _ in SWEEP_MODELS)}; J in {SWEEP_J})"
            )
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pools", required=True, help="directory with <bank>/<model>/<task>_M{M}_J{J}[_il].jsonl and gold/")
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--crossbench", action="store_true", help="also the cross-benchmark report cells (Figure 9)")
    parser.add_argument("--beta-sweep", action="store_true", help="also the guidance-strength sweep (Figure 11)")
    parser.add_argument("--nboot", type=int, default=NBOOT)
    parser.add_argument("--compare", nargs="+", default=None, help="directories of reference tables, e.g. data/tables/fig4")
    args = parser.parse_args()

    pools = Path(args.pools)
    if not pools.is_dir() or not has_pools(pools):
        print(f"no Figure 4 pools in {pools}; {FETCH_HINT}", file=sys.stderr)
        sys.exit(2)

    missing = missing_inputs(pools, args.crossbench, args.beta_sweep)
    if missing:
        for line in missing:
            print(f"{line}; {FETCH_HINT}", file=sys.stderr)
        sys.exit(2)

    frames = analyse(pools, args.crossbench, args.beta_sweep, args.nboot)
    write_tables(Path(args.out), frames)
    if args.compare:
        compare(frames, args.compare)


if __name__ == "__main__":
    main()
