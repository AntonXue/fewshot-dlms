# ported from editdiff/experiments/sec42_mj/mj_analyze.py, experiments/paper_integration/agent_c/c_mj_diagnostics.py (sharpening) and experiments/gold_reward/gr_position.py (mechanism) @ d276fa6
"""Figure 5 (Section 4.2) and its appendix tables, from the frozen pools (CPU only, no model).

The M x J grid of Section 4.2: List Functions-Easy, 64 report families, three diffusion models,
M in {2, 4, 8, 16} demonstrations, J in {1, 2, 4, 8} targets, oracle target lengths, K = 16 infills per
family sampled from P1. A pool file holds one record per family (see run.py):

    <pools>/<model>/list_easy_fam_M{M}_J{J}.jsonl

Each candidate infill k has the texts of its J targets and its two log-likelihoods ell1 = log P1 and
ell0 = log P0. Per family, with weights w = softmax(beta * (ell1 - ell0)) (scoring.pool_weights),

    A_target = sum_k w_k mean_j c_kj      per-target accuracy (A_avg in the paper)
    A_all    = sum_k w_k prod_j c_kj      every target correct

where c_kj is 1 if target j of infill k is correct (grading.py). beta = 0 is ordinary sampling, beta = 0.5
the paper's common strength. Every cell is resampled over families (bootstrap.py): one index matrix per
cell, shared by all of its quantities, so the gains at beta > 0 are paired with beta = 0.

Outputs in --out:
    families.csv     one row per (model, M, J, family, beta)
    cells.csv        one row per (model, M, J, beta): Figure 5 (A_all, A_target and the gain in A_all at beta = 0.5)
    gold.csv         per (model, M, J): the reference answers scored under P1 and P0 (gold reward per token)
    diagnostics.csv  per (model, M, J): pool coverage, reward spread, ESS, prompt sizes
    candidates.csv.gz, coverage.json
    sharpening.csv   M x J sharpening figure (Fig. 10 ACL / Fig. 11 arXiv; Appendix B, Likelihood Sharpening):
                     the sharpening control w = softmax(0.5 * ell1), i.e. the target P1^{3/2} (temperature 2/3),
                     against example guidance at beta = 0.5
    mechanism.csv    mechanism figure (Fig. 8, both builds; Appendix B, Multi-Target Prompting): gold NLL per
                     token by reveal position under P1 and P0 in the cross-benchmark study (M = 4, J = 8); read
                     from --gold-pools when present

--compare DIR prints, for every table also in DIR, the largest |difference| per numeric column, and for
every cell a resampling verdict (README.md, fidelity level 3): is the frozen A_all inside the 95 % CI of
the pools analysed here?

    python experiments/fig5_multi_target/analyze.py --pools artifacts/pools/fig5 --compare data/tables/fig5
"""
import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fewshot_dlms.bootstrap import boot_sd
from fewshot_dlms.bootstrap import cell_indices
from fewshot_dlms.bootstrap import ci
from fewshot_dlms.data import load_families
from fewshot_dlms.grading import answer
from fewshot_dlms.grading import grade
from fewshot_dlms.scoring import family_metrics
from fewshot_dlms.scoring import pool_weights

HERE = Path(__file__).resolve().parent
FETCH_HINT = "run: python -m fewshot_dlms.fetch pools java"

# The grid of Section 4.2, fixed before the run. The model order is the row order of the paper's tables.
TASK = "list_easy_fam"
MODELS = ["llada", "dqwen", "nemotron_pt"]
M_GRID = [2, 4, 8, 16]
J_GRID = [1, 2, 4, 8]
BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]  # beta = 0 is ordinary sampling; 0.1, 0.25 and 1 are sensitivity values
BETA = 0.5  # the paper's common guidance strength
NBOOT = 10000
BOOTSTRAP_SEED = "mtfsp-20261011-mj-bootstrap-v1"  # cell key f"{model}|{M}|{J}"

# Appendix B, Likelihood Sharpening: the sharpening control has the same total exponent 3/2 on P1 as
# Q_0.5 ∝ P1^{3/2} P0^{-1/2}, but no P0: target P1^{3/2}, i.e. temperature 2/3, pool weights ∝ P1^{1/2}. Its
# intervals use their own family resamples, seeded from SHARPENING_SEED (see sharpening_indices).
SHARPENING_TAU = 2 / 3
SHARPENING_SEED = "agentC-appA-20261011-mj-v1"

# Appendix B, Multi-Target Prompting, mechanism figure (Fig. 8, both builds): the cross-benchmark study (M = 4, J = 8,
# K = 64) with its reference answers scored under P1 and P0, one record per family in
# <gold-pools>/<model>/<task>_M4_J8_il.jsonl.
MECHANISM_MODELS = ["llada", "nemotron_pt", "dqwen"]
MECHANISM_TASKS = [
    ("sst2_fam", "SST-2"),
    ("list_easy_fam", "List Functions"),
    ("salt_rep_fam", "SALT"),
    ("movie_dir_fam", "Movie Director"),
]
MECHANISM_M = 4
MECHANISM_J = 8

# The tables of --compare and the columns that identify a row.
KEYS = {
    "cells": ["model", "M", "J", "beta"],
    "gold": ["model", "M", "J"],
    "diagnostics": ["model", "M", "J"],
    "sharpening": ["model", "M", "J"],
    "mechanism": ["model", "source", "J", "position"],
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


def load_cell(pools, model, M, J):
    """The pool records of one cell, by episode id: the first complete record of each family.

    A record is complete when it is for this (M, J) and holds all K of its candidates. A family can appear
    twice in a file (two of the paper's pool files have one such family); the first complete record counts.
    """
    path = Path(pools) / model / f"{TASK}_M{M}_J{J}.jsonl"
    records = {}
    if not path.exists():
        return records
    for record in read_records(path):
        if record["M"] != M or record["J"] != J or len(record["candidates"]) != record["K"]:
            continue
        records.setdefault(record["episode_id"], record)
    return records


def grade_pool(record, model):
    """K x J matrices of correctness and of well-formedness of the candidates' answers (grading.py).

    `model` is the pool's model name ("llada", "nemotron_pt", "dqwen"), passed to `answer` as the paper's
    analysis passed it.
    """
    n = len(record["candidates"])
    J = record["J"]
    correct = np.zeros((n, J))
    parsed = np.zeros((n, J), bool)
    for k, candidate in enumerate(record["candidates"]):
        for j in range(J):
            is_correct, parse_ok = grade(record["grader"], answer(candidate["texts"][j], model), record["gold"][j])
            correct[k, j] = is_correct
            parsed[k, j] = parse_ok
    return correct, parsed


def pool_class(all_correct):
    """'all' if every infill has all targets correct, 'none' if no infill has, 'mixed' otherwise."""
    if all_correct.all():
        return "all"
    if not all_correct.any():
        return "none"
    return "mixed"


def family_summary(record, model):
    """Everything the tables need from one family's pool record."""
    candidates = record["candidates"]
    correct, parsed = grade_pool(record, model)
    ell1 = np.array([candidate["ell1"] for candidate in candidates])
    ell0 = np.array([candidate["ell0"] for candidate in candidates])
    gold_toks = list(record["gold_toks"])
    return {
        "family": record["episode_id"],
        "correct": correct,
        "parsed": parsed,
        "ell1": ell1,
        "ell0": ell0,
        "reward": ell1 - ell0,  # example reward R of each infill
        "target_tokens": int(sum(record["oracle_L"])),
        "n_unique": len({tuple(candidate["toks"]) for candidate in candidates}),
        "prompt_tokens": record["n_tokens"],
        "seconds": record["seconds_gen"] + record["seconds_score"],
        "gold_ell1": record["gold_ell1"],
        "gold_ell0": record["gold_ell0"],
        "gold_T": record["gold_T"],
        "gold_in_pool": int(any(list(candidate["toks"]) == gold_toks for candidate in candidates)),
    }


# ---------------------------------------------------------------------------------------------
# Rows of the tables
# ---------------------------------------------------------------------------------------------


def candidate_rows(model, M, J, f, record):
    """One row per candidate infill (candidates.csv.gz)."""
    rows = []
    for k, candidate in enumerate(record["candidates"]):
        c = f["correct"][k]
        rows.append({
            "model": model,
            "M": M,
            "J": J,
            "family": f["family"],
            "k": k,
            "text_sha": hashlib.sha256(json.dumps(candidate["toks"]).encode()).hexdigest()[:12],
            "ell1": candidate["ell1"],
            "ell0": candidate["ell0"],
            "R": candidate["ell1"] - candidate["ell0"],
            "all_correct": int(c.all()),
            "n_correct": int(c.sum()),
            "parse_all": int(f["parsed"][k].all()),
            "c": "".join(str(int(v)) for v in c),
        })
    return rows


def family_rows(model, M, J, f, record):
    """One row per beta for one family (families.csv): the metrics of scoring.family_metrics and pool facts."""
    c = f["correct"]
    a_all = c.all(1)
    cls = pool_class(a_all)
    mixed = cls == "mixed"
    rows = []
    for beta in BETAS:
        w = pool_weights(f["ell1"], f["ell0"], beta)
        metrics = family_metrics(c, w)
        row = {
            "model": model,
            "M": M,
            "J": J,
            "family": f["family"],
            "beta": beta,
            "A_target": metrics["A_target"],
            "A_all": metrics["A_all"],
            "ess": metrics["ess"],
            "max_w": metrics["max_w"],
            "pool_class": cls,
            "coverage_all": int(a_all.any()),
            "coverage_any": int(c.max() > 0),
            "n_unique": f["n_unique"],
            "T": f["target_tokens"],
            "n_tokens": record["n_tokens"],
            "parse_rate": float(f["parsed"].mean()),
            "reward_mean": float(f["reward"].mean()),
            "reward_sd": float(f["reward"].std()),
            # Cov(R, correct) over the uniform pool = d A / d beta at beta = 0; defined on mixed pools only.
            "cov_R_all": float(np.cov(f["reward"], a_all.astype(float), bias=True)[0, 1]) if mixed else np.nan,
            "cov_R_target": float(np.cov(f["reward"], c.mean(1), bias=True)[0, 1]) if mixed else np.nan,
            "gold_R_per_tok": (record["gold_ell1"] - record["gold_ell0"]) / record["gold_T"],
            "gold_nll1_per_tok": -record["gold_ell1"] / record["gold_T"],
            "gold_nll0_per_tok": -record["gold_ell0"] / record["gold_T"],
            "gold_in_pool": f["gold_in_pool"],
        }
        for j in range(J):
            row[f"acc_t{j + 1}"] = float(w @ c[:, j])
        rows.append(row)
    return rows


def cell_rows(model, M, J, F, idx):
    """One row per beta for one cell (cells.csv): family means with paired bootstrap intervals.

    A_target, A_all at beta; gain_target, gain_all = the same minus beta = 0, resampled with the same
    family indices; families better / worse in A_all than at beta = 0; mean ESS and maximum weight; the
    mean accuracy of each target position.
    """
    weights = {beta: [pool_weights(f["ell1"], f["ell0"], beta) for f in F] for beta in BETAS}
    A_target = {beta: np.array([w @ f["correct"].mean(1) for w, f in zip(weights[beta], F)]) for beta in BETAS}
    A_all = {beta: np.array([w @ f["correct"].all(1) for w, f in zip(weights[beta], F)]) for beta in BETAS}
    rows = []
    for beta in BETAS:
        at, at_lo, at_hi = ci(A_target[beta], idx)
        aa, aa_lo, aa_hi = ci(A_all[beta], idx)
        gt, gt_lo, gt_hi = ci(A_target[beta] - A_target[0.0], idx)
        ga, ga_lo, ga_hi = ci(A_all[beta] - A_all[0.0], idx)
        ess = [1 / (w ** 2).sum() for w in weights[beta]]
        max_w = [w.max() for w in weights[beta]]
        row = {
            "model": model,
            "M": M,
            "J": J,
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
            "families_better_all": int(np.sum(A_all[beta] - A_all[0.0] > 1e-12)),
            "families_worse_all": int(np.sum(A_all[beta] - A_all[0.0] < -1e-12)),
            "ess_mean": float(np.mean(ess)),
            "ess_median": float(np.median(ess)),
            "max_w_mean": float(np.mean(max_w)),
        }
        for j in range(J):
            row[f"acc_t{j + 1}"] = float(np.mean([w @ f["correct"][:, j] for w, f in zip(weights[beta], F)]))
        rows.append(row)
    return rows


def gold_row(model, M, J, F, idx):
    """The reference answers of one cell scored under P1 and P0 (gold.csv).

    Per family, with T the number of target tokens: gold reward per token (ell1 - ell0) / T and the gold
    NLL per token under P1 and P0, with bootstrap intervals; the rank of the gold reward among the pool's
    rewards (1 = highest) and how often the gold infill was sampled.
    """
    reward = np.array([(f["gold_ell1"] - f["gold_ell0"]) / f["gold_T"] for f in F])
    nll1 = np.array([-f["gold_ell1"] / f["gold_T"] for f in F])
    nll0 = np.array([-f["gold_ell0"] / f["gold_T"] for f in F])
    reward_mean, reward_lo, reward_hi = ci(reward, idx)
    nll1_mean, nll1_lo, nll1_hi = ci(nll1, idx)
    nll0_mean, nll0_lo, nll0_hi = ci(nll0, idx)
    return {
        "model": model,
        "M": M,
        "J": J,
        "families": len(F),
        "gold_reward_per_token": reward_mean,
        "gold_rpt_lo": reward_lo,
        "gold_rpt_hi": reward_hi,
        "gold_nll1_per_token": nll1_mean,
        "gold_nll1_lo": nll1_lo,
        "gold_nll1_hi": nll1_hi,
        "gold_nll0_per_token": nll0_mean,
        "gold_nll0_lo": nll0_lo,
        "gold_nll0_hi": nll0_hi,
        "gold_reward_total_mean": float(np.mean([f["gold_ell1"] - f["gold_ell0"] for f in F])),
        "gold_T_mean": float(np.mean([f["gold_T"] for f in F])),
        "gold_in_pool_rate": float(np.mean([f["gold_in_pool"] for f in F])),
        "gold_rank_mean": float(np.mean([(f["reward"] > (f["gold_ell1"] - f["gold_ell0"])).sum() + 1 for f in F])),
        "pool_reward_per_token_mean": float(np.mean([f["reward"].mean() / f["target_tokens"] for f in F])),
    }


def diagnostics_row(model, M, J, K, F, ordinary_A_all):
    """Pool-level facts of one cell (diagnostics.csv).

    Coverage: fraction of families with at least one all-correct infill (the ceiling of any reweighting)
    and headroom = coverage - ordinary A_all. Pool classes all / none / mixed. Reward R = ell1 - ell0 over
    all candidates, and on mixed pools its covariance with correctness (the slope of A at beta = 0).
    """
    classes = collections.Counter(pool_class(f["correct"].all(1)) for f in F)
    mixed = [f for f in F if 0 < f["correct"].all(1).sum() < K]
    rewards = np.concatenate([f["reward"] for f in F])
    cov_all = [np.cov(f["reward"], f["correct"].all(1).astype(float), bias=True)[0, 1] for f in mixed]
    cov_target = [np.cov(f["reward"], f["correct"].mean(1), bias=True)[0, 1] for f in mixed]
    weights = [pool_weights(f["ell1"], f["ell0"], BETA) for f in F]
    coverage_all = float(np.mean([f["correct"].all(1).any() for f in F]))
    return {
        "model": model,
        "M": M,
        "J": J,
        "families": len(F),
        "K": K,
        "coverage_all_correct": coverage_all,
        "coverage_any_correct_target": float(np.mean([f["correct"].max() > 0 for f in F])),
        "pools_all": classes["all"],
        "pools_none": classes["none"],
        "pools_mixed": classes["mixed"],
        "headroom_all": coverage_all - ordinary_A_all.mean(),
        "unique_candidates_mean": float(np.mean([f["n_unique"] for f in F])),
        "parse_rate": float(np.mean([f["parsed"].mean() for f in F])),
        "reward_mean": float(rewards.mean()),
        "reward_sd": float(rewards.std()),
        "reward_q05": float(np.percentile(rewards, 5)),
        "reward_q50": float(np.percentile(rewards, 50)),
        "reward_q95": float(np.percentile(rewards, 95)),
        "reward_per_target_token_mean": float(np.mean([f["reward"].mean() / f["target_tokens"] for f in F])),
        "cov_R_all_mixed_mean": float(np.mean(cov_all)) if cov_all else np.nan,
        "cov_R_all_mixed_pos_frac": float(np.mean(np.array(cov_all) > 0)) if cov_all else np.nan,
        "cov_R_target_mixed_mean": float(np.mean(cov_target)) if cov_target else np.nan,
        "n_mixed": len(mixed),
        "ess_mean_beta_primary": float(np.mean([1 / (w ** 2).sum() for w in weights])),
        "max_w_mean_beta_primary": float(np.mean([w.max() for w in weights])),
        "prompt_tokens_median": float(np.median([f["prompt_tokens"] for f in F])),
        "prompt_tokens_max": int(max(f["prompt_tokens"] for f in F)),
        "target_tokens_mean": float(np.mean([f["target_tokens"] for f in F])),
        "seconds_per_family": float(np.mean([f["seconds"] for f in F])),
    }


def sharpening_indices(model, M, J, n, nboot):
    """Family resamples of the sharpening table (Appendix B, Likelihood Sharpening).

    Seed = the first 16 hex digits of SHA-256(f"{SHARPENING_SEED}|{model}|{M}|{J}"), mod 2**32, for
    numpy's default_rng; one rng.integers(0, n, size=(nboot, n)) draw, as in bootstrap.cell_indices.
    """
    text = f"{SHARPENING_SEED}|{model}|{M}|{J}"
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2 ** 32)
    return np.random.default_rng(seed).integers(0, n, size=(nboot, n))


def sharpening_row(model, M, J, K, F, nboot):
    """Sharpening control vs example guidance in one cell (sharpening.csv, the M x J sharpening figure).

    Per family: A at beta = 0 (ordinary), at beta = 0.5 (guidance, Q ∝ P1^{3/2} P0^{-1/2}) and under the
    sharpening control (Q ∝ P1^{3/2}, pool weights pool_weights(ell1, ell0, 0, tau=2/3) ∝ P1^{1/2}).
    gain_*_power_matched = control minus ordinary; tilt_minus_power_* = guidance minus control.
    """
    idx = sharpening_indices(model, M, J, len(F), nboot)
    values = collections.defaultdict(list)
    for f in F:
        c = f["correct"]
        for name, w in (
            ("b0", pool_weights(f["ell1"], f["ell0"], 0.0)),
            ("b0.5", pool_weights(f["ell1"], f["ell0"], BETA)),
            ("power_matched", pool_weights(f["ell1"], f["ell0"], 0.0, tau=SHARPENING_TAU)),
        ):
            metrics = family_metrics(c, w)
            values[f"A_target_{name}"].append(metrics["A_target"])
            values[f"A_all_{name}"].append(metrics["A_all"])
            if name == "power_matched":
                values["ess_power_matched"].append(metrics["ess"])
    values = {name: np.array(v) for name, v in values.items()}
    row = {"model": model, "M": M, "J": J, "families": len(F), "K": K}
    statistics = [("ess_power_matched", values["ess_power_matched"])]
    for metric in ("A_target", "A_all"):
        statistics.append((f"gain_{metric}_power_matched", values[f"{metric}_power_matched"] - values[f"{metric}_b0"]))
        statistics.append((f"tilt_minus_power_{metric}", values[f"{metric}_b0.5"] - values[f"{metric}_power_matched"]))
    for name, v in statistics:
        mean, lo, hi = ci(v, idx)
        row[name] = mean
        row[f"{name}_lo"] = lo
        row[f"{name}_hi"] = hi
    return row


def mechanism_rows(gold_pools, nboot):
    """Gold NLL per token by reveal position (mechanism.csv, the mechanism figure).

    In the cross-benchmark study (M = 4, J = 8) the reference answers were scored target by target in the
    fixed reveal order, so gold_ell1_t[k] (ell1_t in the gold records) is the log P1 of target k given the
    gold answers of targets 1..k-1, and the same for P0. Per position k, over families: the gold reward per
    token (ell1_t[k] - ell0_t[k]) / L_k with a bootstrap interval, and the gold NLL per token under P1 and P0.
    """
    rows = []
    for model in MECHANISM_MODELS:
        for task, label in MECHANISM_TASKS:
            path = Path(gold_pools) / model / f"{task}_M{MECHANISM_M}_J{MECHANISM_J}_il.jsonl"
            if not path.exists():
                continue
            gold = {}
            for record in read_records(path):
                gold.setdefault(record["episode_id"], record)
            records = list(gold.values())
            seed = f"gold|{model}|{task}|{MECHANISM_J}"
            for k in range(MECHANISM_J):
                reward = np.array([(r["ell1_t"][k] - r["ell0_t"][k]) / r["gold_L"][k] for r in records])
                nll1 = np.array([-r["ell1_t"][k] / r["gold_L"][k] for r in records])
                nll0 = np.array([-r["ell0_t"][k] / r["gold_L"][k] for r in records])
                mean, lo, hi = boot_sd(reward, seed, nboot)
                rows.append({
                    "model": model,
                    "benchmark": label,
                    "source": task,
                    "J": MECHANISM_J,
                    "position": k + 1,
                    "gold_families": len(records),
                    "gold_reward_per_token": mean,
                    "gold_rpt_lo": lo,
                    "gold_rpt_hi": hi,
                    "gold_nll1_per_token": float(nll1.mean()),
                    "gold_nll0_per_token": float(nll0.mean()),
                })
    return rows


# ---------------------------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------------------------


def analyse(pools, models, nboot):
    """All per-family, per-cell and per-candidate tables of the M x J grid, as lists of rows."""
    family_ids = [family["episode_id"] for family in load_families(TASK, "report")]
    tables = {name: [] for name in ("families", "cells", "gold", "diagnostics", "candidates", "sharpening")}
    coverage = {}
    for model in models:
        for M in M_GRID:
            for J in J_GRID:
                records = load_cell(pools, model, M, J)
                fids = [fid for fid in family_ids if fid in records]
                coverage[f"{model}|M{M}|J{J}"] = len(fids)
                if not fids:
                    continue
                Ks = {records[fid]["K"] for fid in fids}
                if len(Ks) != 1:
                    raise ValueError(f"{model} M={M} J={J}: records with different K {sorted(Ks)}")
                K = Ks.pop()
                F = []  # family_summary of every family of the cell, in bank order
                for fid in fids:
                    f = family_summary(records[fid], model)
                    tables["candidates"].extend(candidate_rows(model, M, J, f, records[fid]))
                    tables["families"].extend(family_rows(model, M, J, f, records[fid]))
                    F.append(f)
                idx = cell_indices(BOOTSTRAP_SEED, f"{model}|{M}|{J}", len(F), nboot)
                cells = cell_rows(model, M, J, F, idx)
                tables["cells"].extend(cells)
                tables["gold"].append(gold_row(model, M, J, F, idx))
                ordinary_A_all = np.array([pool_weights(f["ell1"], f["ell0"], 0.0) @ f["correct"].all(1) for f in F])
                tables["diagnostics"].append(diagnostics_row(model, M, J, K, F, ordinary_A_all))
                tables["sharpening"].append(sharpening_row(model, M, J, K, F, nboot))
                ordinary = cells[BETAS.index(0.0)]
                guided = cells[BETAS.index(BETA)]
                print(
                    f"{model:11s} M={M:2d} J={J} fam {len(F):2d} | A_target {ordinary['A_target']:.3f} -> {guided['A_target']:.3f}"
                    f" | A_all {ordinary['A_all']:.3f} -> {guided['A_all']:.3f} | gain_all {guided['gain_all']:+.3f}"
                    f" | goldR/tok {tables['gold'][-1]['gold_reward_per_token']:+.2f}",
                    flush=True,
                )
    return tables, coverage, family_ids


def write_tables(out, tables, coverage, family_ids, models):
    """Write every table to `out` (CSV with pandas' defaults, as in the paper's analysis)."""
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in ("families", "cells", "gold", "diagnostics", "sharpening", "mechanism"):
        if name not in tables:
            continue
        frames[name] = pd.DataFrame(tables[name])
        frames[name].to_csv(out / f"{name}.csv", index=False)
    with gzip.open(out / "candidates.csv.gz", "wt") as handle:
        pd.DataFrame(tables["candidates"]).to_csv(handle, index=False)
    expected_cells = len(models) * len(M_GRID) * len(J_GRID)
    complete = all(v == len(family_ids) for v in coverage.values()) and len(coverage) == expected_cells
    summary = {"families_expected": len(family_ids), "cells": coverage, "complete": complete}
    (out / "coverage.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {', '.join(sorted(frames))} to {out}; cells complete: {complete}")
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
    """Print the largest |difference| per numeric column against the tables in `reference`, then the
    resampling verdicts of every cell: is the frozen A_all inside the 95 % CI computed here, for ordinary
    sampling (beta = 0) and for guidance at BETA? One line each; the guided line names its beta.

    CSVs are parsed with float_precision="round_trip", so a value written by pandas reads back exactly.
    """
    for name, keys in KEYS.items():
        path = Path(reference) / f"{name}.csv"
        if name not in frames or frames[name].empty or not path.exists():
            continue
        theirs = pd.read_csv(path, float_precision="round_trip")
        merged = frames[name].merge(theirs, on=keys, suffixes=("", "_ref"))
        columns = [c for c in theirs.columns if c not in keys and theirs[c].dtype.kind == "f" and c in frames[name]]
        worst = max(((max_abs_diff(merged[c], merged[c + "_ref"]), c) for c in columns), default=(0.0, "-"))
        where = f" ({worst[1]})" if worst[0] > 0 else ""
        print(f"{name}: {len(merged)} rows in common; largest |diff| {worst[0]:.3e}{where}")
        for column in columns:
            print(f"  {column:32s} max |diff| {max_abs_diff(merged[column], merged[column + '_ref']):.3e}")

    path = Path(reference) / "cells.csv"
    if "cells" not in frames or frames["cells"].empty or not path.exists():
        return
    frozen = pd.read_csv(path, float_precision="round_trip")
    frozen = frozen.set_index(["model", "M", "J", "beta"])
    cells = frames["cells"]
    guided = cells[cells.beta == BETA].set_index(["model", "M", "J"])
    for _, ordinary in cells[cells.beta == 0.0].iterrows():
        key = (ordinary.model, ordinary.M, ordinary.J)
        rows = [(0.0, "", ordinary)]
        if key in guided.index:
            rows.append((BETA, f" beta={BETA:g}", guided.loc[key]))
        for beta, label, cell in rows:
            if key + (beta,) not in frozen.index:
                continue
            value = frozen.loc[key + (beta,), "A_all"]
            inside = cell.A_all_lo <= value <= cell.A_all_hi
            verdict = "inside" if inside else "outside"
            print(
                f"{ordinary.model} M={ordinary.M} J={ordinary.J}{label}: frozen A_all {value:.3f}, rerun CI"
                f" [{cell.A_all_lo:.3f}, {cell.A_all_hi:.3f}] -> frozen mean {verdict} the rerun's 95 % CI ({int(cell.families)} families)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pools", required=True, help="directory with <model>/list_easy_fam_M{M}_J{J}.jsonl")
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--nboot", type=int, default=NBOOT)
    parser.add_argument("--gold-pools", default=None, help="cross-benchmark gold records (default: <pools>/crossbench_gold)")
    parser.add_argument("--compare", default=None, help="directory of reference tables, e.g. data/tables/fig5")
    args = parser.parse_args()

    pools = Path(args.pools)
    models = args.models.split(",")
    if not pools.is_dir() or not any((pools / model).is_dir() for model in models):
        print(f"no Figure 5 pools in {pools}; {FETCH_HINT}", file=sys.stderr)
        sys.exit(2)

    tables, coverage, family_ids = analyse(pools, models, args.nboot)
    gold_pools = Path(args.gold_pools) if args.gold_pools else pools / "crossbench_gold"
    if gold_pools.is_dir():
        tables["mechanism"] = mechanism_rows(gold_pools, args.nboot)
    else:
        print(f"no cross-benchmark gold records in {gold_pools}: mechanism.csv not written")
    frames = write_tables(Path(args.out), tables, coverage, family_ids, models)
    if args.compare:
        compare(frames, args.compare)


if __name__ == "__main__":
    main()
