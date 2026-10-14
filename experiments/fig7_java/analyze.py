# ported from editdiff/<the authors' Section 5 run package>: analyze.py, length_metrics.py, zd_analyze.py (zero-demonstration ablation) and s5_pool_audit.py (pool taxonomy) @ d276fa6
"""Figure 7 (Section 5) and its Appendix C tables, from the frozen Java pools and judge verdicts (CPU only).

Section 5: LLaDA-1.5 propagates a developer's helper refactoring. Each of the 127 episodes of data/java shows the
helper, M accepted edits (the demonstrations) and J target sites; cells (M, J) in {1, 2} x {1, 2}; 50 commits.
For every episode and length regime (oracle: the reference token counts; learned: lengths drawn from head C) the
run sampled K = 16 infills from P1. Draw k has its example reward R_k = logp1 - logp0 and a verdict of the judge
(rubric r1, judge.py): y_k = 1 when every target is judged correct; "incorrect" and "uncertain" count 0 (the
upper bound counts "uncertain" as 1). With w = softmax(beta * R) (scoring.pool_weights),

    U_beta(episode, regime) = sum_k w_k y_k        beta = 0 ordinary sampling, beta = 0.5 example guidance

A cell's estimate is the mean over commits of the mean of each commit's episodes. Intervals resample commits:
10,000 resamples, random.Random(0), one randrange per commit in the order the commits first appear, percentiles
at the indices int(0.025 n) and int(0.975 n) of the sorted resampled means (commit_bootstrap). A paired contrast
resamples the per-commit differences.

Inputs (python -m fewshot_dlms.fetch pools java), with defaults next to --pools:
    <pools>/pools_*.jsonl                one record per (episode, regime): K candidates with toks, text_by_target,
                                         logp1, logp0, R and allocation (run.py)
    <judging>/verdicts.jsonl             one verdict per judge packet, keyed by the packet key (judge.py)
    <length>/<head>_s5final_G4.npz       length-head predictions for the target sites (Figure 7, left)
    <length>/prior_s5h_train.json        the training-label length prior
    <zero_demo>/pools/pools_*.jsonl and <zero_demo>/judging/verdicts.jsonl     the zero-demonstration ablation

Outputs in --out:
    metrics_by_episode.csv     one row per (episode, regime): U at every beta, coverage, pool class, AUC of R
    summary_by_cell.csv        Figure 7 (right) and Table 1 of Appendix C: U per cell, regime and rule
    paired_intervals.csv       guidance minus ordinary and learned minus oracle, per stratum and cell
    metrics_by_commit.csv      commit means per cell and regime
    strata.csv                 summary_by_cell.csv in the strata of Appendix C
    beta_sensitivity.csv       U and its gain over ordinary at beta in {0, 0.1, 0.25, 0.5, 1}
    length_fidelity.csv        length mass within +-2 tokens (F2), exact mass (F0), NLL per head over the 80 unique sites
    length_fidelity_by_bin.csv Figure 7 (left): the same by reference length bin, head C and two comparators vs the prior
    zero_demo_main_table.csv   Appendix C: ordinary success with M = 0 vs M = 1 demonstrations
    pool_taxonomy_estimates.csv  Appendix C: estimates on all pools and on the pools with both outcomes
    checkstyle_case.json       Section 5: the Checkstyle pool where guidance moves weight to a wrong draw

--compare DIR prints the largest |difference| per numeric column against the tables in DIR and, per cell, whether
the frozen U lies inside the 95 % CI of the pools analysed here (fidelity level 3).

    python experiments/fig7_java/analyze.py --pools artifacts/java/pools --compare data/tables/fig7
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import pandas as pd

from fewshot_dlms.data import load_java_episodes
from fewshot_dlms.scoring import pool_weights
from judge import RUBRIC
from judge import packet_key
from judge import rubric_sha

HERE = Path(__file__).resolve().parent
FETCH_HINT = "run: python -m fewshot_dlms.fetch pools java"

# The run's design, fixed before any output (the protocol lock of Section 5).
CELLS = ["M1J1", "M1J2", "M2J1", "M2J2"]
REGIMES = ["oracle", "learned"]
BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]  # beta = 0 ordinary; 0.5 the paper's guidance; the rest are the Appendix C grid
BETA = 0.5
RULES = [(0.0, "ordinary"), (0.5, "tilting")]  # "tilting" is the run's name for example guidance at beta = 0.5
NBOOT = 10000
BOOTSTRAP_SEED = 0  # random.Random(0) for every commit bootstrap of Section 5
COMMENT = re.compile(r"^\s*(//|/\*|\*)")  # a Java comment line (padding a candidate with comments)

# Appendix C strata of the main run: (name, predicate on a metrics_by_episode row).
STRATA = [
    ("not_llada_dev", lambda row: not row["llada_dev"]),
    ("llada_dev", lambda row: row["llada_dev"]),
    ("gold_visible", lambda row: row["gold_visible"]),
    ("gold_not_visible", lambda row: not row["gold_visible"]),
    ("literal_copy", lambda row: row["literal_copy"]),
    ("adaptation", lambda row: not row["literal_copy"]),
    ("manifest_A", lambda row: row["in_A"]),
    ("test_code", lambda row: row["test_code"]),
    ("main_code", lambda row: not row["test_code"]),
    ("mixed_pools_diagnostic", lambda row: row["class"] == "mixed"),
]

# Figure 7 (left): the length heads evaluated on the target sites, and the population of the paper.
LENGTH_HEADS = ["s5hc_C", "s5g_B", "fin_llada_m70_100k"]  # head C (used by the run) and two comparators
LENGTH_PRIOR = "prior_s5h_train.json"
LENGTH_BINS = [(1, 8), (9, 16), (17, 32), (33, 80)]
POPULATION = "B"  # the 127 episodes of data/java

# The Checkstyle pool of Section 5.
CHECKSTYLE_EPISODE = "M1J1_chec_b142dda_getMessage"

# The tables of --compare and the columns that identify a row.
KEYS = {
    "summary_by_cell": ["stratum", "cell", "regime", "rule"],
    "paired_intervals": ["stratum", "cell", "contrast"],
    "metrics_by_commit": ["cell", "regime", "commit10"],
    "strata": ["stratum", "cell", "regime", "rule"],
    "beta_sensitivity": ["regime", "beta"],
    "length_fidelity_by_bin": ["population", "head", "bin"],
    "length_fidelity": ["population", "head"],
    "zero_demo_main_table": ["stratum", "block"],
    "pool_taxonomy_estimates": ["regime", "subset"],
}


# ---------------------------------------------------------------------------------------------
# Reading pools and verdicts
# ---------------------------------------------------------------------------------------------


def read_records(path):
    """The JSON records of a file, skipping blank lines and lines that do not parse (an interrupted write)."""
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


def read_main_pools(pools):
    """(episode_id, regime) -> pool record, from <pools>/pools_*.jsonl in sorted file order.

    The paper's run finished 23 units twice (two workers on one list); the copies are identical, and, as in the
    paper's analysis, the later one is kept.
    """
    records = {}
    for path in sorted(Path(pools).glob("pools_*.jsonl")):
        for record in read_records(path):
            records[(record["episode_id"], record["regime"])] = record
    return records


def read_verdicts(path):
    """Packet key -> verdict record (judge.py); empty when the file does not exist (a rerun not yet judged)."""
    if not Path(path).exists():
        return {}
    return {record["key"]: record for record in read_records(path)}


def joint_label(verdict):
    """1.0 for a joint "correct", 0.0 for "incorrect", NaN for "uncertain" (counted separately)."""
    if verdict["joint"] == "correct":
        return 1.0
    if verdict["joint"] == "incorrect":
        return 0.0
    return float("nan")


def auc(scores, labels):
    """P(score of a correct draw > score of an incorrect draw) + half the ties, over all pairs (NaN if a class is empty)."""
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return float(wins / (len(positives) * len(negatives)))


def comment_lines(texts, keys):
    """Number of comment lines in the texts of the given targets."""
    return sum(1 for key in keys for line in texts.get(key, "").split("\n") if COMMENT.match(line))


# ---------------------------------------------------------------------------------------------
# Commit bootstrap
# ---------------------------------------------------------------------------------------------

_INDEX_CACHE = {}


def resample_indices(n, nboot, seed):
    """The nboot x n commit indices of the paper's bootstrap: random.Random(seed), one randrange(n) per commit.

    Every interval starts a fresh generator with the same seed, so the indices depend only on (n, nboot, seed)
    and are drawn once per n.
    """
    key = (n, nboot, seed)
    if key not in _INDEX_CACHE:
        rng = random.Random(seed)
        rows = []
        for _ in range(nboot):
            rows.append([rng.randrange(n) for _ in range(n)])
        _INDEX_CACHE[key] = np.array(rows)
    return _INDEX_CACHE[key]


def commit_bootstrap(values, nboot=NBOOT, seed=BOOTSTRAP_SEED):
    """95 % percentile interval of the mean over commits; `values` maps commit -> value, in first-appearance order.

    Each resample draws len(values) commits with replacement; the interval is the sorted resampled means at the
    indices int(0.025 nboot) and int(0.975 nboot) (analyze.py's boot_ci).
    """
    vals = np.array([values[commit] for commit in values], float)
    idx = resample_indices(len(vals), nboot, seed)
    means = np.sort(vals[idx].mean(1))
    return float(means[int(0.025 * nboot)]), float(means[int(0.975 * nboot)])


def commit_means(rows, column):
    """Commit -> mean of `column` over the commit's rows, commits in first-appearance order."""
    by_commit = collections.defaultdict(list)
    for row in rows:
        by_commit[row["commit10"]].append(row[column])
    return {commit: float(np.mean(values)) for commit, values in by_commit.items()}


def mean_of(values):
    return float(np.mean(list(values)))


# ---------------------------------------------------------------------------------------------
# Main run: one row per (episode, regime)
# ---------------------------------------------------------------------------------------------


def episode_row(episode, record, verdicts, rubric):
    """metrics_by_episode.csv row of one pool: U_beta and the pool facts of Appendix C."""
    candidates = record["candidates"]
    K = len(candidates)
    targets = record["targets"]
    labels = []  # per draw: 1.0, 0.0, NaN (uncertain) or None (no verdict)
    per_target = []  # per draw: fraction of its targets judged correct
    uncertain = 0
    unjudged = 0
    ill_posed = 0
    correct_nonref = 0
    correct_padded = 0
    for candidate in candidates:
        verdict = verdicts.get(packet_key(rubric, episode, candidate["text_by_target"]))
        if verdict is None:
            unjudged += 1
            labels.append(None)
            per_target.append(float("nan"))
            continue
        ill_posed += verdict["instance_ill_posed"]
        y = joint_label(verdict)
        uncertain += math.isnan(y)
        labels.append(y)
        per_target.append(float(np.mean([1.0 if verdict["per_target"].get(key) == "correct" else 0.0 for key in targets])))
        if y == 1.0 and not candidate["allocation_exact"]:
            correct_nonref += 1
            if comment_lines(candidate["text_by_target"], targets) > comment_lines(record.get("gold_text", {}), targets):
                correct_padded += 1
    ys = [float("nan") if label is None else label for label in labels]
    y_low = np.array([0.0 if math.isnan(y) else y for y in ys])
    y_high = np.array([1.0 if math.isnan(y) else y for y in ys])
    per_target_low = np.array([0.0 if math.isnan(p) else p for p in per_target])
    R = [candidate["R"] for candidate in candidates]
    ell1 = [candidate["logp1"] for candidate in candidates]
    ell0 = [candidate["logp0"] for candidate in candidates]

    row = {
        "episode_id": record["episode_id"],
        "cell": episode["cell"],
        "M": episode["M"],
        "J": episode["J"],
        "commit10": episode["commit10"],
        "regime": record["regime"],
        "K": K,
        "n_unique_text": record["n_unique_text"],
        "n_unique_toks": record["n_unique_toks"],
        "n_uncertain_draws": uncertain,
        "n_unjudged_draws": unjudged,
        "n_ill_posed_flags": ill_posed,
        "distinct_allocations": record["distinct_allocations"],
        "allocation_rejections": record["allocation_rejections"],
        "max_p1_replay_absdiff": record["max_p1_replay_absdiff"],
        "exposure": episode["exposure"],
        "llada_dev": episode["exposure_strict_llada_dev"],
        "gold_visible": episode["gold_visible_in_helper_or_demo"],
        "literal_copy": episode["literal_copy_of_demo"],
        "overload_not_called": episode["rendered_overload_not_called"],
        "in_A": episode["in_manifest_A"],
        "test_code": episode["test_code"],
    }
    for beta in BETAS:
        w = pool_weights(ell1, ell0, beta)
        row[f"U{beta}"] = float(w @ y_low)
        row[f"U{beta}_upper"] = float(w @ y_high)
        row[f"T{beta}"] = float(w @ per_target_low)
        if beta == BETA:
            row["ESS0.5"] = float(1 / (w * w).sum())
            row["maxw0.5"] = float(w.max())
    correct_draws = sum(1 for y in ys if y == 1.0)
    row["coverage"] = float(correct_draws > 0)
    row["headroom"] = row["coverage"] - row["U0.0"]
    row["correct_draws"] = correct_draws
    row["unique_correct"] = len({candidate["text_sha"] for candidate, y in zip(candidates, ys) if y == 1.0})
    if correct_draws == K:
        row["class"] = "all_correct"
    elif correct_draws == 0 and uncertain == 0 and unjudged == 0:
        row["class"] = "none_correct"
    else:
        row["class"] = "mixed"
    definite = [(r, y) for r, y in zip(R, ys) if y in (0.0, 1.0)]
    row["auc_R"] = auc([r for r, _ in definite], [y for _, y in definite]) if row["class"] == "mixed" else float("nan")
    row["R_range"] = max(R) - min(R)
    row["modal_share"] = max(collections.Counter(candidate["text_sha"] for candidate in candidates).values()) / K
    row["correct_nonref_length_draws"] = correct_nonref
    row["correct_nonref_padded_draws"] = correct_padded
    row["alloc_exact_draws"] = sum(candidate["allocation_exact"] for candidate in candidates)
    row["mean_signed_alloc_error"] = float(np.mean([sum(candidate["allocation"]) - sum(record["L_ref"]) for candidate in candidates]))
    row["sumL_ref"] = sum(record["L_ref"])
    return row


def stratum_tables(rows, label, K, nboot, out):
    """The summary rows (U per cell, regime and rule) and paired contrasts of one stratum, appended to `out`.

    out["summary"] gets the primary stratum, out["strata"] every other one; out["paired"] gets, per cell and
    regime, guidance minus ordinary, and per cell and rule, learned minus oracle (paired by episode, then commit).
    """
    for cell in CELLS + ["all"]:
        cell_rows = [row for row in rows if cell == "all" or row["cell"] == cell]
        for regime in REGIMES:
            group = [row for row in cell_rows if row["regime"] == regime]
            if not group:
                continue
            for beta, rule in RULES:
                means = commit_means(group, f"U{beta}")
                lo, hi = commit_bootstrap(means, nboot)
                summary = {
                    "stratum": label,
                    "cell": cell,
                    "regime": regime,
                    "rule": rule,
                    "beta": beta,
                    "n_commits": len(means),
                    "n_episodes": len(group),
                    "K": K,
                    "n_unique_patches": sum(row["n_unique_text"] for row in group),
                    "U_commit": mean_of(means.values()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "U_episode": float(np.mean([row[f"U{beta}"] for row in group])),
                    "U_commit_upper_uncertain": mean_of(commit_means(group, f"U{beta}_upper").values()),
                    "per_target_commit": mean_of(commit_means(group, f"T{beta}").values()),
                    "coverage_commit": mean_of(commit_means(group, "coverage").values()),
                    "headroom_commit": mean_of(commit_means(group, "headroom").values()),
                    "all_correct": sum(row["class"] == "all_correct" for row in group),
                    "none_correct": sum(row["class"] == "none_correct" for row in group),
                    "mixed": sum(row["class"] == "mixed" for row in group),
                    "uncertain_draws": sum(row["n_uncertain_draws"] for row in group),
                    "unjudged_draws": sum(row["n_unjudged_draws"] for row in group),
                }
                out["summary" if label == "primary" else "strata"].append(summary)
            guided = commit_means(group, "U0.5")
            ordinary = commit_means(group, "U0.0")
            differences = {commit: guided[commit] - ordinary[commit] for commit in ordinary}
            lo, hi = commit_bootstrap(differences, nboot)
            out["paired"].append({
                "stratum": label,
                "cell": cell,
                "contrast": f"tilting - ordinary ({regime})",
                "n_commits": len(differences),
                "n_episodes": len(group),
                "delta_commit": mean_of(differences.values()),
                "ci_lo": lo,
                "ci_hi": hi,
                "episodes_better": sum(row["U0.5"] - row["U0.0"] > 1e-9 for row in group),
                "episodes_worse": sum(row["U0.5"] - row["U0.0"] < -1e-9 for row in group),
                "delta_episode": float(np.mean([row["U0.5"] - row["U0.0"] for row in group])),
            })
        oracle_of = {row["episode_id"]: row for row in cell_rows if row["regime"] == "oracle"}
        for beta, rule in RULES:
            by_commit = collections.defaultdict(list)
            for row in cell_rows:
                if row["regime"] == "learned" and row["episode_id"] in oracle_of:
                    by_commit[row["commit10"]].append(row[f"U{beta}"] - oracle_of[row["episode_id"]][f"U{beta}"])
            if not by_commit:
                continue
            differences = {commit: float(np.mean(values)) for commit, values in by_commit.items()}
            lo, hi = commit_bootstrap(differences, nboot)
            flat = [value for values in by_commit.values() for value in values]
            out["paired"].append({
                "stratum": label,
                "cell": cell,
                "contrast": f"learned - oracle ({rule})",
                "n_commits": len(differences),
                "n_episodes": len(flat),
                "delta_commit": mean_of(differences.values()),
                "ci_lo": lo,
                "ci_hi": hi,
                "episodes_better": sum(value > 1e-9 for value in flat),
                "episodes_worse": sum(value < -1e-9 for value in flat),
                "delta_episode": float(np.mean(flat)),
            })


def beta_rows(rows, nboot):
    """beta_sensitivity.csv: U over commits at each beta and its paired gain over ordinary sampling."""
    out = []
    for beta in BETAS:
        for regime in REGIMES:
            group = [row for row in rows if row["regime"] == regime]
            if not group:
                continue
            means = commit_means(group, f"U{beta}")
            ordinary = commit_means(group, "U0.0")
            differences = {commit: means[commit] - ordinary[commit] for commit in means}
            lo, hi = commit_bootstrap(differences, nboot)
            out.append({
                "regime": regime,
                "beta": beta,
                "n_commits": len(means),
                "U_commit": mean_of(means.values()),
                "delta_vs_ordinary": mean_of(differences.values()),
                "ci_lo": lo,
                "ci_hi": hi,
            })
    return out


def commit_rows(rows):
    """metrics_by_commit.csv: per cell and regime, each commit's mean U at beta = 0 and 0.5 and its coverage."""
    out = []
    for cell in CELLS:
        for regime in REGIMES:
            by_commit = collections.defaultdict(list)
            for row in rows:
                if row["cell"] == cell and row["regime"] == regime:
                    by_commit[row["commit10"]].append(row)
            for commit, group in sorted(by_commit.items()):
                out.append({
                    "cell": cell,
                    "regime": regime,
                    "commit10": commit,
                    "n_episodes": len(group),
                    "U0.0": np.mean([row["U0.0"] for row in group]),
                    "U0.5": np.mean([row["U0.5"] for row in group]),
                    "coverage": np.mean([row["coverage"] for row in group]),
                })
    return out


def main_run(pools, verdicts, episodes, nboot):
    """All tables of the main run: per episode, per cell, strata, contrasts, beta grid and commit means."""
    rubric = rubric_sha(RUBRIC)
    rows = []
    Ks = set()
    for (episode_id, _), record in sorted(pools.items()):
        if episode_id not in episodes:
            continue
        rows.append(episode_row(episodes[episode_id], record, verdicts, rubric))
        Ks.add(len(record["candidates"]))
    if len(Ks) != 1:
        raise ValueError(f"pools with different K {sorted(Ks)}")
    K = Ks.pop()
    tables = {"summary": [], "strata": [], "paired": []}
    stratum_tables(rows, "primary", K, nboot, tables)
    for label, predicate in STRATA:
        stratum_tables([row for row in rows if predicate(row)], label, K, nboot, tables)
    return {
        "metrics_by_episode": rows,
        "summary_by_cell": tables["summary"],
        "paired_intervals": tables["paired"],
        "strata": tables["strata"],
        "beta_sensitivity": beta_rows(rows, nboot),
        "metrics_by_commit": commit_rows(rows),
    }


# ---------------------------------------------------------------------------------------------
# Pool taxonomy and the Checkstyle case (Appendix C, Section 5)
# ---------------------------------------------------------------------------------------------


def taxonomy_rows(pools, verdicts, episodes):
    """Per (episode, regime): draws by verdict, the binary class (all / no / some correct draws), U at 0 and 0.5."""
    rubric = rubric_sha(RUBRIC)
    rows = []
    for regime in REGIMES:
        for episode_id, _ in sorted(key for key in pools if key[1] == regime and key[0] in episodes):
            record = pools[(episode_id, regime)]
            candidates = record["candidates"]
            labels = []
            for candidate in candidates:
                verdict = verdicts.get(packet_key(rubric, episodes[episode_id], candidate["text_by_target"]))
                labels.append("unjudged" if verdict is None else verdict["joint"])
            y = np.array([1.0 if label == "correct" else 0.0 for label in labels])
            ell1 = [candidate["logp1"] for candidate in candidates]
            ell0 = [candidate["logp0"] for candidate in candidates]
            n_correct = labels.count("correct")
            if n_correct == len(candidates):
                binary = "all_correct"
            elif n_correct == 0:
                binary = "no_correct"
            else:
                binary = "binary_mixed"
            if n_correct == len(candidates):
                ternary = "all_correct"
            elif n_correct == 0 and labels.count("uncertain") == 0 and labels.count("unjudged") == 0:
                ternary = "none_correct"
            else:
                ternary = "mixed"
            U0 = float(pool_weights(ell1, ell0, 0.0) @ y)
            U05 = float(pool_weights(ell1, ell0, BETA) @ y)
            rows.append({
                "episode_id": episode_id,
                "regime": regime,
                "commit10": episodes[episode_id]["commit10"],
                "binary_class": binary,
                "frozen_ternary_class": ternary,
                "definitely_mixed": n_correct > 0 and labels.count("incorrect") > 0,
                "U0": U0,
                "U05": U05,
                "delta": U05 - U0,
            })
    return rows


def taxonomy_estimate(rows, nboot):
    """U at beta = 0 and 0.5 and their paired difference over commits, with commit-bootstrap intervals."""
    ordinary = commit_means(rows, "U0")
    guided = commit_means(rows, "U05")
    differences = {commit: guided[commit] - ordinary[commit] for commit in ordinary}
    lo0, hi0 = commit_bootstrap(ordinary, nboot)
    lo5, hi5 = commit_bootstrap(guided, nboot)
    lod, hid = commit_bootstrap(differences, nboot)
    return {
        "n_episodes": len(rows),
        "n_commits": len(ordinary),
        "U0": mean_of(ordinary.values()),
        "U0_lo": lo0,
        "U0_hi": hi0,
        "U05": mean_of(guided.values()),
        "U05_lo": lo5,
        "U05_hi": hi5,
        "delta": mean_of(differences.values()),
        "delta_lo": lod,
        "delta_hi": hid,
        "episodes_better": sum(row["delta"] > 1e-9 for row in rows),
        "episodes_worse": sum(row["delta"] < -1e-9 for row in rows),
        "episodes_unchanged": sum(abs(row["delta"]) <= 1e-9 for row in rows),
    }


def taxonomy_table(pools, verdicts, episodes, nboot):
    """pool_taxonomy_estimates.csv: every pool, the pools with a correct and a non-correct draw (binary mixed), the
    pools the run's analysis called mixed (uncertain draws included), and the pools with a correct and an incorrect draw."""
    rows = taxonomy_rows(pools, verdicts, episodes)
    subsets = [
        ("all 127 (primary)", lambda row: True),
        ("binary_mixed", lambda row: row["binary_class"] == "binary_mixed"),
        ("frozen ternary 'mixed' (historical)", lambda row: row["frozen_ternary_class"] == "mixed"),
        ("definitely_mixed", lambda row: row["definitely_mixed"]),
    ]
    out = []
    for regime in REGIMES:
        group = [row for row in rows if row["regime"] == regime]
        for name, predicate in subsets:
            selected = [row for row in group if predicate(row)]
            if selected:
                out.append({"regime": regime, "subset": name, **taxonomy_estimate(selected, nboot)})
    return out


def checkstyle_case(pools, verdicts, episodes):
    """The Checkstyle pool of Section 5 (oracle lengths): a correct infill drawn 7 times and an incorrect one drawn
    once whose larger reward R makes guidance move most of the weight to it. None if a draw has no verdict.

    Draws with the same text are merged (multiplicity, summed weights). `correct` is the most frequent correct
    text, `highlighted_incorrect` the non-correct text with the largest R. The ideal target Q_beta ∝ P1 e^{beta R}
    weighs the pair in the ratio exp(Delta log P1 + beta Delta R).
    """
    episode = episodes[CHECKSTYLE_EPISODE]
    record = pools[(CHECKSTYLE_EPISODE, "oracle")]
    candidates = record["candidates"]
    rubric = rubric_sha(RUBRIC)
    keys = [packet_key(rubric, episode, candidate["text_by_target"]) for candidate in candidates]
    if any(key not in verdicts for key in keys):
        return None
    labels = [verdicts[key]["joint"] for key in keys]
    ell1 = [candidate["logp1"] for candidate in candidates]
    ell0 = [candidate["logp0"] for candidate in candidates]
    w0 = pool_weights(ell1, ell0, 0.0)
    w5 = pool_weights(ell1, ell0, BETA)
    unique = {}
    for candidate, label, a0, a5 in zip(candidates, labels, w0, w5):
        text = unique.setdefault(candidate["text_sha"], {
            "text_sha": candidate["text_sha"],
            "verdict": label,
            "multiplicity": 0,
            "logp1": candidate["logp1"],
            "logp0": candidate["logp0"],
            "R": candidate["R"],
            "w_beta0": 0.0,
            "w_beta05": 0.0,
            "draws": [],
        })
        text["multiplicity"] += 1
        text["w_beta0"] += float(a0)
        text["w_beta05"] += float(a5)
        text["draws"].append(candidate["draw"])
    texts = list(unique.values())
    correct = max((text for text in texts if text["verdict"] == "correct"), key=lambda text: text["multiplicity"])
    wrong = max((text for text in texts if text["verdict"] != "correct"), key=lambda text: text["R"])
    d1 = wrong["logp1"] - correct["logp1"]
    dR = wrong["R"] - correct["R"]
    others = [text for text in texts if text is not correct and text is not wrong]
    y = np.array([1.0 if label == "correct" else 0.0 for label in labels])
    return {
        "episode_id": CHECKSTYLE_EPISODE,
        "commit10": episode["commit10"],
        "llada_dev": episode["exposure_strict_llada_dev"],
        "K": len(candidates),
        "L_ref": record["L_ref"],
        "n_unique": len(texts),
        "correct": {key: correct[key] for key in ("text_sha", "multiplicity", "logp1", "logp0", "R", "w_beta0", "w_beta05", "draws")},
        "highlighted_incorrect": {key: wrong[key] for key in ("text_sha", "verdict", "multiplicity", "logp1", "logp0", "R", "w_beta0", "w_beta05", "draws")},
        "other_noncorrect_draws": len(candidates) - correct["multiplicity"] - wrong["multiplicity"],
        "other_noncorrect_verdicts": dict(collections.Counter(text["verdict"] for text in others for _ in range(text["multiplicity"]))),
        "sum_w0": sum(text["w_beta0"] for text in texts),
        "sum_w05": sum(text["w_beta05"] for text in texts),
        "U0": float(w0 @ y),
        "U05": float(w5 @ y),
        "logP1_diff_wrong_minus_correct": d1,
        "P1_ratio_wrong_over_correct": math.exp(d1),
        "R_diff_wrong_minus_correct": dR,
        "idealQ_ratio_wrong_over_correct_beta05": math.exp(d1 + 0.5 * dR),
        "idealQ_ratio_wrong_over_correct_beta1": math.exp(d1 + 1.0 * dR),
        "pool_weight_ratio_wrong_over_correct_total_beta05": wrong["w_beta05"] / correct["w_beta05"],
        "pool_weight_ratio_check": wrong["multiplicity"] / correct["multiplicity"] * math.exp(0.5 * dR),
        "beta_where_ideal_pair_ratio_crosses_1": -d1 / dR,
        "P1_wrong": math.exp(wrong["logp1"]),
        "P1_correct": math.exp(correct["logp1"]),
    }


# ---------------------------------------------------------------------------------------------
# Figure 7 (left): length fidelity of the heads (length_metrics.py)
# ---------------------------------------------------------------------------------------------


def site_metrics(p, L):
    """How a length distribution p over 1..lmax treats the reference length L.

    F0 = p(L), F2 = p(L-2..L+2), nll = -log p(L), eae = expected |length - L|, mode, under = p(< L), over = p(> L).
    The arithmetic follows the paper's script in the stored dtype (float32 for the heads).
    """
    p = np.clip(p, 0, None)
    p = p / p.sum()
    index = L - 1
    lo = max(0, index - 2)
    hi = min(len(p), index + 3)
    return {
        "F0": float(p[index]),
        "F2": float(p[lo:hi].sum()),
        "nll": float(-math.log(max(p[index], 1e-12))),
        "eae": float(sum(p[i] * abs(i + 1 - L) for i in range(len(p)))),
        "mode": int(p.argmax() + 1),
        "under": float(p[:index].sum()),
        "over": float(p[index + 1 :].sum()),
    }


def site_bootstrap(values_by_commit, statistic, nboot=NBOOT, seed=BOOTSTRAP_SEED):
    """Interval of a statistic over sites, resampling commits with all their sites (length_metrics.py's boot).

    random.Random(seed), one choice per commit; the statistic of the pooled sites of the drawn commits; sorted;
    indices int(0.025 n) and int(0.975 n).
    """
    rng = random.Random(seed)
    commits = list(values_by_commit)
    stats = []
    for _ in range(nboot):
        drawn = [rng.choice(commits) for _ in commits]
        stats.append(statistic([value for commit in drawn for value in values_by_commit[commit]]))
    stats.sort()
    return stats[int(0.025 * nboot)], stats[int(0.975 * nboot)]


def interval_text(values_by_commit, nboot):
    return "[%.3f,%.3f]" % site_bootstrap(values_by_commit, np.mean, nboot)


def length_tables(length_dir, episodes, nboot):
    """length_fidelity.csv and length_fidelity_by_bin.csv over the unique target sites of the 127 episodes.

    A unique site is (commit, file, site); its first appearance (episodes in data/java order, targets in order)
    stands for it, since cells share the site's text and length. Each head's distribution at a site is its
    prediction for the episode's compact prompt; the prior is the training-label length distribution.
    """
    prior = np.array(json.loads((Path(length_dir) / LENGTH_PRIOR).read_text())["p"])
    prior = prior / prior.sum()
    overall = []
    by_bin = []
    for head in LENGTH_HEADS:
        stored = np.load(Path(length_dir) / f"{head}_s5final_G4.npz", allow_pickle=False)
        probs = dict(zip(stored["ids"].tolist(), stored["P"]))
        sites = {}
        for episode_id, episode in episodes.items():
            for j, target in enumerate(episode["reference"]["targets"]):
                L = target["oracle_length"]
                site = f"{episode['commit10']}:{episode['file'].split('/')[-1]}:{target['site_id']}"
                row = {"commit10": episode["commit10"], "L": L, **site_metrics(probs[f"{episode_id}|{episode['J']}|{j}"], L)}
                for key, value in site_metrics(prior, L).items():
                    row["prior_" + key] = value
                sites.setdefault(site, row)
        unique = list(sites.values())
        by_commit = collections.defaultdict(list)
        for row in unique:
            by_commit[row["commit10"]].append(row)
        Ls = [row["L"] for row in unique]
        summary = {
            "population": POPULATION,
            "head": head,
            "unique_sites": len(unique),
            "commits": len(by_commit),
            "L_mean": np.mean(Ls),
            "L_sd": np.std(Ls, ddof=1),
            "L_median": np.median(Ls),
            "L_q25": np.percentile(Ls, 25),
            "L_q75": np.percentile(Ls, 75),
            "L_max": max(Ls),
        }
        for key in ("F0", "F2", "nll", "eae", "under", "over"):
            summary[key] = np.mean([row[key] for row in unique])
            summary[key + "_ci"] = interval_text({commit: [row[key] for row in rows] for commit, rows in by_commit.items()}, nboot)
            summary["prior_" + key] = np.mean([row["prior_" + key] for row in unique])
        summary["mode_exact"] = np.mean([row["mode"] == row["L"] for row in unique])
        summary["prior_mode_exact"] = np.mean([row["prior_mode"] == row["L"] for row in unique])
        overall.append(summary)
        for a, b in LENGTH_BINS:
            group = [row for row in unique if a <= row["L"] <= b]
            if not group:
                continue
            bin_commits = collections.defaultdict(list)
            for row in group:
                bin_commits[row["commit10"]].append(row)
            by_bin.append({
                "population": POPULATION,
                "head": head,
                "bin": f"{a}-{b}",
                "unique_sites": len(group),
                "commits": len(bin_commits),
                "F2": np.mean([row["F2"] for row in group]),
                "F2_ci": interval_text({commit: [row["F2"] for row in rows] for commit, rows in bin_commits.items()}, nboot),
                "prior_F2": np.mean([row["prior_F2"] for row in group]),
                "prior_F2_ci": interval_text({commit: [row["prior_F2"] for row in rows] for commit, rows in bin_commits.items()}, nboot),
                "F0": np.mean([row["F0"] for row in group]),
                "prior_F0": np.mean([row["prior_F0"] for row in group]),
                "nll": np.mean([row["nll"] for row in group]),
                "prior_nll": np.mean([row["prior_nll"] for row in group]),
            })
    return {"length_fidelity": overall, "length_fidelity_by_bin": by_bin}


# ---------------------------------------------------------------------------------------------
# Zero-demonstration ablation (Appendix C, zd_analyze.py)
# ---------------------------------------------------------------------------------------------


def zero_demo_bootstrap(table, statistic, nboot=NBOOT, seed=BOOTSTRAP_SEED):
    """(estimate, lo, hi) of the ablation's paired commit bootstrap (zd_analyze.py's boot).

    Commits in sorted order; random.Random(seed), one choice per commit; the statistic of the drawn commits'
    arm means; sorted; lo at index int(0.025 n), hi at int(0.975 n) - 1 (that script's convention).
    """
    commits = sorted(table)
    rng = random.Random(seed)
    estimate = statistic([table[commit] for commit in commits])
    stats = []
    for _ in range(nboot):
        drawn = [table[rng.choice(commits)] for _ in commits]
        stats.append(statistic(drawn))
    stats.sort()
    return estimate, stats[int(0.025 * nboot)], stats[int(0.975 * nboot) - 1]


def plain_mean(values):
    values = list(values)
    return sum(values) / len(values)


def zero_demo_pools(zero_demo):
    """unit_id -> pool record of the ablation (first record per unit: two workers shared one list)."""
    records = {}
    for path in sorted((Path(zero_demo) / "pools").glob("pools_*.jsonl")):
        for record in read_records(path):
            records.setdefault(record["unit_id"], record)
    return records


def zero_demo_table(zero_demo, episodes, nboot):
    """zero_demo_main_table.csv: ordinary success (beta = 0, oracle lengths) with M = 0 vs M = 1 demonstrations.

    Each anchor (an M = 1 episode) was run without its demonstration (M = 0) and with it (M = 1), under one
    neutral instruction. Every draw was judged with the anchor's first demonstration as the packet's example set.
    U0 = fraction of draws judged jointly correct; per commit the mean over its anchors; Delta = M1 - M0, paired.
    Returns None when a draw has no verdict.
    """
    verdicts = read_verdicts(Path(zero_demo) / "judging" / "verdicts.jsonl")
    rubric = rubric_sha(RUBRIC)
    pools = {}
    per_target = {}
    for unit_id, record in zero_demo_pools(zero_demo).items():
        anchor = episodes[record["anchor_episode_id"]]
        ys = []
        upper = []
        texts = []
        target_hits = []
        for candidate in record["candidates"]:
            verdict = verdicts.get(packet_key(rubric, anchor, candidate["text_by_target"]))
            if verdict is None:
                print(f"zero-demo ablation: no verdict for {unit_id} draw {candidate['draw']}; table not written")
                return None
            ys.append(1 if verdict["joint"] == "correct" else 0)
            upper.append(1 if verdict["joint"] in ("correct", "uncertain") else 0)
            texts.append(candidate["text_sha"])
            for target in verdict["per_target"]:
                target_hits.append(1 if verdict["per_target"].get(target) == "correct" else 0)
        K = len(ys)
        counts = collections.Counter(texts)
        shares = [count / K for count in counts.values()]
        entropy = -sum(share * math.log(share) for share in shares)
        if sum(ys) == K:
            pool_class = "all"
        elif sum(ys) == 0:
            pool_class = "none"
        else:
            pool_class = "mixed"
        key = (record["anchor_episode_id"], record["M"])
        pools[key] = {
            "U0": sum(ys) / K,
            "U0_upper": sum(upper) / K,
            "coverage": int(any(ys)),
            "unique_exact": len(counts),
            "effective_patches": math.exp(entropy),
            "pool_class": pool_class,
        }
        per_target[key] = plain_mean(target_hits)
    anchors = sorted({anchor for anchor, _ in pools})
    rows = []
    primary = {J: [anchor for anchor in anchors if episodes[anchor]["J"] == J] for J in (1, 2)}
    blocks = [("primary J=1", primary[1], 1), ("primary J=2", primary[2], 2), ("primary pooled (commit mean over its J=1 and J=2 episodes)", primary[1] + primary[2], 0)]
    for name, selected, J in blocks:
        table = collections.defaultdict(lambda: collections.defaultdict(list))
        for anchor in selected:
            for M in (0, 1):
                table[episodes[anchor]["commit10"]][M].append(pools[(anchor, M)]["U0"])
        table = {commit: {M: sum(arms[M]) / len(arms[M]) for M in (0, 1)} for commit, arms in table.items()}
        row = {"stratum": "primary", "block": name, "J": J, "n_episodes": len(selected), "n_commits": len(table)}
        for M in (0, 1):
            estimate, lo, hi = zero_demo_bootstrap(table, lambda drawn, M=M: plain_mean(x[M] for x in drawn), nboot)
            arm = [pools[(anchor, M)] for anchor in selected]
            row[f"U0_M{M}"] = estimate
            row[f"U0_M{M}_lo"] = lo
            row[f"U0_M{M}_hi"] = hi
            row[f"U0_M{M}_episode_mean"] = plain_mean(pool["U0"] for pool in arm)
            row[f"U0_M{M}_upper_uncertain"] = plain_mean(pool["U0_upper"] for pool in arm)
            row[f"coverage_M{M}"] = plain_mean(pool["coverage"] for pool in arm)
            row[f"unique_M{M}"] = plain_mean(pool["unique_exact"] for pool in arm)
            row[f"eff_patches_M{M}"] = plain_mean(pool["effective_patches"] for pool in arm)
            row[f"all_none_mixed_M{M}"] = "/".join(str(sum(1 for pool in arm if pool["pool_class"] == c)) for c in ("all", "none", "mixed"))
        estimate, lo, hi = zero_demo_bootstrap(table, lambda drawn: plain_mean(x[1] - x[0] for x in drawn), nboot)
        differences = [table[commit][1] - table[commit][0] for commit in sorted(table)]
        row["delta_M1-M0"] = estimate
        row["delta_M1-M0_lo"] = lo
        row["delta_M1-M0_hi"] = hi
        row["commits_better_M1-M0"] = sum(d > 1e-12 for d in differences)
        row["commits_worse_M1-M0"] = sum(d < -1e-12 for d in differences)
        row["commits_tied_M1-M0"] = sum(abs(d) <= 1e-12 for d in differences)
        row["delta_M1-M0_episode_mean"] = plain_mean(pools[(anchor, 1)]["U0"] - pools[(anchor, 0)]["U0"] for anchor in selected)
        if J == 2:
            for M in (0, 1):
                row[f"per_target_M{M}"] = plain_mean(per_target[(anchor, M)] for anchor in selected)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# Writing and --compare
# ---------------------------------------------------------------------------------------------


def write_csv(path, rows):
    """Rows as CSV in the paper's format (csv module; columns in first-appearance order, missing cells empty)."""
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def max_abs_diff(a, b):
    """Largest |a - b| over two aligned float arrays; NaN in both counts as equal, NaN in one as infinite."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    diff[np.isnan(diff)] = np.inf
    return float(diff.max()) if len(diff) else 0.0


def compare(out, reference):
    """Print the largest |difference| per numeric column against `reference`, then, for every primary cell, regime
    and rule, whether the frozen U lies inside the 95 % CI of the pools analysed here."""
    for name, keys in KEYS.items():
        ours_path = Path(out) / f"{name}.csv"
        path = Path(reference) / f"{name}.csv"
        if not ours_path.exists() or not path.exists():
            continue
        ours = pd.read_csv(ours_path, float_precision="round_trip", dtype={"commit10": str})
        theirs = pd.read_csv(path, float_precision="round_trip", dtype={"commit10": str})
        merged = ours.merge(theirs, on=keys, suffixes=("", "_ref"))
        columns = [c for c in theirs.columns if c not in keys and theirs[c].dtype.kind == "f" and c in ours]
        worst = max(((max_abs_diff(merged[c], merged[c + "_ref"]), c) for c in columns), default=(0.0, "-"))
        where = f" ({worst[1]})" if worst[0] > 0 else ""
        print(f"{name}: {len(merged)} rows in common; largest |diff| {worst[0]:.3e}{where}")
    ours_path = Path(out) / "summary_by_cell.csv"
    path = Path(reference) / "summary_by_cell.csv"
    if not ours_path.exists() or not path.exists():
        return
    frozen = pd.read_csv(path, float_precision="round_trip").set_index(["cell", "regime", "rule"])
    for _, cell in pd.read_csv(ours_path, float_precision="round_trip").iterrows():
        key = (cell.cell, cell.regime, cell.rule)
        if key not in frozen.index:
            continue
        value = frozen.loc[key, "U_commit"]
        verdict = "inside" if cell.ci_lo <= value <= cell.ci_hi else "outside"
        print(
            f"{cell.cell} {cell.regime} {cell.rule}: frozen U {value:.3f}, rerun CI [{cell.ci_lo:.3f}, {cell.ci_hi:.3f}]"
            f" -> frozen mean {verdict} the rerun's 95 % CI ({cell.n_commits} commits)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pools", required=True, help="directory with pools_*.jsonl (run.py)")
    parser.add_argument("--verdicts", default=None, help="judge verdicts (default: <pools>/../judging/verdicts.jsonl)")
    parser.add_argument("--length", default=None, help="length-head predictions and prior (default: <pools>/../length)")
    parser.add_argument("--zero-demo", default=None, help="the ablation's pools/ and judging/ (default: <pools>/../zero_demo)")
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--nboot", type=int, default=NBOOT)
    parser.add_argument("--compare", default=None, help="directory of reference tables, e.g. data/tables/fig7")
    args = parser.parse_args()

    pools_dir = Path(args.pools)
    if not pools_dir.is_dir() or not any(pools_dir.glob("pools_*.jsonl")):
        print(f"no Figure 7 pools in {pools_dir}; {FETCH_HINT}", file=sys.stderr)
        sys.exit(2)
    root = pools_dir.parent
    verdicts_path = Path(args.verdicts) if args.verdicts else root / "judging" / "verdicts.jsonl"
    length_dir = Path(args.length) if args.length else root / "length"
    zero_demo = Path(args.zero_demo) if args.zero_demo else root / "zero_demo"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    pools = read_main_pools(pools_dir)
    verdicts = read_verdicts(verdicts_path)
    if not verdicts:
        print(f"no verdicts in {verdicts_path}: every draw counts as unjudged (run judge.py)")
    tables = main_run(pools, verdicts, episodes, args.nboot)
    tables["pool_taxonomy_estimates"] = taxonomy_table(pools, verdicts, episodes, args.nboot)
    for name, rows in tables.items():
        write_csv(out / f"{name}.csv", rows)
    case = checkstyle_case(pools, verdicts, episodes) if (CHECKSTYLE_EPISODE, "oracle") in pools else None
    if case:
        (out / "checkstyle_case.json").write_text(json.dumps(case, indent=1))
    if all((length_dir / f"{head}_s5final_G4.npz").exists() for head in LENGTH_HEADS):
        for name, rows in length_tables(length_dir, episodes, args.nboot).items():
            write_csv(out / f"{name}.csv", rows)
    else:
        print(f"no length-head predictions in {length_dir}: length tables not written")
    if (zero_demo / "pools").is_dir():
        rows = zero_demo_table(zero_demo, episodes, args.nboot)
        if rows:
            write_csv(out / "zero_demo_main_table.csv", rows)
    else:
        print(f"no zero-demonstration pools in {zero_demo}: zero_demo_main_table.csv not written")

    for row in tables["summary_by_cell"]:
        print(
            f"{row['cell']:4s} {row['regime']:7s} {row['rule']:8s} commits={row['n_commits']:2d} episodes={row['n_episodes']:3d}"
            f" U={row['U_commit']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] classes all/none/mixed"
            f" {row['all_correct']}/{row['none_correct']}/{row['mixed']} uncertain={row['uncertain_draws']} unjudged={row['unjudged_draws']}"
        )
    print(f"wrote the Figure 7 tables to {out}")
    if args.compare:
        compare(out, args.compare)


if __name__ == "__main__":
    main()
