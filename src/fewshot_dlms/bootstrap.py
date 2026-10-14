# ported from editdiff/experiments/paper_integration/agent_a/fig5_compute.py (idx_for, ci, boot_sd) and experiments/sec42_mj/mj_analyze.py @ d276fa6
"""Family bootstrap with the paper's exact seed derivation. Do not change the RNG or draw order: CI parity depends on it.

Figures 4 and 5 use this module: they resample whole families (episodes) with replacement. One index matrix is drawn
per cell and shared by every quantity of that cell, so differences between quantities (for example tilted minus
ordinary accuracy) are paired. Figures 6 and 7 keep their own resampling, as their analyze.py files document: Figure 6
an integer-seeded numpy default_rng over families, Figure 7 a commit-level random.Random(0).
Per-pipeline seed strings and cell keys: Fig 4 ("mtfsp-20261003-bootstrap-v1", f"{model}|{src}"),
Fig 5 ("mtfsp-20261011-mj-bootstrap-v1", f"{model}|{M}|{J}").
"""
import hashlib

import numpy as np


def _seed_from_string(s):
    """Base seed: the first 16 hex digits of SHA-256(s), reduced mod 2**32."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) % (2 ** 32)


def cell_indices(seed_string, cell_key, n, nboot=10000):
    """Resample indices (nboot x n) for one cell of n families.

    Seed = (base(seed_string) + first 8 hex digits of SHA-256(cell_key)) mod 2**32, fed to numpy's default_rng;
    the matrix is a single rng.integers(0, n, size=(nboot, n)) draw.
    """
    base = _seed_from_string(seed_string)
    rng = np.random.default_rng((base + int(hashlib.sha256(cell_key.encode()).hexdigest()[:8], 16)) % (2 ** 32))
    return rng.integers(0, n, size=(nboot, n))


def ci(values, idx):
    """(mean, 2.5th, 97.5th percentile of the resampled means) of per-family values under the index matrix idx."""
    v = np.asarray(values, float)
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_sd(values, seed, nboot=10000):
    """Same interval with a private index matrix, used for the gold-reward bands (the `sd_figs.boot` convention).

    The generator is seeded with the first 8 hex digits of SHA-256(seed) (already below 2**32); there is no base seed.
    """
    v = np.asarray(values, float)
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    m = v[rng.integers(0, len(v), size=(nboot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
