"""Unit tests for the family bootstrap, including golden values for the paper's seed derivation.

The golden numbers below were produced by the original analysis code (the seed arithmetic and the draw order
are what the published confidence intervals depend on), so these tests fail if either ever changes.
"""
import numpy as np

from fewshot_dlms.bootstrap import boot_sd
from fewshot_dlms.bootstrap import cell_indices
from fewshot_dlms.bootstrap import ci

FIG4_SEED = "mtfsp-20261003-bootstrap-v1"
FIG5_SEED = "mtfsp-20261011-mj-bootstrap-v1"


def test_indices_are_deterministic_and_shaped():
    a = cell_indices(FIG5_SEED, "llada|4|2", 64, 10000)
    b = cell_indices(FIG5_SEED, "llada|4|2", 64, 10000)
    assert a.shape == (10000, 64)
    assert (a == b).all()
    assert a.max() < 64


def test_default_nboot_is_ten_thousand():
    assert cell_indices("s", "k", 5).shape == (10000, 5)


def test_cell_key_and_seed_string_change_the_draws():
    a = cell_indices("s", "k1", 16, 200)
    assert not (a == cell_indices("s", "k2", 16, 200)).all()
    assert not (a == cell_indices("t", "k1", 16, 200)).all()


def test_ci_brackets_mean():
    v = np.arange(64, dtype=float)
    idx = cell_indices("s", "k", 64, 2000)
    m, lo, hi = ci(v, idx)
    assert lo <= m <= hi
    assert np.isclose(m, v.mean())


def test_paired_gain_uses_same_indices():
    idx = cell_indices("s", "k", 10, 500)
    a = np.random.default_rng(0).random(10)
    b = a + 0.1
    assert np.isclose(ci(b - a, idx)[0], 0.1)


def test_paired_gain_interval_is_the_difference_of_resampled_means():
    idx = cell_indices("s", "k", 10, 500)
    a = np.random.default_rng(1).random(10)
    b = np.random.default_rng(2).random(10)
    _, lo, hi = ci(b - a, idx)
    resampled = b[idx].mean(1) - a[idx].mean(1)
    assert np.isclose(lo, np.percentile(resampled, 2.5))
    assert np.isclose(hi, np.percentile(resampled, 97.5))


def test_fig4_seed_derivation_golden():
    idx = cell_indices(FIG4_SEED, "llada|sst2_fam", 64)
    assert idx.shape == (10000, 64)
    assert int(idx.sum()) == 20142854
    assert idx[0, :8].tolist() == [46, 55, 52, 34, 43, 50, 20, 1]
    assert idx[-1, -8:].tolist() == [63, 24, 41, 62, 34, 61, 35, 59]


def test_fig5_seed_derivation_golden():
    idx = cell_indices(FIG5_SEED, "llada|4|2", 64)
    assert int(idx.sum()) == 20158883
    assert idx[0, :8].tolist() == [36, 35, 9, 4, 56, 17, 57, 1]
    assert idx[-1, -8:].tolist() == [47, 37, 63, 52, 10, 8, 18, 55]


def test_ci_golden():
    idx = cell_indices(FIG4_SEED, "llada|sst2_fam", 64)
    m, lo, hi = ci(np.arange(64) / 63.0, idx)
    assert m == 0.5
    assert lo == 0.4283234126984127
    assert hi == 0.5704365079365079


def test_boot_sd_golden():
    m, lo, hi = boot_sd(np.arange(64) / 63.0, "4.1|llada|sst2_fam|4|rpt")
    assert m == 0.5
    assert lo == 0.4255952380952381
    assert hi == 0.5721726190476191


def test_boot_sd_is_deterministic_and_seed_dependent():
    v = np.random.default_rng(3).random(20)
    assert boot_sd(v, "a", nboot=300) == boot_sd(v, "a", nboot=300)
    assert boot_sd(v, "a", nboot=300) != boot_sd(v, "b", nboot=300)
