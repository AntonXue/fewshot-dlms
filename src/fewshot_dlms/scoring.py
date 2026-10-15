# ported from editdiff/experiments/algo_investigation/fs_analyze_lib.py and experiments/sec42_mj/mj_analyze.py @ d276fa6
"""Example guidance on a pool of K complete infills.

ell1[k] = log P1(infill_k): log-likelihood with demonstration outputs visible.
ell0[k] = log P0(infill_k): the same with demonstration-output tokens masked.
R = ell1 - ell0 is the example reward. Pool weights are softmax(beta * R) over the K samples,
so beta = 0 is unguided sampling and beta = 0.5 gives Q_0.5 ∝ P1^{3/2} P0^{-1/2}.
The sharpening control keeps the same total exponent on P1 without P0: weights softmax((1/tau - 1) ell1), tau = 2/3.
"""
import numpy as np


def example_reward(ell1, ell0):
    """Example reward R = ell1 - ell0 for each of the K samples."""
    return np.asarray(ell1, float) - np.asarray(ell0, float)


def pool_weights(ell1, ell0, beta, tau=1.0):
    """Self-normalised importance weights over the K samples of one pool.

    The samples were drawn from P1, so reweighting them by exp(beta * R) targets
    P1 * exp(beta * R) = P1^(1 + beta) P0^(-beta). The optional temperature tau adds the
    pure sharpening term (1/tau - 1) * ell1, giving

        w = softmax((1/tau - 1) * ell1 + beta * (ell1 - ell0)).

    With the defaults (beta = 0, tau = 1) every weight is 1/K. The maximum is subtracted
    before exponentiating, which leaves the softmax unchanged and keeps it finite.
    """
    ell1 = np.asarray(ell1, float)
    ell0 = np.asarray(ell0, float)
    lw = (1.0 / tau - 1.0) * ell1 + beta * (ell1 - ell0)
    lw = lw - lw.max()
    w = np.exp(lw)
    return w / w.sum()


def family_metrics(correct, w):
    """Expected accuracy of one family (pool) under the weights w.

    `correct` is the K x J 0/1 matrix of per-target correctness and `w` the K pool weights.
      A_target = sum_k w[k] * mean_j correct[k, j]   (average per-target accuracy)
      A_all    = sum_k w[k] * all_j  correct[k, j]   (every target of the infill correct)
      ess      = 1 / sum_k w[k]^2                    (effective sample size of the pool)
      max_w    = max_k w[k]
    """
    c = np.asarray(correct, float)
    w = np.asarray(w, float)
    return {
        "A_target": float(w @ c.mean(1)),
        "A_all": float(w @ c.all(1)),
        "ess": float(1.0 / (w ** 2).sum()),
        "max_w": float(w.max()),
    }
