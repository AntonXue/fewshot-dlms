"""Shared pieces of the GPU tests: replay tolerances and releasing a loaded model.

Why replay is not always exact. A bf16 forward pass gives slightly different logits for the same canvas
depending on the batch it runs in (the GPU picks different matrix kernels), and the differences add up
over the T target tokens of an infill. The frozen run scored many candidates per forward pass:
sampling batched the distinct prefixes of the K draws, and P0 scored the distinct draws together with
the reference answer. `logprob_fixed_order` scores one candidate per forward pass, so it agrees with the
stored values only up to this batch noise.

Scoring with the frozen run's batches removes it. On the authors' platform (REFERENCE_GPU, REFERENCE_TORCH)
LLaDA-1.5 and Nemotron then reproduce every stored log-likelihood bit for bit. dQwen3.5 does not: its
Gated-DeltaNet layers run Triton kernels (flash-linear-attention 0.5.1, as in the paper's runs) whose
configuration is autotuned afresh in every process, so its logits vary from process to process even with
identical batches.
"""
import gc

import torch

# Replay tolerance per model: the largest |replayed - stored| accepted for one candidate's ell1 or ell0
# (in nats) when candidates are scored one at a time. Measured on an H100 NVL over the first 8 report
# families of the frozen M = 4, J = 2 pools (16 candidates x {ell1, ell0} each): the largest deviation was
# 0.16 (LLaDA-1.5), 0.43 (Nemotron) and 0.62 (dQwen3.5, worst of two processes); the mean was 0.01, 0.07
# and 0.05. The deviation does not grow with |ell| (LLaDA's largest is at |ell| = 8), so the tolerance is
# absolute: about 1.6 to 1.9 times the largest measured value.
#
# The causal interfaces of Figure 4 were measured the same way on the frozen single-target M = 4 pools, every
# distinct continuation of the first 8 SST-2 and the first 4 List Functions-Easy report families ({ell1, ell0}
# each; 214 + 150 values for nemotron_ar, 180 + 98 for qwen). Nemotron's causal suffix was exact on SST-2 (one
# answer token, so every forward pass held one sequence anyway) and deviated by at most 0.33 (mean 0.08) on the
# multi-token List Functions answers; Qwen3-8B deviated by at most 0.18 (mean 0.02) on SST-2 and 0.54 (mean
# 0.08) on List Functions. The tolerances are about 1.8 times the largest value.
TOLERANCES = {
    "llada": 0.3,
    "nemotron_pt": 0.75,
    "dqwen": 1.0,
    "nemotron_ar": 0.6,
    "qwen": 1.0,
}

REFERENCE_GPU = "NVIDIA H100 NVL"  # the GPU of the authors' runs
REFERENCE_TORCH = "2.7.1+cu128"  # the torch build of the authors' runs

# Tolerance on the reference platform when scoring with the frozen run's batches. LLaDA-1.5 and Nemotron are
# bit-exact there. dQwen3.5 depends on the process: in 10 separate processes (two H100 NVLs) replaying the
# two fixture records (every candidate's ell1 and ell0, the reference answer's ell0), 7 reproduced
# every value exactly and 3 differed by at most 0.372; the tolerance leaves a margin of about a third.
# The causal interfaces reproduce every one of the values above (and resample the stored draws token for token)
# when the distinct continuations are scored together with the frozen batch sizes.
BATCHED_TOLERANCES = {
    "llada": 0.0,
    "nemotron_pt": 0.0,
    "dqwen": 0.5,
    "nemotron_ar": 0.0,
    "qwen": 0.0,
}


def tolerance(name):
    """The largest |replayed - stored| accepted for model `name` when scoring one candidate at a time."""
    return TOLERANCES[name]


def on_reference_platform():
    """True on the GPU type and torch build the frozen pools were computed with."""
    return torch.cuda.get_device_name() == REFERENCE_GPU and torch.__version__ == REFERENCE_TORCH


def batched_tolerance(name):
    """Tolerance when scoring with the frozen run's batches: BATCHED_TOLERANCES on the reference platform.

    Elsewhere the GPU's own kernels differ from the authors', so the one-at-a-time `tolerance` applies.
    """
    if on_reference_platform():
        return BATCHED_TOLERANCES[name]
    return tolerance(name)


def release(adapter):
    """Free the GPU memory of a loaded adapter, so that the next model fits.

    Every attribute is dropped, not only `model`: Nemotron also keeps the model's layers in `layers`.
    """
    vars(adapter).clear()
    gc.collect()
    torch.cuda.empty_cache()
