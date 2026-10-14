"""Shared pieces of the GPU tests: replay tolerances and releasing a loaded model.

Why replay is not always exact. A bf16 forward pass gives slightly different logits for the same canvas
depending on the batch it runs in (the GPU picks different matrix kernels), and the differences add up
over the T target tokens of an infill. The frozen run scored many candidates per forward pass:
sampling batched the distinct prefixes of the K draws, and P0 scored the distinct draws together with
the reference answer. `logprob_fixed_order` scores one candidate per forward pass, so it agrees with the
stored values only up to this batch noise.

Scoring with the frozen run's batches removes it. On the authors' platform (REFERENCE_GPU, REFERENCE_TORCH)
LLaDA-1.5 and Nemotron then reproduce every stored log-likelihood bit for bit. dQwen3.5 does not: its
Gated-DeltaNet layers run Triton kernels (flash-linear-attention) whose configuration is autotuned
afresh in every process, so its logits vary from process to process even with identical batches.
"""
import gc

import torch

# Replay tolerance per model: the largest |replayed - stored| accepted for one candidate's ell1 or ell0
# (in nats) when candidates are scored one at a time. Measured on an H100 NVL over the first 8 report
# families of the frozen M = 4, J = 2 pools (16 candidates x {ell1, ell0} each): the largest deviation was
# 0.16 (LLaDA-1.5), 0.43 (Nemotron) and 0.62 (dQwen3.5, worst of two processes); the mean was 0.01, 0.07
# and 0.05. The deviation does not grow with |ell| (LLaDA's largest is at |ell| = 8), so the tolerance is
# absolute: about 1.6 to 1.9 times the largest measured value.
TOLERANCES = {
    "llada": 0.3,
    "nemotron_pt": 0.75,
    "dqwen": 1.0,
}

REFERENCE_GPU = "NVIDIA H100 NVL"  # the GPU of the authors' runs
REFERENCE_TORCH = "2.7.1+cu128"  # the torch build of the authors' runs
BIT_EXACT = ("llada", "nemotron_pt")  # models whose replay with the frozen run's batches is exact there


def tolerance(name):
    """The largest |replayed - stored| accepted for model `name` when scoring one candidate at a time."""
    return TOLERANCES[name]


def on_reference_platform():
    """True on the GPU type and torch build the frozen pools were computed with."""
    return torch.cuda.get_device_name() == REFERENCE_GPU and torch.__version__ == REFERENCE_TORCH


def batched_tolerance(name):
    """Tolerance when scoring with the frozen run's batches: 0 where replay is bit-exact, else `tolerance`."""
    if name in BIT_EXACT and on_reference_platform():
        return 0.0
    return tolerance(name)


def release(adapter):
    """Free the GPU memory of a loaded adapter, so that the next model fits.

    Every attribute is dropped, not only `model`: Nemotron also keeps the model's layers in `layers`.
    """
    vars(adapter).clear()
    gc.collect()
    torch.cuda.empty_cache()
