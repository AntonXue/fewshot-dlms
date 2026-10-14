"""CPU checks of the model adapters that need no checkpoint: Nemotron's per-instance attention mask and
the error `adapters.load` raises when a pinned snapshot is missing."""
import pytest
import torch
from torch import nn
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from fewshot_dlms import adapters
from fewshot_dlms.adapters import llada
from fewshot_dlms.adapters import nemotron


class Attention(nn.Module):
    """A stand-in for one attention layer: SDPA reads only `is_causal` from it."""

    is_causal = True


def qkv(seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 2, 5, 4, generator=generator) for _ in range(3)]


def test_two_nemotron_instances_keep_their_own_masks():
    # Two instances register their layers; each layer must read its own instance's mask.
    first_layer = Attention()
    second_layer = Attention()
    first_holder = {"mask": None}
    second_holder = {"mask": None}
    nemotron._register_mask_attention([first_layer], first_holder)
    nemotron._register_mask_attention([second_layer], second_holder)
    prefix_tail = torch.tensor(
        [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool
    )[None, None]
    first_holder["mask"] = prefix_tail
    query, key, value = qkv(0)

    first, _ = nemotron.padmask_sdpa(first_layer, query, key, value, None, is_causal=False)
    second, _ = nemotron.padmask_sdpa(second_layer, query, key, value, None, is_causal=False)

    with_mask, _ = sdpa_attention_forward(first_layer, query, key, value, prefix_tail, is_causal=False)
    without_mask, _ = sdpa_attention_forward(second_layer, query, key, value, None, is_causal=False)
    assert torch.equal(first, with_mask)
    assert torch.equal(second, without_mask)
    assert not torch.equal(first, second)


def test_unregistered_layer_runs_plain_sdpa():
    query, key, value = qkv(1)
    layer = Attention()
    out, _ = nemotron.padmask_sdpa(layer, query, key, value, None, is_causal=False)
    expected, _ = sdpa_attention_forward(layer, query, key, value, None, is_causal=False)
    assert torch.equal(out, expected)


def test_load_names_the_missing_snapshot(monkeypatch):
    def missing_snapshot(self, device):
        raise OSError("simulated: snapshot not in the cache")

    monkeypatch.setattr(llada.LLaDA, "__init__", missing_snapshot)
    monkeypatch.setenv("HF_HOME", "/nonexistent/hf-cache")
    with pytest.raises(OSError) as raised:
        adapters.load("llada", "cpu")
    message = str(raised.value)
    assert llada.LLaDA.model_id in message
    assert llada.LLaDA.revision in message
    assert "HF_HOME=/nonexistent/hf-cache" in message
    assert "README.md" in message
    assert "simulated" in message


def test_load_rejects_unknown_names():
    with pytest.raises(KeyError):
        adapters.load("gpt", "cpu")
