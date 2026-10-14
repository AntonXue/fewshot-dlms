"""CPU checks of the model adapters that need no checkpoint: Nemotron's per-instance attention mask (also with a
prefix/tail and a causal-suffix instance side by side), Qwen3's free-generation support and stop tokens, and the
error `adapters.load` raises when a pinned snapshot is missing. Constructors run on stand-in models and tokenizers."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from fewshot_dlms import adapters
from fewshot_dlms.adapters import llada
from fewshot_dlms.adapters import nemotron
from fewshot_dlms.adapters import qwen_ar
from fewshot_dlms.adapters.base import sample_fixed_order
from fewshot_dlms.canvas import Canvas


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


class FakeTokenizer:
    """Just enough tokenizer for the adapters' constructors: one token per non-space character."""

    eos_token_id = 1
    pad_token_id = 0
    all_special_ids = [0, 1, 2]
    added_tokens_decoder = {}
    vocabulary = ["<pad>", "<|im_end|>", "<|im_start|>", "a", "b", "\n", "c\n\n", "�"] + [" " + c for c in "ABCDEFGHIJ"]

    def __len__(self):
        return len(self.vocabulary)

    def encode(self, text, add_special_tokens=False):
        return [self.vocabulary.index(text)] if text in self.vocabulary else [ord(c) for c in text.strip()]

    def decode(self, ids):
        return "".join(self.vocabulary[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.index(token)


class FakeNemotronModel(nn.Module):
    """The pieces of Nemotron's remote model that the adapter touches: config, encoder layers, attention configs."""

    def __init__(self, n_layers=2):
        super().__init__()
        self.config = SimpleNamespace(dlm_paradigm="bidirectional", mask_token_id=nemotron.MASK_ID, vocab_size=1200)
        # One attention config shared by all layers, as in the model.
        attention_config = SimpleNamespace(_attn_implementation="sdpa")
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.self_attn = Attention()
            layer.self_attn.config = attention_config
            layer.self_attn.diffusion_lm = True
            layers.append(layer)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(layers)


class FakeQwenModel(nn.Module):
    """The pieces of Qwen3-8B that the adapter's constructor touches."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=20)


def test_causal_and_prefix_tail_instances_keep_their_own_layers_and_masks(monkeypatch):
    monkeypatch.setattr(nemotron.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(nemotron.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeNemotronModel())
    prefix_tail = nemotron.Nemotron("cpu", mode="prefix_tail")
    causal = nemotron.Nemotron("cpu", mode="causal")

    assert (prefix_tail.name, prefix_tail.kind) == ("nemotron_pt", "dlm")
    assert (causal.name, causal.kind) == ("nemotron_ar", "ar")
    assert all(layer.self_attn.diffusion_lm for layer in prefix_tail.layers)
    assert not any(layer.self_attn.diffusion_lm for layer in causal.layers)
    for instance in (prefix_tail, causal):
        assert instance.model.config._attn_implementation == "padmask_sdpa"
        for layer in instance.layers:
            assert layer.self_attn.config._attn_implementation == "padmask_sdpa"
            assert nemotron._MASK_HOLDERS[layer.self_attn] is instance.explicit_mask

    # While a prefix/tail mask is set, the causal instance's layers still run the model's own causal attention.
    mask = torch.ones(5, 5, dtype=torch.bool)[None, None]
    mask[..., 0, 1:] = False
    prefix_tail.explicit_mask["mask"] = mask
    query, key, value = qkv(2)
    causal_layer = causal.layers[0].self_attn
    out, _ = nemotron.padmask_sdpa(causal_layer, query, key, value, None)
    expected, _ = sdpa_attention_forward(causal_layer, query, key, value, None)
    assert torch.equal(out, expected)
    tail_layer = prefix_tail.layers[0].self_attn
    out, _ = nemotron.padmask_sdpa(tail_layer, query, key, value, None, is_causal=False)
    expected, _ = sdpa_attention_forward(tail_layer, query, key, value, mask, is_causal=False)
    assert torch.equal(out, expected)


def test_causal_nemotron_has_no_canvas_to_reveal(monkeypatch):
    monkeypatch.setattr(nemotron.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(nemotron.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeNemotronModel())
    causal = nemotron.Nemotron("cpu", mode="causal")
    with pytest.raises(ValueError):
        causal.begin_reveal([3, 4])


def test_nemotron_rejects_unknown_modes():
    with pytest.raises(ValueError):
        nemotron.Nemotron("cpu", mode="block")


def test_qwen_free_generation_support_and_stop_tokens(monkeypatch):
    monkeypatch.setattr(qwen_ar.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(qwen_ar.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: FakeQwenModel())
    model = qwen_ar.Qwen3("cpu")
    end_of_turn = FakeTokenizer.vocabulary.index("<|im_end|>")
    ordinary = model.support("ordinary")
    free = model.support("free")
    assert not ordinary[end_of_turn]
    assert free[end_of_turn]
    assert torch.equal(free ^ ordinary, torch.nn.functional.one_hot(torch.tensor(end_of_turn), model.V).bool())
    newline_tokens = {FakeTokenizer.vocabulary.index("\n"), FakeTokenizer.vocabulary.index("c\n\n")}
    assert model.stop_ids() == newline_tokens | {end_of_turn}
    assert not ordinary[FakeTokenizer.vocabulary.index("�")]


def test_causal_adapters_carry_the_frozen_decoding_defaults(monkeypatch):
    # Figure 4's pools: Qwen3 sampled and scored over the free support, 32 sequences per pass; Nemotron's causal
    # suffix over the ordinary support, sampling 32 and scoring P0 16 per pass.
    monkeypatch.setattr(qwen_ar.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(qwen_ar.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: FakeQwenModel())
    monkeypatch.setattr(nemotron.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(nemotron.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeNemotronModel())
    qwen = qwen_ar.Qwen3("cpu")
    causal = nemotron.Nemotron("cpu", mode="causal")
    assert (qwen.default_support, qwen.sample_chunk, qwen.p0_chunk) == ("free", 32, 32)
    assert (causal.default_support, causal.sample_chunk, causal.p0_chunk) == ("ordinary", 32, 16)


def test_qwen_has_no_canvas_to_reveal(monkeypatch):
    monkeypatch.setattr(qwen_ar.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(qwen_ar.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: FakeQwenModel())
    qwen = qwen_ar.Qwen3("cpu")
    with pytest.raises(ValueError, match="sample_ar"):
        qwen.begin_reveal([3, 4])
    canvas = Canvas(ids=[0, 1, 2], sites=[[2]], E=[1], gold=[""], meta={})
    with pytest.raises(ValueError, match="sample_ar"):
        sample_fixed_order(qwen, canvas, K=2, seed=0)


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
