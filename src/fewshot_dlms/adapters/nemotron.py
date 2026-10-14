# ported from editdiff/experiments/multimodel/mm_adapters.py::Nemotron (mode "block") @ d276fa6
"""Nemotron-Labs-Diffusion-8B with a causal prefix and a bidirectional tail (Appendix B.1).

Nemotron was trained as a block diffusion model: noisy tokens attend bidirectionally within their block
and causally to the clean prefix before it. Both target-count studies of the paper (Figure 5 and the
cross-benchmark sweep) use it through one "prefix/tail" interface, which follows that pattern with a
single tail block. Let b be the first target position of the canvas. Then

    a query at position i < b  attends to the keys k <= i   (causal prefix)
    a query at position i >= b attends to every key         (bidirectional tail: targets, the records
                                                              after them, and the prefix)

b is fixed for a whole reveal, under P1 and P0 alike: the reveal loop announces the order through
`begin_reveal`, and b = order[0]. The tail runs from b to the end of the canvas; it is not cut into
32-token blocks. There is no KV cache: every forward pass recomputes the prefix. The model's diffusion
head reads the hidden state at the masked position itself (no shift). The mask token is id 100.

The checkpoint is nvidia/Nemotron-Labs-Diffusion-8B at a pinned revision, loaded in bf16 with its own
modelling code (trust_remote_code). The attention pattern above is passed to the model through a
registered attention function (`padmask_sdpa`), the authors' shim; see `_register_mask_attention`.
Several Nemotron instances can live in one process: each keeps its own mask.
"""
import weakref

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from fewshot_dlms.adapters.base import Adapter

MASK_ID = 100  # <SPECIAL_100>, the config's mask_token_id
CONTROL_MAX = 999  # ids 0..999 are the control block: every registered special token and the mask
EOS_ID = 11  # <|im_end|>

# Interfaces of this model; the causal-suffix interface ("nemotron_ar") is added with Figure 4.
MODES = {"prefix_tail": "nemotron_pt"}


# Attention layer -> the explicit-mask holder of the Nemotron instance that owns the layer. The attention
# function is registered once per process under one name, so it looks up the calling layer here: every
# instance (the prefix/tail interface, and later the causal-suffix one) sees only its own mask.
_MASK_HOLDERS = weakref.WeakKeyDictionary()


def padmask_sdpa(module, query, key, value, attention_mask, **kwargs):
    """SDPA with the explicit boolean mask of the calling layer's instance, when one is set.

    The holder's "mask" is None or a boolean tensor [B, 1, T, T] (True = may attend). While it is set and
    matches the key length, the layer uses it in place of the model's own mask, with is_causal=False;
    otherwise the layer runs plain SDPA as the model asks.
    """
    holder = _MASK_HOLDERS.get(module)
    mask = holder["mask"] if holder is not None else None
    if mask is not None and mask.shape[-1] == key.shape[-2]:
        kwargs["is_causal"] = False
        return sdpa_attention_forward(module, query, key, value, mask, **kwargs)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def _register_mask_attention(attention_layers, explicit_mask):
    """Route the attention of these layers through `padmask_sdpa`, reading the mask from `explicit_mask`.

    This is how the prefix/tail mask (and the key-validity mask for right padding) reaches the model, which
    is called without an attention mask. The caller also sets each layer's `_attn_implementation`.
    """
    ALL_ATTENTION_FUNCTIONS["padmask_sdpa"] = padmask_sdpa
    for layer in attention_layers:
        _MASK_HOLDERS[layer] = explicit_mask


class Nemotron(Adapter):
    """Nemotron-Labs-Diffusion-8B; `mode="prefix_tail"` is the interface of the target-count studies."""

    model_id = "nvidia/Nemotron-Labs-Diffusion-8B"
    revision = "16c67f0560b912e93e0cabb6e0c4f5c3086d95fc"
    mask_id = MASK_ID
    kind = "dlm"
    close_slot = False

    def __init__(self, device, mode="prefix_tail"):
        if mode not in MODES:
            raise ValueError(f"unknown Nemotron mode {mode!r}; expected one of {sorted(MODES)}")
        self.name = MODES[mode]
        self.device = device
        # trust_remote_code only silences an interactive prompt: the tokenizer is the same with or without it.
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True, dtype=torch.bfloat16
        )
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        config = self.model.config
        assert config.dlm_paradigm == "bidirectional" and config.mask_token_id == MASK_ID
        self.layers = self.model.encoder.layers
        self.explicit_mask = {"mask": None}
        _register_mask_attention([layer.self_attn for layer in self.layers], self.explicit_mask)
        config._attn_implementation = "padmask_sdpa"
        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "padmask_sdpa"
        self.block_start = None  # b, the first target position; set by begin_reveal
        # Every layer in diffusion mode (is_causal=False); the explicit mask makes the prefix causal.
        self.set_mode("dlm")
        self.eos_id = EOS_ID
        self.V = config.vocab_size
        self.control_ids = set(range(CONTROL_MAX + 1))
        self.code_ids = {c: self.encode(" " + c)[0] for c in "ABCDEFGHIJ"}
        assert all(len(self.encode(" " + c)) == 1 for c in "ABCDEFGHIJ")

    def set_mode(self, mode):
        """Switch every attention layer to diffusion ("dlm", non-causal) or autoregressive ("ar", causal) mode."""
        for layer in self.layers:
            layer.self_attn.diffusion_lm = mode == "dlm"
        self.mode = mode

    def marker_ids(self):
        """None beyond the control block: ids 0..999 hold every special token and chat marker."""
        return set()

    def chat_header(self, instruction):
        return self.chat_user_open() + instruction + self.chat_user_close_assistant_open()

    def chat_user_open(self):
        return "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"

    def chat_user_close_assistant_open(self):
        return "<|im_end|>\n<|im_start|>assistant\n<think></think>"

    def chat_end(self):
        return "<|im_end|>\n"

    def begin_reveal(self, order):
        """Fix the start b of the bidirectional tail to the first target position of this reveal."""
        self.block_start = order[0]

    def _hidden(self, batch_ids):
        """Final hidden states [B, T, d] of the canvases under the prefix/tail mask."""
        assert self.block_start is not None, "prefix/tail mode needs block_start (set by begin_reveal)"
        batch, length = batch_ids.shape
        ids = batch_ids.to(self.device)
        # Key validity (the authors' padding patch): every key is real, since the canvases have equal length.
        valid = torch.ones(batch, length, dtype=torch.bool, device=self.device)
        index = torch.arange(length, device=self.device)
        query = index[:, None]
        key = index[None, :]
        b = self.block_start
        allow = (((query < b) & (key <= query)) | (query >= b))[None, None] & valid[:, None, None, :]
        self.explicit_mask["mask"] = allow.expand(batch, 1, length, length)
        try:
            output = self.model(
                input_ids=ids,
                labels=None,
                use_cache=False,
                past_key_values=None,
                use_causal_mask=False,
                output_last_hidden_states_only=True,
            )
        finally:
            self.explicit_mask["mask"] = None
        return output.last_hidden_state

    def logits(self, batch_ids, positions):
        """The diffusion head at `positions` of each canvas under the prefix/tail mask, as fp32."""
        assert self.mode == "dlm"
        with torch.inference_mode():
            hidden = self._hidden(batch_ids)
            rows = torch.arange(len(positions), device=self.device)[:, None]
            columns = torch.tensor(positions, device=self.device)
            selected = hidden[rows, columns]
            batch, n_positions, width = selected.shape
            logits = self.model.diffusion_head(selected.reshape(batch * n_positions, width)).float()
            return logits.reshape(batch, n_positions, -1)
