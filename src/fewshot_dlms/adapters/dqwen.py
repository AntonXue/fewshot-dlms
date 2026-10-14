# ported from editdiff/experiments/multimodel/mm_adapters.py::DQwen9B and experiments/dx/dqwen_adapter.py::{DQwen, MASK_ID, BOS_ID} @ d276fa6
"""dQwen3.5-9B-Base, a hybrid masked diffusion language model that infills the whole canvas (Appendix B.1).

dQwen3.5 is adapted from Qwen3.5: its full-attention layers (every fourth layer) attend bidirectionally
over the canvas, its Gated-DeltaNet layers stay causal. It is a base model, so the canvas is plain text
with no chat template: the instruction, a blank line, then the records.

Logit shift, as in the model's own forward pass: the model prepends a BOS token and returns
logits[:, i] = lm_head(hidden state at index i of the BOS-prefixed sequence), the state that sits on
input token i - 1. Those are its predictions for the token at input position i, so the adapter reads
logits[b, position] at the masked position itself, with no further shift. The mask token is
<|fim_middle|>.

The checkpoint is UT-IFML/dQwen3.5-9B-Base at a pinned revision, loaded in bf16 with its own modelling
code (trust_remote_code).

Kernels. The Gated-DeltaNet layers come from transformers' Qwen3.5 code, which picks their kernels from
what is installed: the Triton kernels of flash-linear-attention (fla) when it is importable with CUDA,
torch code otherwise, and the causal-conv1d convolution when that package is importable. The choices give
different numerics. The paper's dQwen pools used fla 0.5.1 and no causal-conv1d (PAPER_KERNELS; install
with `pip install -e .[gpu]`); the adapter warns when the environment differs. Even then fla autotunes its
kernels in every process, so dQwen's log-likelihoods vary slightly from process to process.
"""
import importlib.metadata
import warnings

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.utils.import_utils import is_causal_conv1d_available
from transformers.utils.import_utils import is_flash_linear_attention_available

from fewshot_dlms.adapters.base import Adapter

MASK_ID = 248061  # <|fim_middle|>
BOS_ID = 248073  # the BOS the model prepends to every input

# The linear-attention kernels of the paper's dQwen runs: the version of each package in use, None if unused.
PAPER_KERNELS = {"fla": "0.5.1", "causal_conv1d": None}


def _version(distribution):
    """The installed version of a distribution, or "unknown" if its metadata cannot be found."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def linear_attention_kernels():
    """The kernels transformers will use for dQwen's Gated-DeltaNet layers here, as in PAPER_KERNELS.

    "fla" is the version of fla-core (the package that provides `fla`, installed by flash-linear-attention)
    when transformers uses its Triton kernels, else None; "causal_conv1d" likewise.
    """
    fla = _version("fla-core") if is_flash_linear_attention_available() else None
    conv = _version("causal-conv1d") if is_causal_conv1d_available() else None
    return {"fla": fla, "causal_conv1d": conv}


class DQwen(Adapter):
    """dQwen3.5-9B-Base on a plain-text canvas: bidirectional attention layers, causal linear layers."""

    name = "dqwen"
    model_id = "UT-IFML/dQwen3.5-9B-Base"
    revision = "f193fa20e341504f8d9e2b4b8342048c466e6590"
    mask_id = MASK_ID
    kind = "dlm"
    close_slot = False

    def __init__(self, device):
        self.device = device
        self.kernels = linear_attention_kernels()
        if self.kernels != PAPER_KERNELS:
            warnings.warn(
                f"dQwen's linear-attention layers will run with {self.kernels}, but the paper's dQwen pools used "
                f"{PAPER_KERNELS} (flash-linear-attention 0.5.1, no causal-conv1d; `pip install -e .[gpu]` and "
                "uninstall causal-conv1d). Log-likelihoods and samples will differ from the frozen pools.",
                stacklevel=2,
            )
        # trust_remote_code only silences an interactive prompt: the tokenizer is the same with or without it.
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True, dtype=torch.bfloat16
        )
        self.model = model.to(device).eval()
        # The model's own guard: under transformers 5 the rotary buffer can load as garbage without an error.
        self.model._dg_assert_finite_rope()
        self.bos_id = BOS_ID
        self.eos_id = int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else None
        self.pad_id = int(self.tok.pad_token_id) if self.tok.pad_token_id is not None else None
        self.V = int(self.model.lm_head.out_features)
        special_ids = {i for i in self.tok.all_special_ids if i is not None}
        optional_ids = {i for i in (self.eos_id, self.pad_id) if i is not None}
        self.control_ids = {self.mask_id, self.bos_id} | optional_ids | special_ids
        assert all(len(self.encode(" " + c)) == 1 for c in "ABCD"), "code tokens must be single tokens"
        self.code_ids = {c: self.encode(" " + c)[0] for c in "ABCDEFGHIJ" if len(self.encode(" " + c)) == 1}
        assert all(c in self.code_ids for c in "ABCDEFGHIJ")

    def chat_header(self, instruction):
        return instruction + "\n\n"

    def chat_user_open(self):
        return ""

    def chat_user_close_assistant_open(self):
        return "\n"

    def chat_end(self):
        return ""

    def logits(self, batch_ids, positions):
        """One forward pass of the hybrid model over the B canvases; the bf16 logits at `positions`, as fp32."""
        with torch.inference_mode():
            ids = batch_ids.to(self.device)
            output = self.model(input_ids=ids)
            rows = torch.arange(len(positions), device=self.device)[:, None]
            columns = torch.tensor(positions, device=self.device)
            return output.logits[rows, columns].float()
