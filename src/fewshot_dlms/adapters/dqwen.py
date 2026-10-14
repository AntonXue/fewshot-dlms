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
"""
import torch
from transformers import AutoModel
from transformers import AutoTokenizer

from fewshot_dlms.adapters.base import Adapter

MASK_ID = 248061  # <|fim_middle|>
BOS_ID = 248073  # the BOS the model prepends to every input


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
