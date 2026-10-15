# ported from editdiff/experiments/multimodel/mm_adapters.py::Qwen3 and experiments/algo_investigation/dr_worker.py::{run_qwen, free_steps, IM_END} @ d276fa6
"""Qwen3-8B, the autoregressive comparison of the single-target study (Figure 5; Appendix B.1, Experimental Setup).

Qwen3-8B continues the prompt of canvas.ar_prompt_ids left to right with its own causal attention
(SDPA, no KV cache), one token per step (adapters.base.sample_ar). The logits for the token at position p
come from the hidden state at p - 1, so the next token after a sequence of length T is read at T - 1.

The continuation has free length: it ends at the first token whose text contains a newline, or at
<|im_end|>, and keeps that token (`stop_ids`); after FREE_MAX_NEW tokens without one it is cut and marked
truncated. To make <|im_end|> drawable, the continuation is sampled over the "free" support: the ordinary
support plus <|im_end|>. Grading keeps the first line of the text (grading.answer with model "qwen").
P0 deletes the demonstration outputs from the prompt.

The checkpoint is Qwen/Qwen3-8B at a pinned revision, loaded in bf16. Thinking is switched off by the chat
template's own flag (enable_thinking=False), which opens the assistant turn with an empty
<think></think> block.
"""
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from fewshot_dlms.adapters.base import Adapter

END_OF_TURN = "<|im_end|>"
FREE_MAX_NEW = 200  # token budget of a free-length continuation (the frozen pools' max_new)
SPLIT = "XSPLITX"  # stand-in content that cuts the rendered chat template into its two halves


class Qwen3(Adapter):
    """Qwen/Qwen3-8B as a causal model: next-token logits and Qwen's chat template without thinking."""

    name = "qwen"
    model_id = "Qwen/Qwen3-8B"
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    mask_id = -1
    kind = "ar"
    # How the Figure 5 pools were decoded: over the free support, 32 sequences per forward pass both while
    # sampling and while scoring P0 (read by adapters.base.sample_ar / logprob_ar / decode_ar).
    default_support = "free"
    sample_chunk = 32
    p0_chunk = 32

    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.V = int(self.model.config.vocab_size)
        self.eos_id = int(self.tok.eos_token_id)
        self.pad_id = int(self.tok.pad_token_id)
        specials = {i for i in self.tok.all_special_ids if i is not None}
        self.control_ids = {self.eos_id, self.pad_id} | specials
        self.code_ids = {c: self.encode(" " + c)[0] for c in "ABCDEFGHIJ"}
        assert all(len(self.encode(" " + c)) == 1 for c in "ABCDEFGHIJ")
        self.end_of_turn_id = self.tok.convert_tokens_to_ids(END_OF_TURN)

    def _template(self, content):
        """The chat template with `content` as the user turn, the assistant turn opened, thinking off."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

    def chat_header(self, instruction):
        return self._template(instruction)

    def chat_user_open(self):
        return self._template(SPLIT).split(SPLIT)[0]

    def chat_user_close_assistant_open(self):
        return self._template(SPLIT).split(SPLIT)[1]

    def chat_end(self):
        return END_OF_TURN + "\n"

    def begin_reveal(self, order):
        """Qwen3-8B has no canvas: it continues a prompt left to right (adapters.base.sample_ar / logprob_ar)."""
        raise ValueError("Qwen3-8B is a causal model with no canvas to reveal; use sample_ar / logprob_ar")

    def support(self, kind, codes=None):
        """As Adapter.support, plus kind "free": the ordinary support and <|im_end|>, for free-length decoding."""
        if kind != "free":
            return super().support(kind, codes)
        mask = super().support("ordinary")
        mask[self.end_of_turn_id] = True
        return mask

    def stop_ids(self):
        """The tokens that end a free-length continuation: <|im_end|> and every token whose text has a newline."""
        if not hasattr(self, "_stop_ids"):
            stops = {self.end_of_turn_id}
            for i in range(len(self.tok)):
                if "\n" in self.tok.decode([i]):
                    stops.add(i)
            self._stop_ids = stops
        return self._stop_ids

    def logits(self, batch_ids, positions):
        """The logits for the token at each of `positions` (from the hidden state one position before), as fp32.

        A position may be T = the sequence length: the next token after the sequence (all the decoding loop
        asks for).
        """
        with torch.inference_mode():
            hidden = self.model.model(input_ids=batch_ids.to(self.device)).last_hidden_state
            rows = torch.arange(len(positions), device=self.device)[:, None]
            columns = torch.tensor(positions, device=self.device) - 1
            selected = hidden[rows, columns]
            batch, n_positions, width = selected.shape
            logits = self.model.lm_head(selected.reshape(batch * n_positions, width)).float()
            return logits.reshape(batch, n_positions, -1)
