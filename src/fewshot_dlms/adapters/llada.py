# ported from editdiff/experiments/multimodel/mm_adapters.py::LLaDA15, experiments/llada_pilot/adapters.py::LLaDA15, experiments/sec42_mj/mj_worker.py (transformers 5 loader shims) and experiments/length_pilot/lp_common.py::MASK_ID @ d276fa6
"""LLaDA-1.5, a masked diffusion language model that infills the whole canvas (Appendix B, Experimental Setup).

Interface: the canvas is one sequence under the chat template, and the model attends over all of it in
both directions (its own forward pass, no attention mask). The logits at a position are the model's
distribution for the token at that same position (no shift). The mask token is <|mdm_mask|>.

The checkpoint is GSAI-ML/LLaDA-1.5 at a pinned revision, loaded in bf16 with its own modelling code
(trust_remote_code). That code was written for transformers 4.x; `_patch_for_transformers_5` applies the
shims the authors used to run it under transformers 5.13.
"""
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from fewshot_dlms.adapters.base import Adapter

MASK_ID = 126336  # <|mdm_mask|>
BOS_ID = 126080  # <|startoftext|>


def _patch_for_transformers_5(model_id, revision):
    """Make LLaDA-1.5's remote code (written for transformers 4.46) load under transformers 5.

    Shim 1: transformers 5 reads the class attribute `all_tied_weights_keys`, which the remote code never
    sets. LLaDA-1.5 has weight_tying = false, so the correct value is an empty mapping.
    Shim 2: transformers 5 calls `tie_weights` with keyword arguments that the 4.x-era signature rejects;
    the model is untied, so it is replaced by a no-op.
    (Shim 3, `config.use_cache`, is applied after loading, in LLaDA.__init__.)
    """
    if int(transformers.__version__.split(".")[0]) < 5:
        return
    cls = get_class_from_dynamic_module("modeling_llada.LLaDAModelLM", model_id, revision=revision)
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    cls.tie_weights = lambda self, *args, **kwargs: None


class LLaDA(Adapter):
    """LLaDA-1.5 on a fully bidirectional canvas."""

    name = "llada"
    model_id = "GSAI-ML/LLaDA-1.5"
    revision = "84346fd91ba60252d260022201ad6fc5a3468fb2"
    mask_id = MASK_ID
    kind = "dlm"
    close_slot = True  # the authors' slot rule: a one-target canvas ends the turn right after the masks

    def __init__(self, device):
        _patch_for_transformers_5(self.model_id, self.revision)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True)
        # bf16 weights; `dtype` is transformers 5's name for the `torch_dtype` argument in the authors' code.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True, dtype=torch.bfloat16
        )
        self.model = model.to(device).eval()
        # Shim 3: transformers 5 no longer gives the config a `use_cache` default; the remote code reads it.
        if not hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        assert int(self.model.config.mask_token_id) == MASK_ID
        self.eos_id = int(self.model.config.eos_token_id)
        self.bos_id = BOS_ID
        self.V = int(self.model.config.vocab_size)
        special_ids = {i for i in self.tok.all_special_ids if i is not None}
        self.control_ids = {self.mask_id, self.eos_id, self.bos_id} | special_ids
        self.code_ids = {c: self.encode(" " + c)[0] for c in "ABCDEFGHIJ" if len(self.encode(" " + c)) == 1}
        assert all(c in self.code_ids for c in "ABCDEFGHIJ")

    def _template(self, content):
        """The chat template around one user turn, up to the start of the assistant turn."""
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # The authors' fallback; the template of the pinned revision renders the same text.
            return (
                f"<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n{content}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    def chat_header(self, instruction):
        return self._template(instruction)

    def chat_user_open(self):
        return self._template("XSPLITX").split("XSPLITX")[0]

    def chat_user_close_assistant_open(self):
        return self._template("XSPLITX").split("XSPLITX")[1]

    def chat_end(self):
        return "<|eot_id|>"

    def logits(self, batch_ids, positions):
        """One bidirectional forward pass over the B canvases; the bf16 logits read at `positions`, as fp32."""
        with torch.inference_mode():
            ids = batch_ids.to(self.device)
            output = self.model(input_ids=ids)
            logits = output.logits if hasattr(output, "logits") else output[0]
            rows = torch.arange(len(positions), device=self.device)[:, None]
            columns = torch.tensor(positions, device=self.device)
            return logits[rows, columns].float()
