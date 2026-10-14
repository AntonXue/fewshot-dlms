"""Model adapters. `base.py` holds the fixed-reveal-order loop (Lemma 3.1) and the `Adapter` interface;
each model file only loads its pinned checkpoint and supplies `Adapter.logits` and the chat framing.

The model interfaces of Appendix B, Experimental Setup:

    name         model                              interface                                   file
    llada        GSAI-ML/LLaDA-1.5                  bidirectional canvas                        llada.py
    nemotron_pt  nvidia/Nemotron-Labs-Diffusion-8B  causal prefix + bidirectional tail          nemotron.py
    nemotron_ar  nvidia/Nemotron-Labs-Diffusion-8B  causal suffix (Figure 4)                    nemotron.py
    dqwen        UT-IFML/dQwen3.5-9B-Base           bidirectional canvas                        dqwen.py
    qwen         Qwen/Qwen3-8B                      free-length causal continuation (Figure 4)  qwen_ar.py

The canvas interfaces fill masked canvases (base.sample_fixed_order); the two causal ones continue a
prompt left to right (base.sample_ar).

Each checkpoint is loaded with `from_pretrained(model_id, revision=<pinned commit>)`, so Hugging Face
takes it from the cache under $HF_HOME when it is there and downloads it otherwise (about 15, 16, 17 and
16 GB). Without network access the four pinned snapshots must already be in that cache; set
HF_HUB_OFFLINE=1 to make sure nothing is downloaded. The revisions are listed in README.md.
"""
import os

NAMES = ("llada", "nemotron_pt", "nemotron_ar", "dqwen", "qwen")


def adapter_class(name):
    """The adapter class of model `name` and the keyword arguments it is built with."""
    if name == "llada":
        from fewshot_dlms.adapters.llada import LLaDA

        return LLaDA, {}
    if name == "nemotron_pt":
        from fewshot_dlms.adapters.nemotron import Nemotron

        return Nemotron, {"mode": "prefix_tail"}
    if name == "nemotron_ar":
        from fewshot_dlms.adapters.nemotron import Nemotron

        return Nemotron, {"mode": "causal"}
    if name == "dqwen":
        from fewshot_dlms.adapters.dqwen import DQwen

        return DQwen, {}
    if name == "qwen":
        from fewshot_dlms.adapters.qwen_ar import Qwen3

        return Qwen3, {}
    raise KeyError(f"unknown model {name!r}; expected one of {', '.join(NAMES)}")


def load(name, device):
    """Load the adapter `name` (see the table above) onto `device`, e.g. "cuda".

    Raises OSError naming the checkpoint and the cache when Hugging Face cannot provide the pinned snapshot.
    """
    cls, kwargs = adapter_class(name)
    try:
        return cls(device, **kwargs)
    except OSError as error:
        cache = os.environ.get("HF_HOME", "~/.cache/huggingface (the default; HF_HOME is not set)")
        offline = os.environ.get("HF_HUB_OFFLINE", "0")
        raise OSError(
            f"could not load {cls.model_id} at revision {cls.revision} for the adapter {name!r}. "
            f"Hugging Face cache: HF_HOME={cache}; HF_HUB_OFFLINE={offline}. Download the snapshot with network "
            "access, or point HF_HOME at a cache that holds it (README.md, Requirements, lists the "
            f"checkpoints). Original error: {error}"
        ) from error
