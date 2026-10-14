"""Model adapters. `base.py` holds the fixed-reveal-order loop (Lemma 3.1) and the `Adapter` interface;
each model file only loads its pinned checkpoint and supplies `Adapter.logits` and the chat framing.

The model interfaces of Appendix B.1:

    name         model                              interface                            file
    llada        GSAI-ML/LLaDA-1.5                  bidirectional canvas                 llada.py
    nemotron_pt  nvidia/Nemotron-Labs-Diffusion-8B  causal prefix + bidirectional tail   nemotron.py
    dqwen        UT-IFML/dQwen3.5-9B-Base           bidirectional canvas                 dqwen.py

Each checkpoint is loaded with `from_pretrained(model_id, revision=<pinned commit>)`, so Hugging Face
takes it from the cache under $HF_HOME when it is there and downloads it otherwise (about 15, 16 and
17 GB). Without network access the three pinned snapshots must already be in that cache; set
HF_HUB_OFFLINE=1 to make sure nothing is downloaded. The revisions are listed in README.md.
"""


def load(name, device):
    """Load the adapter `name` (see the table above) onto `device`, e.g. "cuda"."""
    if name == "llada":
        from fewshot_dlms.adapters.llada import LLaDA

        return LLaDA(device)
    if name == "nemotron_pt":
        from fewshot_dlms.adapters.nemotron import Nemotron

        return Nemotron(device, mode="prefix_tail")
    if name == "dqwen":
        from fewshot_dlms.adapters.dqwen import DQwen

        return DQwen(device)
    raise KeyError(f"unknown model {name!r}; expected one of llada, nemotron_pt, dqwen")
