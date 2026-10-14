"""Model adapters. `base.py` holds the fixed-reveal-order loop (Lemma 3.1) and the `Adapter` interface;
each model file only loads its pinned checkpoint and supplies `Adapter.logits`."""
