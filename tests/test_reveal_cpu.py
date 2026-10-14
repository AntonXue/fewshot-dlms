"""Unit tests for the fixed-reveal-order loop (adapters/base.py) with toy models on CPU.

`Toy` is a unigram model: its logits do not depend on the canvas, so every log-probability can be
checked by hand. `Recorder` returns the same logits but keeps every canvas it was shown, so a test can
read off the reveal order and the P0 masking.
"""
import numpy as np
import pytest
import torch

from fewshot_dlms.adapters.base import Adapter
from fewshot_dlms.adapters.base import logprob_fixed_order
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.adapters.base import sample_fixed_order
from fewshot_dlms.adapters.base import site_sums
from fewshot_dlms.canvas import Canvas

MASK = 4


class Toy(Adapter):
    """Unigram model over a 5-token vocab: logits independent of context, so log-probs are checkable by hand."""

    name = "toy"
    model_id = "toy"
    revision = "0"
    mask_id = MASK

    def __init__(self):
        self.table = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1, 1e-9]))

    def encode(self, s):
        return [ord(c) % 4 for c in s]

    def logits(self, batch_ids, positions):
        return self.table.expand(batch_ids.shape[0], max(len(p) for p in positions), 5).clone()

    def support(self, kind):
        return torch.tensor([True, True, True, True, False])


class Recorder(Toy):
    """The unigram model, keeping each (canvas, position) pair it is asked about."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def logits(self, batch_ids, positions):
        for ids, where in zip(batch_ids.tolist(), positions):
            self.seen.append((ids, where))
        return super().logits(batch_ids, positions)


def canvas():
    """Two target sites, [2, 3] and [5]; one demonstration-output position, 1."""
    return Canvas(ids=[0, 1, 4, 4, 2, 4], sites=[[2, 3], [5]], E=[1], gold=["", ""], meta={})


def test_logprob_of_sample_matches_sampler():
    cv = canvas()
    inf = sample_fixed_order(Toy(), cv, K=3, seed=0)
    for x in inf:
        total, per = logprob_fixed_order(Toy(), cv, x.toks, hide_E=False)
        assert np.isclose(total, x.ell1_sample, atol=1e-5)
        assert len(per) == 3


def test_unigram_logprob_is_sum_of_table():
    cv = canvas()
    total, _ = logprob_fixed_order(Toy(), cv, [[0, 1], [2]], hide_E=False)
    assert np.isclose(total, float(np.log(0.4) + np.log(0.3) + np.log(0.2)), atol=1e-6)


def test_sampling_is_seeded():
    cv = canvas()
    a = sample_fixed_order(Toy(), cv, 4, seed=7)
    b = sample_fixed_order(Toy(), cv, 4, seed=7)
    assert [x.toks for x in a] == [x.toks for x in b]


def test_infill_has_one_token_list_per_site_and_per_token_terms():
    cv = canvas()
    for x in sample_fixed_order(Toy(), cv, K=5, seed=3):
        assert [len(site) for site in x.toks] == [2, 1]
        assert all(0 <= token < MASK for site in x.toks for token in site)
        assert len(x.ell1_tokens) == 3
        assert x.ell1_sample == sum(x.ell1_tokens)


def test_reveal_order_is_ascending_position_one_token_per_step():
    model = Recorder()
    logprob_fixed_order(model, canvas(), [[0, 1], [2]], hide_E=False)
    assert reveal_order(canvas()) == [2, 3, 5]
    assert model.seen == [
        ([0, 1, 4, 4, 2, 4], [2]),
        ([0, 1, 0, 4, 2, 4], [3]),
        ([0, 1, 0, 1, 2, 4], [5]),
    ]


def test_hide_E_masks_the_demonstration_outputs_and_nothing_else():
    model = Recorder()
    logprob_fixed_order(model, canvas(), [[0, 1], [2]], hide_E=True)
    assert [ids for ids, _ in model.seen] == [
        [0, MASK, 4, 4, 2, 4],
        [0, MASK, 0, 4, 2, 4],
        [0, MASK, 0, 1, 2, 4],
    ]


def test_flat_tokens_in_reveal_order_score_like_per_site_tokens():
    # The frozen pools store an infill as one flat list in reveal order.
    per_site = logprob_fixed_order(Toy(), canvas(), [[0, 1], [2]], hide_E=False)
    flat = logprob_fixed_order(Toy(), canvas(), [0, 1, 2], hide_E=False)
    assert per_site == flat


def test_logprob_rejects_a_wrong_number_of_tokens():
    with pytest.raises(ValueError):
        logprob_fixed_order(Toy(), canvas(), [[0], [2]], hide_E=False)
    with pytest.raises(ValueError):
        logprob_fixed_order(Toy(), canvas(), [0, 1], hide_E=False)


def test_site_sums_are_the_per_target_terms():
    _, per = logprob_fixed_order(Toy(), canvas(), [[0, 1], [2]], hide_E=False)
    assert site_sums(canvas(), per) == [per[0] + per[1], per[2]]


def test_cost_counts_forwards_sequences_and_tokens():
    model = Toy()
    logprob_fixed_order(model, canvas(), [[0, 1], [2]], hide_E=False)
    cost = model.cost()
    assert (cost["forwards"], cost["sequences"], cost["tokens"]) == (3, 3, 18)


def test_candidates_on_different_canvases_follow_their_own_order():
    # Per-candidate target lengths (Section 4.3): candidate k is revealed on canvas which[k].
    long_canvas = [0, 1, 4, 4, 2, 4]
    short_canvas = [0, 1, 4, 2, 4]
    generator = torch.Generator().manual_seed(0)
    traces, logprobs = reveal(
        Toy(),
        [long_canvas, short_canvas],
        [[2, 3, 5], [2, 4]],
        which=[0, 1, 0],
        allowed=Toy().support("ordinary"),
        generator=generator,
    )
    assert [len(trace) for trace in traces] == [3, 2, 3]
    assert [len(terms) for terms in logprobs] == [3, 2, 3]
