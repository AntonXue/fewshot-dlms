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


class Neighbour(Toy):
    """A context-dependent model: the token at a position follows the token to its left, l.

    It is l with probability 0.4, l + 1 with 0.3, l + 2 with 0.2 and l + 3 with 0.1 (mod 4); a
    masked left neighbour counts as l = 0. Revealed tokens therefore change every later factor.
    """

    def logits(self, batch_ids, positions):
        base = torch.tensor([0.4, 0.3, 0.2, 0.1])
        rows = []
        for ids, where in zip(batch_ids.tolist(), positions):
            left = ids[where[0] - 1]
            shift = 0 if left == MASK else left
            probs = torch.cat([torch.roll(base, shift), torch.tensor([1e-9])])
            rows.append(torch.log(probs).expand(len(where), 5))
        return torch.stack(rows)


class Announced(Toy):
    """The unigram model, logging begin_reveal and logits calls in the order they happen."""

    def __init__(self):
        super().__init__()
        self.events = []

    def begin_reveal(self, order):
        self.events.append(("begin_reveal", list(order)))

    def logits(self, batch_ids, positions):
        self.events.append(("logits", positions[0][0]))
        return super().logits(batch_ids, positions)


def canvas():
    """Two target sites, [2, 3] and [5]; one demonstration-output position, 1."""
    return Canvas(ids=[0, 1, 4, 4, 2, 4], sites=[[2, 3], [5]], E=[1], gold=["", ""], meta={})


def reference_sample(model, cv, K, seed, allowed):
    """The authors' draw order, written out for one canvas: one generator for the whole call; at each
    step the candidates are grouped by the tokens they revealed so far, in order of first appearance,
    and each group gets one multinomial call whose draws go to its members in candidate order."""
    generator = torch.Generator().manual_seed(seed)
    order = sorted(position for site in cv.sites for position in site)
    traces = [[] for _ in range(K)]
    for step, position in enumerate(order):
        groups = {}
        for k in range(K):
            groups.setdefault(tuple(traces[k]), []).append(k)
        for prefix, members in groups.items():
            ids = list(cv.ids)
            for revealed, token in zip(order, prefix):
                ids[revealed] = token
            logits = model.logits(torch.tensor([ids]), [[position]])[0, 0]
            logits = logits.masked_fill(~allowed, float("-inf"))
            probs = torch.log_softmax(logits, -1).exp()
            draws = torch.multinomial(probs, len(members), replacement=True, generator=generator).tolist()
            for k, token in zip(members, draws):
                traces[k].append(token)
    return traces


def flat(infill):
    """The tokens of an infill in reveal order (the sites of these canvases are in ascending order)."""
    return [token for site in infill.toks for token in site]


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


# ---------------------------------------------------------------------------------------------
# Support masking, the seeded draw order and the begin_reveal hook
# ---------------------------------------------------------------------------------------------

ALLOW_123 = torch.tensor([False, True, True, True, False])  # excludes id 0 and the mask


def zero_canvas():
    """Like canvas(), but the left neighbour of each site is token 0, so id 0 is the likeliest there."""
    return Canvas(ids=[0, 0, 4, 4, 0, 4], sites=[[2, 3], [5]], E=[1], gold=["", ""], meta={})


def test_excluded_ids_are_never_drawn():
    unrestricted = sample_fixed_order(Neighbour(), zero_canvas(), K=200, seed=0, support=None)
    restricted = sample_fixed_order(Neighbour(), zero_canvas(), K=200, seed=0, support=ALLOW_123)
    assert any(0 in flat(x) for x in unrestricted)
    assert not any(0 in flat(x) for x in restricted)


def test_scores_are_renormalised_over_the_support():
    # Position 2: left 0, so p = (0.4, 0.3, 0.2, 0.1); without id 0 token 1 has 0.3 / 0.6.
    # Position 3: left 1, so p = (0.1, 0.4, 0.3, 0.2); token 2 has 0.3 / 0.9. Position 5: as position 2.
    infill = [[1, 2], [1]]
    total, per = logprob_fixed_order(Neighbour(), zero_canvas(), infill, hide_E=False, support=ALLOW_123)
    expected = [np.log(0.3 / 0.6), np.log(0.3 / 0.9), np.log(0.3 / 0.6)]
    assert np.allclose(per, expected, atol=1e-6)
    assert np.isclose(total, sum(expected), atol=1e-6)


def test_draws_follow_the_authors_order():
    allowed = Neighbour().support("ordinary")
    for seed in (0, 7):
        infills = sample_fixed_order(Neighbour(), canvas(), K=12, seed=seed, chunk=5)
        assert [flat(x) for x in infills] == reference_sample(Neighbour(), canvas(), 12, seed, allowed)


def test_another_seed_draws_other_tokens():
    a = sample_fixed_order(Neighbour(), canvas(), K=12, seed=0)
    b = sample_fixed_order(Neighbour(), canvas(), K=12, seed=1)
    assert [flat(x) for x in a] != [flat(x) for x in b]


def test_begin_reveal_is_called_once_before_the_first_forward_pass():
    sampler = Announced()
    sample_fixed_order(sampler, canvas(), K=3, seed=0)
    scorer = Announced()
    logprob_fixed_order(scorer, canvas(), [[0, 1], [2]], hide_E=True)
    for model in (sampler, scorer):
        assert model.events[0] == ("begin_reveal", [2, 3, 5])
        assert [name for name, _ in model.events].count("begin_reveal") == 1
        assert [where for name, where in model.events[1:]][:1] == [2]
