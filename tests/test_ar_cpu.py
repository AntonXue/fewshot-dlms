"""Unit tests for left-to-right decoding (adapters/base.py: decode_ar, sample_ar, logprob_ar) with toy models on CPU.

`Bigram` is a causal toy: the next token depends only on the token before it, so every log-probability can be
checked by hand. Token 4 plays the newline (a stop token), token 5 is outside the support. `Recorder` keeps every
batch of sequences it is asked about, so a test can read off the batches of sampling and scoring.
"""
import pytest
import torch

from fewshot_dlms.adapters.base import Adapter
from fewshot_dlms.adapters.base import decode_ar
from fewshot_dlms.adapters.base import logprob_ar
from fewshot_dlms.adapters.base import sample_ar
from fewshot_dlms.adapters.base import sample_fixed_order
from fewshot_dlms.canvas import Canvas

STOP = 4
EXCLUDED = 5
SUPPORT = torch.tensor([True, True, True, True, True, False])


def bigram_table():
    """Row l: log-probabilities of the next token after token l (row 6: after an empty sequence)."""
    probs = torch.tensor(
        [
            [0.30, 0.30, 0.10, 0.10, 0.15, 0.05],
            [0.10, 0.30, 0.30, 0.10, 0.15, 0.05],
            [0.10, 0.10, 0.30, 0.30, 0.15, 0.05],
            [0.30, 0.10, 0.10, 0.30, 0.15, 0.05],
            [0.20, 0.20, 0.20, 0.20, 0.15, 0.05],
            [0.20, 0.20, 0.20, 0.20, 0.15, 0.05],
            [0.40, 0.30, 0.10, 0.05, 0.10, 0.05],
        ]
    )
    return torch.log(probs)


class Bigram(Adapter):
    """A causal toy over 6 tokens: the logits for the token at position p depend only on the token at p - 1."""

    name = "bigram"
    model_id = "bigram"
    revision = "0"
    kind = "ar"

    def __init__(self):
        self.table = bigram_table()

    def logits(self, batch_ids, positions):
        rows = []
        for ids, where in zip(batch_ids.tolist(), positions):
            rows.append(torch.stack([self.table[ids[p - 1] if p > 0 else 6] for p in where]))
        return torch.stack(rows)

    def support(self, kind, codes=None):
        return SUPPORT.clone()


class Recorder(Bigram):
    """The bigram toy, keeping the sequences of each forward pass (one list per call)."""

    def __init__(self):
        super().__init__()
        self.batches = []

    def logits(self, batch_ids, positions):
        self.batches.append([tuple(ids) for ids in batch_ids.tolist()])
        for ids, where in zip(batch_ids.tolist(), positions):
            assert where == [len(ids)], "decode_ar asks for the next token after each sequence"
        return super().logits(batch_ids, positions)


def by_hand(prompt, toks):
    """log P(toks | prompt) under the bigram toy, renormalised over SUPPORT."""
    table = bigram_table().masked_fill(~SUPPORT, float("-inf"))
    table = torch.log_softmax(table, -1)
    previous = prompt[-1]
    total = 0.0
    for token in toks:
        total += float(table[previous, token])
        previous = token
    return total


def reference_sample(model, prompt, K, seed, max_new, stop_ids):
    """The authors' free-length sampler written out (dr_worker.free_steps): one generator for the whole call; at
    each step the unfinished continuations are grouped by their tokens so far, in order of first appearance, and
    each group gets one multinomial call whose draws go to its members in continuation order."""
    generator = torch.Generator().manual_seed(seed)
    allowed = SUPPORT
    traces = [[] for _ in range(K)]
    done = [False] * K
    step = 0
    while not all(done):
        groups = {}
        for k in range(K):
            if not done[k]:
                groups.setdefault(tuple(traces[k][:step]), []).append(k)
        for prefix, members in groups.items():
            sequence = list(prompt) + list(prefix)
            logits = model.logits(torch.tensor([sequence]), [[len(sequence)]])[0, 0]
            probs = torch.log_softmax(logits.masked_fill(~allowed, float("-inf")), -1).exp()
            draws = torch.multinomial(probs, len(members), replacement=True, generator=generator).tolist()
            for k, token in zip(members, draws):
                traces[k].append(token)
                if token in stop_ids or len(traces[k]) >= max_new:
                    done[k] = True
        step += 1
    return traces


PROMPT = [6, 0, 1, 2]
PROMPT_P0 = [6, 0, 2]  # PROMPT with token 1 (a demonstration output) deleted


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------


def test_sample_ar_draws_follow_the_authors_order():
    drawn = sample_ar(Bigram(), PROMPT, K=24, seed=7, max_new=6, stop_ids={STOP})
    expected = reference_sample(Bigram(), PROMPT, K=24, seed=7, max_new=6, stop_ids={STOP})
    assert [infill.toks[0] for infill in drawn] == expected


def test_sampling_is_seeded():
    first = sample_ar(Bigram(), PROMPT, K=16, seed=3, max_new=6, stop_ids={STOP})
    again = sample_ar(Bigram(), PROMPT, K=16, seed=3, max_new=6, stop_ids={STOP})
    other = sample_ar(Bigram(), PROMPT, K=16, seed=4, max_new=6, stop_ids={STOP})
    assert [infill.toks for infill in first] == [infill.toks for infill in again]
    assert [infill.toks for infill in first] != [infill.toks for infill in other]


def test_free_length_continuations_end_at_a_stop_token_or_the_budget():
    drawn = sample_ar(Bigram(), PROMPT, K=64, seed=0, max_new=3, stop_ids={STOP})
    assert {len(infill.toks[0]) for infill in drawn} <= {1, 2, 3}
    for infill in drawn:
        trace = infill.toks[0]
        stops = [position for position, token in enumerate(trace) if token == STOP]
        if stops:
            assert stops == [len(trace) - 1], "the stop token ends the continuation and is kept"
            assert not infill.truncated
        else:
            assert len(trace) == 3 and infill.truncated
    assert any(infill.truncated for infill in drawn) and not all(infill.truncated for infill in drawn)


def test_without_stop_ids_every_continuation_has_the_fixed_length():
    # Nemotron's causal suffix: exactly max_new tokens (the oracle length), nothing is truncated.
    drawn = sample_ar(Bigram(), PROMPT, K=32, seed=1, max_new=4, stop_ids=())
    assert all(len(infill.toks[0]) == 4 for infill in drawn)
    assert not any(infill.truncated for infill in drawn)
    assert any(STOP in infill.toks[0] for infill in drawn), "a stop-like token does not end the suffix"


def test_excluded_ids_are_never_drawn():
    drawn = sample_ar(Bigram(), PROMPT, K=64, seed=2, max_new=5, stop_ids={STOP})
    assert all(EXCLUDED not in infill.toks[0] for infill in drawn)


def test_ell1_sample_is_the_sum_of_its_terms_and_matches_the_hand_value():
    drawn = sample_ar(Bigram(), PROMPT, K=8, seed=5, max_new=4, stop_ids={STOP})
    for infill in drawn:
        assert len(infill.ell1_tokens) == len(infill.toks[0])
        assert infill.ell1_sample == pytest.approx(sum(infill.ell1_tokens), abs=1e-12)
        assert infill.ell1_sample == pytest.approx(by_hand(PROMPT, infill.toks[0]), abs=1e-5)


def test_sample_ar_rejects_a_canvas_model():
    model = Bigram()
    model.kind = "dlm"
    with pytest.raises(ValueError):
        sample_ar(model, PROMPT, K=2, seed=0, max_new=2, stop_ids=())


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------


def test_logprob_of_sample_matches_sampler():
    drawn = sample_ar(Bigram(), PROMPT, K=16, seed=11, max_new=5, stop_ids={STOP})
    for infill in drawn:
        total, terms = logprob_ar(Bigram(), PROMPT, infill.toks)
        assert total == infill.ell1_sample
        assert terms == infill.ell1_tokens


def test_flat_tokens_score_like_infill_tokens():
    total, terms = logprob_ar(Bigram(), PROMPT, [0, 3, STOP])
    same_total, same_terms = logprob_ar(Bigram(), PROMPT, [[0, 3, STOP]])
    assert (total, terms) == (same_total, same_terms)
    assert total == pytest.approx(by_hand(PROMPT, [0, 3, STOP]), abs=1e-5)


def test_hide_E_prompt_scores_under_the_P0_prompt():
    model = Recorder()
    total, _ = logprob_ar(model, PROMPT, [3, 1], hide_E_prompt_ids=PROMPT_P0)
    assert total == pytest.approx(by_hand(PROMPT_P0, [3, 1]), abs=1e-5)
    assert [batch[0] for batch in model.batches] == [tuple(PROMPT_P0), tuple(PROMPT_P0 + [3])]


def test_hide_E_prompt_must_delete_tokens_of_the_prompt():
    with pytest.raises(ValueError):
        logprob_ar(Bigram(), PROMPT, [3], hide_E_prompt_ids=[6, 2, 0])
    with pytest.raises(ValueError):
        logprob_ar(Bigram(), PROMPT, [3], hide_E_prompt_ids=PROMPT + [1])


def test_sampling_without_a_generator_is_refused():
    # Every draw must come from a seeded generator, never from torch's global RNG.
    with pytest.raises(ValueError, match="Generator"):
        decode_ar(Bigram(), PROMPT, 2, SUPPORT, max_new=3, stop_ids={STOP})


def test_scoring_the_distinct_draws_together_repeats_the_sampler_batches():
    # The frozen pools scored P0 (and replayed P1) with the distinct draws in first-occurrence order in one call;
    # the forward passes then hold the same sequences as during sampling, batch for batch.
    sampler = Recorder()
    traces, terms, _ = decode_ar(
        sampler, PROMPT, 32, SUPPORT, generator=torch.Generator().manual_seed(9), max_new=4, stop_ids={STOP}
    )
    distinct = [list(trace) for trace in {tuple(trace): None for trace in traces}]
    scorer = Recorder()
    _, replayed, _ = decode_ar(scorer, PROMPT, len(distinct), SUPPORT, forced=distinct)
    assert scorer.batches == sampler.batches
    by_trace = {tuple(trace): sum(trace_terms) for trace, trace_terms in zip(traces, terms)}
    for trace, trace_terms in zip(distinct, replayed):
        assert sum(trace_terms) == by_trace[tuple(trace)]


def test_chunk_limits_the_batch_size():
    model = Recorder()
    forced = [[a, b] for a in range(4) for b in range(4)]
    decode_ar(model, PROMPT, len(forced), SUPPORT, forced=forced, chunk=3)
    assert max(len(batch) for batch in model.batches) == 3
    assert sum(len(batch) for batch in model.batches) == 1 + 4  # one empty prefix, then the four first tokens


def test_decode_ar_counts_its_forward_passes():
    model = Bigram()
    decode_ar(model, PROMPT, 2, SUPPORT, forced=[[0, 1], [0, 2]])
    assert model.cost()["forwards"] == 2
    assert model.cost()["sequences"] == 2
    assert model.cost()["tokens"] == len(PROMPT) + len(PROMPT) + 1


# ---------------------------------------------------------------------------------------------
# Per-adapter defaults: support and batch sizes
# ---------------------------------------------------------------------------------------------


class FreeBigram(Recorder):
    """The recording bigram toy whose ordinary support excludes the stop token and whose "free" support admits it,
    like Qwen3-8B's <|im_end|>; free generation is its default, with small batches."""

    default_support = "free"
    sample_chunk = 3
    p0_chunk = 2

    def support(self, kind, codes=None):
        mask = SUPPORT.clone()
        if kind != "free":
            mask[STOP] = False
        return mask


def test_logprob_ar_uses_the_adapters_default_support():
    model = FreeBigram()
    total, _ = logprob_ar(model, PROMPT, [0, STOP])
    assert total == pytest.approx(by_hand(PROMPT, [0, STOP]), abs=1e-5)
    ordinary, _ = logprob_ar(model, PROMPT, [0, STOP], support="ordinary")
    assert ordinary == float("-inf")


def test_sample_ar_uses_the_adapters_default_support_and_chunk():
    model = FreeBigram()
    drawn = sample_ar(model, PROMPT, K=32, seed=0, max_new=3, stop_ids={STOP})
    assert any(infill.toks[0][-1] == STOP for infill in drawn)
    assert max(len(batch) for batch in model.batches) == FreeBigram.sample_chunk
    explicit = sample_ar(FreeBigram(), PROMPT, K=32, seed=0, max_new=3, stop_ids={STOP}, support="free", chunk=3)
    assert [infill.toks for infill in drawn] == [infill.toks for infill in explicit]


def test_decode_ar_defaults_to_the_sampling_chunk():
    model = FreeBigram()
    forced = [[a, b] for a in range(4) for b in range(4)]
    decode_ar(model, PROMPT, len(forced), forced=forced)
    assert max(len(batch) for batch in model.batches) == FreeBigram.sample_chunk
    model.batches.clear()
    decode_ar(model, PROMPT, len(forced), forced=forced, chunk=model.p0_chunk)
    assert max(len(batch) for batch in model.batches) == FreeBigram.p0_chunk


def test_base_adapter_defaults():
    assert Adapter.default_support == "ordinary"
    assert Adapter.sample_chunk == 16 and Adapter.p0_chunk == 16


def test_fixed_order_loop_rejects_a_causal_adapter_with_a_message():
    # A causal adapter has no canvas: the reveal loop's begin_reveal hook stops it before any forward pass.
    class NoCanvas(Bigram):
        def begin_reveal(self, order):
            raise ValueError("no canvas to reveal; use sample_ar / logprob_ar")

    canvas = Canvas(ids=[0, 1, 2], sites=[[2]], E=[1], gold=[""], meta={})
    with pytest.raises(ValueError):
        sample_fixed_order(NoCanvas(), canvas, K=2, seed=0)
