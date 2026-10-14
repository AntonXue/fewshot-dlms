"""Replay (fidelity level 1 of README.md): the ported adapters reproduce the authors' stored log-likelihoods.

data/fixtures/fig5/<model>_M4_J2.jsonl holds two Figure 5 pool records per model (List Functions-Easy,
M = 4 demonstrations, J = 2 targets, K = 16 candidates). Each candidate stores its token ids in reveal
order and the two log-likelihoods of the paper: ell1 = log P1 (demonstration outputs visible) and
ell0 = log P0 (demonstration outputs masked at E). The tests rebuild each canvas with this repository's
canvas.py and the model's tokenizer, check that it is the frozen prompt token for token, and score the
stored tokens with the fixed-order loop of Lemma 3.1 (adapters/base.py):

  - one candidate at a time with logprob_fixed_order (the first N_CANDIDATES distinct candidates of each
    record), which agrees up to bf16 batch noise;
  - with the frozen run's own batches, which on the authors' platform is exact for LLaDA-1.5 and
    Nemotron (tests/gpu_util.py explains both tolerances). For dQwen3.5 this test also checks that the
    linear-attention kernels are those of the paper's runs (flash-linear-attention 0.5.1).

The two causal interfaces of Figure 4 (Appendix B, Experimental Setup) are checked the same way on
data/fixtures/fig4/<model>_sst2_M4.jsonl: two single-target pool records per model (SST-2, M = 4,
K = 64 continuations). nemotron_ar is Nemotron's causal suffix of the oracle length; qwen is Qwen3-8B's
free-length continuation, which stops at a newline or <|im_end|>. Here P0 deletes the demonstration
outputs from the prompt. The tests rebuild both prompts with canvas.ar_prompt_ids, compare them with the
decoded prompts the frozen run stored (by SHA-256: the fixtures do not redistribute SST-2 text), re-score
the stored continuations (one at a time with logprob_ar, and in the frozen run's batches with decode_ar),
and draw the 64 continuations again from the frozen run's seed.

GPU only (`make test-gpu`); needs the four pinned checkpoints (see README.md) and, for the Figure 4
records, the rebuilt SST-2 text (`python data/benchmarks/build.py`).
"""
import hashlib
import json

import pytest
import torch

from fewshot_dlms.adapters import load
from fewshot_dlms.adapters.base import context_ids
from fewshot_dlms.adapters.base import decode_ar
from fewshot_dlms.adapters.base import logprob_ar
from fewshot_dlms.adapters.base import logprob_fixed_order
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.adapters.base import sample_ar
from fewshot_dlms.adapters.dqwen import PAPER_KERNELS
from fewshot_dlms.canvas import ar_prompt_ids
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.data import read_jsonl
from tests.gpu_util import batched_tolerance
from tests.gpu_util import on_reference_platform
from tests.gpu_util import release
from tests.gpu_util import tolerance

pytestmark = pytest.mark.gpu

POOL = {
    "llada": "fig5/llada_M4_J2.jsonl",
    "nemotron_pt": "fig5/nemotron_pt_M4_J2.jsonl",
    "dqwen": "fig5/dqwen_M4_J2.jsonl",
}
N_RECORDS = 2  # pool records (families) per model
N_CANDIDATES = 4  # distinct candidates scored one at a time per record (fewer if the record has fewer)

# The causal interfaces of Figure 4 and their fixtures. How the frozen run decoded them (the support of every
# step, the batch sizes of sampling and of P0 scoring) is on the adapters: default_support, sample_chunk, p0_chunk.
AR_POOL = {
    "nemotron_ar": "fig4/nemotron_ar_sst2_M4.jsonl",
    "qwen": "fig4/qwen_sst2_M4.jsonl",
}


@pytest.fixture(scope="module")
def adapter(request):
    """The model of request.param, loaded once per module and released before the next one is loaded."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    model = load(request.param, "cuda")
    yield model
    release(model)


@pytest.fixture(scope="module")
def families():
    return {family["episode_id"]: family for family in load_families("list_easy_fam", "report")}


def records_of(root, name):
    return read_jsonl(root / "data" / "fixtures" / POOL[name])[:N_RECORDS]


def canvas_of(adapter, families, record):
    family = families[record["episode_id"]]
    return build_canvas(adapter, family, record["M"], record["J"], record["oracle_L"])


def sha256_of(ids):
    """The prompt hash of the frozen run: SHA-256 of the JSON list of token ids."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def distinct_candidates(record, n):
    """The first n candidates of a record with distinct token ids, in stored order."""
    seen = set()
    chosen = []
    for candidate in record["candidates"]:
        toks = tuple(candidate["toks"])
        if toks not in seen and len(chosen) < n:
            seen.add(toks)
            chosen.append(candidate)
    return chosen


def assert_close(replayed, stored, limit, where):
    assert abs(replayed - stored) <= limit, (where, replayed, stored, limit)


@pytest.mark.parametrize("adapter", list(POOL), indirect=True)
def test_canvas_is_the_frozen_prompt(adapter, families, root):
    for record in records_of(root, adapter.name):
        canvas = canvas_of(adapter, families, record)
        assert sha256_of(context_ids(adapter, canvas, hide_E=False)) == record["prompt_sha256_p1"]
        assert sha256_of(context_ids(adapter, canvas, hide_E=True)) == record["prompt_sha256_p0"]
        assert reveal_order(canvas) == record["T_positions"]
        assert len(canvas.E) == record["n_demo_output_tokens_hidden"]


@pytest.mark.parametrize("adapter", list(POOL), indirect=True)
def test_replay_reproduces_stored_loglik(adapter, families, root):
    limit = tolerance(adapter.name)
    for record in records_of(root, adapter.name):
        canvas = canvas_of(adapter, families, record)
        for candidate in distinct_candidates(record, N_CANDIDATES):
            ell1, _ = logprob_fixed_order(adapter, canvas, candidate["toks"], hide_E=False)
            ell0, _ = logprob_fixed_order(adapter, canvas, candidate["toks"], hide_E=True)
            where = (adapter.name, record["episode_id"], "one at a time")
            assert_close(ell1, candidate["ell1"], limit, where)
            assert_close(ell0, candidate["ell0"], limit, where)
        if record["block_start"] is not None:
            assert adapter.block_start == record["block_start"]


@pytest.mark.parametrize("adapter", list(POOL), indirect=True)
def test_replay_with_the_frozen_batches(adapter, families, root, record_property):
    """Score in the batches of the frozen run: exact on the authors' platform for LLaDA and Nemotron.

    The frozen run sampled P1 with all K draws in one reveal, so each forward pass held the distinct
    prefixes of the draws; replaying the distinct draws together forms the same batches. It scored P0 for
    the distinct draws and the reference answer together, and P1 of the reference answer on its own.
    """
    name = adapter.name
    limit = batched_tolerance(name)
    if name == "dqwen":
        record_property("dqwen_kernels", adapter.kernels)
        if on_reference_platform():
            assert adapter.kernels == PAPER_KERNELS
    allowed = adapter.support("ordinary")
    for record in records_of(root, name):
        canvas = canvas_of(adapter, families, record)
        order = reveal_order(canvas)
        distinct = list({tuple(candidate["toks"]): None for candidate in record["candidates"]})
        p1 = context_ids(adapter, canvas, hide_E=False)
        p0 = context_ids(adapter, canvas, hide_E=True)
        _, p1_terms = reveal(adapter, [p1], [order], [0] * len(distinct), allowed, forced=distinct)
        with_gold = distinct + [tuple(record["gold_toks"])]
        _, p0_terms = reveal(adapter, [p0], [order], [0] * len(with_gold), allowed, forced=with_gold)
        ell1 = {toks: sum(terms) for toks, terms in zip(distinct, p1_terms)}
        ell0 = {toks: sum(terms) for toks, terms in zip(distinct, p0_terms)}
        where = (name, record["episode_id"], "frozen batches")
        for candidate in record["candidates"]:
            toks = tuple(candidate["toks"])
            assert_close(ell1[toks], candidate["ell1"], limit, where)
            assert_close(ell0[toks], candidate["ell0"], limit, where)
        gold_ell0 = sum(p0_terms[-1])
        gold_ell1, _ = logprob_fixed_order(adapter, canvas, record["gold_toks"], hide_E=False)
        assert_close(gold_ell0, record["gold_ell0"], limit, where)
        assert_close(gold_ell1, record["gold_ell1"], limit, where)


# ---------------------------------------------------------------------------------------------
# The causal interfaces of Figure 4
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sst2_families():
    try:
        return {family["episode_id"]: family for family in load_families("sst2_fam", "report")}
    except FileNotFoundError:
        pytest.skip("SST-2 text not built: run `python data/benchmarks/build.py`")


def ar_records_of(root, name):
    return read_jsonl(root / "data" / "fixtures" / AR_POOL[name])[:N_RECORDS]


def ar_prompts(adapter, families, record):
    """The P1 and P0 prompt ids of a single-target record (target T1, the record's M demonstrations)."""
    family = families[record["episode_id"]]
    p1 = ar_prompt_ids(adapter, family, record["M"], 1)
    p0 = ar_prompt_ids(adapter, family, record["M"], 1, hide_E=True)
    return p1, p0


def sha256_of_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ar_seed(name, record):
    """The frozen run's sampling seed of a single-target record (the proposal key, reduced to 31 bits)."""
    parts = ("algo_investigation", "proposal", name, record["episode_id"], record["M"], record["J"])
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:16], 16) % (2**31)


@pytest.mark.parametrize("adapter", list(AR_POOL), indirect=True)
def test_ar_prompt_is_the_frozen_prompt(adapter, sst2_families, root):
    for record in ar_records_of(root, adapter.name):
        p1, p0 = ar_prompts(adapter, sst2_families, record)
        checks = record["checks"]
        assert sha256_of_text(adapter.decode(p1)) == checks["p1_prompt_text_sha256"]
        assert sha256_of_text(adapter.decode(p0)) == checks["p0_prompt_text_sha256"]
        assert len(p1) - len(p0) == checks["P0_prefix_shorter_by_tokens"]
        oracle_length = len(adapter.encode(" " + record["gold"][0]))
        assert record["oracle_L"] == [oracle_length]
        if adapter.name == "nemotron_ar":
            assert record["n_tokens"] == len(p1) + oracle_length
            assert all(len(candidate["toks"]) == oracle_length for candidate in record["candidates"])
        else:
            assert record["n_tokens"] == len(p1)


@pytest.mark.parametrize("adapter", list(AR_POOL), indirect=True)
def test_ar_replay_reproduces_stored_loglik(adapter, sst2_families, root):
    limit = tolerance(adapter.name)
    for record in ar_records_of(root, adapter.name):
        p1, p0 = ar_prompts(adapter, sst2_families, record)
        for candidate in distinct_candidates(record, N_CANDIDATES):
            ell1, _ = logprob_ar(adapter, p1, candidate["toks"])
            ell0, _ = logprob_ar(adapter, p1, candidate["toks"], hide_E_prompt_ids=p0)
            where = (adapter.name, record["episode_id"], "one at a time")
            assert_close(ell1, candidate["ell1"], limit, where)
            assert_close(ell0, candidate["ell0"], limit, where)


@pytest.mark.parametrize("adapter", list(AR_POOL), indirect=True)
def test_ar_replay_with_the_frozen_batches(adapter, sst2_families, root):
    """Score the distinct continuations together, in first-occurrence order: the frozen run's batches.

    Sampling ran all K draws in one call, so each forward pass held the distinct prefixes of the draws; the
    distinct draws in first-occurrence order form the same batches. P0 was scored the same way.
    """
    name = adapter.name
    limit = batched_tolerance(name)
    for record in ar_records_of(root, name):
        p1, p0 = ar_prompts(adapter, sst2_families, record)
        distinct = list({tuple(candidate["toks"]): None for candidate in record["candidates"]})
        n = len(distinct)
        _, p1_terms, _ = decode_ar(adapter, p1, n, forced=distinct)
        _, p0_terms, _ = decode_ar(adapter, p0, n, forced=distinct, chunk=adapter.p0_chunk)
        ell1 = {toks: sum(terms) for toks, terms in zip(distinct, p1_terms)}
        ell0 = {toks: sum(terms) for toks, terms in zip(distinct, p0_terms)}
        where = (name, record["episode_id"], "frozen batches")
        for candidate in record["candidates"]:
            toks = tuple(candidate["toks"])
            assert_close(ell1[toks], candidate["ell1"], limit, where)
            assert_close(ell0[toks], candidate["ell0"], limit, where)
            assert candidate["ell1_t"] == [candidate["ell1"]] and candidate["ell0_t"] == [candidate["ell0"]]


@pytest.mark.parametrize("adapter", list(AR_POOL), indirect=True)
def test_ar_sampling_reproduces_the_stored_draws(adapter, sst2_families, root):
    """Draw the K continuations again from the record's seed: on the authors' platform, the same tokens."""
    if not on_reference_platform():
        pytest.skip("draws depend on the GPU's numerics; reproduced only on the authors' platform")
    name = adapter.name
    for record in ar_records_of(root, name):
        p1, _ = ar_prompts(adapter, sst2_families, record)
        if name == "nemotron_ar":
            max_new = record["oracle_L"][0]
            stop_ids = ()
        else:
            max_new = record["max_new"]
            stop_ids = adapter.stop_ids()
        drawn = sample_ar(adapter, p1, record["K"], ar_seed(name, record), max_new, stop_ids)
        stored = record["candidates"]
        assert [infill.toks[0] for infill in drawn] == [candidate["toks"] for candidate in stored]
        assert [infill.truncated for infill in drawn] == [candidate["truncated"] for candidate in stored]
        where = (name, record["episode_id"], "resampled")
        for infill, candidate in zip(drawn, stored):
            assert_close(infill.ell1_sample, candidate["ell1"], batched_tolerance(name), where)
            assert adapter.decode(infill.toks[0]) == candidate["texts"][0]
