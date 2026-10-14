"""Replay (fidelity level 1 of README.md): the ported adapters reproduce the authors' stored log-likelihoods.

data/fixtures/fig5/<model>_M4_J2.jsonl holds two Figure 5 pool records per model (List Functions-Easy,
M = 4 demonstrations, J = 2 targets, K = 16 candidates). Each candidate stores its token ids in reveal
order and the two log-likelihoods of the paper: ell1 = log P1 (demonstration outputs visible) and
ell0 = log P0 (demonstration outputs masked at E). The tests rebuild each canvas with this repository's
canvas.py and the model's tokenizer, check that it is the frozen prompt token for token, and score the
stored tokens with the fixed-order loop of Lemma 3.1 (adapters/base.py):

  - one candidate at a time with logprob_fixed_order, which agrees up to bf16 batch noise;
  - with the frozen run's own batches, which on the authors' platform is exact for LLaDA-1.5 and
    Nemotron (tests/gpu_util.py explains both tolerances).

GPU only (`make test-gpu`); needs the three pinned checkpoints (see README.md).
"""
import hashlib
import json

import pytest
import torch

from fewshot_dlms.adapters import load
from fewshot_dlms.adapters.base import context_ids
from fewshot_dlms.adapters.base import logprob_fixed_order
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.data import read_jsonl
from tests.gpu_util import batched_tolerance
from tests.gpu_util import release
from tests.gpu_util import tolerance

pytestmark = pytest.mark.gpu

POOL = {
    "llada": "fig5/llada_M4_J2.jsonl",
    "nemotron_pt": "fig5/nemotron_pt_M4_J2.jsonl",
    "dqwen": "fig5/dqwen_M4_J2.jsonl",
}
N_RECORDS = 2  # pool records (families) per model
N_CANDIDATES = 4  # candidates scored one at a time per record


@pytest.fixture(scope="module", params=list(POOL))
def adapter(request):
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


def assert_close(replayed, stored, limit, where):
    assert abs(replayed - stored) <= limit, (where, replayed, stored, limit)


def test_canvas_is_the_frozen_prompt(adapter, families, root):
    for record in records_of(root, adapter.name):
        canvas = canvas_of(adapter, families, record)
        assert sha256_of(context_ids(adapter, canvas, hide_E=False)) == record["prompt_sha256_p1"]
        assert sha256_of(context_ids(adapter, canvas, hide_E=True)) == record["prompt_sha256_p0"]
        assert reveal_order(canvas) == record["T_positions"]
        assert len(canvas.E) == record["n_demo_output_tokens_hidden"]


def test_replay_reproduces_stored_loglik(adapter, families, root):
    limit = tolerance(adapter.name)
    for record in records_of(root, adapter.name):
        canvas = canvas_of(adapter, families, record)
        for candidate in record["candidates"][:N_CANDIDATES]:
            ell1, _ = logprob_fixed_order(adapter, canvas, candidate["toks"], hide_E=False)
            ell0, _ = logprob_fixed_order(adapter, canvas, candidate["toks"], hide_E=True)
            where = (adapter.name, record["episode_id"], "one at a time")
            assert_close(ell1, candidate["ell1"], limit, where)
            assert_close(ell0, candidate["ell0"], limit, where)
        if record["block_start"] is not None:
            assert adapter.block_start == record["block_start"]


def test_replay_with_the_frozen_batches(adapter, families, root):
    """Score in the batches of the frozen run: exact on the authors' platform for LLaDA and Nemotron.

    The frozen run sampled P1 with all K draws in one reveal, so each forward pass held the distinct
    prefixes of the draws; replaying the distinct draws together forms the same batches. It scored P0 for
    the distinct draws and the reference answer together, and P1 of the reference answer on its own.
    """
    name = adapter.name
    limit = batched_tolerance(name)
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
