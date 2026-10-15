"""Replay (fidelity level 1 of README.md) for Section 5: the ported Java canvas and LLaDA-1.5 reproduce the stored pools.

Two stored pool records of the paper's Java run are replayed: the oracle pool of M2J2_clan_c95ee37_next (two
demonstrations, two targets, four distinct infills) and the learned-length pool of
M1J2_clan_5d146cd_createChildBackgroundInitializer (three allocations of head C, so three canvases). For every
candidate the canvas of its allocation is rebuilt with canvas.build_java_canvas and must be the stored prompt
(prompt_sha, n_tokens). The stored token ids are then teacher-forced in the batches of the paper's run (per
allocation, the distinct infills in first-draw order; P1 on the canvas, P0 with the accepted edits E masked) over
the Java runs' support (run.java_support). On the reference platform the paper's run reproduced its own
sample-time log-likelihoods exactly (max_p1_replay_absdiff = 0.0 in every record), so logp1 and logp0 must be
reproduced within that recorded value; elsewhere the one-at-a-time tolerance of tests/gpu_util.py applies.

The last test evaluates length head C (artifacts/heads/java_head_C.pt) on the two episodes and compares its
distributions with the stored predictions, and the learned allocations it implies with the stored ones.

GPU only (`make test-gpu`); needs LLaDA-1.5 and the fetched Java artifacts.
"""
import collections
import importlib.util
import sys

import numpy as np
import pytest
import torch

from fewshot_dlms.adapters import load
from fewshot_dlms.adapters.base import context_ids
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.canvas import build_java_canvas
from fewshot_dlms.data import load_java_episodes
from fewshot_dlms.lengths import compact_prompt
from tests.gpu_util import on_reference_platform
from tests.gpu_util import release
from tests.gpu_util import tolerance

pytestmark = pytest.mark.gpu

RECORDS = [
    ("M2J2_clan_c95ee37_next", "oracle"),
    ("M1J2_clan_5d146cd_createChildBackgroundInitializer", "learned"),
]
HEAD_TOLERANCE = 1e-4  # largest |p_live - p_stored| per length; the stored predictions are float32


def load_script(root, name):
    path = root / "experiments" / "java_refactoring" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"java_refactoring_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def java(artifact_root):
    path = artifact_root / "java"
    if not (path / "pools").exists():
        pytest.skip("Java artifacts not fetched: run `python -m fewshot_dlms.fetch pools java`")
    return path


@pytest.fixture(scope="module")
def adapter():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    model = load("llada", "cuda")
    yield model
    release(model)


@pytest.fixture(scope="module")
def run(root):
    return load_script(root, "run")


@pytest.fixture(scope="module")
def records(root, java):
    pools = load_script(root, "analyze").read_main_pools(java / "pools")
    return [pools[key] for key in RECORDS]


@pytest.fixture(scope="module")
def episodes():
    return {episode["episode_id"]: episode for episode in load_java_episodes()}


def by_allocation(record):
    """Stored candidates grouped by allocation, groups and members in stored order (the run's canvases)."""
    groups = collections.OrderedDict()
    for candidate in record["candidates"]:
        groups.setdefault(tuple(candidate["allocation"]), []).append(candidate)
    return groups


def test_canvases_are_the_frozen_prompts(adapter, run, records, episodes):
    for record in records:
        episode = episodes[record["episode_id"]]
        for allocation, candidates in by_allocation(record).items():
            canvas = build_java_canvas(adapter, episode, episode["M"], episode["J"], list(allocation))
            for candidate in candidates:
                assert run.prompt_sha(canvas.ids) == candidate["prompt_sha"], (record["episode_id"], allocation)
                assert len(canvas.ids) == candidate["n_tokens"]


def test_replay_in_the_frozen_batches_reproduces_stored_loglik(adapter, run, records, episodes, record_property):
    allowed = run.java_support(adapter)
    for record in records:
        limit = record["max_p1_replay_absdiff"] if on_reference_platform() else tolerance("llada")
        episode = episodes[record["episode_id"]]
        worst = 0.0
        for allocation, candidates in by_allocation(record).items():
            canvas = build_java_canvas(adapter, episode, episode["M"], episode["J"], list(allocation))
            order = reveal_order(canvas)
            distinct = list(dict.fromkeys(tuple(candidate["toks"]) for candidate in candidates))
            p1 = context_ids(adapter, canvas, hide_E=False)
            p0 = context_ids(adapter, canvas, hide_E=True)
            _, p1_terms = reveal(adapter, [p1], [order], [0] * len(distinct), allowed, forced=distinct, chunk=run.CHUNK)
            _, p0_terms = reveal(adapter, [p0], [order], [0] * len(distinct), allowed, forced=distinct, chunk=run.CHUNK)
            ell1 = {toks: float(sum(terms)) for toks, terms in zip(distinct, p1_terms)}
            ell0 = {toks: float(sum(terms)) for toks, terms in zip(distinct, p0_terms)}
            for candidate in candidates:
                toks = tuple(candidate["toks"])
                worst = max(worst, abs(ell1[toks] - candidate["logp1"]), abs(ell0[toks] - candidate["logp0"]))
        record_property(f"replay_max_abs_diff_{record['regime']}", worst)
        assert worst <= limit, (record["episode_id"], worst, limit)


def test_head_c_predictions_match_the_stored_ones(adapter, run, records, episodes, artifact_root, java, record_property):
    """Head C on the two episodes' compact prompts: the stored distributions, and the stored learned allocations."""
    head_path = artifact_root / "heads" / "java_head_C.pt"
    if not head_path.exists():
        pytest.skip("Java head C not fetched: run `python -m fewshot_dlms.fetch heads`")
    pytest.importorskip("flash_attn")
    from fewshot_dlms.lengths import LengthHead

    stored = run.read_head_probs(java / "length" / run.HEAD_PROBS_FILE)
    head = LengthHead.load(adapter, head_path, "cuda")
    selected = [episodes[record["episode_id"]] for record in records]
    examples = []
    for episode in selected:
        canvas = build_java_canvas(adapter, episode, episode["M"], episode["J"], [1] * episode["J"])
        examples.append(compact_prompt(canvas, adapter.mask_id))
    predictions = head.predict_examples(examples)
    live = {}
    worst = 0.0
    for episode, prediction in zip(selected, predictions):
        for j in range(episode["J"]):
            key = f"{episode['episode_id']}|{episode['J']}|{j}"
            live[key] = prediction[j].astype(np.float32).astype(np.float64)
            worst = max(worst, float(np.abs(live[key] - stored[key]).max()))
    record_property("head_max_abs_diff", worst)
    assert worst <= HEAD_TOLERANCE
    learned = records[1]
    allocations, rejections = run.learned_allocations(episodes[learned["episode_id"]], live, learned["K"])
    by_draw = {candidate["draw"]: candidate["allocation"] for candidate in learned["candidates"]}
    assert [list(allocation) for allocation in allocations] == [by_draw[k] for k in range(learned["K"])]
    assert rejections == learned["allocation_rejections"]
