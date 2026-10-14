"""The ported length head reproduces the authors' frozen LLaDA-1.5 head (Section 4.3, Appendix B, Length Allocation).

data/fixtures/fig6/head_predictions_llada.json holds, for the first ten report families of List
Functions-Easy at M = 4 demonstrations and J = 2 targets, the frozen head's distributions over the lengths
1..256 at both target sites, as the authors' evaluation computed them (tools/extract_fixtures.py). The tests
rebuild each canvas with canvas.py at its oracle lengths, check that the compact prompt the head reads is the
frozen one token for token (through its SHA-256), and run the ported head (lengths.LengthHead on the
LLaDA-1.5 adapter) on it:

  - one canvas per forward pass (LengthHead.predict), and
  - all ten compact prompts packed into one stream (LengthHead.predict_examples), which must not leak
    attention across prompts;
  - and LengthHead.sample with the Section 4.3 seed, which must redraw the authors' allocations.

Each site's distribution must lie within total-variation distance TV_LIMIT of the stored one. The frozen
evaluation packed 32 prompts per forward pass; the head pads a short pack with a filler segment so that the GPU
runs the same bf16 kernels (LengthHead.predict_examples). On the authors' platform the distributions then
agree to about 1e-6; without the filler a single prompt would differ by up to 0.07.

GPU only (`make test-gpu`); needs flash-attn, the pinned LLaDA-1.5 checkpoint (README.md) and the head
checkpoint at artifacts/heads/llada.pt (the authors' final.pt of run fin_llada_m70_100k, re-saved without the run
directory pickled in its training config; tensors identical, SHA-256 in the fixture's head.checkpoint_sha256).
"""
import hashlib
import json

import numpy as np
import pytest
import torch

from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.lengths import LengthHead
from fewshot_dlms.lengths import compact_prompt
from fewshot_dlms.lengths import section43_seed
from tests.gpu_util import on_reference_platform
from tests.gpu_util import release

pytestmark = pytest.mark.gpu

TV_LIMIT = 1e-3  # largest total-variation distance accepted between a ported and a stored site distribution
# Off the authors' platform, the least share of redrawn allocations that must equal the authors'. A draw can change
# only when its uniform falls within the total-variation distance of a CDF step, so with J = 2 sites under
# TV_LIMIT at most 2 * TV_LIMIT = 0.2 % of the draws are expected to change; on the authors' platform none do.
MIN_SAME_ALLOCATIONS = 0.99
HEAD = "heads/llada.pt"


def total_variation(p, q):
    """Total-variation distance between the rows of two [J, lmax] probability arrays: one value per site."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(1)


def sha256_of(ids):
    """The prompt hash of the authors' runs: SHA-256 of the JSON list of token ids."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


@pytest.fixture(scope="module")
def fixture(root):
    path = root / "data" / "fixtures" / "fig6" / "head_predictions_llada.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def head_path(artifact_root):
    path = artifact_root / HEAD
    if not path.exists():
        pytest.skip(f"no head checkpoint at artifacts/{HEAD}")
    return path


@pytest.fixture(scope="module")
def head(head_path):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    head = LengthHead.load("llada", str(head_path), "cuda")
    yield head
    adapter = head.adapter
    vars(head).clear()
    release(adapter)


@pytest.fixture(scope="module")
def canvases(head, fixture):
    families = {family["episode_id"]: family for family in load_families(fixture["task"], fixture["bank"])}
    out = []
    for record in fixture["canvases"]:
        family = families[record["episode_id"]]
        out.append(build_canvas(head.adapter, family, fixture["M"], fixture["J"], record["oracle_L"]))
    return out


def test_checkpoint_is_the_frozen_head(head_path, fixture):
    """The published head, or the authors' original final.pt it was re-saved from (the same tensors)."""
    digest = hashlib.sha256()
    with open(head_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    assert digest.hexdigest() in (fixture["head"]["checkpoint_sha256"], fixture["head"]["trained_checkpoint_sha256"])


def test_head_reads_the_frozen_compact_prompts(head, fixture, canvases):
    assert head.backbone == fixture["head"]["backbone"]
    assert head.lmax == fixture["head"]["lmax"]
    for record, canvas in zip(fixture["canvases"], canvases):
        compact = compact_prompt(canvas, head.adapter.mask_id)
        assert sha256_of(compact["input_ids"]) == record["compact_sha256"]
        assert compact["mask_positions"] == record["mask_positions"]


def test_head_predictions_match_frozen(head, fixture, canvases, record_property):
    largest = 0.0
    for record, canvas in zip(fixture["canvases"], canvases):
        predicted = head.predict(canvas)
        stored = np.asarray(record["probs"], dtype=np.float64)
        assert predicted.shape == stored.shape
        distances = total_variation(predicted, stored)
        largest = max(largest, float(distances.max()))
        assert distances.max() <= TV_LIMIT, (record["episode_id"], distances.tolist())
    record_property("max_tv_one_at_a_time", largest)


def test_packed_predictions_match_frozen(head, fixture, canvases, record_property):
    examples = [compact_prompt(canvas, head.adapter.mask_id) for canvas in canvases]
    predictions = head.predict_examples(examples)
    largest = 0.0
    for record, predicted in zip(fixture["canvases"], predictions):
        distances = total_variation(predicted, record["probs"])
        largest = max(largest, float(distances.max()))
        assert distances.max() <= TV_LIMIT, (record["episode_id"], distances.tolist())
    record_property("max_tv_packed", largest)


def test_sample_redraws_the_frozen_allocations(head, fixture, canvases, record_property):
    same = 0
    total = 0
    for record, canvas in zip(fixture["canvases"], canvases):
        seed = section43_seed(fixture["bank"], head.backbone, record["episode_id"], fixture["J"])
        draws = head.sample(canvas, len(record["allocations"]), seed)
        same += sum(draw == stored for draw, stored in zip(draws, record["allocations"]))
        total += len(record["allocations"])
    record_property("same_allocations", f"{same}/{total}")
    if on_reference_platform():
        assert same == total
    else:
        assert same >= MIN_SAME_ALLOCATIONS * total
