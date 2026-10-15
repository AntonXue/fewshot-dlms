"""Target lengths (Section 4.3): allocation modes, the compact prompt the length head reads, and sampling.

CPU only; no model, no tokenizer download. The canvases use `FakeAdapter`. The frozen head's distributions
and the allocations the authors drew from them are in data/fixtures/length_allocation/head_predictions_llada.json
(tools/extract_fixtures.py); tests/test_gpu_head.py checks that the ported head reproduces the distributions.
"""
import hashlib
import json

import numpy as np
import pytest
import torch

from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.lengths import ALLOC_TAG
from fewshot_dlms.lengths import L_MAX
from fewshot_dlms.lengths import LengthHead
from fewshot_dlms.lengths import allocate
from fewshot_dlms.lengths import compact_prompt
from fewshot_dlms.lengths import hashed_uniform
from fewshot_dlms.lengths import inverse_cdf
from fewshot_dlms.lengths import sample_allocations
from fewshot_dlms.lengths import section43_seed
from tests.fakes import FakeAdapter

FAMILY = load_families("list_easy_fam", "report")[0]


@pytest.fixture(scope="module")
def head_fixture(root):
    path = root / "data" / "fixtures" / "length_allocation" / "head_predictions_llada.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_allocate_oracle_returns_the_reference_lengths():
    assert allocate("oracle", [3, 5]) == [3, 5]


def test_allocate_offset_clips():
    assert allocate("offset", [1, 3], delta=-2) == [1, 1]
    assert allocate("offset", [1, 3], delta=+2) == [3, 5]
    assert allocate("offset", [255, 10], delta=+2) == [256, 12]
    assert allocate("offset", [3, 5], delta=+2, lmax=6) == [5, 6]


def test_allocate_offset_zero_is_the_oracle():
    assert allocate("offset", [4, 9, 1], delta=0) == allocate("oracle", [4, 9, 1])


def test_allocate_learned_returns_the_head_sample():
    assert allocate("learned", [3, 5], head_sample=[7, 2]) == [7, 2]


def test_allocate_learned_checks_the_head_sample():
    with pytest.raises(ValueError):
        allocate("learned", [3, 5])
    with pytest.raises(ValueError):
        allocate("learned", [3, 5], head_sample=[7])
    with pytest.raises(ValueError):
        allocate("learned", [3, 5], head_sample=[0, 2])
    with pytest.raises(ValueError):
        allocate("learned", [3, 5], head_sample=[3, 257])


def test_allocate_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        allocate("prior", [3, 5])


# ---------------------------------------------------------------------------------------------
# The compact prompt the head reads
# ---------------------------------------------------------------------------------------------


def test_compact_prompt_has_one_mask_per_site():
    adapter = FakeAdapter()
    canvas = build_canvas(adapter, FAMILY, 4, 2, [3, 5])
    compact = compact_prompt(canvas, adapter.mask_id)
    ids = compact["input_ids"]
    assert ids.count(adapter.mask_id) == 2
    assert [ids[position] for position in compact["mask_positions"]] == [adapter.mask_id] * 2
    assert len(ids) == len(canvas.ids) - (3 - 1) - (5 - 1)


def test_compact_prompt_is_the_canvas_with_one_token_per_target():
    adapter = FakeAdapter()
    for lengths in ([3, 5], [1, 1], [12, 2]):
        canvas = build_canvas(adapter, FAMILY, 4, 2, lengths)
        compact = compact_prompt(canvas, adapter.mask_id)
        one_token = build_canvas(adapter, FAMILY, 4, 2, [1, 1])
        assert compact["input_ids"] == one_token.ids
        assert compact["mask_positions"] == [site[0] for site in one_token.sites]


def test_compact_prompt_keeps_the_demonstration_outputs():
    adapter = FakeAdapter()
    canvas = build_canvas(adapter, FAMILY, 4, 2, [3, 5])
    text = adapter.decode(compact_prompt(canvas, adapter.mask_id)["input_ids"])
    for demo in FAMILY["demos"][:4]:
        assert f"Output: {demo['output']}" in text


def test_compact_prompt_rejects_a_site_without_masks():
    adapter = FakeAdapter()
    canvas = build_canvas(adapter, FAMILY, 4, 2, [3, 5])
    canvas.ids[canvas.sites[1][2]] = 65
    with pytest.raises(ValueError):
        compact_prompt(canvas, adapter.mask_id)


# ---------------------------------------------------------------------------------------------
# Sampling allocations from per-site distributions
# ---------------------------------------------------------------------------------------------


def test_hashed_uniform_is_the_authors_formula():
    key = ("lpx-alloc-v1", "report", "llada", "list_easy_fam:report:000", 2, 0, 1)
    digest = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
    assert hashed_uniform(*key) == int(digest[:16], 16) / 2**64
    assert 0.0 <= hashed_uniform(*key) < 1.0


def test_inverse_cdf_picks_the_length_whose_mass_covers_u():
    probs = [0.5, 0.25, 0.25, 0.0]
    assert inverse_cdf(probs, 0.0) == 1
    assert inverse_cdf(probs, 0.49) == 1
    assert inverse_cdf(probs, 0.5) == 2
    assert inverse_cdf(probs, 0.8) == 3
    assert inverse_cdf(probs, 0.999999) == 3


def test_inverse_cdf_normalises_and_skips_zero_mass():
    assert inverse_cdf([0.0, 2.0, 2.0], 0.0) == 2
    assert inverse_cdf([0.0, 2.0, 2.0], 0.6) == 3


def test_sample_allocations_shape_and_determinism():
    probs = np.full((3, L_MAX), 1.0 / L_MAX)
    draws = sample_allocations(probs, 5, seed=7)
    assert len(draws) == 5
    assert all(len(allocation) == 3 for allocation in draws)
    assert all(1 <= length <= L_MAX for allocation in draws for length in allocation)
    assert draws == sample_allocations(probs, 5, seed=7)
    assert draws != sample_allocations(probs, 5, seed=8)
    assert sample_allocations(probs, 8, seed=7)[:5] == draws


def test_sample_allocations_follow_the_distribution():
    probs = np.zeros((1, L_MAX))
    probs[0, 2] = 0.75  # length 3
    probs[0, 9] = 0.25  # length 10
    draws = [allocation[0] for allocation in sample_allocations(probs, 4000, seed=0)]
    assert set(draws) == {3, 10}
    assert abs(draws.count(3) / len(draws) - 0.75) < 0.03


def test_sampler_reproduces_the_frozen_allocations(head_fixture):
    """The authors' Section 4.3 allocations, redrawn from the frozen LLaDA-1.5 head's stored distributions.

    The authors sampled from the float32 predictions read back as float64, so the stored values are too.
    """
    backbone = head_fixture["head"]["backbone"]
    for record in head_fixture["canvases"]:
        probs = np.asarray(record["probs"], dtype=np.float32).astype(np.float64)
        seed = section43_seed(head_fixture["bank"], backbone, record["episode_id"], head_fixture["J"])
        assert sample_allocations(probs, len(record["allocations"]), seed) == record["allocations"]


def test_head_sample_rounds_through_float32_as_the_pipelines_do():
    """LengthHead.sample draws from the head's float64 probabilities rounded to float32 and back, the two-step path of
    the length_allocation and java_refactoring run.py. The vector puts site 0's CDF 1e-9 below candidate 0's uniform, where the rounding
    decides the length."""
    seed = section43_seed("report", "llada", FAMILY["episode_id"], 1)
    u = hashed_uniform(ALLOC_TAG, *seed, 0, 0)
    probs = np.array([[u - 1e-9, 1 - (u - 1e-9)]], dtype=np.float64)
    head = LengthHead.__new__(LengthHead)  # no model: predict returns the vector
    head.predict = lambda canvas: probs
    pipelines = sample_allocations(probs.astype(np.float32).astype(np.float64), 8, seed)
    assert head.sample(None, 8, seed) == pipelines
    assert sample_allocations(probs, 8, seed) != pipelines


def test_head_fixture_is_well_formed(head_fixture):
    assert len(head_fixture["canvases"]) == 10
    for record in head_fixture["canvases"]:
        probs = np.asarray(record["probs"], dtype=np.float64)
        assert probs.shape == (head_fixture["J"], head_fixture["head"]["lmax"])
        assert np.allclose(probs.sum(1), 1.0, atol=1e-5)
        assert len(record["oracle_L"]) == head_fixture["J"]


# ---------------------------------------------------------------------------------------------
# Loading a head checkpoint
# ---------------------------------------------------------------------------------------------


class NamedAdapter:
    """An adapter stand-in with nothing but a name: enough for the checks LengthHead.load makes first."""

    def __init__(self, name):
        self.name = name


def test_load_rejects_a_head_of_another_backbone(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"config": {"backbone": "nemotron", "lmax": L_MAX}, "branch": {}}, path)
    with pytest.raises(ValueError, match="nemotron head"):
        LengthHead.load(NamedAdapter("llada"), str(path), "cpu")


def test_load_rejects_an_adapter_without_a_head(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"config": {"backbone": "llada", "lmax": L_MAX}, "branch": {}}, path)
    with pytest.raises(ValueError, match="no length head"):
        LengthHead.load(NamedAdapter("fake"), str(path), "cpu")
