"""Tests for data/java: the 127 helper-propagation episodes of Section 5 and the Java canvas built from them.

The episodes are the paper's evaluation population ("manifest B"): 127 episodes from 50 commits of
14 public Java repositories, in four cells of (M, J) in {1, 2} x {1, 2}. Everything here runs on CPU.
`test_java_canvas_reproduces_frozen_prompt_hashes` additionally needs the LLaDA-1.5 tokenizer in the
local Hugging Face cache; it is skipped when the tokenizer is not there.
"""
import collections
import csv
import hashlib
import json
import re

import pytest

from fewshot_dlms.canvas import build_java_canvas
from fewshot_dlms.data import JAVA_DIR
from fewshot_dlms.data import load_java_episodes
from tests.fakes import FakeAdapter

EPISODES = load_java_episodes()
BY_ID = {episode["episode_id"]: episode for episode in EPISODES}

# Episodes per cell and commits per cell of manifest B (Appendix C, population flow).
CELL_EPISODES = {"M1J1": 52, "M1J2": 28, "M2J1": 28, "M2J2": 19}
CELL_COMMITS = {"M1J1": 50, "M1J2": 26, "M2J1": 26, "M2J2": 17}

# The rules of the G4 population: replacement <= 80 LLaDA tokens, enclosing method <= 200 lines,
# and an oracle canvas that fits the 2,048-token context.
MAX_REPLACEMENT_TOKENS = 80
MAX_METHOD_LINES = 200
MAX_CANVAS_TOKENS = 2048

# Patterns that must not match any file under data/java (lower-cased): absolute paths of the
# authors' machines, run metadata that names a machine, the private source tree, and the
# vocabulary of the authors' internal working documents.
PRIVATE_PATTERNS = [
    r"/(?:hdd|ssd)\d",
    r"/home/",
    r"/mnt/",
    r'"host":',
    r'"gpu_env":',
    r"editdiff",
    r"scratch/",
    r"hand-?\s?off",
    r"sign-?off",
]

LLADA_REVISION = "84346fd91ba60252d260022201ad6fc5a3468fb2"


def read_csv(name):
    """The rows of data/java/<name> as dictionaries."""
    with (JAVA_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sites_of(episode):
    """Every site an episode shows the model: its demonstrations, then its targets."""
    return episode["demos"] + episode["targets_visible"]


def test_127_episodes_50_commits_14_repos():
    assert len(EPISODES) == 127
    assert len({episode["episode_id"] for episode in EPISODES}) == 127
    assert len({episode["commit"] for episode in EPISODES}) == 50
    assert len({episode["repository"] for episode in EPISODES}) == 14


def test_cells_have_the_frozen_population_sizes():
    episodes_per_cell = collections.Counter(episode["cell"] for episode in EPISODES)
    assert dict(episodes_per_cell) == CELL_EPISODES
    for cell, expected in CELL_COMMITS.items():
        commits = {episode["commit"] for episode in EPISODES if episode["cell"] == cell}
        assert len(commits) == expected


def test_every_episode_is_admitted_and_cell_matches_M_and_J():
    for episode in EPISODES:
        assert episode["in_manifest_B"] is True
        assert episode["cell"] == f"M{episode['M']}J{episode['J']}"
        assert len(episode["demos"]) == episode["M"]
        assert len(episode["targets_visible"]) == episode["J"]
        assert len(episode["reference"]["targets"]) == episode["J"]
        assert len(episode["oracle_lengths"]) == episode["J"]


def test_records_carry_no_run_metadata():
    for episode in EPISODES:
        for key in ("host", "gpu_env", "wall_time"):
            assert key not in episode


def test_cell_ledger_lists_all_158_candidates_and_marks_the_127():
    rows = read_csv("episode_cells.csv")
    assert len(rows) == 158
    admitted = {row["episode_id"] for row in rows if row["in_manifest_B"] == "True"}
    assert admitted == set(BY_ID)


def test_every_dropped_episode_has_a_reason_in_the_exclusion_ledger():
    rows = read_csv("exclusion_ledger.csv")
    dropped_ids = {row["episode_id"] for row in rows if row["manifest"] == "B"}
    candidates = {row["episode_id"] for row in read_csv("episode_cells.csv")}
    assert dropped_ids == candidates - set(BY_ID)
    assert len(dropped_ids) == 31
    assert all(row["reason"].strip() for row in rows)


def test_reference_validity_records_a_verdict_for_every_candidate():
    rows = read_csv("reference_validity.csv")
    assert len(rows) == 158
    verdicts = collections.Counter(row["adjudication_verdict"] for row in rows)
    assert verdicts["exclude"] == 10
    excluded = {row["episode_id"] for row in rows if row["adjudication_verdict"] == "exclude"}
    assert not excluded & set(BY_ID)


def test_source_groups_cover_every_episode():
    groups = {row["group_id"] for row in read_csv("source_groups.csv")}
    assert len(groups) == 338
    assert {episode["source_group_id"] for episode in EPISODES} <= groups


def test_helper_text_matches_its_recorded_hash():
    for episode in EPISODES:
        digest = hashlib.sha256(episode["helper_text"].encode("utf-8")).hexdigest()[:16]
        assert digest == episode["helper_text_sha"], episode["episode_id"]


def test_episodes_obey_the_g4_rules():
    for episode in EPISODES:
        assert max(episode["oracle_lengths"]) <= MAX_REPLACEMENT_TOKENS, episode["episode_id"]
        assert episode["oracle_canvas_tokens"] <= MAX_CANVAS_TOKENS, episode["episode_id"]
        for site in sites_of(episode):
            n_lines_before = len(site["method_before"].split("\n"))
            n_lines_after = n_lines_before - (site["i2"] - site["i1"]) + (site["j2"] - site["j1"])
            assert n_lines_after <= MAX_METHOD_LINES, (episode["episode_id"], site["site_id"])


def test_site_spans_sit_where_the_coordinates_say():
    for episode in EPISODES:
        for site in sites_of(episode):
            lines = site["method_before"].split("\n")
            assert "\n".join(lines[site["i1"] : site["i2"]]) == site["span_old"]
        for demo in episode["demos"]:
            assert len(demo["span_new"].split("\n")) == demo["j2"] - demo["j1"]
        references = episode["reference"]["targets"]
        for site, reference in zip(episode["targets_visible"], references, strict=True):
            assert reference["site_id"] == site["site_id"]
            assert len(reference["span_new"].split("\n")) == site["j2"] - site["j1"]


def test_load_java_episodes_returns_independent_copies():
    first = load_java_episodes()
    first[0]["episode_id"] = "changed"
    assert load_java_episodes()[0]["episode_id"] != "changed"


def test_java_canvas_masks_targets_and_keeps_helper():
    adapter = FakeAdapter()
    episode = next(e for e in EPISODES if e["M"] == 2 and e["J"] == 2)
    canvas = build_java_canvas(adapter, episode, 2, 2, episode["oracle_lengths"])
    text = adapter.decode(canvas.ids)

    assert len(canvas.sites) == episode["J"]
    assert [len(site) for site in canvas.sites] == episode["oracle_lengths"]
    for site in canvas.sites:
        assert all(canvas.ids[position] == adapter.mask_id for position in site)
    assert not {p for site in canvas.sites for p in site} & set(canvas.E)

    assert episode["helper_text"].rstrip() in text
    for demo in episode["demos"]:
        assert demo["span_new"] in text
    accepted = adapter.decode([canvas.ids[position] for position in canvas.E])
    assert accepted == "".join(demo["span_new"] for demo in episode["demos"])

    references = [reference["span_new"] for reference in episode["reference"]["targets"]]
    assert canvas.gold == references
    visible = episode["gold_visible_in_helper_or_demo"] or episode["literal_copy_of_demo"]
    if not visible:
        for reference in references:
            assert reference not in text


def test_no_reference_is_in_the_canvas_unless_the_record_says_it_is_visible():
    adapter = FakeAdapter()
    for episode in EPISODES:
        canvas = build_java_canvas(adapter, episode, episode["M"], episode["J"], episode["oracle_lengths"])
        text = adapter.decode(canvas.ids)
        for reference in episode["reference"]["targets"]:
            if reference["span_new"] in text:
                assert episode["gold_visible_in_helper_or_demo"] or episode["literal_copy_of_demo"], episode["episode_id"]


def test_canvas_does_not_depend_on_the_reference():
    adapter = FakeAdapter()
    episode = next(e for e in EPISODES if e["J"] == 1)
    changed = json.loads(json.dumps(episode))
    changed["reference"]["targets"][0]["span_new"] = "        throw new Error();"
    original = build_java_canvas(adapter, episode, episode["M"], episode["J"], [3])
    other = build_java_canvas(adapter, changed, changed["M"], changed["J"], [3])
    assert original.ids == other.ids
    assert original.gold != other.gold


def test_java_data_directory_has_no_private_strings():
    files = sorted(path for path in JAVA_DIR.iterdir() if path.is_file())
    assert {path.name for path in files} >= {
        "episodes.jsonl",
        "episode_cells.csv",
        "exclusion_ledger.csv",
        "reference_validity.csv",
        "source_groups.csv",
        "PROVENANCE.md",
        "LICENSES.md",
    }
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        for pattern in PRIVATE_PATTERNS:
            assert re.search(pattern, text) is None, f"{pattern!r} matches {path.name}"


def test_licenses_name_every_repository_and_commit():
    text = (JAVA_DIR / "LICENSES.md").read_text(encoding="utf-8")
    commits_of = collections.defaultdict(set)
    for episode in EPISODES:
        commits_of[episode["repository"]].add(episode["commit"])
    for repository, commits in commits_of.items():
        assert repository in text
        for commit in commits:
            assert commit[:10] in text, (repository, commit)


# ---------------------------------------------------------------------------------------------
# Token-level fidelity to the authors' renderer
# ---------------------------------------------------------------------------------------------


class LladaTokenizerAdapter:
    """The tokenizer half of the LLaDA-1.5 adapter: the same chat template, token ids and mask id."""

    name = "llada"
    mask_id = 126336

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def chat_header(self, instruction):
        messages = [{"role": "user", "content": instruction}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def chat_end(self):
        return "<|eot_id|>"


def prompt_sha(ids):
    """The 12-hex-digit hash the authors recorded for each rendered prompt."""
    return hashlib.sha1(json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()[:12]


@pytest.fixture(scope="module")
def llada_adapter():
    """The LLaDA-1.5 tokenizer from the local Hugging Face cache; the test is skipped when it is not there."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        huggingface_hub.snapshot_download("GSAI-ML/LLaDA-1.5", revision=LLADA_REVISION, local_files_only=True)
    except Exception as error:
        pytest.skip(f"LLaDA-1.5 is not in the local Hugging Face cache: {type(error).__name__}")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-1.5",
        revision=LLADA_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    return LladaTokenizerAdapter(tokenizer)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_java_canvas_reproduces_frozen_prompt_hashes(llada_adapter):
    """Both prompts the authors hashed per episode: oracle lengths, and one mask per target."""
    for episode in EPISODES:
        M, J = episode["M"], episode["J"]
        oracle = build_java_canvas(llada_adapter, episode, M, J, episode["oracle_lengths"])
        assert prompt_sha(oracle.ids) == episode["prompt_sha_oracle"], episode["episode_id"]
        assert len(oracle.ids) == episode["oracle_canvas_tokens"], episode["episode_id"]
        compact = build_java_canvas(llada_adapter, episode, M, J, [1] * J)
        assert prompt_sha(compact.ids) == episode["prompt_sha_compact"], episode["episode_id"]
        assert len(compact.ids) == episode["compact_tokens"], episode["episode_id"]
