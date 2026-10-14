# ported from editdiff/experiments/multimodel/mm_common.py, experiments/sec42_mj/mj_worker.py, experiments/sec42_mj/mj_build.py::extend and experiments/example_tilting/et_common.py @ d276fa6
"""From a frozen family to the canvas a diffusion language model fills in.

A family (see data.py) holds an instruction, 16 ordered demonstrations D1..D16 and 8 ordered targets
T1..T8. An episode of the paper uses the first M demonstrations and the first J targets. The canvas
puts all of them into one token sequence, and the model fills the J targets in one pass:

    <chat header + instruction>
    Input: <demonstration input>
    Output: <demonstration output>
    Input: <target input>
    Output: <mask> <mask> ... <mask>        <- one mask per token of the target's answer
    ...                                     <- demonstrations and targets are interleaved
    <end of the assistant turn>

Three things on a Canvas carry the paper's quantities:

  sites  the mask positions of each target. The model's infill is a token string for every site.
  E      the demonstration-output positions. log P1 is the log-likelihood of an infill with the
         canvas as is; log P0 is the same with the tokens at E replaced by the mask token, so that
         the demonstrations show their inputs but not their outputs. R = log P1 - log P0 is the
         example reward of the paper (see scoring.py).
  gold   the reference answers. They are for grading and never appear in `ids`.

Where the records sit: `master_layout`. This module builds the Figure 5 canvas ("layout A" of the
authors' code, one chat turn, J >= 1 targets) for tasks whose outputs are text: List Functions-Easy
(checked against the authors' run), SALT and MIT Movie. The classification canvas of SST-2 (opaque
output codes) and the one-target canvas of Figure 4 are added with their figures.
"""
from dataclasses import dataclass
import hashlib
import random

# A family has 16 demonstrations and 8 targets. The published layout (meta.master_placement) places only
# D1..D4 and T1..T8 on 12 slots; the M x J grid extends it to all 16 demonstrations on 24 slots.
N_DEMOS = 16
N_TARGETS = 8
BASE_DEMOS = 4
BASE_SLOTS = BASE_DEMOS + N_TARGETS  # 12
N_NEW_DEMOS = N_DEMOS - BASE_DEMOS  # 12: D5..D16
N_SLOTS = N_DEMOS + N_TARGETS  # 24

# Seed of the extension to 16 demonstrations that the M x J grid (Figure 5) uses.
GRID_PLACEMENT_SEED = "mtfsp-20261011-mj-placement-v1"

# "interleaved": records in the order of the master layout (the paper's setting).
# "demos_first": all demonstrations, then all targets. "demos_after": all targets, then all demonstrations.
PLACEMENTS = ("interleaved", "demos_first", "demos_after")


@dataclass
class Canvas:
    """One episode as the model sees it. All positions index into `ids`."""

    ids: list[int]  # token ids; the mask id at every target position
    sites: list[list[int]]  # per target j: its positions (contiguous), in prompt order
    E: list[int]  # demonstration-output positions (masked for P0)
    gold: list[str]  # reference answers, one per target, in the order of `sites` (never in ids)
    meta: dict  # task, episode_id, M, J, lengths, placement, model, slots


# ---------------------------------------------------------------------------------------------
# Which records, in which order
# ---------------------------------------------------------------------------------------------


def _seed_from(*parts):
    """A 64-bit integer from the first 16 hex digits of the SHA-256 of the parts joined by '|'."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _is_published_layout(layout):
    """True for a 12-slot layout that starts with D1 and holds each of D1..D4 and T1..T8 once."""
    expected = [f"D{i}" for i in range(1, BASE_DEMOS + 1)] + [f"T{j}" for j in range(1, N_TARGETS + 1)]
    return layout[:1] == ["D1"] and sorted(layout) == sorted(expected)


def extend_layout(base_layout, function_id):
    """The 24-slot master layout of the M x J grid, from the published 12-slot layout.

    The published layout (D1..D4 and T1..T8 on 12 slots, first slot D1) is extended by the demonstrations
    D5..D16. They take 12 of the slots 1..23, drawn with a generator seeded from the function id; the
    original 12 records fill the remaining slots in their original relative order. Hence deleting D5..D16
    from the result gives back the published layout, and a cell (M, J) of the grid is the 24-slot layout
    with D(M+1)..D16 and T(J+1)..T8 removed and nothing else moved.
    """
    if not _is_published_layout(base_layout):
        raise ValueError(f"not a published {BASE_SLOTS}-slot layout: {base_layout}")

    rng = random.Random(_seed_from(GRID_PLACEMENT_SEED, function_id))
    new_demo_slots = set(rng.sample(range(1, N_SLOTS), N_NEW_DEMOS))
    base_records = iter(base_layout)
    last_demo = BASE_DEMOS
    layout = []
    for slot in range(N_SLOTS):
        if slot in new_demo_slots:
            last_demo += 1
            layout.append(f"D{last_demo}")
        else:
            layout.append(next(base_records))
    return layout


def master_layout(family, M):
    """The slot labels ('D1', 'T1', ...) of a family in prompt order, before removing unused records.

    With M <= 4 this is the published 12-slot layout stored in the family (meta.master_placement). With
    M > 4 it is the 24-slot extension of the M x J grid, which exists for List Functions-Easy only. The
    two agree on D1..D4 and T1..T8, so every cell with M <= 4 is the same prompt in either layout.
    """
    base_layout = family["meta"]["master_placement"]
    if M <= BASE_DEMOS:
        return list(base_layout)

    function_id = family["meta"].get("function_id")
    if function_id is None:
        raise ValueError(
            f"{family['episode_id']}: no frozen layout for more than {BASE_DEMOS} demonstrations "
            "(only List Functions-Easy has one)"
        )
    return extend_layout(base_layout, function_id)


def active_episode(family, M, J, placement="interleaved"):
    """The records an episode uses and the order they appear in.

    Takes the first M demonstrations and the first J targets of the family, so cells are nested: the
    demonstrations of M = 4 are the first four of M = 8. Returns

      demos, targets  the used records ({id, input, output}), in family order (D1..DM, T1..TJ)
      entries         the same records in prompt order; each entry adds a "slot" ('D3', 'T1', ...)
                      and a "role" ('demo' or 'target')

    `placement` is "interleaved" (the master layout; the paper's setting), "demos_first" or
    "demos_after". Targets keep the order T1..TJ in every placement.
    """
    if not 1 <= M <= len(family["demos"]):
        raise ValueError(f"M must be between 1 and {len(family['demos'])}, got {M}")
    if not 1 <= J <= len(family["targets"]):
        raise ValueError(f"J must be between 1 and {len(family['targets'])}, got {J}")
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}; expected one of {PLACEMENTS}")

    demos = family["demos"][:M]
    targets = family["targets"][:J]
    demo_slots = [f"D{i}" for i in range(1, M + 1)]
    target_slots = [f"T{j}" for j in range(1, J + 1)]

    if placement == "interleaved":
        used = set(demo_slots + target_slots)
        order = [slot for slot in master_layout(family, M) if slot in used]
    elif placement == "demos_first":
        order = demo_slots + target_slots
    else:
        order = target_slots + demo_slots

    records = dict(zip(demo_slots + target_slots, demos + targets))
    entries = []
    for slot in order:
        role = "demo" if slot.startswith("D") else "target"
        entries.append({**records[slot], "slot": slot, "role": role})
    return {
        "episode_id": family["episode_id"],
        "M": M,
        "J": J,
        "placement": placement,
        "demos": demos,
        "targets": targets,
        "entries": entries,
    }


# ---------------------------------------------------------------------------------------------
# The token sequence
# ---------------------------------------------------------------------------------------------


def build_canvas(adapter, family, M, J, lengths, placement="interleaved"):
    """Compile an episode into a Canvas with `lengths[j]` mask tokens at target j.

    `adapter` supplies the tokenizer and the chat framing: `encode(text)`, `mask_id` and the methods
    `chat_header(instruction)` and `chat_end()`. Every piece of text is tokenized on its own and the
    token lists are concatenated, so positions are exact by construction:

      chat header + instruction
      for each record in prompt order:
          "Input: <input>\\nOutput:"
          demonstration -> " <output>" tokens (their positions are collected in E)
          target        -> lengths[j] mask tokens (their positions form sites[j])
          "\\n\\n"
      chat end

    `lengths` is the number of answer tokens given to each target, in the order T1..TJ. The paper's
    oracle setting passes the token count of each reference answer, tokenized as " <output>"; Section 4.3
    passes predicted lengths. Raises ValueError unless there is one positive length per target.
    """
    if len(lengths) != J:
        raise ValueError(f"expected {J} target lengths (one per target), got {len(lengths)}")
    if any(length < 1 for length in lengths):
        raise ValueError(f"every target needs at least one mask token, got lengths {list(lengths)}")

    episode = active_episode(family, M, J, placement)

    ids = list(adapter.encode(adapter.chat_header(family["instruction"])))
    sites = []
    E = []
    gold = []
    target_index = 0  # which target comes next; targets appear in the order T1..TJ
    for entry in episode["entries"]:
        ids.extend(adapter.encode(f"Input: {entry['input']}\nOutput:"))
        if entry["role"] == "demo":
            output_ids = adapter.encode(" " + entry["output"])
            E.extend(range(len(ids), len(ids) + len(output_ids)))
            ids.extend(output_ids)
        else:
            length = int(lengths[target_index])
            sites.append(list(range(len(ids), len(ids) + length)))
            ids.extend([adapter.mask_id] * length)
            gold.append(entry["output"])
            target_index += 1
        ids.extend(adapter.encode("\n\n"))
    ids.extend(adapter.encode(adapter.chat_end()))

    meta = {
        "task": family["source"],
        "episode_id": family["episode_id"],
        "M": M,
        "J": J,
        "lengths": [int(length) for length in lengths],
        "placement": placement,
        "model": getattr(adapter, "name", None),
        "slots": [entry["slot"] for entry in episode["entries"]],
    }
    return Canvas(ids=ids, sites=sites, E=E, gold=gold, meta=meta)
