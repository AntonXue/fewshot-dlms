# ported from editdiff/experiments/multimodel/mm_common.py, experiments/sec42_mj/mj_worker.py, experiments/sec42_mj/mj_build.py::extend, experiments/example_tilting/et_common.py, for build_ar_prompt experiments/multimodel/mm_common.py::compile_ar with experiments/preflight/pf_common.py::{compile_ar_perhole,AR_PERHOLE_NOTE} and experiments/algo_investigation/dr_worker.py::make_aep, and, for build_java_canvas, experiments/swe_explore/sx_common.py::{site_parts,render} with the Section 5 package's scratch/fr_lib.py @ d276fa6
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

`build_ar_prompt` and `ar_prompt_ids` build the prompt of the two causal interfaces of Figure 4 (Appendix
B.1): Nemotron's fixed-length causal suffix and Qwen3-8B's free-length continuation. The same records as
the canvas, demonstrations first, end at the target's "Output:"; the model continues from there. P0
deletes the demonstration outputs from the prompt instead of masking them.

`build_java_canvas` builds the canvas of Section 5 (Appendix C) from a Java episode of data/java: the
helper the developer introduced, then the demonstrations (accepted before/after edits of other sites that
call the helper), then the target sites with a mask span in place of the new code.
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


# ---------------------------------------------------------------------------------------------
# The single-target prompt of a causal model (Figure 4)
# ---------------------------------------------------------------------------------------------

# The per-hole instruction note (placeholder_note=True), with the hole number filled in twice.
AR_PERHOLE_NOTE = (
    " Some records have their output replaced by a numbered placeholder [[k]]."
    " Write the output for placeholder [[%d]] only, as '[[%d]]: <output>'."
)


def _ar_pieces(adapter, family, M, j, placeholder_note):
    """The single-target prompt as (text, is_demonstration_output) pieces, in prompt order.

    Each piece is tokenized on its own (`ar_prompt_ids`). Raises ValueError unless 1 <= M <= the number of
    demonstrations and 1 <= j <= the number of targets.
    """
    if not 1 <= M <= len(family["demos"]):
        raise ValueError(f"M must be between 1 and {len(family['demos'])}, got {M}")
    if not 1 <= j <= len(family["targets"]):
        raise ValueError(f"j must be between 1 and {len(family['targets'])}, got {j}")

    target = family["targets"][j - 1]
    pieces = []
    if placeholder_note:
        opening = adapter.chat_user_open() + family["instruction"] + AR_PERHOLE_NOTE % (1, 1) + "\n\n"
        pieces.append((opening, False))
    else:
        pieces.append((adapter.chat_header(family["instruction"]), False))
    for demo in family["demos"][:M]:
        pieces.append((f"Input: {demo['input']}\nOutput:", False))
        pieces.append((" " + demo["output"], True))
        pieces.append(("\n\n", False))
    pieces.append((f"Input: {target['input']}\nOutput:", False))
    if placeholder_note:
        pieces.append((" [[1]]", False))
        pieces.append((adapter.chat_user_close_assistant_open(), False))
        pieces.append(("[[1]]:", False))
    return pieces


def build_ar_prompt(adapter, family, M, j, placeholder_note=False):
    """The prompt text a causal model continues for target j of a family, and the reference answer.

    The prompt holds the first M demonstrations and then target Tj (j = 1 in Figure 4), as the canvas holds
    them, and stops where the target's answer would start:

      chat header + instruction
      for each demonstration: "Input: <input>\nOutput:" " <output>" "\n\n"
      "Input: <target input>\nOutput:"

    Nemotron's causal suffix (`nemotron_ar`) continues it with exactly as many tokens as " <reference>" has
    (the oracle length); Qwen3-8B (`qwen`) continues it freely until a stop token. Every record sits in the
    assistant turn, which stays open.

    placeholder_note=True gives the per-hole wording of the authors' preflight instead: the instruction and the
    records in the user turn, the instruction told that outputs replaced by a numbered placeholder [[k]] are to be
    written as '[[1]]: <output>', the target's output shown as "[[1]]", and the assistant turn opened with the
    forced "[[1]]:". It is not used for any figure of the paper (no autoregressive multi-target baseline was run).

    Returns (prompt text, [reference answer of target j]). The text is for reading and checking; the model gets
    `ar_prompt_ids`, which tokenizes the pieces one by one (a real tokenizer can merge across piece boundaries).
    `adapter` supplies `chat_header`, `chat_user_open`, `chat_user_close_assistant_open`. Raises ValueError
    unless 1 <= M <= the number of demonstrations and 1 <= j <= the number of targets.
    """
    pieces = _ar_pieces(adapter, family, M, j, placeholder_note)
    text = "".join(piece for piece, _ in pieces)
    return text, [family["targets"][j - 1]["output"]]


def ar_prompt_ids(adapter, family, M, j, placeholder_note=False, hide_E=False):
    """The token ids of `build_ar_prompt`: P1's prompt, or P0's with hide_E=True.

    Every piece of text is tokenized on its own and the token lists are concatenated, as for the canvas.
    hide_E=True deletes each demonstration's " <output>" tokens and keeps everything else, the "Output:"
    labels and the blank lines included: P0 of a causal model (Appendix B.1).
    """
    ids = []
    for piece, is_demonstration_output in _ar_pieces(adapter, family, M, j, placeholder_note):
        if hide_E and is_demonstration_output:
            continue
        ids.extend(adapter.encode(piece))
    return ids


# ---------------------------------------------------------------------------------------------
# Java helper propagation (Section 5)
# ---------------------------------------------------------------------------------------------

# The instruction of every canvas of the main run, and the neutral instruction of the zero-demonstration
# ablation (Appendix C), which is the same for M = 0, 1 and 2 so that the arms differ only in their demonstrations.
JAVA_INSTRUCTION = (
    "Apply the refactoring demonstrated in the example to every target site. "
    "Each target's <new> block must be completed with the transformed code."
)
JAVA_ZERO_DEMO_INSTRUCTION = (
    "Complete each target <new> span by applying the intended refactoring using the helper shown in the "
    "source context; use any accepted before/after edits if provided."
)

# A site shows the lines of its enclosing method around the changed span: at most 20 before it and 12 after
# it. Longer context is cut at the far end and marked with a "// ..." line.
JAVA_CONTEXT_BEFORE = 20
JAVA_CONTEXT_AFTER = 12
JAVA_CUT_MARKER = "    // ..."


def _java_context(site):
    """The method lines before and after the changed span of a site, as text that surrounds the span.

    Returns (head, tail). `head` is the lines before the span, each followed by a newline, and `tail` is the
    lines after it, each preceded by a newline; either is "" when there is no such line. The lines are the
    same before and after the commit (data/java records `method_before` only; tools/export_java.py checks it).
    """
    lines = site["method_before"].split("\n")
    head_lines = lines[: site["i1"]]
    tail_lines = lines[site["i2"] :]
    if len(head_lines) > JAVA_CONTEXT_BEFORE:
        head_lines = [JAVA_CUT_MARKER] + head_lines[-JAVA_CONTEXT_BEFORE:]
    if len(tail_lines) > JAVA_CONTEXT_AFTER:
        tail_lines = tail_lines[:JAVA_CONTEXT_AFTER] + [JAVA_CUT_MARKER]
    head = "\n".join(head_lines)
    tail = "\n".join(tail_lines)
    return (head + "\n" if head else ""), ("\n" + tail if tail else "")


def build_java_canvas(adapter, episode, M, J, lengths, instruction=JAVA_INSTRUCTION):
    """Compile a Java helper-propagation episode into a Canvas with `lengths[j]` mask tokens at target j.

    `episode` is a record of `data.load_java_episodes()`. The first `M` of its demonstrations and the first
    `J` of its targets are used (M = 0 gives the canvas of the zero-demonstration ablation). `adapter`
    supplies `encode`, `mask_id`, `chat_header(instruction)` and `chat_end()`. Every piece of text is
    tokenized on its own and the token lists are concatenated, so positions are exact by construction:

      chat header + instruction
      "// <repository> <commit[:10]> <file>\n// helper, already present in the file:\n<helper>\n\n"
      for each demonstration:
          "// --- <File.java> : <method> ---\n<old>\n" + context + old span + context + "\n</old>\n<new>\n"
          context + new span (their positions are collected in E) + context + "\n</new>\n\n"
      for each target:
          the same block with `lengths[j]` mask tokens in place of the new span (the positions of sites[j])
      chat end

    The demonstrations are the accepted before/after edits of the paper, so E marks the accepted edits
    that P0 hides. The reference answers (`episode["reference"]`) are returned in `gold` and never enter
    `ids`. `lengths` is the number of answer tokens given to each target: the paper's oracle setting passes
    `episode["oracle_lengths"]`, the learned setting passes the lengths drawn from the length head.
    Raises ValueError unless 0 <= M <= len(demos), 1 <= J <= len(targets) and there is one positive length
    per target.
    """
    demos = episode["demos"]
    targets = episode["targets_visible"]
    if not 0 <= M <= len(demos):
        raise ValueError(f"M must be between 0 and {len(demos)}, got {M}")
    if not 1 <= J <= len(targets):
        raise ValueError(f"J must be between 1 and {len(targets)}, got {J}")
    if len(lengths) != J:
        raise ValueError(f"expected {J} target lengths (one per target), got {len(lengths)}")
    if any(length < 1 for length in lengths):
        raise ValueError(f"every target needs at least one mask token, got lengths {list(lengths)}")

    file_name = episode["file"].split("/")[-1]
    header = f"{episode['repository']} {episode['commit'][:10]} {episode['file']}"

    ids = list(adapter.encode(adapter.chat_header(instruction)))
    helper_block = f"// {header}\n// helper, already present in the file:\n{episode['helper_text'].rstrip()}\n\n"
    ids.extend(adapter.encode(helper_block))

    E = []
    for demo in demos[:M]:
        head, tail = _java_context(demo)
        ids.extend(adapter.encode(f"// --- {file_name} : {demo['method']} ---\n<old>\n" + head))
        ids.extend(adapter.encode(demo["span_old"]))
        ids.extend(adapter.encode(tail + "\n</old>\n<new>\n"))
        ids.extend(adapter.encode(head))
        output_ids = adapter.encode(demo["span_new"])
        E.extend(range(len(ids), len(ids) + len(output_ids)))
        ids.extend(output_ids)
        ids.extend(adapter.encode(tail + "\n</new>\n\n"))

    sites = []
    for j, target in enumerate(targets[:J]):
        head, tail = _java_context(target)
        ids.extend(adapter.encode(f"// --- {file_name} : {target['method']} ---\n<old>\n" + head))
        ids.extend(adapter.encode(target["span_old"]))
        ids.extend(adapter.encode(tail + "\n</old>\n<new>\n"))
        ids.extend(adapter.encode(head))
        length = int(lengths[j])
        sites.append(list(range(len(ids), len(ids) + length)))
        ids.extend([adapter.mask_id] * length)
        ids.extend(adapter.encode(tail + "\n</new>\n\n"))
    ids.extend(adapter.encode(adapter.chat_end()))

    gold = [reference["span_new"] for reference in episode["reference"]["targets"][:J]]
    meta = {
        "task": "java_helper",
        "episode_id": episode["episode_id"],
        "M": M,
        "J": J,
        "lengths": [int(length) for length in lengths],
        "placement": "demos_first",
        "model": getattr(adapter, "name", None),
        "slots": [f"D{i}" for i in range(1, M + 1)] + [f"T{j}" for j in range(1, J + 1)],
    }
    return Canvas(ids=ids, sites=sites, E=E, gold=gold, meta=meta)
