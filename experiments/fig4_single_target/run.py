# ported from editdiff/experiments/algo_investigation/dr_worker.py::{seed_of, run_canvas, run_ar_oracle, free_steps, run_qwen} and experiments/gold_reward/gr_worker.py::{gold_canvas, gold_ar, gold_qwen, stop_ids} @ d276fa6
"""Sample and score the pools of Figure 4 (Section 4.1) and of the cross-benchmark study (Fig. 9, both builds) on a
GPU: one process, one model, one GPU.

For every family and cell, K answers are sampled from P1 and scored under P0, as in the paper:

  Section 4.1 (default): the first M demonstrations, then the target T1 (J = 1), M in {1, 2, 4, 8, 16}.
    llada, dqwen   canvas: the target gets its oracle length (token count of " <answer>"); the K infills are drawn by
                   fixed-order decoding (Lemma 3.1), all K in one reveal, 32 canvases per forward pass; P0 masks the
                   demonstration outputs in place and scores the distinct infills together.
    nemotron_ar    Nemotron's causal suffix: K continuations of exactly the oracle length, left to right at temperature 1
                   (32 sequences per forward pass); P0 deletes the demonstration outputs from the prompt and scores the
                   distinct continuations together (16 per pass).
    qwen           Qwen3-8B, free length: each continuation stops at its first token that holds a newline or at
                   <|im_end|> (200-token budget), sampled over the ordinary support plus <|im_end|>; P0 as above (32 per pass).
  --crossbench: the cross-benchmark study, M = 4 and J in {1, 2, 4, 8} targets placed in each family's frozen 12-slot
    master layout (canvas.master_layout), for the three canvas models llada, nemotron_pt (causal prefix, bidirectional
    tail from the first target) and dqwen.

The seed of a family's K answers is the first 16 hex digits of SHA-256("algo_investigation|proposal|<model>|<episode>|
<M>|<J>") mod 2^31; the cross-benchmark cells append "|('placement', 'master')" (`proposal_seed`). One torch.Generator
with this seed serves all K draws. The batches above are the authors': bf16 logits depend on the batch they are computed
in (tests/gpu_util.py), so these are what reproduce the stored values.

--score-gold also scores the reference answer under P1 and P0 (Figure 4's gold row): canvases teacher-force it in the
same reveal order; the causal suffix scores its oracle-length tokens; Qwen3-8B scores its tokens and then the probability
of stopping right after them (any pure terminator: <|im_end|> or a whitespace token with a newline), so its T counts the
stop. On the first --audit-n families the reference is scored together with the first two distinct sampled answers,
whose rescored log-likelihoods are kept as a score-equivalence check (gold_equiv), as the authors' gold scoring did.

Records go to <out>/<bank>/<model>/<task>_M{M}_J{J}[_il].jsonl, one JSON line per family, appended as they finish.
Every record carries the protocol_hash of its cell (cell_config: model, revision, bank, task, M, J, K, chunk, rules; a
function of that cell alone). A rerun with the same protocol skips the families already in the file, so an interrupted
run resumes; a pool file that holds records of another protocol (another K, chunk, revision, ...) is refused with exit
status 2, naming the file, before the model is loaded, so that analyze.py never reads two protocols as one pool.
<out>/<bank>/<model>/manifest_<hash>.json lists every cell's configuration and hash, the seeds and the library
versions. A family that fails is reported and counted; the run exits with status 1 if any did. analyze.py reads the
pools.

    CUDA_VISIBLE_DEVICES=0 python experiments/fig4_single_target/run.py --model llada --tasks list_easy_fam --M 4 \\
        --K 64 --families 8 --score-gold --out artifacts/reruns/fig4
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers

from fewshot_dlms.adapters import adapter_class
from fewshot_dlms.adapters import load
from fewshot_dlms.adapters.base import context_ids
from fewshot_dlms.adapters.base import decode_ar
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.adapters.base import sample_ar
from fewshot_dlms.adapters.base import sample_fixed_order
from fewshot_dlms.adapters.base import site_sums
from fewshot_dlms.canvas import active_episode
from fewshot_dlms.canvas import ar_prompt_ids
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import BENCHMARKS
from fewshot_dlms.data import load_families

# How each model is run (Appendix B, Experimental Setup), and which adapter it loads.
INTERFACE = {
    "llada": "canvas",
    "dqwen": "canvas",
    "nemotron_pt": "canvas",
    "nemotron_ar": "causal_suffix",
    "qwen": "free_length",
}
MODELS41 = ["llada", "nemotron_ar", "dqwen", "qwen"]
MODELS42 = ["llada", "nemotron_pt", "dqwen"]
M_GRID = [1, 2, 4, 8, 16]
J_GRID = [1, 2, 4, 8]
CROSSBENCH_M = 4
K = 64
CHUNK = 32  # canvases per forward pass while sampling and scoring, as in the authors' run
FREE_MAX_NEW = 200  # Qwen3-8B's token budget
EQUIV_N = 2  # distinct sampled answers rescored with the reference answer (score-equivalence check)

# The record names of the interfaces (the `interface` field of the authors' records).
INTERFACE_NAME = {
    "canvas": "{model}_canvas_oracle_length",
    "causal_suffix": "nemotron_causal_suffix_oracle_length",
    "free_length": "qwen3_free_length_stop_at_newline_or_im_end",
}


def proposal_seed(model, episode_id, M, J, master):
    """The seed of a family's K answers (see the module doc). `master` adds the master-layout placement to the key.
    Parts are joined with "|" as str() writes them."""
    parts = ["algo_investigation", "proposal", model, episode_id, M, J]
    if master:
        parts.append(("placement", "master"))
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2 ** 31)


def oracle_lengths(adapter, family, J):
    """The token count of each reference answer " <output>" of targets T1..TJ under the model's tokenizer."""
    return [len(adapter.encode(" " + target["output"])) for target in family["targets"][:J]]


def first_distinct(traces, n):
    """The first n distinct token lists of `traces`, in order of first occurrence."""
    return list({tuple(trace): None for trace in traces})[:n]


# ---------------------------------------------------------------------------------------------
# Canvas models (llada, dqwen, nemotron_pt)
# ---------------------------------------------------------------------------------------------


def run_canvas(adapter, model, family, M, J, master, n_candidates, chunk, audit, score_gold):
    """One family's pool on a canvas: sample K infills from P1, score the distinct ones under P0; optionally the reference.

    `model` is the name in the seed and the records (the adapter's name for the three canvas models)."""
    placement = "interleaved" if master else "demos_first"
    t0 = time.time()
    lengths = oracle_lengths(adapter, family, J)
    canvas = build_canvas(adapter, family, M, J, lengths, placement=placement)
    order = reveal_order(canvas)
    p1 = context_ids(adapter, canvas, hide_E=False)
    p0 = context_ids(adapter, canvas, hide_E=True)
    allowed = adapter.support("ordinary")

    seed = proposal_seed(model, family["episode_id"], M, J, master)
    infills = sample_fixed_order(adapter, canvas, n_candidates, seed, support=allowed, chunk=chunk)
    traces = [[token for site in infill.toks for token in site] for infill in infills]
    seconds_gen = time.time() - t0

    t1 = time.time()
    distinct = list({tuple(trace): None for trace in traces})
    _, p0_terms = reveal(adapter, [p0], [order], [0] * len(distinct), allowed, forced=distinct, chunk=chunk)
    ell0_terms = dict(zip(distinct, p0_terms))
    seconds_score = time.time() - t1

    checks = None
    if audit:
        _, replay = reveal(adapter, [p1], [order], [0] * len(distinct), allowed, forced=distinct, chunk=chunk)
        ell1_by_infill = {tuple(trace): infill.ell1_sample for trace, infill in zip(traces, infills)}
        E = set(canvas.E)
        checks = {
            "replay_P1_max_abs_seq": max(abs(sum(terms) - ell1_by_infill[infill]) for infill, terms in zip(distinct, replay)),
            "P0_differs_only_on_demo_output_spans": all(p0[i] == adapter.mask_id for i in E) and all(p1[i] == p0[i] for i in range(len(p0)) if i not in E),
            "targets_masked_in_P1_and_P0": all(p1[i] == adapter.mask_id and p0[i] == adapter.mask_id for i in order),
            "allocation_equals_oracle_L": [len(site) for site in canvas.sites] == lengths,
            "n_demo_output_tokens_hidden": len(E),
            "block_start": getattr(adapter, "block_start", None),
            "first_target_position": order[0],
            "p1_prompt_text": adapter.decode(p1),
            "p0_prompt_text": adapter.decode(p0),
        }

    candidates = []
    for trace, infill in zip(traces, infills):
        terms0 = ell0_terms[tuple(trace)]
        candidates.append({
            "toks": trace,
            "texts": [adapter.decode(site) for site in infill.toks],
            "ell1": infill.ell1_sample,
            "ell0": float(sum(terms0)),
            "ell1_t": site_sums(canvas, infill.ell1_tokens),
            "ell0_t": site_sums(canvas, terms0),
            "truncated": False,
        })
    body = {
        "interface": INTERFACE_NAME["canvas"].format(model=model),
        "n_tokens": len(canvas.ids),
        "oracle_L": lengths,
        "alloc_L": lengths,
        "candidates": candidates,
        "n_unique": len(distinct),
        "checks": checks,
        "seconds_gen": seconds_gen,
        "seconds_score": seconds_score,
        "seed": seed,
        "record_labels": canvas.meta["slots"],
        "block_start": getattr(adapter, "block_start", None),
    }
    if score_gold:
        body.update(gold_canvas(adapter, canvas, p1, p0, order, allowed, chunk, first_distinct(traces, EQUIV_N) if audit else [], candidates))
    return body


def gold_canvas(adapter, canvas, p1, p0, order, allowed, chunk, saved, candidates):
    """The reference infill (the tokens of " <answer>" of every target, in reveal order) teacher-forced under P1 and P0,
    in one reveal with the `saved` sampled infills, whose rescored values are kept as gold_equiv."""
    gold = [token for answer in canvas.gold for token in adapter.encode(" " + answer)]
    if len(gold) != len(order):
        raise ValueError("the oracle allocation must equal the tokenization of the reference answers")
    forced = [tuple(gold)] + saved
    _, terms1 = reveal(adapter, [p1], [order], [0] * len(forced), allowed, forced=forced, chunk=chunk)
    _, terms0 = reveal(adapter, [p0], [order], [0] * len(forced), allowed, forced=forced, chunk=chunk)
    by_toks = {tuple(candidate["toks"]): candidate for candidate in candidates}
    equiv = []
    for toks, rescored1, rescored0 in zip(saved, terms1[1:], terms0[1:]):
        equiv.append({
            "ell1_saved": by_toks[toks]["ell1"],
            "ell1_rescored": float(sum(rescored1)),
            "ell0_saved": by_toks[toks]["ell0"],
            "ell0_rescored": float(sum(rescored0)),
            "T_saved": len(toks),
        })
    return {
        "gold_toks": gold,
        "gold_T": len(gold),
        "gold_L": [len(site) for site in canvas.sites],
        "gold_ell1": float(sum(terms1[0])),
        "gold_ell0": float(sum(terms0[0])),
        "gold_ell1_t": site_sums(canvas, terms1[0]),
        "gold_ell0_t": site_sums(canvas, terms0[0]),
        "gold_equiv": equiv,
    }


# ---------------------------------------------------------------------------------------------
# Causal models (nemotron_ar, qwen), single target
# ---------------------------------------------------------------------------------------------


def run_causal(adapter, model, family, M, n_candidates, audit, score_gold):
    """One family's pool of a causal interface (J = 1): K continuations of P1's prompt, the distinct ones scored under
    P0's prompt (demonstration outputs deleted), in the frozen run's batches."""
    interface = INTERFACE[model]
    t0 = time.time()
    p1 = ar_prompt_ids(adapter, family, M, 1)
    p0 = ar_prompt_ids(adapter, family, M, 1, hide_E=True)
    length = oracle_lengths(adapter, family, 1)[0]
    support = adapter.default_support
    sample_chunk = adapter.sample_chunk
    p0_chunk = adapter.p0_chunk
    if interface == "free_length":
        max_new = FREE_MAX_NEW
        stop_ids = adapter.stop_ids()
    else:
        max_new = length
        stop_ids = ()

    seed = proposal_seed(model, family["episode_id"], M, 1, False)
    draws = sample_ar(adapter, p1, n_candidates, seed, max_new, stop_ids, support=support, chunk=sample_chunk)
    traces = [infill.toks[0] for infill in draws]
    seconds_gen = time.time() - t0

    t1 = time.time()
    distinct = list({tuple(trace): None for trace in traces})
    _, p0_terms, _ = decode_ar(adapter, p0, len(distinct), support, forced=distinct, chunk=p0_chunk)
    ell0 = {toks: float(sum(terms)) for toks, terms in zip(distinct, p0_terms)}
    seconds_score = time.time() - t1

    checks = None
    if audit:
        replay_chunk = sample_chunk if interface == "free_length" else p0_chunk
        _, replay, _ = decode_ar(adapter, p1, len(distinct), support, forced=distinct, chunk=replay_chunk)
        ell1_by_draw = {tuple(infill.toks[0]): infill.ell1_sample for infill in draws}
        checks = {
            "replay_P1_max_abs_seq": max(abs(sum(terms) - ell1_by_draw[toks]) for toks, terms in zip(distinct, replay)),
            "P0_prefix_shorter_by_tokens": len(p1) - len(p0),
            "p1_prompt_text": adapter.decode(p1),
            "p0_prompt_text": adapter.decode(p0),
        }

    candidates = []
    for infill in draws:
        toks = infill.toks[0]
        candidates.append({
            "toks": toks,
            "texts": [adapter.decode(toks)],
            "ell1": infill.ell1_sample,
            "ell0": ell0[tuple(toks)],
            "ell1_t": [infill.ell1_sample],
            "ell0_t": [ell0[tuple(toks)]],
            "truncated": bool(infill.truncated),
        })
    body = {
        "interface": INTERFACE_NAME[interface],
        "n_tokens": len(p1) if interface == "free_length" else len(p1) + length,
        "oracle_L": [length],
        "candidates": candidates,
        "n_unique": len(distinct),
        "checks": checks,
        "seconds_gen": seconds_gen,
        "seconds_score": seconds_score,
        "seed": seed,
    }
    if interface == "free_length":
        body["max_new"] = FREE_MAX_NEW
    if score_gold:
        saved = first_distinct(traces, EQUIV_N) if audit else []
        if interface == "free_length":
            body.update(gold_free_length(adapter, family, p1, p0, saved, candidates))
        else:
            body.update(gold_causal_suffix(adapter, family, p1, p0, saved, candidates))
    return body


def equivalence(saved, terms1, terms0, candidates):
    """Rescored vs sampled log-likelihoods of the saved answers (the score-equivalence check)."""
    by_toks = {tuple(candidate["toks"]): candidate for candidate in candidates}
    rows = []
    for toks, rescored1, rescored0 in zip(saved, terms1, terms0):
        rows.append({
            "ell1_saved": by_toks[toks]["ell1"],
            "ell1_rescored": float(sum(rescored1)),
            "ell0_saved": by_toks[toks]["ell0"],
            "ell0_rescored": float(sum(rescored0)),
            "T_saved": len(toks),
        })
    return rows


def gold_causal_suffix(adapter, family, p1, p0, saved, candidates):
    """Nemotron's causal suffix: the reference tokens teacher-forced under P1's and P0's prompt, with the saved answers."""
    gold = adapter.encode(" " + family["targets"][0]["output"])
    forced = [tuple(gold)] + saved
    _, terms1, _ = decode_ar(adapter, p1, len(forced), forced=forced, chunk=CHUNK)
    _, terms0, _ = decode_ar(adapter, p0, len(forced), forced=forced, chunk=CHUNK)
    return {
        "gold_toks": gold,
        "gold_T": len(gold),
        "gold_L": [len(gold)],
        "gold_ell1": float(sum(terms1[0])),
        "gold_ell0": float(sum(terms0[0])),
        "gold_ell1_t": [float(sum(terms1[0]))],
        "gold_ell0_t": [float(sum(terms0[0]))],
        "gold_equiv": equivalence(saved, terms1[1:], terms0[1:], candidates),
    }


def pure_terminators(adapter):
    """The tokens after which a free-length answer is exactly the reference text: <|im_end|> and the tokens whose text
    is only whitespace holding a newline (the sampler also stops on a token such as "]\\n", which would add text)."""
    stops = {adapter.end_of_turn_id}
    for token in range(len(adapter.tok)):
        text = adapter.decode([token])
        if "\n" in text and text.strip() == "":
            stops.add(token)
    return sorted(stops)


def gold_free_length(adapter, family, p1, p0, saved, candidates):
    """Qwen3-8B: log P of the reference tokens, plus log P of stopping right after them (logsumexp over the pure
    terminators, at temperature 1 over the free support), under P1's and P0's prompt. T = reference tokens + 1."""
    gold = adapter.encode(" " + family["targets"][0]["output"])
    allowed = adapter.support(adapter.default_support)
    stops = torch.tensor(pure_terminators(adapter))
    parts = {}
    for name, prompt in (("1", p1), ("0", p0)):
        _, terms, _ = decode_ar(adapter, prompt, 1, forced=[gold])
        sequence = list(prompt) + list(gold)
        logits = adapter.counted_logits(torch.tensor([sequence], dtype=torch.long), [[len(sequence)]])[0, 0].float()
        logprobs = torch.log_softmax(logits.masked_fill(~allowed.to(logits.device), float("-inf")), -1)
        parts[name] = (float(sum(terms[0])), float(torch.logsumexp(logprobs[stops.to(logprobs.device)], 0)))
    equiv = []
    if saved:
        _, saved1, _ = decode_ar(adapter, p1, len(saved), forced=saved)
        _, saved0, _ = decode_ar(adapter, p0, len(saved), forced=saved)
        equiv = equivalence(saved, saved1, saved0, candidates)
    ell1 = parts["1"][0] + parts["1"][1]
    ell0 = parts["0"][0] + parts["0"][1]
    return {
        "gold_toks": gold,
        "gold_T": len(gold) + 1,
        "gold_L": [len(gold) + 1],
        "gold_ell1": ell1,
        "gold_ell0": ell0,
        "gold_ell1_t": [ell1],
        "gold_ell0_t": [ell0],
        "gold_ell1_content": parts["1"][0],
        "gold_ell0_content": parts["0"][0],
        "gold_logp_stop1": parts["1"][1],
        "gold_logp_stop0": parts["0"][1],
        "gold_n_stop_ids": len(stops),
        "gold_equiv": equiv,
    }


# ---------------------------------------------------------------------------------------------
# Configuration, protocol hash, manifest
# ---------------------------------------------------------------------------------------------


def seed_rule(master):
    """The seed rule of a cell, as text (proposal_seed)."""
    key = "algo_investigation|proposal|<model>|<episode_id>|<M>|<J>"
    if master:
        key += "|('placement', 'master')"
    return f"sha256('{key}')[:16] mod 2**31"


def cell_config(model, model_id, revision, bank, task, M, J, master, n_candidates, chunk, score_gold):
    """Everything that decides the records of one cell, and nothing else (not which other cells or families were
    requested, where, or on which device): its hash is the `protocol_hash` of the cell's records."""
    interface = INTERFACE[model]
    config = {
        "study": "fig4_single_target" if not master else "crossbench",
        "model": model,
        "interface": interface,
        "model_id": model_id,
        "revision": revision,
        "bank": bank,
        "task": task,
        "M": M,
        "J": J,
        "K": n_candidates,
        "placement": "master layout (meta.master_placement, 12 slots)" if master else "demonstrations first, target last",
        "temperature": 1.0,
        "seed_rule": seed_rule(master),
        "score_gold": bool(score_gold),
    }
    if interface == "canvas":
        config["lengths"] = "oracle: token count of ' ' + answer under the model's tokenizer"
        config["support"] = "ordinary"
        config["reveal_order"] = "ascending canvas position, one token per step"
        config["chunk"] = {"sample": chunk, "p0": chunk, "gold": chunk}
    elif interface == "causal_suffix":
        config["lengths"] = "oracle"
        config["support"] = "ordinary"
        config["chunk"] = {"sample": 32, "p0": 16, "gold": CHUNK}
    else:
        config["lengths"] = "free: stop at the first token with a newline or <|im_end|>"
        config["max_new"] = FREE_MAX_NEW
        config["support"] = "free (ordinary + <|im_end|>)"
        config["chunk"] = {"sample": 32, "p0": 32}
    return config


def protocol_hash(config):
    """First 16 hex digits of SHA-256 of the configuration as sorted JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def write_manifest(directory, cells, plan, model):
    """The run's manifest: every cell's configuration, hash, families and seeds; library and GPU versions; the command."""
    digest = hashlib.sha256(json.dumps(sorted(cell["protocol_hash"] for cell in cells.values())).encode()).hexdigest()[:16]
    seeds = {}
    for _, families, M, J, name in plan:
        config = cells[name]["config"]
        master = config["study"] == "crossbench"
        seeds[name] = {family["episode_id"]: proposal_seed(model, family["episode_id"], M, J, master) for family in families}
    manifest = {
        "cells": cells,
        "seeds": seeds,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "command": sys.argv,
    }
    path = directory / f"manifest_{digest}.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


class ProtocolMismatch(Exception):
    """A pool file holds records of another protocol than the run that would append to it."""


def done_episodes(path, digest):
    """Episode ids already in a pool file, which a run of protocol `digest` resumes after.

    Every record of the file must carry protocol_hash == digest; otherwise ProtocolMismatch names the file and the
    hashes found. Lines that do not parse (an interrupted write) are ignored, so their family is sampled again.
    """
    if not path.exists():
        return set()
    done = set()
    others = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("protocol_hash") == digest:
                done.add(record["episode_id"])
            else:
                others.add(str(record.get("protocol_hash")))
    if others:
        raise ProtocolMismatch(
            f"{path} holds records of protocol {', '.join(sorted(others))}, but this run is protocol {digest} "
            "(another K, chunk, revision or code). Write to another --out, or move the file away to start the pool afresh."
        )
    return done


def append_record(path, record):
    """Append one JSON line in a single write, so that a killed run leaves complete lines only. If the file ends in a
    cut-off line (no final newline), the write starts with a newline, so the record never joins that line."""
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    size = os.fstat(fd).st_size
    lead = b"\n" if size and os.pread(fd, 1, size - 1) != b"\n" else b""
    os.write(fd, lead + (json.dumps(record) + "\n").encode())
    os.close(fd)


def int_list(text):
    return [int(value) for value in text.split(",")]


def task_list(text):
    tasks = text.split(",")
    for task in tasks:
        if task not in BENCHMARKS:
            raise argparse.ArgumentTypeError(f"unknown task {task!r}; expected some of {','.join(BENCHMARKS)}")
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(INTERFACE))
    parser.add_argument("--tasks", type=task_list, default=list(BENCHMARKS), help="e.g. sst2_fam,list_easy_fam")
    parser.add_argument("--M", type=int_list, default=None, help="demonstration counts (default 1,2,4,8,16; --crossbench: 4)")
    parser.add_argument("--J", type=int_list, default=J_GRID, help="--crossbench: target counts, e.g. 1,2,4,8")
    parser.add_argument("--K", type=int, default=K, help="answers sampled per family")
    parser.add_argument("--bank", default="report", choices=["dev", "report"])
    parser.add_argument("--families", type=int, default=None, help="only the first N families of the bank")
    parser.add_argument("--crossbench", action="store_true", help="the cross-benchmark study (M = 4, master layout, J targets)")
    parser.add_argument("--score-gold", action="store_true", help="also score the reference answers under P1 and P0")
    parser.add_argument("--out", default="artifacts/reruns/fig4")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="canvases per forward pass (canvas models)")
    parser.add_argument("--audit-n", type=int, default=2, help="first families of the bank that also run the audit checks")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.crossbench and args.model not in MODELS42:
        parser.error(f"--crossbench runs the canvas models {', '.join(MODELS42)}")
    if not args.crossbench and args.model not in MODELS41:
        parser.error(f"Section 4.1 runs {', '.join(MODELS41)} (nemotron_pt is the --crossbench interface)")
    Ms = args.M or ([CROSSBENCH_M] if args.crossbench else M_GRID)
    Js = args.J if args.crossbench else [1]

    adapter_cls, _ = adapter_class(args.model)
    out = Path(args.out) / args.bank / args.model

    # Every cell's pool, checked before the model is loaded: resume a pool of the same protocol only.
    plan = []
    cells = {}
    done_of = {}
    for task in args.tasks:
        families = load_families(task, args.bank)[: args.families]
        for M in Ms:
            for J in Js:
                config = cell_config(
                    args.model, adapter_cls.model_id, adapter_cls.revision, args.bank, task, M, J, args.crossbench, args.K, args.chunk, args.score_gold
                )
                name = f"{task}_M{M}_J{J}" + ("_il" if args.crossbench else "")
                cells[name] = {"config": config, "protocol_hash": protocol_hash(config)}
                plan.append((task, families, M, J, name))
                try:
                    done_of[name] = done_episodes(out / f"{name}.jsonl", cells[name]["protocol_hash"])
                except ProtocolMismatch as error:
                    print(f"refusing to append: {error}", file=sys.stderr)
                    sys.exit(2)

    started = time.time()
    adapter = load(args.model, args.device)
    print(f"[fig4] {args.model} {adapter.model_id} @ {adapter.revision} loaded in {time.time() - started:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(out, cells, plan, args.model)
    print(f"[fig4] manifest {manifest}", flush=True)

    failures = []
    for task, families, M, J, name in plan:
        path = out / f"{name}.jsonl"
        digest = cells[name]["protocol_hash"]
        done = done_of[name]
        written = 0
        cell_start = time.time()
        for index, family in enumerate(families):
            if family["episode_id"] in done:
                continue
            M_used = min(M, family["max_M"], len(family["demos"]))
            audit = index < args.audit_n
            try:
                if INTERFACE[args.model] == "canvas":
                    body = run_canvas(adapter, args.model, family, M_used, J, args.crossbench, args.K, args.chunk, audit, args.score_gold)
                else:
                    body = run_causal(adapter, args.model, family, M_used, args.K, audit, args.score_gold)
            except Exception as error:  # one failed family must not stop the cell; it is counted and retried on the next run
                print(f"FAILED {family['episode_id']} {name}: {error}\n{traceback.format_exc()[-1500:]}", flush=True)
                failures.append((name, family["episode_id"]))
                torch.cuda.empty_cache()
                continue
            episode = active_episode(family, M_used, J, "interleaved" if args.crossbench else "demos_first")
            record = {
                "episode_id": family["episode_id"],
                "source": task,
                "rule_id": family["rule_id"],
                "category": family.get("category"),
                "model": args.model,
                "model_name": adapter.model_id,
                "revision": adapter.revision,
                "M_requested": M,
                "M": M_used,
                "J": J,
                "K": args.K,
                "K_base": args.K,
                "placement": "master" if args.crossbench else "demos_first",
                "p": 1.0,
                "tau": 1.0,
                "target_offset": 0,
                "beta_convention": "zero_is_ordinary",
                "protocol_hash": digest,
                "gold": [target["output"] for target in episode["targets"]],
                "target_inputs": [target["input"] for target in episode["targets"]],
                "grader": family["grader"],
                "status": "scored",
                **body,
            }
            append_record(path, record)
            written += 1
            gold_note = f" | gold R {body['gold_ell1'] - body['gold_ell0']:+.2f}" if "gold_ell1" in body else ""
            replay_note = f" replay {body['checks']['replay_P1_max_abs_seq']:.1e}" if body["checks"] else ""
            print(
                f"[{args.model}] {name} {family['episode_id']} tokens={body['n_tokens']} L={body['oracle_L']} distinct={body['n_unique']}"
                f" gen {body['seconds_gen']:.1f}s score {body['seconds_score']:.1f}s{replay_note}{gold_note}",
                flush=True,
            )
        print(f"CELL_DONE {args.model} {name} new={written} {time.time() - cell_start:.0f}s", flush=True)
    if failures:
        print(f"RUN_FAILED {args.model}: {len(failures)} families or cells failed: {failures}", flush=True)
        sys.exit(1)
    print(f"RUN_DONE {args.model} {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
