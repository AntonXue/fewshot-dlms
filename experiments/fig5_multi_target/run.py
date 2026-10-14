# ported from editdiff/experiments/sec42_mj/mj_worker.py @ d276fa6
"""Sample and score the pools of Figure 5 (Section 4.2) on a GPU: one process, one model, one GPU.

For every family and cell (M demonstrations, J targets) of List Functions-Easy, the canvas has the oracle
length of every target (the token count of " <answer>" under the model's tokenizer). Then, as in the paper:

  1. Sample K infills from P1 by fixed-order decoding (Lemma 3.1), all K in one reveal: one torch
     Generator seeded with proposal_seed(model, episode, M, J). ell1 = log P1 of each infill, summed
     while sampling, and its per-target parts ell1_t.
  2. Score the distinct infills under P0 (demonstration outputs masked, canvas.E), together with the
     reference infill (gold) in the same reveal: ell0, ell0_t and gold_ell0.
  3. Score the reference infill alone under P1: gold_ell1.

Scoring in these batches is what makes the stored ell1 / ell0 reproducible: bf16 logits depend on the
batch they are computed in (tests/gpu_util.py). On the authors' platform this reproduces the frozen
LLaDA-1.5 and Nemotron pools exactly; dQwen3.5 varies from process to process (autotuned kernels).

Records go to <out>/<model>/list_easy_fam_M{M}_J{J}.jsonl, one JSON line per family, appended as they
finish (a family already in the file is skipped, so an interrupted run resumes). <out>/<model>/manifest_<hash>.json
records the configuration, its hash, the seeds and the library versions. analyze.py reads the pools.

    python experiments/fig5_multi_target/run.py --model llada --M 4 --J 2 --K 16 --families 8 --out artifacts/reruns/fig5
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

from fewshot_dlms.adapters import load
from fewshot_dlms.adapters.base import context_ids
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.adapters.base import sample_fixed_order
from fewshot_dlms.adapters.base import site_sums
from fewshot_dlms.adapters.base import split_by_site
from fewshot_dlms.canvas import GRID_PLACEMENT_SEED
from fewshot_dlms.canvas import active_episode
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families

TASK = "list_easy_fam"
MODELS = ["llada", "nemotron_pt", "dqwen"]
M_GRID = [2, 4, 8, 16]
J_GRID = [1, 2, 4, 8]
K = 16
CHUNK = 16  # most canvases per forward pass, as in the authors' run

# How each model attends over the canvas (the `attention` field of the authors' records; Appendix B, Experimental Setup).
ATTENTION = {
    "llada": "full bidirectional (LLaDA architecture; no attention mask)",
    "nemotron_pt": (
        "all layers diffusion_lm=True with an explicit boolean mask: b = first target position; queries i < b "
        "attend to valid keys k <= i; queries i >= b attend to every valid key (prefix + tail); no KV cache; "
        "prefix states recomputed on every forward (P0 included)"
    ),
    "dqwen": (
        "dQwen3.5 hybrid: full_attention layers (every 4th) bidirectional over the BOS-prefixed canvas "
        "(is_causal=False), Gated-DeltaNet layers causal; no mask (equal lengths only)"
    ),
}


def proposal_seed(model, episode_id, M, J):
    """The seed of a cell's K proposals: SHA-256 of "sec42_mj|proposal|<model>|<episode>|<M>|<J>|master24",
    first 16 hex digits, mod 2**31. The same seed gives the same draws for the same logits."""
    text = "|".join(str(part) for part in ("sec42_mj", "proposal", model, episode_id, M, J, "master24"))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2 ** 31)


def oracle_lengths(adapter, family, J):
    """The token count of each reference answer " <output>" of targets T1..TJ under the model's tokenizer."""
    return [len(adapter.encode(" " + target["output"])) for target in family["targets"][:J]]


def prompt_sha(ids):
    """SHA-256 of the JSON list of token ids: the prompt hash stored in the records."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def gold_infill(adapter, canvas):
    """The reference infill in reveal order: the tokens of " <answer>" of each target, site after site."""
    return [token for gold in canvas.gold for token in adapter.encode(" " + gold)]


def audit_checks(adapter, canvas, lengths, distinct, ell1_by_infill, chunk):
    """The authors' checks on the first families of each cell.

    The distinct infills replayed under P1 give back the ell1 summed while sampling; P0 differs from P1
    exactly at the demonstration outputs E; every target position is masked in both; each site has the
    oracle length; the reference tokens decode to " <answer>".
    """
    order = reveal_order(canvas)
    p1 = context_ids(adapter, canvas, hide_E=False)
    p0 = context_ids(adapter, canvas, hide_E=True)
    allowed = adapter.support("ordinary")
    _, replay = reveal(adapter, [p1], [order], [0] * len(distinct), allowed, forced=distinct, chunk=chunk)
    E = set(canvas.E)
    gold_sites = split_by_site(canvas, gold_infill(adapter, canvas))
    return {
        "replay_P1_max_abs_seq": max(abs(sum(terms) - ell1_by_infill[infill]) for infill, terms in zip(distinct, replay)),
        "P0_differs_only_on_demo_output_spans": all(p0[i] == adapter.mask_id for i in E) and all(p1[i] == p0[i] for i in range(len(p0)) if i not in E),
        "targets_masked_in_P1_and_P0": all(p1[i] == adapter.mask_id and p0[i] == adapter.mask_id for i in order),
        "allocation_equals_oracle_L": [len(site) for site in canvas.sites] == lengths,
        "n_demo_output_tokens_hidden": len(E),
        "n_demo_spans": canvas.meta["M"],
        "reveal_ascending": order == sorted(order),
        "block_start": getattr(adapter, "block_start", None),
        "first_target_position": order[0],
        "gold_decodes_to_gold_text": [adapter.decode(site) == " " + gold for site, gold in zip(gold_sites, canvas.gold)],
        "p1_prompt_text": adapter.decode(p1),
        "p0_prompt_text": adapter.decode(p0),
    }


def run_canvas(adapter, family, M, J, n_candidates, chunk, audit):
    """Sample and score one family's pool for the cell (M, J); returns the record body (see the module doc)."""
    model = adapter.name
    episode = active_episode(family, M, J)
    t0 = time.time()
    lengths = oracle_lengths(adapter, family, J)
    canvas = build_canvas(adapter, family, M, J, lengths)
    order = reveal_order(canvas)
    p1 = context_ids(adapter, canvas, hide_E=False)
    p0 = context_ids(adapter, canvas, hide_E=True)
    gold = gold_infill(adapter, canvas)
    assert len(gold) == len(order)
    allowed = adapter.support("ordinary")

    seed = proposal_seed(model, family["episode_id"], M, J)
    infills = sample_fixed_order(adapter, canvas, n_candidates, seed, support=allowed, chunk=chunk)
    traces = [[token for site in infill.toks for token in site] for infill in infills]
    seconds_gen = time.time() - t0

    t1 = time.time()
    distinct = list({tuple(trace): None for trace in traces})
    _, p0_terms = reveal(adapter, [p0], [order], [0] * (len(distinct) + 1), allowed, forced=distinct + [tuple(gold)], chunk=chunk)
    ell0_terms = dict(zip(distinct, p0_terms[:-1]))
    gold_ell0_terms = p0_terms[-1]
    _, gold_p1_terms = reveal(adapter, [p1], [order], [0], allowed, forced=[gold], chunk=chunk)
    gold_ell1_terms = gold_p1_terms[0]
    seconds_score = time.time() - t1

    checks = None
    if audit:
        ell1_by_infill = {tuple(trace): infill.ell1_sample for trace, infill in zip(traces, infills)}
        checks = audit_checks(adapter, canvas, lengths, distinct, ell1_by_infill, chunk)

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
    entries = episode["entries"]
    return {
        "interface": f"{model}_canvas_oracle_length",
        "attention": ATTENTION.get(model),
        "n_tokens": len(canvas.ids),
        "oracle_L": lengths,
        "alloc_L": lengths,
        "T_positions": order,
        "reveal_order": "ascending_canvas_position_one_token_per_step",
        "record_labels": canvas.meta["slots"],
        "demo_ids": [entry["id"] for entry in entries if entry["role"] == "demo"],
        "target_ids": [entry["id"] for entry in entries if entry["role"] == "target"],
        "prompt_sha256_p1": prompt_sha(p1),
        "prompt_sha256_p0": prompt_sha(p0),
        "seed": seed,
        "block_start": getattr(adapter, "block_start", None),
        "n_demo_output_tokens_hidden": len(canvas.E),
        "gold_toks": gold,
        "gold_T": len(gold),
        "gold_ell1": float(sum(gold_ell1_terms)),
        "gold_ell0": float(sum(gold_ell0_terms)),
        "gold_ell1_t": site_sums(canvas, gold_ell1_terms),
        "gold_ell0_t": site_sums(canvas, gold_ell0_terms),
        "candidates": candidates,
        "n_unique": len(distinct),
        "checks": checks,
        "seconds_gen": seconds_gen,
        "seconds_score": seconds_score,
    }


def config_of(args, adapter):
    """Everything that decides the records (not which families, where, or on which device): its hash is the
    `protocol_hash` of every record."""
    return {
        "study": "fig5_multi_target",
        "task": TASK,
        "bank": args.bank,
        "model": adapter.name,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "M": args.M,
        "J": args.J,
        "K": args.K,
        "chunk": args.chunk,
        "placement": f"interleaved: 24-slot master layout ({GRID_PLACEMENT_SEED})",
        "lengths": "oracle: token count of ' ' + answer under the model's tokenizer",
        "support": "ordinary",
        "reveal_order": "ascending canvas position, one token per step",
        "seed_rule": "sha256('sec42_mj|proposal|model|episode_id|M|J|master24')[:16] mod 2**31",
    }


def protocol_hash(config):
    """First 16 hex digits of SHA-256 of the configuration as sorted JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def write_manifest(path, config, digest, families, args):
    """The run's manifest: configuration and hash, the seed of every (cell, family), library and GPU versions."""
    seeds = {}
    for M in args.M:
        for J in args.J:
            seeds[f"M{M}_J{J}"] = {family["episode_id"]: proposal_seed(config["model"], family["episode_id"], M, J) for family in families}
    manifest = {
        "config": config,
        "protocol_hash": digest,
        "families": [family["episode_id"] for family in families],
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
    path.write_text(json.dumps(manifest, indent=1))


def done_episodes(path):
    """Episode ids already in a pool file (an interrupted run resumes after them)."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["episode_id"])
    return done


def append_record(path, record):
    """Append one JSON line in a single write, so that a killed run leaves complete lines only."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.write(fd, (json.dumps(record) + "\n").encode())
    os.close(fd)


def int_list(text):
    return [int(value) for value in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--M", type=int_list, default=M_GRID, help="demonstration counts, e.g. 2,4,8,16")
    parser.add_argument("--J", type=int_list, default=J_GRID, help="target counts, e.g. 1,2,4,8")
    parser.add_argument("--K", type=int, default=K, help="infills sampled per family")
    parser.add_argument("--families", type=int, default=None, help="only the first N families of the bank")
    parser.add_argument("--bank", default="report", choices=["dev", "report"])
    parser.add_argument("--out", default="artifacts/reruns/fig5")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--audit-n", type=int, default=2, help="families per cell that also run the audit checks")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    families = load_families(TASK, args.bank)[: args.families]
    started = time.time()
    adapter = load(args.model, args.device)
    print(f"[fig5] {args.model} {adapter.model_id} @ {adapter.revision} loaded in {time.time() - started:.0f}s", flush=True)
    config = config_of(args, adapter)
    digest = protocol_hash(config)
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / f"manifest_{digest}.json", config, digest, families, args)

    for M in args.M:
        for J in args.J:
            path = out / f"{TASK}_M{M}_J{J}.jsonl"
            done = done_episodes(path)
            written = 0
            job_start = time.time()
            for family in families:
                if family["episode_id"] in done:
                    continue
                try:
                    body = run_canvas(adapter, family, M, J, args.K, args.chunk, audit=written < args.audit_n)
                except Exception as error:  # one failed family must not stop the cell; it is retried on the next run
                    print(f"FAILED {family['episode_id']} M={M} J={J}: {error}\n{traceback.format_exc()[-1500:]}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                record = {
                    "episode_id": family["episode_id"],
                    "source": TASK,
                    "rule_id": family["rule_id"],
                    "function_id": family["meta"]["function_id"],
                    "model": args.model,
                    "model_name": adapter.model_id,
                    "revision": adapter.revision,
                    "M": M,
                    "J": J,
                    "K": args.K,
                    "placement": "master24",
                    "placement_seed_mj": GRID_PLACEMENT_SEED,
                    "p": 1.0,
                    "tau": 1.0,
                    "beta_convention": "zero_is_ordinary",
                    "protocol_hash": digest,
                    "gold": [target["output"] for target in family["targets"][:J]],
                    "target_inputs": [target["input"] for target in family["targets"][:J]],
                    "grader": family["grader"],
                    "status": "scored",
                    **body,
                }
                append_record(path, record)
                written += 1
                print(
                    f"[{args.model}] M={M} J={J} {family['episode_id']} tokens={body['n_tokens']} L={body['oracle_L']}"
                    f" distinct={body['n_unique']} gen {body['seconds_gen']:.1f}s score {body['seconds_score']:.1f}s"
                    + (f" replay {body['checks']['replay_P1_max_abs_seq']:.1e}" if body["checks"] else "")
                    + f" | gold R {body['gold_ell1'] - body['gold_ell0']:+.1f}",
                    flush=True,
                )
            print(f"CELL_DONE {args.model} M={M} J={J} new={written} {time.time() - job_start:.0f}s", flush=True)
    print(f"RUN_DONE {args.model} {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
