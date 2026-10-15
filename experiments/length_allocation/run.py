# ported from editdiff/experiments/algo_investigation/dr_worker.py::{seed_of, run_canvas_pc} and experiments/length_project/lpx_alloc.py (alloc) @ d276fa6
"""Sample the pools of Figure 6 (Section 4.3) on a GPU: one process, one model, one GPU.

Every family of List Functions-Easy (report bank) gets a canvas with M = 4 demonstrations and J targets in the
published 12-slot layout. Each of the K candidates gets its own target lengths L^(k) = (L_1, ..., L_J), the
*allocation*, from one of three modes (lengths.allocate):

    oracle   L_j = L*_j, the token count of " <answer>" under the model's tokenizer
    learned  L^(k) drawn from the frozen length head (Appendix B, Length Allocation): the head reads the compact
             prompt once (lengths.compact_prompt) and candidate k, target j draws L = F_j^{-1}(u_kj) with the hashed
             uniform u_kj of lengths.sample_allocations under the seed (bank, backbone, family, J)
             (lengths.section43_seed)
    offset   L_j = L*_j + delta at every target. A family with some L*_j + delta < 1 is skipped and counted, never
             clipped (the paper's rule; lengths.allocate would clip to 1)

Candidates with the same allocation share a canvas; candidate k is revealed on the canvas of its own allocation
(adapters.base.reveal with one context per distinct allocation, Lemma 3.1, ascending positions, one token per step).
All K draws come from one torch Generator seeded with proposal_seed(model, family, M, J), the seed of the oracle
pools, so the three modes use common random numbers for the content. Only log P1 is computed (beta = 0: Section 4.3
reweights nothing); ell0 is NaN in the records, as in the authors' records.

K defaults to the paper's K = 64 / J (J = 1, 2, 4, 8 -> 64, 32, 16, 8), 64 sampled targets per family in every cell;
the paper's oracle arm used the K = 64 pools of Sections 4.1/4.2 (pass --K 64). With the paper's K, chunk 32 and an
H100 NVL this reproduces the frozen LLaDA-1.5 and Nemotron pools token for token (README.md, Fidelity).

Records go to <out>/<model>/list_easy_fam_M4_J{J}_<condition>.jsonl (condition = oracle, learned, delta-2, delta-1,
delta0, delta+1, delta+2), one JSON line per family, appended as families finish. Every record carries the protocol_hash
of its pool (pool_config: model, revision, condition, J, K, chunk, head, rules). A rerun with the same protocol skips the
families already in the file, so an interrupted run resumes; a file that holds records of another protocol (another K,
head, ...) is refused with exit status 2, so that analyze.py never reads two protocols as one pool; so is a learned run
without its head file (fetch heads). A family that fails (head, canvas or sampling) is reported and retried on the next
run; the run then exits with status 1. <out>/<model>/manifest_<hash>.json holds the
run's configuration, the hash of every pool, every seed, and the library and GPU versions. analyze.py reads the pools.

    python experiments/length_allocation/run.py --model llada --mode learned --J 1,2,4,8 --out artifacts/reruns/length_allocation
    python experiments/length_allocation/run.py --model llada --mode offset --delta=-2,-1,1,2 --J 1,2 --out artifacts/reruns/length_allocation
"""
import argparse
import collections
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
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.canvas import active_episode
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.lengths import BACKBONE_OF_ADAPTER
from fewshot_dlms.lengths import LengthHead
from fewshot_dlms.lengths import allocate
from fewshot_dlms.lengths import sample_allocations
from fewshot_dlms.lengths import section43_seed

TASK = "list_easy_fam"
MODELS = ["llada", "nemotron_pt", "dqwen"]
MODES = ["oracle", "learned", "offset"]
M = 4  # demonstrations: the first four of every family (Section 4.3)
J_GRID = [1, 2, 4, 8]
TARGETS_PER_FAMILY = 64  # K = 64 / J candidates, so every cell samples 64 targets per family
CHUNK = 32  # most canvases per forward pass, as in the authors' run
PLACEMENT = "master"  # the family's published 12-slot layout (meta.master_placement), the same at every J


def default_K(J):
    """The paper's K at J targets: 64 / J."""
    return TARGETS_PER_FAMILY // J


def proposal_seed(model, episode_id, M, J):
    """The seed of a family's K content draws (dr_worker.seed_of with the master placement): the first 16 hex digits of
    SHA-256 of "algo_investigation|proposal|<model>|<episode>|<M>|<J>|('placement', 'master')", mod 2**31."""
    parts = ("algo_investigation", "proposal", model, episode_id, M, J, ("placement", PLACEMENT))
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2 ** 31)


def condition_name(mode, delta):
    """The pool name of an allocation mode: 'oracle', 'learned', or 'delta-2', ..., 'delta0', ..., 'delta+2'."""
    if mode != "offset":
        return mode
    return "delta0" if delta == 0 else f"delta{delta:+d}"


def offset_lengths(oracle_L, delta):
    """L*_j + delta at every target, or None when some target would get fewer than one token (the family is skipped)."""
    lengths = [int(length) + delta for length in oracle_L]
    if min(lengths) < 1:
        return None
    if allocate("offset", oracle_L, delta=delta) != lengths:
        raise ValueError(f"offset {delta} of {list(oracle_L)} exceeds the length support")
    return lengths


def oracle_lengths(adapter, family, J):
    """The token count of each reference answer " <output>" of targets T1..TJ under the model's tokenizer."""
    return [len(adapter.encode(" " + target["output"])) for target in family["targets"][:J]]


def allocations_of(mode, family, J, K, oracle_L, canvas, head, delta):
    """The K allocations of a family (K lists of J lengths), or None if the offset skips the family.

    learned: the head's distributions are rounded to float32 before drawing, as they were stored in the authors'
    evaluation file, so the draws are exactly lpx_alloc.py's.
    """
    if mode == "oracle":
        return [allocate("oracle", oracle_L) for _ in range(K)]
    if mode == "offset":
        lengths = offset_lengths(oracle_L, delta)
        if lengths is None:
            return None
        return [list(lengths) for _ in range(K)]
    probs = head.predict(canvas).astype(np.float32).astype(np.float64)
    draws = sample_allocations(probs, K, section43_seed("report", head.backbone, family["episode_id"], J))
    return [allocate("learned", oracle_L, head_sample=draw) for draw in draws]


def prompt_sha(ids):
    """SHA-256 of the JSON list of token ids: the prompt hash stored in the records."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def run_family(adapter, family, J, K, allocations, chunk, audit):
    """Sample one family's pool: candidate k on the canvas of its allocation (dr_worker.run_canvas_pc, without P0)."""
    t0 = time.time()
    allocations = [tuple(int(length) for length in allocation) for allocation in allocations]
    distinct = list(collections.OrderedDict.fromkeys(allocations))
    index_of = {allocation: i for i, allocation in enumerate(distinct)}
    which = [index_of[allocation] for allocation in allocations]
    canvases = [build_canvas(adapter, family, M, J, list(allocation)) for allocation in distinct]
    for canvas, allocation in zip(canvases, distinct):
        assert [len(site) for site in canvas.sites] == list(allocation)
    orders = [reveal_order(canvas) for canvas in canvases]
    first_target = orders[0][0]
    assert all(order[0] == first_target for order in orders), "the first target position must not depend on the lengths"
    contexts = [context_ids(adapter, canvas, hide_E=False) for canvas in canvases]
    allowed = adapter.support("ordinary")

    seed = proposal_seed(adapter.name, family["episode_id"], M, J)
    generator = torch.Generator().manual_seed(seed)
    traces, logprobs = reveal(adapter, contexts, orders, which, allowed, generator=generator, chunk=chunk)
    seconds_gen = time.time() - t0

    checks = None
    if audit:
        pairs = list(collections.OrderedDict.fromkeys((which[k], tuple(traces[k])) for k in range(K)))
        _, replay = reveal(adapter, contexts, orders, [u for u, _ in pairs], allowed, forced=[list(t) for _, t in pairs], chunk=chunk)
        ell1_of = {(which[k], tuple(traces[k])): sum(logprobs[k]) for k in range(K)}
        checks = {
            "replay_P1_max_abs_seq": max(abs(sum(terms) - ell1_of[pair]) for pair, terms in zip(pairs, replay)),
            "targets_masked": all(contexts[u][p] == adapter.mask_id for u in range(len(canvases)) for p in orders[u]),
            "allocation_fixed": all(len(trace) == sum(distinct[u]) for u, trace in pairs),
            "n_unique_allocations": len(distinct),
            "p1_prompt_text_first_allocation": adapter.decode(contexts[0]),
        }

    candidates = []
    for k in range(K):
        lengths = allocations[k]
        texts = []
        ell1_t = []
        start = 0
        for length in lengths:
            texts.append(adapter.decode(traces[k][start : start + length]))
            ell1_t.append(float(sum(logprobs[k][start : start + length])))
            start += length
        candidates.append({
            "toks": traces[k],
            "texts": texts,
            "ell1": float(sum(logprobs[k])),
            "ell0": float("nan"),
            "ell1_t": ell1_t,
            "ell0_t": [float("nan")] * J,
            "alloc_L": list(lengths),
            "truncated": False,
        })
    return {
        "interface": f"{adapter.name}_canvas_per_candidate_allocation",
        "alloc_mode": "per_candidate",
        "p0_replayed": False,
        "n_tokens": float(np.mean([len(canvases[u].ids) for u in which])),
        "n_tokens_per_unique_alloc": [len(canvas.ids) for canvas in canvases],
        "prompt_sha256_first_allocation": prompt_sha(contexts[0]),
        "alloc_L": [list(allocation) for allocation in allocations],
        "n_unique_alloc": len(distinct),
        "candidates": candidates,
        "n_unique": len({tuple(trace) for trace in traces}),
        "seed": seed,
        "block_start": getattr(adapter, "block_start", None),
        "checks": checks,
        "seconds_gen": seconds_gen,
        "seconds_score": 0.0,
    }


def run_config(args, model_id, revision, head_sha):
    """Everything that decides the records of this run (not which families, where, or on which device)."""
    return {
        "study": "fig6_lengths",
        "task": TASK,
        "bank": "report",
        "model": args.model,
        "model_id": model_id,
        "revision": revision,
        "M": M,
        "mode": args.mode,
        "delta": args.delta if args.mode == "offset" else None,
        "J": args.J,
        "K": args.K,
        "chunk": args.chunk,
        "head_sha256": head_sha,
        "placement": "published 12-slot layout (meta.master_placement)",
        "support": "ordinary",
        "reveal_order": "ascending canvas position, one token per step, per-candidate canvases",
        "seed_rule": "sha256(\"algo_investigation|proposal|model|episode_id|M|J|('placement', 'master')\")[:16] mod 2**31",
        "allocation_rule": "learned: L = F^-1(u), u = sha256('lpx-alloc-v1|report|backbone|family|J|k|j')[:16] / 2**64; offset: skip if L* + delta < 1",
    }


def pool_config(config, condition, J, K):
    """What decides the records of one pool file: the run's configuration narrowed to the pool's condition, J and K.

    Its hash is the protocol_hash of the pool's records, so a pool is resumed only by a run with the same protocol,
    whatever other J or offsets that run also samples.
    """
    narrowed = {key: value for key, value in config.items() if key not in ("J", "K", "delta")}
    return {**narrowed, "condition": condition, "J": J, "K": K}


def protocol_hash(config):
    """First 16 hex digits of SHA-256 of the configuration as sorted JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def write_manifest(path, config, digest, pool_hashes, families, args, Ks):
    """The run's manifest: configuration and hash, the hash of every pool, the seeds of every (J, family), versions."""
    seeds = {}
    for J in args.J:
        seeds[f"J{J}"] = {family["episode_id"]: proposal_seed(config["model"], family["episode_id"], M, J) for family in families}
    manifest = {
        "config": config,
        "protocol_hash": digest,
        "pool_protocol_hash": pool_hashes,
        "K_by_J": {str(J): Ks[J] for J in args.J},
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
            "(another K, mode, head or code). Write to another --out, or move the file away to start the pool afresh."
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--delta", type=int_list, default=[0], help="offset mode: one or more offsets, e.g. --delta=-2,-1,1,2")
    parser.add_argument("--J", type=int_list, default=J_GRID, help="target counts, e.g. 1,2,4,8")
    parser.add_argument("--K", type=int, default=None, help="candidates per family (default: 64 / J, the paper's K)")
    parser.add_argument("--families", type=int, default=None, help="only the first N families of the report bank")
    parser.add_argument("--head", default=None, help="learned mode: the head checkpoint (default artifacts/heads/<backbone>.pt)")
    parser.add_argument("--out", default="artifacts/reruns/length_allocation")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--audit-n", type=int, default=2, help="families per pool that also replay their draws under P1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(not -2 <= delta <= 2 for delta in args.delta):
        parser.error("--delta takes offsets in -2..2")
    Ks = {J: (args.K if args.K else default_K(J)) for J in args.J}

    families = load_families(TASK, "report")[: args.families]
    head_path = None
    head_sha = None
    if args.mode == "learned":
        head_path = args.head or f"artifacts/heads/{BACKBONE_OF_ADAPTER[args.model]}.pt"
        if not Path(head_path).is_file():
            print(f"no length head at {head_path}; run: python -m fewshot_dlms.fetch heads", file=sys.stderr)
            sys.exit(2)
        head_sha = sha256_file(head_path)
    adapter_cls, _ = adapter_class(args.model)
    config = run_config(args, adapter_cls.model_id, adapter_cls.revision, head_sha)
    digest = protocol_hash(config)
    out = Path(args.out) / args.model

    # Every pool this run appends to, checked before the model is loaded: resume a pool of the same protocol only.
    pools = []
    deltas = args.delta if args.mode == "offset" else [None]
    for delta in deltas:
        condition = condition_name(args.mode, delta)
        for J in args.J:
            K = Ks[J]
            path = out / f"{TASK}_M{M}_J{J}_{condition}.jsonl"
            pool_digest = protocol_hash(pool_config(config, condition, J, K))
            try:
                done = done_episodes(path, pool_digest)
            except ProtocolMismatch as error:
                print(f"refusing to append: {error}", file=sys.stderr)
                sys.exit(2)
            pools.append((delta, condition, J, K, path, pool_digest, done))

    started = time.time()
    adapter = load(args.model, args.device)
    print(f"[length_allocation] {args.model} {adapter.model_id} @ {adapter.revision} loaded in {time.time() - started:.0f}s", flush=True)
    head = None
    if args.mode == "learned":
        head = LengthHead.load(adapter, head_path, args.device)
        print(f"[length_allocation] length head {head_path} (sha256 {head_sha[:16]})", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pool_hashes = {path.name: pool_digest for _, _, _, _, path, pool_digest, _ in pools}
    write_manifest(out / f"manifest_{digest}.json", config, digest, pool_hashes, families, args, Ks)

    failed = 0
    for delta, condition, J, K, path, pool_digest, done in pools:
        written = 0
        skipped = 0
        job_start = time.time()
        for family in families:
            if family["episode_id"] in done:
                continue
            try:
                oracle_L = oracle_lengths(adapter, family, J)
                canvas = build_canvas(adapter, family, M, J, oracle_L)
                allocations = allocations_of(args.mode, family, J, K, oracle_L, canvas, head, delta)
                if allocations is None:
                    skipped += 1
                    continue
                body = run_family(adapter, family, J, K, allocations, args.chunk, audit=written < args.audit_n)
            except Exception as error:  # a failed family (head, canvas, sampling) must not stop the pool; it is retried
                print(f"FAILED {family['episode_id']} J={J} {condition}: {error}\n{traceback.format_exc()[-1500:]}", flush=True)
                torch.cuda.empty_cache()
                failed += 1
                continue
            episode = active_episode(family, M, J)
            record = {
                "episode_id": family["episode_id"],
                "source": TASK,
                "rule_id": family["rule_id"],
                "model": args.model,
                "model_name": adapter.model_id,
                "revision": adapter.revision,
                "M": M,
                "J": J,
                "K": K,
                "condition": condition,
                "delta": delta,
                "head_sha256": head_sha,
                "placement": PLACEMENT,
                "record_labels": [entry["slot"] for entry in episode["entries"]],
                "p": 1.0,
                "tau": 1.0,
                "beta_convention": "zero_is_ordinary",
                "protocol_hash": pool_digest,
                "gold": [target["output"] for target in family["targets"][:J]],
                "target_inputs": [target["input"] for target in family["targets"][:J]],
                "grader": family["grader"],
                "status": "scored",
                "oracle_L": oracle_L,
                **body,
            }
            append_record(path, record)
            written += 1
            print(
                f"[{args.model}] {condition} J={J} K={K} {family['episode_id']} L*={oracle_L} allocations={body['n_unique_alloc']}"
                f" distinct={body['n_unique']} gen {body['seconds_gen']:.1f}s"
                + (f" replay {body['checks']['replay_P1_max_abs_seq']:.1e}" if body["checks"] else ""),
                flush=True,
            )
        print(f"CELL_DONE {args.model} {condition} J={J} new={written} skipped={skipped} {time.time() - job_start:.0f}s", flush=True)
    print(f"RUN_DONE {args.model} {time.time() - started:.0f}s -> {out}; failed families: {failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
