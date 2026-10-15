# ported from editdiff/<the authors' Section 5 run package>: fr_gen.py (main run) and zd_gen.py (zero-demonstration ablation) @ d276fa6
"""Sample and score the Java pools of Section 5 (Figure 8) on a GPU: LLaDA-1.5, one process, one GPU.

Main run (--study main). For every episode of data/java in the chosen cells (M demonstrations, the paper's D; J
targets) and a length regime:
  oracle   every target gets its reference token count (episode["oracle_lengths"]): one canvas;
  learned  each of the K candidates draws its own target lengths from the Java predictor (length head C;
           Appendix C.2): candidate k, target j gets F_j^{-1}(u) with u = SHA-256("s5-final-alloc-v1" | protocol
           hash | episode | J | k | j | redraw) / 2^64 (allocation_uniform); a draw whose canvas would exceed 2,048
           tokens is redrawn with the next redraw counter (learned_allocations). One canvas per distinct allocation.
For each distinct allocation, as in the paper's run:
  1. Sample its candidates from P1 by fixed-order decoding (Lemma 3.1), all in one reveal, with one
     torch.Generator seeded with generation_seed(regime, episode, allocation); logp1_sampletime is the sum taken
     while sampling.
  2. Teacher-force the distinct infills, in first-draw order, on the same canvas under P1 (logp1) and under P0,
     the canvas with the demonstrations' accepted edits E masked (logp0); R = logp1 - logp0. Both use the batched
     loop of sampling, so with K <= 16 (one chunk) the replayed logp1 equals logp1_sampletime exactly.

Zero-demonstration ablation (--study zero_demo, Appendix C.2). Each M = 1 episode is an anchor; it is run with no
demonstration (M = 0), with its demonstration (M = 1) and, when the cell M = 2 holds the same targets, with both
demonstrations of that episode (M = 2), all under the neutral instruction canvas.JAVA_ZERO_DEMO_INSTRUCTION, at
oracle lengths, K draws from P1 per arm, no P0 (beta = 0 only). Seed: zero_demo_seed(unit).

Records are appended as episodes finish:
    main       <out>/pools_<regime>.jsonl          analyze.py --pools <out>
    zero_demo  <out>/pools_zero_demo.jsonl          analyze.py --zero-demo <out>/..
Every record carries the config_hash of its pool (pool_config: model, revision, study, regime, K, chunk, head, rules;
not which cells or episodes the run selects); protocol_hash stays the paper's run hash, from which the seeds derive. A
rerun with the same configuration skips the units already in the file, so an interrupted run resumes; a file that
holds records of another configuration (another K, chunk, head, ...; or the frozen pools, which carry no config_hash)
is refused with exit status 2, naming the file, before the model is loaded, so that analyze.py never reads two
configurations as one pool. A unit that raises is reported and counted, the others are written, and the run ends with
RUN_FAILED and exit status 1; the next run samples only the failed units. <out>/manifest_<hash>.json records the
run's configuration and hash, the pool's hash, every seed and the library versions.

    python experiments/java_refactoring/run.py --regime oracle --M 1,2 --J 1,2 --K 16 --out artifacts/reruns/java_refactoring/pools
    python experiments/java_refactoring/run.py --regime learned --head artifacts/heads/java_head_C.pt --out artifacts/reruns/java_refactoring/pools
    python experiments/java_refactoring/run.py --study zero_demo --M 0,1,2 --J 1,2 --out artifacts/reruns/java_refactoring/zero_demo/pools
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
from types import SimpleNamespace

import numpy as np
import torch
import transformers

from fewshot_dlms.adapters import adapter_class
from fewshot_dlms.adapters import load
from fewshot_dlms.adapters.base import context_ids
from fewshot_dlms.adapters.base import reveal
from fewshot_dlms.adapters.base import reveal_order
from fewshot_dlms.adapters.base import sample_fixed_order
from fewshot_dlms.canvas import JAVA_ZERO_DEMO_INSTRUCTION
from fewshot_dlms.canvas import build_java_canvas
from fewshot_dlms.data import load_java_episodes
from fewshot_dlms.lengths import compact_prompt
from fewshot_dlms.lengths import hashed_uniform
from fewshot_dlms.lengths import inverse_cdf

MODEL = "llada"  # Section 5 uses LLaDA-1.5 only
STUDIES = ("main", "zero_demo")
REGIMES = ("oracle", "learned")
K = 16
CHUNK = 16  # most canvases per forward pass, as in the paper's run
CTX_CAP = 2048  # most canvas tokens of a learned allocation (draws above it are redrawn)

# The protocol hashes of the paper's two Java runs; every seed below is derived from them.
PROTOCOL_HASH = "0e1d3c9040b09598"
ZERO_DEMO_PROTOCOL_HASH = "c2a07e4a701f0a41"
ALLOC_TAG = "s5-final-alloc-v1"

# The Java predictor, length head C (Appendix C.2): the SHA-256 of the published checkpoint
# (artifacts/heads/java_head_C.pt) and the file of its stored predictions in java/length. The published file is the
# authors' final.pt re-saved without the run directory pickled in its training config, tensors identical; the stored
# pools and predictions record the original file's SHA-256 (TRAINED_HEAD_CKPT_SHA) in head_ckpt_sha / ckpt_sha, and a
# live run records the file it reads.
HEAD_CKPT_SHA = "b2efefded5ac382b5fc67ace4ee6d9f0ebfa15b820f2431b6f67cdb09721cebf"
TRAINED_HEAD_CKPT_SHA = "ea54c8b92110f4c5c4dbe838721d23961775082712247f8099b74e8d7d312309"
HEAD_PROBS_FILE = "s5hc_C_s5final_G4.npz"


# ---------------------------------------------------------------------------------------------
# Seeds and learned lengths
# ---------------------------------------------------------------------------------------------


def key12(*parts):
    """An integer from the first 12 hex digits of the SHA-256 of the parts joined by '|' (the run's key())."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def generation_seed(regime, episode_id, allocation):
    """The seed of the candidates that share one allocation: key12(protocol hash, "s5final", regime, episode,
    allocation as a tuple) mod 2**31."""
    return key12(PROTOCOL_HASH, "s5final", regime, episode_id, tuple(int(L) for L in allocation)) % (2 ** 31)


def zero_demo_seed(unit):
    """The seed of one ablation arm: key12(ablation hash, "demo_ablation", anchor, target sites joined by '+', M) mod 2**31."""
    parts = (ZERO_DEMO_PROTOCOL_HASH, "demo_ablation", unit["anchor_episode_id"], "+".join(unit["target_sites"]), unit["M"])
    return key12(*parts) % (2 ** 31)


def allocation_uniform(episode_id, J, k, j, redraw):
    """The uniform of candidate k, target j, redraw r of a learned allocation."""
    return hashed_uniform(ALLOC_TAG, PROTOCOL_HASH, episode_id, J, k, j, redraw)


def learned_allocations(episode, probs, n_candidates):
    """The target lengths of K candidates drawn from the Java predictor's (head C's) per-site distributions, and the number of redraws.

    probs maps "<episode_id>|<J>|<j>" to the head's probabilities over the lengths 1..256 (float64). Candidate k
    takes, per target j, the inverse CDF at allocation_uniform(episode, J, k, j, r), starting at r = 0; if the
    canvas would exceed CTX_CAP tokens (the oracle canvas minus the reference lengths plus the drawn lengths), the
    whole allocation is redrawn with r + 1.
    """
    episode_id = episode["episode_id"]
    J = episode["J"]
    sites = [np.clip(np.asarray(probs[f"{episode_id}|{J}|{j}"], dtype=np.float64), 0, None) for j in range(J)]
    base = episode["oracle_canvas_tokens"] - sum(episode["oracle_lengths"])
    allocations = []
    rejections = 0
    for k in range(n_candidates):
        redraw = 0
        while True:
            allocation = tuple(inverse_cdf(sites[j], allocation_uniform(episode_id, J, k, j, redraw)) for j in range(J))
            if base + sum(allocation) <= CTX_CAP:
                break
            redraw += 1
            rejections += 1
        allocations.append(allocation)
    return allocations, rejections


def read_head_probs(path):
    """Stored head predictions (an .npz with ids and P): "<episode_id>|<J>|<j>" -> float64 probabilities."""
    stored = np.load(path, allow_pickle=False)
    return dict(zip(stored["ids"].tolist(), stored["P"].astype(np.float64)))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def head_probs(adapter, head_path, episodes, device):
    """Head C's distributions for the target sites of `episodes`, computed on the GPU (lengths.LengthHead).

    The head reads each episode's compact prompt (one mask per target). The probabilities are rounded to float32
    and back, the precision in which the paper's run stored them before drawing.
    """
    from fewshot_dlms.lengths import LengthHead

    head = LengthHead.load(adapter, head_path, device)
    examples = []
    for episode in episodes:
        canvas = build_java_canvas(adapter, episode, episode["M"], episode["J"], [1] * episode["J"])
        examples.append(compact_prompt(canvas, adapter.mask_id))
    predictions = head.predict_examples(examples)
    probs = {}
    for episode, prediction in zip(episodes, predictions):
        stored = prediction.astype(np.float32).astype(np.float64)
        for j in range(episode["J"]):
            probs[f"{episode['episode_id']}|{episode['J']}|{j}"] = stored[j]
    return probs


# ---------------------------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------------------------


def java_support(adapter):
    """The token support of the paper's Java runs: the ordinary support plus the rows of the output layer beyond
    the tokenizer.

    LLaDA-1.5 has 126,464 logit rows and a tokenizer of 126,349 tokens. The adapter of the Section 5 runs did not
    exclude the 115 rows beyond the tokenizer (they decode to nothing), so every factor of the frozen Java pools
    is normalised over them too; adapter.support("ordinary") of the other figures excludes them. Their mass is
    tiny but not zero: with the ordinary support a rerun draws the same tokens and its log-likelihoods differ from
    the stored ones by about 1e-5 nats; with this support they are equal.
    """
    allowed = adapter.support("ordinary").clone()
    allowed[len(adapter.tok) :] = True
    allowed[sorted(adapter.control_ids)] = False
    return allowed


def prompt_sha(ids):
    """The prompt hash of the main run's records: the first 12 hex digits of SHA-1 of the JSON list of token ids."""
    return hashlib.sha1(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:12]


def short_sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def target_keys(J):
    return [f"t{j + 1}" for j in range(J)]


def run_main_episode(adapter, episode, regime, n_candidates, chunk, probs):
    """Sample and score one episode's pool in one regime; returns the record body (see the module docstring)."""
    episode_id = episode["episode_id"]
    M = episode["M"]
    J = episode["J"]
    keys = target_keys(J)
    L_ref = [int(L) for L in episode["oracle_lengths"]]
    reference = build_java_canvas(adapter, episode, M, J, L_ref)
    if prompt_sha(reference.ids) != episode["prompt_sha_oracle"]:
        raise ValueError(f"{episode_id}: the oracle canvas is not the paper's prompt")
    if regime == "oracle":
        allocations = [tuple(L_ref)] * n_candidates
        rejections = 0
    else:
        allocations, rejections = learned_allocations(episode, probs, n_candidates)
    groups = collections.OrderedDict()
    for k, allocation in enumerate(allocations):
        groups.setdefault(allocation, []).append(k)
    allowed = java_support(adapter)
    candidates = []
    seconds_gen = 0.0
    seconds_score = 0.0
    for allocation, draws in groups.items():
        if list(allocation) == L_ref:
            canvas = reference
        else:
            canvas = build_java_canvas(adapter, episode, M, J, list(allocation))
        order = reveal_order(canvas)
        p1 = context_ids(adapter, canvas, hide_E=False)
        p0 = context_ids(adapter, canvas, hide_E=True)
        t0 = time.time()
        infills = sample_fixed_order(adapter, canvas, len(draws), generation_seed(regime, episode_id, allocation), support=allowed, chunk=chunk)
        seconds_gen += time.time() - t0
        traces = [tuple(token for site in infill.toks for token in site) for infill in infills]
        distinct = list(dict.fromkeys(traces))
        t0 = time.time()
        _, p1_terms = reveal(adapter, [p1], [order], [0] * len(distinct), allowed, forced=distinct, chunk=chunk)
        _, p0_terms = reveal(adapter, [p0], [order], [0] * len(distinct), allowed, forced=distinct, chunk=chunk)
        seconds_score += time.time() - t0
        ell1 = {trace: float(sum(terms)) for trace, terms in zip(distinct, p1_terms)}
        ell0 = {trace: float(sum(terms)) for trace, terms in zip(distinct, p0_terms)}
        for k, trace, infill in zip(draws, traces, infills):
            texts = {key: adapter.decode(site) for key, site in zip(keys, infill.toks)}
            candidates.append({
                "draw": k,
                "allocation": list(allocation),
                "allocation_exact": list(allocation) == L_ref,
                "prompt_sha": prompt_sha(canvas.ids),
                "n_tokens": len(canvas.ids),
                "toks": list(trace),
                "toks_sha": short_sha1(json.dumps(list(trace))),
                "text_by_target": texts,
                "text_sha": short_sha1("\t".join(texts[key] for key in keys)),
                "logp1": ell1[trace],
                "logp1_sampletime": infill.ell1_sample,
                "logp0": ell0[trace],
                "R": ell1[trace] - ell0[trace],
                "p1_replay_absdiff_canvas": abs(ell1[trace] - infill.ell1_sample),
            })
    return {
        "episode_id": episode_id,
        "cell": episode["cell"],
        "M": M,
        "J": J,
        "commit10": episode["commit10"],
        "regime": regime,
        "targets": keys,
        "L_ref": L_ref,
        "distinct_allocations": len(groups),
        "allocation_rejections": rejections,
        "candidates": candidates,
        "n_unique_text": len({candidate["text_sha"] for candidate in candidates}),
        "n_unique_toks": len({candidate["toks_sha"] for candidate in candidates}),
        "seconds_gen": round(seconds_gen, 1),
        "seconds_score": round(seconds_score, 1),
        "max_p1_replay_absdiff": max(candidate["p1_replay_absdiff_canvas"] for candidate in candidates),
    }


def zero_demo_units(episodes, arms, Js):
    """The ablation's units: per anchor (an M = 1 episode with J in Js, in data/java order), one unit per arm M.

    M = 0 and M = 1 use the anchor's record; M = 2 uses the cell-(2, J) episode of the same helper group with the
    same target sites (its first demonstration is the anchor's), and anchors without one have no M = 2 unit.
    """
    by_group = {(episode["source_group_id"], episode["cell"]): episode for episode in episodes}
    units = []
    for anchor in episodes:
        if anchor["M"] != 1 or anchor["J"] not in Js:
            continue
        for M in arms:
            if M in (0, 1):
                source = anchor
            else:
                source = by_group.get((anchor["source_group_id"], f"M2J{anchor['J']}"))
                if source is None or source["target_sites"] != anchor["target_sites"]:
                    continue
            units.append({
                "unit_id": f"{anchor['episode_id']}|M{M}",
                "anchor_episode_id": anchor["episode_id"],
                "arm_source_episode": source["episode_id"],
                "M": M,
                "J": anchor["J"],
                "commit10": anchor["commit10"],
                "target_sites": list(anchor["target_sites"]),
            })
    return units


def run_zero_demo_unit(adapter, unit, source, n_candidates, chunk):
    """Sample one ablation arm (unguided sampling from P1 at oracle lengths, neutral instruction)."""
    M = unit["M"]
    J = unit["J"]
    keys = target_keys(J)
    L_ref = [int(L) for L in source["oracle_lengths"]]
    canvas = build_java_canvas(adapter, source, M, J, L_ref, instruction=JAVA_ZERO_DEMO_INSTRUCTION)
    seed = zero_demo_seed(unit)
    t0 = time.time()
    infills = sample_fixed_order(adapter, canvas, n_candidates, seed, support=java_support(adapter), chunk=chunk)
    seconds = time.time() - t0
    candidates = []
    for k, infill in enumerate(infills):
        trace = [token for site in infill.toks for token in site]
        texts = {key: adapter.decode(site) for key, site in zip(keys, infill.toks)}
        candidates.append({
            "draw": k,
            "allocation": L_ref,
            "toks": trace,
            "toks_sha": short_sha1(json.dumps(trace)),
            "text_by_target": texts,
            "text_sha": short_sha1("\t".join(texts[key] for key in keys)),
            "logp1_sampletime": infill.ell1_sample,
        })
    return {
        **unit,
        "protocol_hash": ZERO_DEMO_PROTOCOL_HASH,
        "instruction_sha256": hashlib.sha256(JAVA_ZERO_DEMO_INSTRUCTION.encode()).hexdigest(),
        "seed": seed,
        "targets": keys,
        "L_ref": L_ref,
        "prompt_sha": prompt_sha(canvas.ids),
        "n_tokens": len(canvas.ids),
        "demo_sites_shown": source["demo_sites"][:M],
        "candidates": candidates,
        "n_unique_text": len({candidate["text_sha"] for candidate in candidates}),
        "n_unique_toks": len({candidate["toks_sha"] for candidate in candidates}),
        "seconds_gen": round(seconds, 1),
    }


# ---------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------


def config_of(args, adapter, head_sha):
    """Everything that decides the records of a run (not which episodes, where, or on which device). `adapter` is the
    loaded adapter or anything with its name, model_id and revision; pool_config narrows the result to the pool."""
    return {
        "study": f"fig7_java/{args.study}",
        "model": adapter.name,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "regime": args.regime if args.study == "main" else "oracle",
        "M": args.M,
        "J": args.J,
        "K": args.K,
        "chunk": args.chunk,
        "protocol_hash": PROTOCOL_HASH if args.study == "main" else ZERO_DEMO_PROTOCOL_HASH,
        "head_ckpt_sha": head_sha,
        "head_probs": "stored" if args.head_probs else ("computed" if head_sha else None),
        "support": "ordinary plus the logit rows beyond the tokenizer (the Java runs)",
        "reveal_order": "ascending canvas position, one token per step",
    }


def config_hash(config):
    """First 16 hex digits of SHA-256 of the configuration as sorted JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def pool_config(config):
    """What decides the records of the pool file: the run's configuration without the cells (M, J) it selects.

    A pool file holds every selected cell of one regime (or every arm of the ablation), and a unit's record does not
    depend on which other units the run samples. Its hash is the config_hash of the pool's records, so a pool is resumed
    only by a run with the same configuration, whatever cells that run selects.
    """
    return {key: value for key, value in config.items() if key not in ("M", "J")}


def write_manifest(path, config, digest, pool_hashes, seeds, args):
    """The run's manifest: configuration and hash, the pool's hash, the seed of every unit, library and GPU versions."""
    manifest = {
        "config": config,
        "config_hash": digest,
        "pool_config_hash": pool_hashes,
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
    """A pool file holds records of another configuration than the run that would append to it."""


def done_units(path, field, digest):
    """Values of `field` (episode_id or unit_id) already in a pool file, which a run of pool configuration `digest`
    resumes after.

    Every record of the file must carry config_hash == digest; otherwise ProtocolMismatch names the file and the hashes
    found. Lines that do not parse (an interrupted write) are ignored, so their unit is sampled again.
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
            if record.get("config_hash") == digest:
                done.add(record[field])
            else:
                others.add(str(record.get("config_hash")))
    if others:
        raise ProtocolMismatch(
            f"{path} holds records of configuration {', '.join(sorted(others))}, but this run's pool is {digest} "
            "(another K, chunk, head or code). Write to another --out, or move the file away to start the pool afresh."
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


def head_sha_of(args):
    """The SHA-256 of length head C that a learned main run uses (the stored predictions' ckpt_sha with --head-probs,
    else the checkpoint file's), or "" for the oracle regime and the ablation. Read before the model is loaded."""
    if args.study != "main" or args.regime != "learned":
        return ""
    if args.head_probs:
        if not Path(args.head_probs).is_file():
            print(f"no stored head predictions at {args.head_probs}; run: python -m fewshot_dlms.fetch java", file=sys.stderr)
            sys.exit(2)
        return str(np.load(args.head_probs, allow_pickle=False)["ckpt_sha"])
    if not Path(args.head).is_file():
        print(f"no length head at {args.head}; run: python -m fewshot_dlms.fetch heads", file=sys.stderr)
        sys.exit(2)
    head_sha = file_sha256(args.head)
    if head_sha not in (HEAD_CKPT_SHA, TRAINED_HEAD_CKPT_SHA):
        print(f"warning: {args.head} is not the paper's Java predictor, head C (sha256 {head_sha[:12]}...)", flush=True)
    return head_sha


def pool_path(args, out):
    """The pool file of the run: <out>/pools_<regime>.jsonl, or <out>/pools_zero_demo.jsonl for the ablation."""
    return out / (f"pools_{args.regime}.jsonl" if args.study == "main" else "pools_zero_demo.jsonl")


def run_main(args, adapter, episodes, out, config, done, head_sha):
    """The main run: one record per selected episode in one regime, skipping the episodes in `done`. Returns the number
    of episodes that failed."""
    selected = [e for e in episodes if e["M"] in args.M and e["J"] in args.J][: args.families]
    probs = None
    if args.regime == "learned":
        if args.head_probs:
            probs = read_head_probs(args.head_probs)
        else:
            probs = head_probs(adapter, args.head, selected, args.device)
    digest = config_hash(config)
    pool_digest = config_hash(pool_config(config))
    path = pool_path(args, out)
    seeds = {}
    if args.regime == "oracle":
        seeds = {e["episode_id"]: generation_seed("oracle", e["episode_id"], e["oracle_lengths"]) for e in selected}
    write_manifest(out / f"manifest_{digest}.json", config, digest, {path.name: pool_digest}, seeds, args)
    failed = 0
    for episode in selected:
        if episode["episode_id"] in done:
            continue
        try:
            body = run_main_episode(adapter, episode, args.regime, args.K, args.chunk, probs)
        except Exception as error:  # one failed episode must not stop the run; it is retried on the next run
            print(f"FAILED {episode['episode_id']} {args.regime}: {error}\n{traceback.format_exc()[-1500:]}", flush=True)
            torch.cuda.empty_cache()
            failed += 1
            continue
        record = {
            **body,
            "tag": args.tag,
            "protocol_hash": PROTOCOL_HASH,
            "config_hash": pool_digest,
            "head_ckpt_sha": head_sha,
            "K": args.K,
        }
        append_record(path, record)
        print(
            f"[{args.regime}] {episode['episode_id']} J={episode['J']} allocations={body['distinct_allocations']}"
            f" redraws={body['allocation_rejections']} distinct texts={body['n_unique_text']}"
            f" replay {body['max_p1_replay_absdiff']:.1e} gen {body['seconds_gen']:.0f}s score {body['seconds_score']:.0f}s",
            flush=True,
        )
    return failed


def run_zero_demo(args, adapter, episodes, out, config, done):
    """The zero-demonstration ablation: one record per (anchor, arm M), skipping the units in `done`. Returns the number
    of units that failed."""
    by_id = {episode["episode_id"]: episode for episode in episodes}
    units = zero_demo_units(episodes, args.M, args.J)
    anchors = list(dict.fromkeys(unit["anchor_episode_id"] for unit in units))
    keep = set(anchors[: args.families])
    units = [unit for unit in units if unit["anchor_episode_id"] in keep]
    digest = config_hash(config)
    pool_digest = config_hash(pool_config(config))
    path = pool_path(args, out)
    seeds = {unit["unit_id"]: zero_demo_seed(unit) for unit in units}
    write_manifest(out / f"manifest_{digest}.json", config, digest, {path.name: pool_digest}, seeds, args)
    failed = 0
    for unit in units:
        if unit["unit_id"] in done:
            continue
        try:
            body = run_zero_demo_unit(adapter, unit, by_id[unit["arm_source_episode"]], args.K, args.chunk)
        except Exception as error:  # one failed unit must not stop the run; it is retried on the next run
            print(f"FAILED {unit['unit_id']}: {error}\n{traceback.format_exc()[-1500:]}", flush=True)
            torch.cuda.empty_cache()
            failed += 1
            continue
        append_record(path, {**body, "tag": args.tag, "config_hash": pool_digest, "K": args.K})
        print(f"[M{unit['M']}] {unit['unit_id']} J={unit['J']} distinct texts={body['n_unique_text']} gen {body['seconds_gen']:.0f}s", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", default="main", choices=STUDIES)
    parser.add_argument("--regime", default="oracle", choices=REGIMES, help="target lengths of the main run")
    parser.add_argument("--M", type=int_list, default=None, help="main: cells' M in {1, 2}; zero_demo: arms in {0, 1, 2}")
    parser.add_argument("--J", type=int_list, default=[1, 2], help="targets J in {1, 2}")
    parser.add_argument("--K", type=int, default=K, help="infills sampled per episode (per arm)")
    parser.add_argument("--families", type=int, default=None, help="only the first N episodes (anchors) in data/java order")
    parser.add_argument("--head", default="artifacts/heads/java_head_C.pt", help="the Java predictor, length head C (learned regime)")
    parser.add_argument("--head-probs", default=None, help="stored head predictions (.npz) instead of running the head")
    parser.add_argument("--out", default=None, help="pool directory (default artifacts/reruns/java_refactoring/pools, or .../zero_demo/pools)")
    parser.add_argument("--tag", default="rerun")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.M is None:
        args.M = [1, 2] if args.study == "main" else [0, 1, 2]
    if args.study == "main" and not set(args.M) <= {1, 2}:
        parser.error("the main run has cells with M in {1, 2}; M = 0 is the zero_demo study")
    if args.study == "zero_demo" and not set(args.M) <= {0, 1, 2}:
        parser.error("the zero-demonstration ablation has arms M in {0, 1, 2}")
    default_out = "artifacts/reruns/java_refactoring/pools" if args.study == "main" else "artifacts/reruns/java_refactoring/zero_demo/pools"
    out = Path(args.out or default_out)

    # The pool this run appends to, checked before the model is loaded: resume a pool of the same configuration only.
    episodes = load_java_episodes()
    adapter_cls, _ = adapter_class(MODEL)
    head_sha = head_sha_of(args)
    config = config_of(args, SimpleNamespace(name=MODEL, model_id=adapter_cls.model_id, revision=adapter_cls.revision), head_sha)
    path = pool_path(args, out)
    try:
        done = done_units(path, "episode_id" if args.study == "main" else "unit_id", config_hash(pool_config(config)))
    except ProtocolMismatch as error:
        print(f"refusing to append: {error}", file=sys.stderr)
        sys.exit(2)

    started = time.time()
    adapter = load(MODEL, args.device)
    print(f"[java_refactoring] {adapter.model_id} @ {adapter.revision} loaded in {time.time() - started:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if args.study == "main":
        failed = run_main(args, adapter, episodes, out, config, done, head_sha)
    else:
        failed = run_zero_demo(args, adapter, episodes, out, config, done)
    if failed:
        print(f"RUN_FAILED {args.study} {time.time() - started:.0f}s -> {out}; failed families: {failed}", flush=True)
        sys.exit(1)
    print(f"RUN_DONE {args.study} {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
