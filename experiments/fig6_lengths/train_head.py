# ported from editdiff/experiments/length_project/lpx_train.py, experiments/length_project/lpx_common.py::{h64, render_fsp, ans_len, MAX_CONTEXT} and experiments/length_project/lpx_fsp.py (rendering of the logical episodes) @ d276fa6
"""Train a length head on a frozen diffusion model (Section 4.3, Appendix B.5). Optional: the paper's heads ship frozen.

The head (lengths.Branch on lengths.LengthHead's trunk) is a trainable copy of the backbone's last transformer block and
final norm plus a linear categorical head over the lengths 1..lmax, read at one placeholder mask per target of a compact
prompt. The backbone stays frozen; no gradient reaches it (checked after training). The recipe of the paper, with the
defaults taken from frozen_config.json (the authors' freeze record):

    data       two sources of compact prompts with exact span lengths as labels, mixed 70 / 30 per update:
               fsp   structured few-shot prompts (List Functions outside List Functions-Easy, SALT outside the
                     evaluation categories, MIT Movie fields other than Director, ATIS fields other than the evaluation
                     ones); J uniform on {1, 2, 4, 8}, M in {2, 4, 8} with probabilities 0.2 / 0.6 / 0.2
               edit  Java code edits from 28 repositories disjoint from Section 5: the pre-commit file with every hunk
                     of a related edit group as <old>...</old><new>...</new>, one placeholder per target block
               prompts above 2,048 tokens or with a target longer than lmax are dropped at load
    schedule   U = ceil(100,000 / 64) = 1,563 updates of 64 prompts; each update draws from ONE source; the per-source
               update counts are U * share rounded by largest remainder, in an order shuffled with the seed
               h64("lpx-schedule", seed, mix, prompts); each source is read through an epoch-wise shuffled iterator
               (seed h64("lpx-iter", seed, source))
    loss       prompt-normalised cross-entropy: (1 / B) sum_i (1 / J_i) sum_j -log q(L*_ij)
    optimiser  AdamW (betas 0.9 / 0.999, eps 1e-8), weight decay 0.01 on matrices only, learning rates 3e-5 (copied
               block and norm) and 3e-4 (head), 5 % linear warm-up then cosine decay, gradient clipping at 1.0, seed 0,
               bf16 autocast with fp32 master weights; LLaDA-1.5 packs prompts into microbatches of at most 24,576 tokens,
               the other backbones run one prompt per forward pass
    model      the final checkpoint (no selection on development data)

The training files are the authors' rendered prompts (one per backbone tokenizer), in --data:

    fsp_<backbone>_train.jsonl, edit_<backbone>_train.jsonl   rows {"id", "J", "input_ids", "mask_positions", "labels", "n_tokens", ...}
    listbench_dev_<backbone>.jsonl                            List Functions-Easy dev bank, M = 4, J in {1, 2, 4, 8} (evaluation)
    edit_<backbone>_dev.jsonl                                 repository-held-out code edits (evaluation, first 1,500 after a seeded shuffle)
    fsp_logical_train.jsonl                                   optional: the FSP episodes before rendering (--check-render)

A row's compact prompt is the canvas of canvas.py with ONE mask per target (render_fsp below re-renders an FSP episode).
Every --eval-every updates and at the end, the head is evaluated on the two development sets (lengths.site_metrics).
Outputs in --out: final.pt (the checkpoint format of lengths.LengthHead.load), config.json, log.jsonl (every 20 updates),
evals.jsonl, probs_<devset>.npz and summary.json.

    python experiments/fig6_lengths/train_head.py --backbone llada --data artifacts/head_training --out runs/llada_head
    python experiments/fig6_lengths/train_head.py --backbone llada --steps 200 --data artifacts/head_training --out /tmp/head_smoke
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from fewshot_dlms.adapters import load
from fewshot_dlms.canvas import build_canvas
from fewshot_dlms.data import load_families
from fewshot_dlms.lengths import BACKBONE_OF_ADAPTER
from fewshot_dlms.lengths import compact_prompt
from fewshot_dlms.lengths import fresh_head
from fewshot_dlms.lengths import length_mass
from fewshot_dlms.lengths import site_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN_CONFIG = HERE / "frozen_config.json"
FIXTURE = ROOT / "data" / "fixtures" / "fig6" / "head_predictions_llada.json"
ADAPTER_OF_BACKBONE = {backbone: adapter for adapter, backbone in BACKBONE_OF_ADAPTER.items()}
MAX_CONTEXT = 2048  # longest training prompt, in tokens (lpx_common.MAX_CONTEXT)
LOG_EVERY = 20
EVAL_GROUP = 64  # LLaDA-1.5 development prompts per packed group, shortest first
CODE_DEV_SEED = 7  # the code-edit development set is the first code_dev_max rows after random.Random(7).shuffle


def h64(*parts):
    """A 64-bit integer from the first 16 hex digits of SHA-256 of the parts joined by '|' (lpx_common.h64)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def render_fsp(adapter, episode):
    """The compact prompt of a logical FSP episode for one backbone (lpx_common.render_fsp): the canvas of canvas.py
    with one placeholder mask per target, the labels L*_j = token count of " <output>", and the demonstration lengths.

    `episode` has "instruction" and "entries" in prompt order, each {"role": "demo" | "target", "input", "output"}.
    """
    ids = list(adapter.encode(adapter.chat_header(episode["instruction"])))
    mask_positions = []
    labels = []
    demo_lengths = []
    for entry in episode["entries"]:
        ids.extend(adapter.encode(f"Input: {entry['input']}\nOutput:"))
        output_ids = adapter.encode(" " + entry["output"])
        if entry["role"] == "demo":
            ids.extend(output_ids)
            demo_lengths.append(len(output_ids))
        else:
            mask_positions.append(len(ids))
            ids.append(adapter.mask_id)
            labels.append(len(output_ids))
        ids.extend(adapter.encode("\n\n"))
    ids.extend(adapter.encode(adapter.chat_end()))
    return {"input_ids": ids, "mask_positions": mask_positions, "labels": labels, "n_tokens": len(ids), "demo_L": demo_lengths}


def check_render(adapter, data_dir, backbone, n):
    """Re-render the first n logical FSP training episodes and compare them with the shipped rows, token for token."""
    path = Path(data_dir) / "fsp_logical_train.jsonl"
    if not path.exists():
        print(f"--check-render: no {path}; skipped")
        return
    logical = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if len(logical) >= n:
                break
            logical.append(json.loads(line))
    wanted = {episode["id"] for episode in logical}
    rendered = {}
    with open(Path(data_dir) / f"fsp_{backbone}_train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                rendered[row["id"]] = row
            if len(rendered) == len(wanted):
                break
    same = 0
    for episode in logical:
        ours = render_fsp(adapter, episode)
        theirs = rendered[episode["id"]]
        keys = ("input_ids", "mask_positions", "labels", "demo_L")
        if all(ours[key] == theirs[key] for key in keys):
            same += 1
        else:
            print(f"--check-render: {episode['id']} differs from the shipped row")
    print(f"--check-render: {same}/{len(logical)} FSP episodes re-rendered identically for {backbone}", flush=True)


def load_sources(data_dir, backbone, mix, lmax):
    """The training rows of every source, without prompts over MAX_CONTEXT tokens or targets over lmax; and the drop counts."""
    data = {}
    dropped = {}
    for source in mix:
        rows = read_jsonl(Path(data_dir) / f"{source}_{backbone}_train.jsonl")
        keep = [row for row in rows if max(row["labels"]) <= lmax and row["n_tokens"] <= MAX_CONTEXT]
        dropped[source] = len(rows) - len(keep)
        data[source] = keep
    return data, dropped


def update_schedule(mix_text, mix, prompts, ppu, seed):
    """The source of every update: U = ceil(prompts / ppu) updates, U * share per source by largest remainder, shuffled."""
    U = math.ceil(prompts / ppu)
    raw = {source: U * share for source, share in mix.items()}
    counts = {source: int(math.floor(value)) for source, value in raw.items()}
    for source in sorted(raw, key=lambda s: -(raw[s] - counts[s]))[: U - sum(counts.values())]:
        counts[source] += 1
    schedule = [source for source in sorted(counts) for _ in range(counts[source])]
    random.Random(h64("lpx-schedule", seed, mix_text, prompts)).shuffle(schedule)
    return U, counts, schedule


class EpochIterator:
    """Indices 0..n-1 in a fresh seeded shuffle per epoch; `take(k)` continues across epochs."""

    def __init__(self, n, seed):
        self.n = n
        self.rng = random.Random(seed)
        self.epoch = 0
        self.order = []
        self.i = 0

    def take(self, k):
        out = []
        while len(out) < k:
            if self.i >= len(self.order):
                self.order = list(range(self.n))
                self.rng.shuffle(self.order)
                self.i = 0
                self.epoch += 1
            out.append(self.order[self.i])
            self.i += 1
        return out


def microbatches(examples, backbone, tokens_per_micro):
    """LLaDA-1.5: consecutive prompts packed up to tokens_per_micro tokens; other backbones: one prompt at a time."""
    if backbone != "llada":
        for example in examples:
            yield [example]
        return
    current = []
    tokens = 0
    for example in examples:
        if current and tokens + example["n_tokens"] > tokens_per_micro:
            yield current
            current = []
            tokens = 0
        current.append(example)
        tokens += example["n_tokens"]
    if current:
        yield current


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


def evaluate(head, devsets, args, out=None):
    """site_metrics of the head on every development set (shortest prompts first); with `out`, also save the probabilities."""
    head.branch.eval()
    results = {}
    with torch.no_grad():
        for name, rows in devsets.items():
            if not rows:
                continue
            order = sorted(range(len(rows)), key=lambda i: rows[i]["n_tokens"])
            group_size = EVAL_GROUP if head.backbone == "llada" else 1
            probs = []
            for start in range(0, len(order), group_size):
                group = [rows[i] for i in order[start : start + group_size]]
                for batch in microbatches(group, head.backbone, args.tokens_per_micro):
                    probs.append(head.log_probs(batch).exp().double().cpu().numpy())
            P = np.concatenate(probs)
            y = np.array([length for i in order for length in rows[i]["labels"]])
            Js = [len(rows[i]["labels"]) for i in order]
            metrics = site_metrics(P, y, args.lmax)
            offsets = np.cumsum([0] + Js)
            nll = -np.log(np.clip(P[np.arange(len(y)), np.clip(y, 1, args.lmax) - 1], 1e-300, None))
            metrics["nll_prompt"] = float(np.mean([nll[offsets[i] : offsets[i + 1]].mean() for i in range(len(Js))]))
            if name == "list_dev":
                for J in (1, 2, 4, 8):
                    parts = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(Js)) if rows[order[i]]["J"] == J]
                    selected = np.concatenate(parts or [np.array([], int)])
                    if len(selected):
                        per_J = site_metrics(P[selected], y[selected], args.lmax)
                        metrics[f"J{J}"] = {key: per_J[key] for key in ("P0", "P1", "P2", "eae", "mode_exact")}
            results[name] = metrics
            if out is not None:
                ids = np.array([rows[i].get("id", rows[i].get("episode_id")) + f"|{rows[i]['J']}" for i in order])
                np.savez_compressed(out / f"probs_{name}.npz", P=P.astype(np.float32), y=y, ids=ids, J=np.array(Js))
    head.branch.train()
    return results


def fixture_check(head):
    """Exact-length mass P_0 of the trained head and of the frozen LLaDA-1.5 head on the 10 report canvases of the
    fixture data/fixtures/fig6/head_predictions_llada.json (M = 4, J = 2): a sanity check of a retrain, not parity."""
    if head.backbone != "llada" or not FIXTURE.exists():
        return None
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    families = {family["episode_id"]: family for family in load_families(fixture["task"], fixture["bank"])}
    trained = []
    frozen = []
    head.branch.eval()
    for entry in fixture["canvases"]:
        canvas = build_canvas(head.adapter, families[entry["episode_id"]], fixture["M"], fixture["J"], entry["oracle_L"])
        prompt = compact_prompt(canvas, head.adapter.mask_id)
        assert hashlib.sha256(json.dumps(prompt["input_ids"]).encode()).hexdigest() == entry["compact_sha256"]
        y = np.array(entry["oracle_L"])
        stored = np.asarray(entry["probs"], dtype=np.float64)
        stored /= stored.sum(1, keepdims=True)
        trained.extend(length_mass(head.predict(canvas), y, 0))
        frozen.extend(length_mass(stored, y, 0))
    head.branch.train()
    return {"sites": len(trained), "P0_trained": float(np.mean(trained)), "P0_frozen": float(np.mean(frozen))}


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------


def frozen_defaults():
    """The training configuration of the paper's heads (identical for the three backbones)."""
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    return config["backbones"]["llada"]["config"]


def main():
    defaults = frozen_defaults()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backbone", required=True, choices=sorted(ADAPTER_OF_BACKBONE))
    parser.add_argument("--data", required=True, help="directory with the rendered training and development files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mix", default=defaults["mix"])
    parser.add_argument("--prompts", type=int, default=defaults["prompts"])
    parser.add_argument("--steps", type=int, default=0, help="stop after this many updates (0: all); the schedule is unchanged")
    parser.add_argument("--lmax", type=int, default=defaults["lmax"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--lr-block", type=float, default=defaults["lr_block"])
    parser.add_argument("--lr-head", type=float, default=defaults["lr_head"])
    parser.add_argument("--wd", type=float, default=defaults["wd"])
    parser.add_argument("--ppu", type=int, default=defaults["ppu"], help="prompts per update")
    parser.add_argument("--tokens-per-micro", type=int, default=defaults["tokens_per_micro"])
    parser.add_argument("--warmup-frac", type=float, default=defaults["warmup_frac"])
    parser.add_argument("--eval-every", type=int, default=defaults["eval_every"])
    parser.add_argument("--code-dev-max", type=int, default=defaults["code_dev_max"])
    parser.add_argument("--check-render", type=int, default=0, help="re-render the first N logical FSP episodes and compare")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mix = {source: float(share) for source, share in (part.split(":") for part in args.mix.split(","))}
    total = sum(mix.values())
    mix = {source: share / total for source, share in mix.items()}
    data, dropped = load_sources(args.data, args.backbone, mix, args.lmax)
    U, counts, schedule = update_schedule(args.mix, mix, args.prompts, args.ppu, args.seed)
    iterators = {source: EpochIterator(len(rows), h64("lpx-iter", args.seed, source)) for source, rows in data.items()}

    head = fresh_head(load(ADAPTER_OF_BACKBONE[args.backbone], args.device), lmax=args.lmax, seed=args.seed)
    branch = head.branch
    if args.check_render:
        check_render(head.adapter, args.data, args.backbone, args.check_render)
    optimizer = torch.optim.AdamW(branch.param_groups(args.lr_block, args.lr_head, args.wd), betas=(0.9, 0.999), eps=1e-8)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    warmup = max(1, round(args.warmup_frac * U))

    def set_lr(step):
        if step < warmup:
            factor = (step + 1) / warmup
        else:
            factor = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, U - warmup)))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor

    devsets = {"list_dev": read_jsonl(Path(args.data) / f"listbench_dev_{args.backbone}.jsonl")}
    code_dev_path = Path(args.data) / f"edit_{args.backbone}_dev.jsonl"
    code_dev = read_jsonl(code_dev_path) if code_dev_path.exists() else []
    random.Random(CODE_DEV_SEED).shuffle(code_dev)
    devsets["code_dev"] = code_dev[: args.code_dev_max]

    config = {
        **{key: value for key, value in vars(args).items() if key not in ("data", "out", "device", "check_render")},
        "mix_normalised": mix,
        "updates": U,
        "updates_per_source": counts,
        "train_prompts_available": {source: len(rows) for source, rows in data.items()},
        "dropped_gt_lmax_or_ctx": dropped,
        "warmup": warmup,
        "architecture": f"frozen {args.backbone} -> trainable copy of the last decoder layer + final norm -> categorical head over 1..{args.lmax}",
        "loss": "prompt-level: mean over prompts of the per-prompt mean site NLL",
        "trainable_params": sum(p.numel() for p in branch.parameters()),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps({key: config[key] for key in ("backbone", "mix_normalised", "updates", "updates_per_source", "dropped_gt_lmax_or_ctx", "trainable_params")}), flush=True)

    # A run writes its logs afresh: a rerun into the same --out replaces them instead of appending to the old ones.
    with open(out / "log.jsonl", "w", encoding="utf-8") as log, open(out / "evals.jsonl", "w", encoding="utf-8") as eval_log:
        started = time.time()
        branch.train()
        n_updates = min(U, args.steps) if args.steps else U
        for u in range(n_updates):
            source = schedule[u]
            examples = [data[source][i] for i in iterators[source].take(args.ppu)]
            set_lr(u)
            optimizer.zero_grad(set_to_none=True)
            loss_prompt = 0.0
            nll_sum = 0.0
            n_sites = 0
            for batch in microbatches(examples, args.backbone, args.tokens_per_micro):
                log_q = head.log_probs(batch)
                y = torch.tensor([length for example in batch for length in example["labels"]], device=args.device)
                nll = -log_q.gather(1, (y - 1)[:, None]).squeeze(1)
                weights = [1.0 / (len(example["labels"]) * len(examples)) for example in batch for _ in example["labels"]]
                loss = (nll * torch.tensor(weights, device=args.device)).sum()
                loss.backward()
                loss_prompt += float(loss)
                nll_sum += float(nll.sum())
                n_sites += len(y)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0))
            optimizer.step()
            if (u + 1) % LOG_EVERY == 0:
                entry = {
                    "update": u + 1,
                    "source": source,
                    "loss_prompt": loss_prompt,
                    "nll_site": nll_sum / n_sites,
                    "grad_norm": grad_norm,
                    "lr_block": optimizer.param_groups[0]["lr"],
                    "s": time.time() - started,
                    "epochs": {s: it.epoch for s, it in iterators.items()},
                }
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(f"update {u + 1}/{n_updates} {source:4s} loss {loss_prompt:.4f} nll/site {nll_sum / n_sites:.4f} grad {grad_norm:.2f} {time.time() - started:.0f}s", flush=True)
            if (u + 1) % args.eval_every == 0 and u + 1 < n_updates:
                results = evaluate(head, devsets, args)
                eval_log.write(json.dumps({"update": u + 1, "s": time.time() - started, **results}) + "\n")
                eval_log.flush()
                print("EVAL", u + 1, json.dumps({k: {m: round(v[m], 4) for m in ("P0", "P1", "P2", "nll_site", "eae", "mode_exact")} for k, v in results.items()}), flush=True)

        assert not any(p.grad is not None for p in head.adapter.model.parameters()), "a gradient reached the frozen backbone"
        results = evaluate(head, devsets, args, out=out)
        eval_log.write(json.dumps({"update": n_updates, "s": time.time() - started, "final": True, **results}) + "\n")
        eval_log.flush()
    checkpoint = out / "final.pt"
    torch.save({"branch": branch.state_dict(), "config": config}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fixture = fixture_check(head)
    summary = {
        "final_eval": results,
        "updates_run": n_updates,
        "train_seconds": time.time() - started,
        "seconds_per_update": (time.time() - started) / max(1, n_updates),
        "checkpoint_sha256": digest,
        "source_epochs": {s: it.epoch for s, it in iterators.items()},
        "fixture_exact_length_mass": fixture,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps({k: {m: round(v[m], 4) for m in ("P0", "P1", "P2", "nll_site", "nll_prompt", "eae", "mode_exact")} for k, v in results.items()}), flush=True)
    if fixture:
        print(f"fixture: exact-length mass {fixture['P0_trained']:.3f} (this head) vs {fixture['P0_frozen']:.3f} (frozen head), {fixture['sites']} sites", flush=True)


if __name__ == "__main__":
    main()
