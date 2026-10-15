"""Walk through one pool of the paper: the canvas, K candidates with ell1 / ell0 / R / w_0.5, and the two picks.

One List Functions-Easy family (family 0 of the report bank), M = 2 demonstrations, J = 2 targets, and K = 8
candidate infills from LLaDA-1.5. The script prints

  1. the canvas, with the J target sites marked (the model fills their mask tokens in one pass);
  2. one row per candidate: ell1 = log P1 (demonstration outputs visible), ell0 = log P0 (demonstration
     outputs masked), the example reward R = ell1 - ell0, the guided pool weight w_0.5 = softmax(0.5 R), and
     whether each target and all targets together are correct;
  3. the ordinary pick (beta = 0: every candidate has weight 1/K) and the guided pick (beta = 0.5: weights
     w_0.5, the target Q_0.5 ∝ P1^{3/2} P0^{-1/2}), and the expected accuracy of the pool under both.

    python examples/walkthrough.py --cached     # a stored pool: no GPU, no torch, instant
    CUDA_VISIBLE_DEVICES=0 python examples/walkthrough.py          # samples and scores the pool on one GPU (minutes)
    CUDA_VISIBLE_DEVICES=0 python examples/walkthrough.py --save-fixture data/fixtures/walkthrough.jsonl

--cached reads data/fixtures/walkthrough.jsonl (--fixture names another stored pool), the pool that the live
command wrote, and prints the same table; a live run prints the table first, then --save-fixture writes the
record. The live run is Figure 5's own sampling and scoring (experiments/multi_target/run.py::run_canvas)
for one family and one cell, so its record is a Figure 5 pool record.

"Ordinary" is what scoring.family_metrics calls beta = 0: the K candidates are draws from P1 and carry equal
weights, so an ordinary answer is a uniform draw from the pool. A guided answer is a draw from the pool with
weights w_0.5. Both picks below use the same uniform random number, so a single run compares them on one draw;
the expected accuracies are the quantities that the paper averages over families.
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fewshot_dlms.data import load_families
from fewshot_dlms.grading import answer
from fewshot_dlms.grading import grade
from fewshot_dlms.scoring import example_reward
from fewshot_dlms.scoring import family_metrics
from fewshot_dlms.scoring import pool_weights

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "walkthrough.jsonl"
MULTI_TARGET_RUN = ROOT / "experiments" / "multi_target" / "run.py"

MODEL = "llada"
BANK = "report"
FAMILY = 0  # the first family of List Functions-Easy
M = 2  # demonstrations
J = 2  # targets
K = 8  # candidates in the pool
BETA = 0.5  # the paper's common guidance strength
PICK_SEED = 0  # the uniform random number shared by the two picks


# ---------------------------------------------------------------------------------------------
# Live: sample and score the pool on a GPU
# ---------------------------------------------------------------------------------------------


def load_multi_target_run():
    """The module experiments/multi_target/run.py (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("multi_target_run", MULTI_TARGET_RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canvas_text(adapter, canvas):
    """The canvas as text, with each target site shown as <<T1 | 6 mask tokens>> instead of its mask tokens."""
    pieces = []
    start = 0
    for j, site in enumerate(canvas.sites):
        pieces.append(adapter.decode(canvas.ids[start : site[0]]))
        pieces.append(f"<<T{j + 1} | {len(site)} mask tokens>>")
        start = site[-1] + 1
    pieces.append(adapter.decode(canvas.ids[start:]))
    return "".join(pieces)


def device_name(device):
    """The provenance name of the device the model is loaded on: the GPU's name for a CUDA device (its own index, not
    the current device's), else the device type ("cpu")."""
    import torch

    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


def pool_protocol_hash(run, adapter):
    """The protocol_hash of the walkthrough's pool: Figure 5's hash of the cell (M, J) at K candidates
    (run.pool_config), as experiments/multi_target/run.py writes it. `adapter` needs only name, model_id and
    revision."""
    settings = SimpleNamespace(bank=BANK, M=[M], J=[J], K=K, chunk=run.CHUNK)
    return run.protocol_hash(run.pool_config(run.config_of(settings, adapter), M, J))


def live_record(device):
    """Sample and score the pool on a GPU, with the calls of Figure 5, and return it as one record."""
    import torch
    import transformers

    from fewshot_dlms.adapters import load
    from fewshot_dlms.canvas import build_canvas

    run = load_multi_target_run()
    family = load_families(run.TASK, BANK)[FAMILY]
    adapter = load(MODEL, device)
    print(f"loaded {adapter.model_id} @ {adapter.revision[:8]}; sampling {K} candidates ...", file=sys.stderr)

    body = run.run_canvas(adapter, family, M, J, K, run.CHUNK, audit=True)
    for key in ("p1_prompt_text", "p0_prompt_text"):  # the audit's prompt texts; canvas_text shows the canvas
        body["checks"].pop(key)
    canvas = build_canvas(adapter, family, M, J, body["oracle_L"])
    return {
        "episode_id": family["episode_id"],
        "source": run.TASK,
        "rule_id": family["rule_id"],
        "function_id": family["meta"]["function_id"],
        "model": adapter.name,
        "model_name": adapter.model_id,
        "revision": adapter.revision,
        "M": M,
        "J": J,
        "K": K,
        "beta_convention": "zero_is_ordinary",
        "protocol_hash": pool_protocol_hash(run, adapter),
        "gold": [target["output"] for target in family["targets"][:J]],
        "target_inputs": [target["input"] for target in family["targets"][:J]],
        "grader": family["grader"],
        **body,
        "canvas_text": canvas_text(adapter, canvas),
        "provenance": {
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "seed": body["seed"],
            "gpu": device_name(device),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "date": datetime.date.today().isoformat(),
            "command": "CUDA_VISIBLE_DEVICES=<gpu> python examples/walkthrough.py --save-fixture data/fixtures/walkthrough.jsonl",
        },
    }


# ---------------------------------------------------------------------------------------------
# The report: the same code prints a live pool and a stored one
# ---------------------------------------------------------------------------------------------


def correctness(record):
    """The K x J matrix of 0/1 per-target correctness of the candidates (grading.py)."""
    correct = np.zeros((len(record["candidates"]), record["J"]))
    for k, candidate in enumerate(record["candidates"]):
        for j, (text, gold) in enumerate(zip(candidate["texts"], record["gold"])):
            is_correct, _ = grade(record["grader"], answer(text, record["model"]), gold)
            correct[k, j] = is_correct
    return correct


def draw(weights, u):
    """The candidate that the uniform number u selects when candidates are drawn with these weights."""
    index = int(np.searchsorted(np.cumsum(weights), u, side="right"))
    return min(index, len(weights) - 1)


def answers_of(record, k):
    """The J answers of candidate k as one short string."""
    texts = record["candidates"][k]["texts"]
    return " | ".join(answer(text, record["model"]).strip() for text in texts)


def print_report(record):
    """Print the canvas, the table of candidates and the ordinary and guided picks of one pool record."""
    candidates = record["candidates"]
    ell1 = np.array([candidate["ell1"] for candidate in candidates])
    ell0 = np.array([candidate["ell0"] for candidate in candidates])
    reward = example_reward(ell1, ell0)
    w_ordinary = pool_weights(ell1, ell0, 0.0)
    w_guided = pool_weights(ell1, ell0, BETA)
    correct = correctness(record)
    joint = correct.all(1)
    provenance = record["provenance"]

    print(f"Pool: {record['episode_id']}  M = {record['M']} demonstrations, J = {record['J']} targets, "
          f"K = {record['K']} candidates, {record['n_tokens']} canvas tokens")
    print(f"      {provenance['model_id']} @ {provenance['revision'][:8]}, seed {provenance['seed']}, "
          f"{provenance['gpu']}, torch {provenance['torch']}, {provenance['date']}")
    print()
    print("1. The canvas (the model fills the masked sites <<T..>> in one pass; P0 also masks the demonstration outputs)")
    print("-" * 100)
    print(record["canvas_text"].rstrip())
    print("-" * 100)
    print()

    print("2. The pool: K candidates drawn from P1, each scored with (ell1) and without (ell0) the demonstration outputs")
    for j, gold in enumerate(record["gold"]):
        print(f"   reference T{j + 1}: {gold}")
    print(f"   reference answers: ell1 {record['gold_ell1']:.2f}, ell0 {record['gold_ell0']:.2f}, "
          f"R {record['gold_ell1'] - record['gold_ell0']:+.2f}")
    print()
    targets = "".join(f"{'T' + str(j + 1):>4}" for j in range(record["J"]))
    print(f"{'k':>3} {'ell1':>9} {'ell0':>9} {'R':>8} {'w_0.5':>7} {targets} {'joint':>6}  answers")
    for k in range(len(candidates)):
        per_target = "".join(f"{int(value):>4}" for value in correct[k])
        print(f"{k:>3} {ell1[k]:>9.2f} {ell0[k]:>9.2f} {reward[k]:>8.2f} {w_guided[k]:>7.4f} "
              f"{per_target} {int(joint[k]):>6}  {answers_of(record, k)}")
    print(f"    (T1.. and joint: 1 = correct. {len({tuple(c['toks']) for c in candidates})} distinct candidates "
          f"of {len(candidates)}; w_0.5 = softmax(0.5 R), each ordinary weight is 1/{len(candidates)})")
    print()

    u = float(np.random.default_rng(PICK_SEED).random())
    print(f"3. Ordinary pick (beta = 0) and guided pick (Q_0.5, beta = {BETA}); one draw each, same u = {u:.4f}")
    for label, weights in (("ordinary  Q_0  ", w_ordinary), (f"guided    Q_{BETA}", w_guided)):
        k = draw(weights, u)
        verdict = " ".join(f"T{j + 1} {'correct' if correct[k, j] else 'wrong'}" for j in range(record["J"]))
        print(f"   {label}: candidate {k}, {answers_of(record, k)}   ({verdict}; "
              f"all targets {'correct' if joint[k] else 'wrong'})")
    print("   (Identical picks are expected: both draws use the same u and often land on the same candidate; the "
          "expected accuracies below carry the comparison.)")
    print()
    print("   Expected accuracy of the pool under each weighting (what the paper averages over families):")
    print(f"   {'':<10} {'A_all':>7} {'A_avg':>7} {'ESS':>6}")
    for label, weights in (("Q_0", w_ordinary), (f"Q_{BETA}", w_guided)):
        metrics = family_metrics(correct, weights)
        print(f"   {label:<10} {metrics['A_all']:>7.3f} {metrics['A_target']:>7.3f} {metrics['ess']:>6.2f}")
    print("   (A_all: every target of the answer correct; A_avg: mean accuracy per target; ESS: 1 / sum w^2)")


def read_pool(path):
    """The one pool record of a stored walkthrough file."""
    with open(path, encoding="utf-8") as handle:
        return json.loads(handle.readline())


def save_fixture(path, record):
    """Write the pool record as the one line of a walkthrough file."""
    Path(path).write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cached", action="store_true", help="print the stored pool; no GPU, no torch")
    fixture_help = f"--cached: the stored pool to read (default {FIXTURE.relative_to(ROOT)})"
    parser.add_argument("--fixture", default=None, help=fixture_help)
    parser.add_argument("--save-fixture", default=None, help="live run: also write the pool record to this file")
    parser.add_argument("--device", default="cuda", help="live run: torch device (choose the GPU with CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()
    if args.cached and args.save_fixture:
        parser.error("--save-fixture needs a live run, not --cached")
    if args.fixture and not args.cached:
        parser.error("--fixture names the stored pool that --cached reads; a live run samples its own")
    if args.save_fixture and not Path(args.save_fixture).parent.is_dir():
        parser.error(f"--save-fixture: the directory {Path(args.save_fixture).parent} does not exist")

    if args.cached:
        print_report(read_pool(args.fixture or FIXTURE))
        return
    record = live_record(args.device)
    print_report(record)
    if args.save_fixture:
        save_fixture(args.save_fixture, record)


if __name__ == "__main__":
    main()
