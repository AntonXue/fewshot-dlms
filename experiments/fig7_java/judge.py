# ported from editdiff/private-docs-redacted/20261011-012221-sec5-final-helper-run/scratch/{fr_packets.py, gather_judgments.py, calib_controls.py, calib_score.py} and judging/rubric_r1.md @ d276fa6
"""The judge of Section 5 (Appendix C): blind packets, rubric r1, JSON verdicts.

A Java candidate is correct when it applies the demonstrated refactoring to the target; there is no test oracle
for most sites, so an LLM judge decides. For every distinct candidate text of an episode the judge sees one
blind packet: the helper the developer introduced, the episode's accepted demonstrations (before and after), and
for each target its old region and the candidate's new region in place, with up to 20 lines of context before
and 12 after (as in the canvas). The targets appear in a seeded random order labelled "Target 1", "Target 2";
the packet shows no reference answer, length regime, guidance weight, reward or score. The judge follows
rubric_r1.md (verbatim from the paper's run) and answers with a JSON verdict per target and a joint verdict:
"correct" only if every target is correct, "incorrect" if any is, else "uncertain".

The paper's verdicts were produced by Claude Sonnet 5.5 running as Claude Code sub-agents (several packets of
different episodes per sub-agent call, 2,089 packets in 272 calls; the 46 calibration controls one per call, 46/46
matching their expected verdicts). This script calls the Anthropic API instead (model claude-sonnet-5-5 by default, one packet per call,
rubric r1 as the system prompt). It is a re-implementation of the instrument, not a replay: an API call is not a
Claude Code sub-agent, and LLM verdicts vary between calls, so re-judging is approximate. The original verdicts
ship with the artifacts (java/judging/verdicts.jsonl) and analyze.py reads them by default. The pools of the
zero-demonstration ablation (run.py --study zero_demo) are not re-judged: its verdicts ship with the artifacts
(java/zero_demo/judging/verdicts.jsonl), and judge.py exits with status 2 on its pool files.

Packet identity. The cache key of a packet is the first 20 hex digits of
SHA-256("<rubric sha16>|<episode_id>|<demo sites joined by '|'>|<candidate texts as sorted JSON>"); a text judged
once is never judged again, also across regimes. The target order uses random.Random(key12("s5final-judge", key)).

    python experiments/fig7_java/judge.py --pools artifacts/reruns/fig7/pools --out artifacts/reruns/fig7/judging/verdicts.jsonl
    python experiments/fig7_java/judge.py --calibrate --limit 5      # the calibration controls of Appendix C (costs API calls)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from fewshot_dlms.data import load_java_episodes
from fewshot_dlms.data import read_jsonl

HERE = Path(__file__).resolve().parent
RUBRIC = HERE / "rubric_r1.md"
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000
CONTEXT_BEFORE = 20  # lines of the method shown before the changed region (as in canvas.py)
CONTEXT_AFTER = 12  # lines shown after it
CUT_MARKER = "    // ..."
VERDICTS = ("correct", "incorrect", "uncertain")
JUDGE_LABEL = "Claude Sonnet 5.5 via the Anthropic API ({model}), rubric r1"
ZERO_DEMO_MESSAGE = (
    "the zero-demo ablation ships the paper's verdicts; re-judging it is not supported "
    "(its packets show the anchor's demonstration for every arm; analyze.py reads java/zero_demo/judging/verdicts.jsonl)"
)
TASK = (
    "Judge the packet below with the rubric in the system prompt. Answer with the verdict JSON of the rubric's "
    "schema and nothing else.\n\n"
)


# ---------------------------------------------------------------------------------------------
# Keys and seeds
# ---------------------------------------------------------------------------------------------


def key12(*parts):
    """An integer from the first 12 hex digits of the SHA-256 of the parts joined by '|' (the run's key())."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rubric_sha(path):
    """The first 16 hex digits of the SHA-256 of the rubric file: part of every packet key."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def packet_key(rubric, episode, texts):
    """The cache key of a packet: one per (rubric, episode, demonstration sites, candidate texts)."""
    demo_sites = "|".join(episode["demo_sites"])
    text = f"{rubric}|{episode['episode_id']}|{demo_sites}|{json.dumps(texts, sort_keys=True)}"
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def packet_seed(key):
    """The seed of a packet's target order."""
    return key12("s5final-judge", key)


# ---------------------------------------------------------------------------------------------
# Packets (fr_packets.py)
# ---------------------------------------------------------------------------------------------


def window(site, which, text=None):
    """The method around a site with its region marked: <old>span_old</old>, or <new>...</new> with the accepted
    span_new (demonstrations) or a candidate's text (targets).

    At most CONTEXT_BEFORE lines before and CONTEXT_AFTER after, cut at the far end with CUT_MARKER. The lines
    around the region are the same before and after the commit (data/java keeps method_before only).
    """
    lines = site["method_before"].split("\n")
    head = lines[: site["i1"]]
    tail = lines[site["i2"] :]
    if len(head) > CONTEXT_BEFORE:
        head = [CUT_MARKER] + head[-CONTEXT_BEFORE:]
    if len(tail) > CONTEXT_AFTER:
        tail = tail[:CONTEXT_AFTER] + [CUT_MARKER]
    if which == "old":
        region = site["span_old"]
    elif text is not None:
        region = text
    else:
        region = site["span_new"]
    return "\n".join(head + [f"<{which}>", region, f"</{which}>"] + tail)


def packet(episode, texts, seed):
    """The blind packet of one candidate: (text, label map "Target n" -> target key).

    texts maps target keys ("t1", "t2") to the candidate's new regions. The episode's demonstrations are the
    accepted examples; the targets are shown in the order random.Random(seed).shuffle gives.
    """
    rng = random.Random(seed)
    targets = episode["targets_visible"]
    keys = [f"t{j + 1}" for j in range(len(targets))]
    order = list(range(len(keys)))
    rng.shuffle(order)
    header = f"{episode['repository']} {episode['commit'][:10]} {episode['file']}"
    file_name = header.split()[-1].split("/")[-1]
    parts = [f"// Helper introduced by the developer (file {file_name}):", episode["helper_text"].rstrip(), ""]
    for i, demo in enumerate(episode["demos"]):
        parts += [f"// Accepted demonstration {i + 1}: method {demo['method']}", "// BEFORE:", window(demo, "old")]
        parts += ["// AFTER (accepted):", window(demo, "new"), ""]
    labels = {}
    for n, index in enumerate(order):
        target = targets[index]
        labels[f"Target {n + 1}"] = keys[index]
        parts += [f"// Target {n + 1}: method {target['method']}", "// BEFORE (region to rewrite marked <old>):", window(target, "old")]
        parts += ["// CANDIDATE (the proposed <new> region in place):", window(target, "new", texts[keys[index]]), ""]
    return "\n".join(parts), labels


def read_pools(pools):
    """The pool records of <pools>/pools_*.jsonl in sorted file order, one per (episode, regime) (the first)."""
    records = []
    seen = set()
    for path in sorted(glob.glob(str(Path(pools) / "pools_*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # the last line of a file still being written
                unit = (record["episode_id"], record["regime"])
                if unit in seen:
                    continue
                seen.add(unit)
                records.append(record)
    return records


def holds_zero_demo_records(pools):
    """True when a pool file of `pools` holds records of the zero-demonstration ablation (keyed by unit, no regime)."""
    for path in sorted(glob.glob(str(Path(pools) / "pools_*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "unit_id" in record and "regime" not in record:
                    return True
    return False


def report_packets(records, episodes, rubric):
    """One packet per distinct (episode, candidate texts), numbered in order of first appearance."""
    packets = {}
    for record in records:
        episode = episodes[record["episode_id"]]
        for candidate in record["candidates"]:
            key = packet_key(rubric, episode, candidate["text_by_target"])
            if key in packets:
                continue
            text, labels = packet(episode, candidate["text_by_target"], packet_seed(key))
            packets[key] = {
                "key": key,
                "n": len(packets),
                "episode_id": record["episode_id"],
                "label_map": labels,
                "text": text,
                "candidate_texts": candidate["text_by_target"],
                "text_sha": candidate["text_sha"],
            }
    return list(packets.values())


def calibration_packet(episode, control):
    """The packet of a calibration control (calib_controls.py): its texts, target order seeded by (episode, control)."""
    return packet(episode, control["texts"], key12("s5final-calib", episode["episode_id"], control["control"]))


# ---------------------------------------------------------------------------------------------
# Verdicts (gather_judgments.py)
# ---------------------------------------------------------------------------------------------


def parse_verdict(reply, label_map):
    """The verdict in a judge's reply: the JSON object from the first '{' to the last '}'.

    Raises ValueError unless the reply follows the rubric's schema: a "targets" list of objects with a string
    "label", one per target, every target verdict and the joint verdict one of correct / incorrect / uncertain.
    Labels ("Target 1") are mapped back to target keys.
    """
    text = reply.strip()
    try:
        text = text[text.index("{") : text.rindex("}") + 1]
        response = json.loads(text)
    except ValueError as error:
        raise ValueError(f"no JSON verdict in the reply: {error}") from error
    if not isinstance(response, dict):
        raise ValueError("the verdict is not a JSON object")
    targets = response.get("targets", [])
    if not isinstance(targets, list) or not all(isinstance(target, dict) for target in targets):
        raise ValueError("'targets' is not a list of objects")
    if not all(isinstance(target.get("label"), str) for target in targets):
        raise ValueError("a target label is not a string")
    if response.get("joint") not in VERDICTS:
        raise ValueError(f"joint verdict {response.get('joint')!r}")
    if any(target.get("verdict") not in VERDICTS for target in targets):
        raise ValueError("a target verdict is not correct / incorrect / uncertain")
    if len(targets) != len(label_map) or any(target.get("label") not in label_map for target in targets):
        raise ValueError(f"expected one verdict for each of {sorted(label_map)}")
    return {
        "joint": response["joint"],
        "per_target": {label_map[target["label"]]: target["verdict"] for target in targets},
        "failure_tags": {label_map[target["label"]]: target.get("failure_tag", "") for target in targets},
        "instance_ill_posed": bool(response.get("instance_ill_posed", False)),
        "ill_posed_reason": response.get("ill_posed_reason", ""),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "response": response,
    }


# ---------------------------------------------------------------------------------------------
# The API call
# ---------------------------------------------------------------------------------------------


def require_api_key():
    """Exit with status 2 and a message unless the environment holds Anthropic API credentials."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return
    print("judge.py calls the Anthropic API: set ANTHROPIC_API_KEY (the paper's verdicts need no key: they ship with the artifacts)", file=sys.stderr)
    sys.exit(2)


def make_client():
    """An Anthropic client with the credentials of the environment (ANTHROPIC_API_KEY); needs the `judge` extra."""
    require_api_key()
    try:
        import anthropic
    except ImportError as error:
        raise ImportError("judge.py needs the Anthropic SDK: pip install -e '.[judge]'") from error
    return anthropic.Anthropic(max_retries=5)


def call_judge(client, model, rubric_text, packet_text):
    """One judge call: rubric r1 as the system prompt, the packet as the user turn; returns the reply text.

    The model's default sampling is used (no temperature). No fallback model is configured, so every verdict
    comes from `model`; a declined request raises RuntimeError and the packet stays unjudged.
    """
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=rubric_text,
        messages=[{"role": "user", "content": TASK + packet_text}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"the judge declined the packet ({getattr(response, 'stop_details', None)})")
    return "".join(block.text for block in response.content if block.type == "text")


def judged_keys(path):
    """Packet keys already in a verdict file (an interrupted run resumes after them)."""
    if not Path(path).exists():
        return set()
    return {record["key"] for record in read_jsonl(path)}


def append_record(path, record):
    """Append one JSON line in a single write, so that a killed run leaves complete lines only."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.write(fd, (json.dumps(record) + "\n").encode())
    os.close(fd)


def judge_one(client, args, rubric_text, text, labels):
    """Judge one packet; returns the parsed verdict, or None (printed) when the judge declines or its reply has no
    valid verdict (the packet stays unjudged and is retried on the next run).

    API errors propagate: the SDK has already retried rate limits, server errors and lost connections, and a
    rejected request (credentials, billing, model name) would fail for every packet. The run resumes from --out.
    """
    try:
        reply = call_judge(client, args.model, rubric_text, text)
        return parse_verdict(reply, labels)
    except (RuntimeError, ValueError) as error:
        print(f"  no verdict: {error}", flush=True)
        return None


def judge_report(args, episodes):
    """Judge every packet of the pools that has no verdict yet in --out."""
    rubric = rubric_sha(args.rubric)
    rubric_text = Path(args.rubric).read_text()
    packets = report_packets(read_pools(args.pools), episodes, rubric)
    done = judged_keys(args.out)
    todo = [p for p in packets if p["key"] not in done][: args.limit]
    print(f"{len(packets)} packets, {len(done)} already judged, {len(todo)} to judge with {args.model}", flush=True)
    client = make_client() if todo else None
    for p in todo:
        verdict = judge_one(client, args, rubric_text, p["text"], p["label_map"])
        if verdict is None:
            continue
        record = {
            "n": p["n"],
            "key": p["key"],
            "episode_id": p["episode_id"],
            "judge_model": JUDGE_LABEL.format(model=args.model),
            "rubric_sha": rubric,
            "packet_sha256": sha256_text(p["text"]),
            **verdict,
        }
        append_record(args.out, record)
        print(f"packet {p['n']} {p['episode_id']}: {verdict['joint']}", flush=True)


def judge_calibration(args, episodes):
    """The calibration controls of Appendix C: reference, no-op, partial edit, argument swap, polarity flip,
    wrong receiver (expected incorrect) and an equivalent reordering or a harmless comment (expected correct).

    Prints, per control, the verdict, the expected verdict and the paper judge's verdict (calibration.csv).
    """
    rubric_text = Path(args.rubric).read_text()
    controls = read_jsonl(args.controls)[: args.limit]
    paper = {}
    paper_path = Path(args.controls).parents[1] / "calibration.csv"
    if paper_path.exists():
        with open(paper_path) as handle:
            paper = {int(row["n"]): row["judged"] for row in csv.DictReader(handle)}
    client = make_client()
    agree_expected = 0
    agree_paper = 0
    judged = 0
    for control in controls:
        text, labels = calibration_packet(episodes[control["episode_id"]], control)
        verdict = judge_one(client, args, rubric_text, text, labels)
        if verdict is None:
            continue
        judged += 1
        agree_expected += verdict["joint"] == control["expected"]
        agree_paper += verdict["joint"] == paper.get(control["n"])
        record = {"n": control["n"], "episode_id": control["episode_id"], "control": control["control"], "expected": control["expected"]}
        record.update({"paper_judge": paper.get(control["n"]), "judge_model": JUDGE_LABEL.format(model=args.model), **verdict})
        append_record(args.out, record)
        print(f"control {control['n']:3d} {control['control']:20s} expected {control['expected']:9s} paper {paper.get(control['n'])!s:9s} now {verdict['joint']}", flush=True)
    print(f"calibration: {judged} judged; {agree_expected} match the expected verdict, {agree_paper} match the paper's judge")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pools", default="artifacts/reruns/fig7/pools", help="directory with pools_*.jsonl (run.py)")
    parser.add_argument("--rubric", default=str(RUBRIC))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default=None, help="verdicts JSONL (appended; default <pools>/../judging/verdicts.jsonl)")
    parser.add_argument("--calibrate", action="store_true", help="judge the 46 calibration controls instead of the pools")
    parser.add_argument("--controls", default="artifacts/java/judging/calibration/controls.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="judge at most this many packets")
    args = parser.parse_args()

    episodes = {episode["episode_id"]: episode for episode in load_java_episodes()}
    if args.calibrate:
        if not Path(args.controls).exists():
            print(f"no calibration controls at {args.controls}; run: python -m fewshot_dlms.fetch pools java", file=sys.stderr)
            sys.exit(2)
        require_api_key()
        args.out = args.out or "artifacts/reruns/fig7/judging/calibration_verdicts.jsonl"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        judge_calibration(args, episodes)
        return
    if not any(Path(args.pools).glob("pools_*.jsonl")):
        print(f"no pools in {args.pools}", file=sys.stderr)
        sys.exit(2)
    if holds_zero_demo_records(args.pools):
        print(ZERO_DEMO_MESSAGE, file=sys.stderr)
        sys.exit(2)
    require_api_key()
    args.out = args.out or str(Path(args.pools).parent / "judging" / "verdicts.jsonl")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    judge_report(args, episodes)


if __name__ == "__main__":
    main()
