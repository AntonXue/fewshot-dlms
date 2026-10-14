"""Assemble the Hugging Face dataset UT-IFML/fewshot-dlms: the frozen artifacts the experiments read.

Maintainer tool, run once per release; users download its result with `python -m fewshot_dlms.fetch`, which places
the tree under artifacts/. The script copies the authors' pools, length heads and head-training data from their
private run directories (the SOURCES block) into one upload directory with exactly that layout:

  pools/fig4/{report,dev}/<model>/<task>_M{M}_J1.jsonl         Section 4.1 pools (llada, nemotron_ar, dqwen, qwen)
  pools/fig4/{report,dev}/<model>/<task>_M4_J{J}_il.jsonl      cross-benchmark pools (llada, nemotron_pt, dqwen)
  pools/fig4/gold/<model>/<task>_M{M}_J1.jsonl                 reference answers scored under P1 and P0
  pools/fig5/<model>/list_easy_fam_M{M}_J{J}.jsonl             the M x J grid
  pools/fig5/crossbench_gold/<model>/<task>_M4_J8_il.jsonl     reference answers of the J = 8 cross-benchmark cells
  pools/fig6/<model>/list_easy_fam_M4_J{J}_<condition>.jsonl   oracle, learned and offset lengths; length_head_report.*
  java/{pools,judging,length,zero_demo}/                      Section 5: pools, verdicts, head predictions, ablation
  heads/{llada,nemotron,dqwen,java_head_C}.pt                 the four length heads
  head_training/                                              the rendered training prompts of the three benchmark
                                                              heads, the heads' configs, head C's corpus manifests
  README.md                                                   the data card (artifacts_upload/README.md, copied)
  SHA256SUMS                                                  "<sha256>  <path>" of every other file, sorted by path

What changes on the way; everything else is copied byte for byte:
  - JSONL records lose the keys in RUN_METADATA (the machine, the moment and the private paths of a run). A line
    without such a key, and every file without one, is kept byte for byte.
  - Records of SST-2 and MIT Movie lose the benchmark text, as in data/fixtures: target_inputs and the decoded
    prompts in checks become SHA-256 digests under the same key + "_sha256"
    (extract_fixtures.without_benchmark_text). The reference answers and the sampled candidates stay.
  - The training configs of the heads lose CONFIG_DROPPED; head C's training corpus ships as manifests, every field
    of a row except MANIFEST_DROPPED (the token ids).
  - The head checkpoints are re-saved (torch.save) with CONFIG_DROPPED removed from the training config pickled in
    them; their tensors are checked to be identical, byte for byte, to the authors' final.pt. The re-saved files have
    new SHA-256s; the stored pools and predictions keep recording the original ones.
Then it hashes the tree into SHA256SUMS (also written to the repository root, where fetch.py verifies downloads
against it) and checks it: no record keeps a key of RUN_METADATA, no SST-2 or MIT Movie record keeps prompt text,
and no file matches PRIVATE_PATTERN (in JSONL files, sampled candidate texts excepted). It exits with status 1 if a check fails or a source is missing.

Re-running is safe: unchanged outputs are not rewritten and files that are no longer planned are removed. The
upload directory is outside the repository (large files); artifacts_upload/ in the repository holds the data card,
which is tracked, and git-ignored links to the upload directory's top-level entries.

Usage:
    python tools/assemble_artifacts.py            # assemble, hash and check
    python tools/assemble_artifacts.py --check    # hash and check the existing tree only
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from extract_fixtures import without_benchmark_text  # tools/ is on sys.path when this script runs
from fewshot_dlms.data import TEXT_FREE_BENCHMARKS

# ---- SOURCES: the authors' private run directories (machine-specific; the public release replaces this block) ----
FINAL_SWEEPS = Path("/disk-a-redacted/user-redacted/final_sweeps_4_1_4_2_4_4_20261003-062825/pools")
GOLD_PROBE = Path("/disk-a-redacted/user-redacted/gold_reward_20261007-173712/report")
MJ_GRID = Path("/disk-a-redacted/user-redacted/sec42_mj_20261011-070018/pools")
LENGTH_PROJECT = Path("/disk-a-redacted/user-redacted/length_project_20261007-061559")
JAVA_LENGTH = Path("/disk-a-redacted/user-redacted/sec5_length_20261010-205040")
JAVA_RUN = Path("/private-repo-redacted/private-docs-redacted/20261011-012221-sec5-final-helper-run")
JAVA_ZERO_DEMO = Path("/private-repo-redacted/private-docs-redacted/20261011-055211-s5-zero-demo-ablation")
UPLOAD_DIR = Path("/disk-a-redacted/user-redacted/fewshot_dlms_upload")
# ---- end SOURCES ----

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "artifacts_upload" / "README.md"
LINK_DIR = ROOT / "artifacts_upload"

TASKS = ("sst2_fam", "list_easy_fam", "salt_rep_fam", "movie_dir_fam")
SECTION41_MODELS = ("llada", "nemotron_ar", "dqwen", "qwen")  # Section 4.1: one target, M = 1..16
CANVAS_MODELS = ("llada", "nemotron_pt", "dqwen")  # the three DLM canvases (cross-benchmark, M x J grid, lengths)
DEMO_COUNTS = (1, 2, 4, 8, 16)
TARGET_COUNTS = (1, 2, 4, 8)
GRID_DEMO_COUNTS = (2, 4, 8, 16)
# Section 4.3: model directory -> (head backbone, the length project's run of that head)
HEAD_RUNS = {
    "llada": ("llada", "fin_llada_m70_100k"),
    "nemotron_pt": ("nemotron", "fin_nemotron_m70_100k"),
    "dqwen": ("dqwen", "fin_dqwen_m70_100k_g2"),
}
OFFSETS = ("-2", "-1", "+1", "+2")  # Section 4.3 offsets; dQwen3.5 also has its same-session delta0 rerun
JAVA_POOLS = ("learned_0", "learned_0m", "learned_0m2", "learned_0r", "learned_1", "learned_1r", "oracle_0", "oracle_1")
JAVA_LENGTH_HEADS = ("s5hc_C", "s5g_B")  # head C and the layout-matched comparator; the third is fin_llada_m70_100k
ZERO_DEMO_POOLS = ("primary", "primary_rev", "ladder")
HEAD_C_CORPUS = ("s5g", "s5h")

# Keys of a record that describe the machine, the moment or the private paths of a run, not its result.
RUN_METADATA = ("host", "hostname", "gpu_env", "gpu", "device", "pid", "cwd", "root", "wall_time", "pool_file",
                "lengths_file")
CONFIG_DROPPED = ("root",)  # a head's training config: the run directory
MANIFEST_DROPPED = ("input_ids", "mask_positions")  # head C's corpus rows: the tokenised code
# Text that must not reach the tree: machine paths and internal working notes (generic, no names).
PRIVATE_PATTERN = re.compile(rb"/(?:hdd|ssd)\d|/home/|/mnt/|hand-off|\.claude")
GROUPS = ("pools/fig4", "pools/fig5", "pools/fig6", "java", "heads", "head_training", "README.md")
TOP_LEVEL = ("pools", "java", "heads", "head_training", "SHA256SUMS")
MAX_BYTES = 12 * 10 ** 9


def plan():
    """Every file of the tree except README.md and SHA256SUMS: a list of (path in the tree, source, kind).

    kind is "copy" (bytes), "jsonl" (records without RUN_METADATA and benchmark text), "config" (JSON without
    CONFIG_DROPPED), "manifest" (JSONL rows without MANIFEST_DROPPED) or "head" (a checkpoint re-saved without
    CONFIG_DROPPED in its training config).
    """
    files = []
    for bank in ("report", "dev"):
        for model in SECTION41_MODELS:
            for task in TASKS:
                for M in DEMO_COUNTS:
                    name = f"{bank}/{model}/{task}_M{M}_J1.jsonl"
                    files.append((f"pools/fig4/{name}", FINAL_SWEEPS / name, "jsonl"))
        for model in CANVAS_MODELS:
            for task in TASKS:
                for J in TARGET_COUNTS:
                    name = f"{bank}/{model}/{task}_M4_J{J}_il.jsonl"
                    files.append((f"pools/fig4/{name}", FINAL_SWEEPS / name, "jsonl"))
    for model in SECTION41_MODELS:
        for task in TASKS:
            for M in DEMO_COUNTS:
                name = f"{model}/{task}_M{M}_J1.jsonl"
                files.append((f"pools/fig4/gold/{name}", GOLD_PROBE / name, "jsonl"))
    for model in CANVAS_MODELS:
        for M in GRID_DEMO_COUNTS:
            for J in TARGET_COUNTS:
                name = f"{model}/list_easy_fam_M{M}_J{J}.jsonl"
                files.append((f"pools/fig5/{name}", MJ_GRID / name, "jsonl"))
        for task in TASKS:
            name = f"{model}/{task}_M4_J8_il.jsonl"
            files.append((f"pools/fig5/crossbench_gold/{name}", GOLD_PROBE / name, "jsonl"))
    for model, (backbone, run) in HEAD_RUNS.items():
        dest = f"pools/fig6/{model}"
        ds_run = LENGTH_PROJECT / "ds_run" / "pools" / "report" / model
        for J in TARGET_COUNTS:
            files.append((f"{dest}/list_easy_fam_M4_J{J}_oracle.jsonl", FINAL_SWEEPS / "report" / model / f"list_easy_fam_M4_J{J}_il.jsonl", "jsonl"))
            files.append((f"{dest}/list_easy_fam_M4_J{J}_learned.jsonl", ds_run / f"list_easy_fam_M4_J{J}_pc_learned_{run}.jsonl", "jsonl"))
        offsets = OFFSETS + ("0",) if model == "dqwen" else OFFSETS
        for J in (1, 2):
            for delta in offsets:
                files.append((f"{dest}/list_easy_fam_M4_J{J}_delta{delta}.jsonl", ds_run / f"list_easy_fam_M4_J{J}_pc_delta{delta}.jsonl", "jsonl"))
        files.append((f"{dest}/length_head_report.npz", LENGTH_PROJECT / "eval" / f"{run}_listbench_report.npz", "copy"))
        files.append((f"{dest}/length_head_report.jsonl", LENGTH_PROJECT / "data" / f"listbench_report_{backbone}.jsonl", "jsonl"))
    for name in JAVA_POOLS:
        files.append((f"java/pools/pools_{name}.jsonl", JAVA_RUN / "generation" / "report" / f"pools_{name}.jsonl", "jsonl"))
    for name in ("verdicts.jsonl", "packet_index.jsonl"):
        files.append((f"java/judging/{name}", JAVA_RUN / "judging" / name, "jsonl"))
        files.append((f"java/zero_demo/judging/{name}", JAVA_ZERO_DEMO / "judging" / name, "jsonl"))
    files.append(("java/judging/calibration.csv", JAVA_RUN / "judging" / "calibration.csv", "copy"))
    files.append(("java/judging/calibration/controls.jsonl", JAVA_RUN / "judging" / "calibration" / "controls.jsonl", "jsonl"))
    for head in JAVA_LENGTH_HEADS:
        files.append((f"java/length/{head}_s5final_G4.npz", JAVA_LENGTH / "eval" / f"{head}_s5final_G4.npz", "copy"))
    files.append(("java/length/fin_llada_m70_100k_s5final_G4.npz", LENGTH_PROJECT / "eval" / "fin_llada_m70_100k_s5final_G4.npz", "copy"))
    files.append(("java/length/prior_s5h_train.json", JAVA_RUN / "length" / "prior_s5h_train.json", "copy"))
    for name in ZERO_DEMO_POOLS:
        files.append((f"java/zero_demo/pools/pools_{name}.jsonl", JAVA_ZERO_DEMO / "generation" / "report" / f"pools_{name}.jsonl", "jsonl"))
    for backbone, run in HEAD_RUNS.values():
        files.append((f"heads/{backbone}.pt", LENGTH_PROJECT / "runs" / run / "final.pt", "head"))
        files.append((f"head_training/configs/{backbone}.json", LENGTH_PROJECT / "runs" / run / "config.json", "config"))
        for name in (f"fsp_{backbone}_train", f"edit_{backbone}_train", f"listbench_dev_{backbone}", f"edit_{backbone}_dev"):
            files.append((f"head_training/{name}.jsonl", LENGTH_PROJECT / "data" / f"{name}.jsonl", "jsonl"))
    files.append(("head_training/fsp_logical_train.jsonl", LENGTH_PROJECT / "data" / "fsp_logical_train.jsonl", "jsonl"))
    files.append(("heads/java_head_C.pt", JAVA_LENGTH / "runs" / "s5hc_C" / "final.pt", "head"))
    files.append(("head_training/configs/java_head_C.json", JAVA_LENGTH / "runs" / "s5hc_C" / "config.json", "config"))
    for corpus in HEAD_C_CORPUS:
        for split in ("train", "dev"):
            files.append((f"head_training/java_head_C/{corpus}_{split}_manifest.jsonl", JAVA_LENGTH / "data" / f"{corpus}_llada_{split}.jsonl", "manifest"))
    files.append(("head_training/java_head_C/corpus_audit.json", JAVA_LENGTH / "corpus_audit.json", "copy"))
    return files


def jsonl_lines(data):
    """The lines of a JSONL file's bytes, split at "\\n" only (str.splitlines would also split inside strings)."""
    return data.decode("utf-8").split("\n")


def dump_like(record, line):
    """One JSON line (without "\\n") for `record`, escaping non-ASCII text only if the original line did."""
    return json.dumps(record, ensure_ascii=line.isascii())


def clean_record(record):
    """The record without RUN_METADATA and, for SST-2 and MIT Movie, without benchmark text.

    The candidates are model output and stay as sampled: a sampled answer may continue the prompt's pattern
    ("... Input: ..."), which without_benchmark_text's own check would take for prompt text, so they are set
    aside while the rest of the record is hashed.
    """
    record = {key: value for key, value in record.items() if key not in RUN_METADATA}
    if record.get("source") not in TEXT_FREE_BENCHMARKS:
        return record
    if "candidates" not in record:
        return without_benchmark_text(record)
    hashed = without_benchmark_text(dict(record, candidates=[]))
    hashed["candidates"] = record["candidates"]
    return hashed


def cleaned_jsonl(source):
    """The bytes of a cleaned JSONL file, or None when the source needs no change (then it is copied)."""
    data = source.read_bytes()
    needles = [f'"{key}":'.encode() for key in RUN_METADATA] + [f'"{task}"'.encode() for task in TEXT_FREE_BENCHMARKS]
    if not any(needle in data for needle in needles):
        return None
    lines = []
    changed = False
    for line in jsonl_lines(data):
        if not line.strip():
            lines.append(line)
            continue
        record = json.loads(line)
        cleaned = clean_record(record)
        if cleaned == record:
            lines.append(line)
        else:
            lines.append(dump_like(cleaned, line))
            changed = True
    return "\n".join(lines).encode("utf-8") if changed else None


def manifest_jsonl(source):
    """head C's corpus rows without the token ids."""
    lines = []
    for line in jsonl_lines(source.read_bytes()):
        if line.strip():
            record = {key: value for key, value in json.loads(line).items() if key not in MANIFEST_DROPPED}
            lines.append(dump_like(record, line) + "\n")
    return "".join(lines).encode("utf-8")


def cleaned_config(source):
    config = json.loads(source.read_text(encoding="utf-8"))
    config = {key: value for key, value in config.items() if key not in CONFIG_DROPPED}
    return (json.dumps(config, indent=1) + "\n").encode("utf-8")


def same_tensors(original, resaved):
    """True if two state dicts hold the same names, dtypes, shapes and bytes (torch.equal and the raw bytes)."""
    if list(original) != list(resaved):
        return False
    for name, tensor in original.items():
        other = resaved[name]
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            return False
        if not torch.equal(tensor.contiguous().view(torch.uint8), other.contiguous().view(torch.uint8)):
            return False
    return True


def resaved_head(source, dest):
    """Write the head checkpoint `source` to `dest` without CONFIG_DROPPED in its training config.

    The checkpoint is {"branch": state dict, "config": training config} (lengths.LengthHead.load). It is saved under
    its final file name, which torch.save writes into the archive, so the bytes depend only on the content. The new
    file must load with weights_only=True, hold the same tensors byte for byte and the same config minus
    CONFIG_DROPPED, and no string of the config may match PRIVATE_PATTERN. Returns True if `dest` was written.
    """
    original = torch.load(source, map_location="cpu", weights_only=True)
    config = {key: value for key, value in original["config"].items() if key not in CONFIG_DROPPED}
    if PRIVATE_PATTERN.search(json.dumps(config).encode("utf-8")):
        raise ValueError(f"{source}: the training config still holds a private path")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
        staged = Path(staging) / dest.name
        torch.save(dict(original, config=config), staged)
        resaved = torch.load(staged, map_location="cpu", weights_only=True)
        if list(resaved) != list(original) or resaved["config"] != config:
            raise ValueError(f"{dest}: re-saved checkpoint differs outside the tensors")
        if not same_tensors(original["branch"], resaved["branch"]):
            raise ValueError(f"{dest}: re-saved tensors differ from {source}")
        if dest.is_file() and dest.stat().st_size == staged.stat().st_size and dest.read_bytes() == staged.read_bytes():
            return False
        os.replace(staged, dest)
    return True


def write_bytes(dest, data):
    """Write `data` to `dest` unless it already holds exactly these bytes; returns True if written."""
    if dest.is_file() and dest.stat().st_size == len(data) and dest.read_bytes() == data:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, dest)
    return True


def copy_file(source, dest):
    """Copy `source` to `dest` with its modification time, unless dest already has the same size and time."""
    if dest.is_file() and not dest.is_symlink():
        here = dest.stat()
        there = source.stat()
        if here.st_size == there.st_size and here.st_mtime_ns == there.st_mtime_ns:
            return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, dest)
    return True


def assemble(dest, files):
    """Write every planned file into `dest`, copy the data card, and remove files that are no longer planned."""
    written = 0
    for path, source, kind in files:
        target = dest / path
        if kind == "copy":
            written += copy_file(source, target)
        elif kind == "jsonl":
            data = cleaned_jsonl(source)
            written += copy_file(source, target) if data is None else write_bytes(target, data)
        elif kind == "manifest":
            written += write_bytes(target, manifest_jsonl(source))
        elif kind == "config":
            written += write_bytes(target, cleaned_config(source))
        elif kind == "head":
            written += resaved_head(source, target)
        else:
            raise ValueError(kind)
    written += write_bytes(dest / "README.md", CARD.read_bytes())
    planned = {dest / path for path, _, _ in files}
    for top in TOP_LEVEL[:-1]:
        for path in sorted((dest / top).rglob("*"), reverse=True):
            if path.is_file() and path not in planned:
                print(f"removed (no longer planned): {path.relative_to(dest)}")
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    print(f"assembled {len(files) + 1} files into the upload directory ({written} written, the rest unchanged)")


def tree_files(dest):
    """Every file of the tree that SHA256SUMS lists: all but SHA256SUMS itself, temporary files and .cache/."""
    paths = []
    for path in dest.rglob("*"):
        relative = path.relative_to(dest).as_posix()
        if relative.startswith(".cache/") or relative == "SHA256SUMS" or relative.endswith(".tmp"):
            continue
        if path.is_file():
            paths.append(relative)
    return sorted(paths)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sums(dest, paths):
    """SHA256SUMS in the tree and at the repository root, in the format of sha256sum: '<hex>  <path>'."""
    text = "".join(f"{sha256_file(dest / path)}  {path}\n" for path in paths)
    write_bytes(dest / "SHA256SUMS", text.encode("utf-8"))
    write_bytes(ROOT / "SHA256SUMS", text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sampled_matches(data):
    """The sampled candidate texts of a JSONL file that match PRIVATE_PATTERN, or None if any match lies elsewhere.

    A model's sample can contain a path it made up (a Nemotron pool of Figure 6 holds a made-up home-directory path);
    such text is model output, which the analysis grades, and stays. Any match outside candidates' texts fails.
    """
    samples = []
    for line in jsonl_lines(data):
        if not PRIVATE_PATTERN.search(line.encode("utf-8")):
            continue
        record = json.loads(line)
        candidates = record.get("candidates") or []
        texts = [text for candidate in candidates for text in candidate.get("texts", [])]
        rest = dict(record, candidates=[{key: value for key, value in candidate.items() if key != "texts"} for candidate in candidates])
        if PRIVATE_PATTERN.search(json.dumps(rest).encode("utf-8")):
            return None
        samples.extend(text for text in texts if PRIVATE_PATTERN.search(text.encode("utf-8")))
    return samples


def check(dest, paths):
    """The checks of the module doc; returns the list of failures."""
    failures = []
    for path in paths:
        data = (dest / path).read_bytes()
        hits = sorted({match.group().decode() for match in PRIVATE_PATTERN.finditer(data)})
        samples = sampled_matches(data) if hits and path.endswith(".jsonl") else None
        if hits and samples:
            print(f"note: {path}: {hits} only inside sampled candidate texts (model output, kept): {samples}")
        elif hits:
            failures.append(f"{path}: private text {hits}")
        if not path.endswith(".jsonl"):
            continue
        for number, line in enumerate(jsonl_lines(data), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            kept = sorted(set(record) & set(RUN_METADATA))
            if kept:
                failures.append(f"{path}:{number}: run metadata {kept}")
                break
            if record.get("source") in TEXT_FREE_BENCHMARKS:
                rest = json.dumps({key: value for key, value in record.items() if key != "candidates"})
                if "target_inputs" in record or "Input:" in rest:
                    failures.append(f"{path}:{number}: benchmark text of {record['source']}")
                    break
    return failures


def group_of(path):
    return next(group for group in GROUPS if path == group or path.startswith(group + "/"))


def summary(dest, paths, sums_sha):
    """The per-group table of ARTIFACTS.md: files and bytes."""
    counts = {group: [0, 0] for group in GROUPS}
    for path in paths:
        counts[group_of(path)][0] += 1
        counts[group_of(path)][1] += (dest / path).stat().st_size
    total = sum(size for _, size in counts.values())
    print("| group | files | bytes | size |")
    print("|---|---|---|---|")
    for group, (count, size) in counts.items():
        print(f"| `{group}` | {count} | {size:,} | {size / 1e9:.3f} GB |")
    print(f"| total | {len(paths)} | {total:,} | {total / 1e9:.3f} GB |")
    print(f"SHA-256 of SHA256SUMS: {sums_sha}")
    return total


def link_top_level(dest):
    """Git-ignored links artifacts_upload/<entry> -> <dest>/<entry>, so the repository shows the uploaded tree."""
    if LINK_DIR.resolve() == dest.resolve():
        return
    for entry in TOP_LEVEL:
        link = LINK_DIR / entry
        if link.is_symlink():
            if Path(os.readlink(link)) == dest / entry:
                continue
            link.unlink()
        elif link.exists():
            raise SystemExit(f"{link} exists and is not a link; remove it first")
        link.symlink_to(dest / entry)


def main():
    print("warning: tools/assemble_artifacts.py reads the authors' private run directories (SOURCES block)", file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dest", type=Path, default=UPLOAD_DIR, help="the upload directory")
    parser.add_argument("--check", action="store_true", help="only hash and check the existing tree")
    args = parser.parse_args()
    files = plan()
    if not args.check:
        missing = [str(source) for _, source, _ in files if not source.is_file()]
        if missing or not CARD.is_file():
            print("missing sources:\n  " + "\n  ".join(missing + ([str(CARD)] if not CARD.is_file() else [])), file=sys.stderr)
            sys.exit(1)
        assemble(args.dest, files)
    paths = tree_files(args.dest)
    expected = sorted([path for path, _, _ in files] + ["README.md"])
    if paths != expected:
        print(f"tree differs from the plan: extra {sorted(set(paths) - set(expected))}, missing {sorted(set(expected) - set(paths))}", file=sys.stderr)
        sys.exit(1)
    sums_sha = write_sums(args.dest, paths)
    link_top_level(args.dest)
    failures = check(args.dest, paths)
    total = summary(args.dest, paths, sums_sha)
    if total > MAX_BYTES:
        failures.append(f"tree is {total / 1e9:.1f} GB, more than {MAX_BYTES / 1e9:.0f} GB")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print("CHECKS_FAILED" if failures else "CHECKS_OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
