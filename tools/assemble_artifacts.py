"""Assemble the Hugging Face dataset UT-IFML/fewshot-dlms: the frozen artifacts the experiments read.

Maintainer tool, run once per release; users download its result with `python -m fewshot_dlms.fetch`, which places
the tree under artifacts/. The script copies the authors' pools, length heads and head-training data from their
private run directories (tools/sources.local.json, not in the repository) into one upload directory with
exactly that layout:

  pools/single_target/{report,dev}/<model>/<task>_M{M}_J1.jsonl            Section 4.1 pools (llada, nemotron_ar, dqwen, qwen)
  pools/single_target/{report,dev}/<model>/<task>_M4_J{J}_il.jsonl         cross-benchmark pools (llada, nemotron_pt, dqwen)
  pools/single_target/gold/<model>/<task>_M{M}_J1.jsonl                    reference answers scored under P1 and P0
  pools/multi_target/<model>/list_easy_fam_M{M}_J{J}.jsonl                 the M x J grid
  pools/multi_target/crossbench_gold/<model>/<task>_M4_J8_il.jsonl         reference answers of the J = 8 cross-benchmark cells
  pools/length_allocation/<model>/list_easy_fam_M4_J{J}_<condition>.jsonl  oracle, learned and offset lengths;
                                                                           length_head_report.*
  java/{pools,judging,length,zero_demo}/                                   Section 5: pools, verdicts, head predictions, ablation
  heads/{llada,nemotron,dqwen,java_head_C}.pt                              the four length heads
  head_training/                                                           the rendered training prompts of the three benchmark
                                                                           heads, the heads' configs, head C's corpus manifests
  README.md                                                                the data card (artifacts_upload/README.md, copied)
  SHA256SUMS                                                               "<sha256>  <path>" of every other file, sorted by path

What changes on the way; everything else is copied byte for byte:
  - JSONL records lose the keys in RUN_METADATA (the machine, the moment and the private paths of a run). A line
    without such a key, and every file without one, is kept byte for byte.
  - Records of SST-2 and MIT Movie lose the benchmark text, as in data/fixtures: target_inputs and the decoded
    prompts in checks become SHA-256 digests under the same key + "_sha256"
    (extract_fixtures.without_benchmark_text). The reference answers and the sampled candidates stay.
  - The structured few-shot training files (head_training/fsp_*.jsonl: the logical episodes and the three rendered
    files) lose the rows whose "source" is in EXCLUDED_FSP_SOURCES (MIT Movie and ATIS); every kept line is copied
    byte for byte, in its order (fsp_jsonl). The frozen heads were trained on the full set.
  - The training configs of the heads lose CONFIG_DROPPED; head C's training corpus ships as manifests, every field
    of a row except MANIFEST_DROPPED (the token ids).
  - The head checkpoints are re-saved (torch.save) with HEAD_CONFIG_DROPPED removed from the training config pickled
    in them; their tensors are checked to be identical, byte for byte, to the authors' final.pt. The re-saved files have
    new SHA-256s; the stored pools and predictions keep recording the original ones.
Then it hashes the tree into SHA256SUMS (also written to the repository root, where fetch.py verifies downloads
against it) and checks it: no record keeps a key of RUN_METADATA, no SST-2 or MIT Movie record keeps prompt text,
no FSP file holds a row of EXCLUDED_FSP_SOURCES and the four FSP files list the same ids in the same order, and no
file, text or binary, holds private text (private_hits: fewshot_dlms.leakcheck plus LOCAL_PRIVATE; in JSONL
files, sampled candidate texts excepted). With --check it also requires the tree to match the repository's SHA256SUMS
and writes nothing. It exits with status 1 if a check fails or a source is missing, and with status 2 if
tools/sources.local.json is missing or incomplete (with --check it needs only upload_dir).

Re-running is safe: unchanged outputs are not rewritten; files that are no longer planned and leftover temporary
files (*.tmp) are removed, and any other file outside the plan stops the run. The
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
from fewshot_dlms.leakcheck import find_private

# The authors' private run directories are not in the repository: tools/sources.local.json (git-ignored) maps each
# key of SOURCE_KEYS to a directory; tools/sources.example.json lists the keys with placeholder paths.
SOURCES_FILE = Path(__file__).resolve().parent / "sources.local.json"
SOURCE_KEYS = (
    "final_sweeps",  # the Section 4.1 and cross-benchmark pools: <bank>/<model>/<task>_M{M}_J{J}[_il].jsonl
    "gold_probe",  # the reference answers scored under P1 and P0: <model>/<task>_M{M}_J{J}[_il].jsonl
    "mj_grid",  # the M x J grid: <model>/list_easy_fam_M{M}_J{J}.jsonl
    "length_project",  # Section 4.3: ds_run/, eval/, data/, runs/<head run>/
    "java_length",  # head C of Section 5: eval/, data/, runs/s5hc_C/, corpus_audit.json
    "java_run",  # the Section 5 run: generation/report/, judging/, length/
    "java_zero_demo",  # the zero-demonstration ablation: generation/report/, judging/
    "upload_dir",  # the default --dest: the upload directory, outside the repository
)

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
CONFIG_DROPPED = ("root", "gpu")  # a head's training config (JSON): the run directory and the GPU index
HEAD_CONFIG_DROPPED = ("root",)  # the config pickled in a head checkpoint: the run directory only; weights unchanged
MANIFEST_DROPPED = ("input_ids", "mask_positions")  # head C's corpus rows: the tokenised code
# The values of a structured few-shot (FSP) row's "source" whose rows the public head_training/ omits: MIT Movie
# ("movies", fields other than Director) and ATIS queries. The two corpora keep their own terms, and the code repository
# itself ships MIT Movie as record ids only (data/benchmarks/build.py rebuilds the text), so neither their plain text
# (fsp_logical_train.jsonl) nor their token ids (fsp_<backbone>_train.jsonl) are redistributed. The frozen heads were
# trained on all 100,000 prompts and are shipped as trained.
EXCLUDED_FSP_SOURCES = ("movies", "atis")
FSP_PREFIX = "head_training/fsp_"  # the logical FSP episodes and their three renderings, one row per episode id
# Text that must not reach the tree: the generic machine paths and metadata keys of fewshot_dlms.leakcheck, and
# (this site's own) the vocabulary of internal working notes. No names.
LOCAL_PRIVATE = re.compile(r"hand-off|\.claude")
GROUPS = ("pools/single_target", "pools/multi_target", "pools/length_allocation", "java", "heads", "head_training", "README.md")
TOP_LEVEL = ("pools", "java", "heads", "head_training", "SHA256SUMS")
MAX_BYTES = 12 * 10 ** 9


def load_sources(path=SOURCES_FILE, required=SOURCE_KEYS):
    """The directories of `path` (JSON: key -> directory) for the keys of SOURCE_KEYS it holds; exits with status 2
    without the file or without a key of `required`."""
    if not path.is_file():
        print(f"no {path.name}: copy tools/sources.example.json to {path} and set the authors' run directories",
              file=sys.stderr)
        sys.exit(2)
    sources = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in required if key not in sources]
    if missing:
        print(f"{path.name} lacks the keys {', '.join(missing)} (see tools/sources.example.json)", file=sys.stderr)
        sys.exit(2)
    return {key: Path(sources[key]) for key in SOURCE_KEYS if key in sources}


def plan(sources):
    """Every file of the tree except README.md and SHA256SUMS: a list of (path in the tree, source, kind).

    `sources` maps each key of SOURCE_KEYS to its directory (load_sources).

    kind is "copy" (bytes), "jsonl" (records without RUN_METADATA and benchmark text), "fsp" (the lines of the rows
    outside EXCLUDED_FSP_SOURCES, byte for byte), "config" (JSON without CONFIG_DROPPED), "manifest" (JSONL rows
    without MANIFEST_DROPPED) or "head" (a checkpoint re-saved without HEAD_CONFIG_DROPPED in its training config).
    """
    final_sweeps = sources["final_sweeps"]
    gold_probe = sources["gold_probe"]
    mj_grid = sources["mj_grid"]
    length_project = sources["length_project"]
    java_length = sources["java_length"]
    java_run = sources["java_run"]
    java_zero_demo = sources["java_zero_demo"]
    files = []
    for bank in ("report", "dev"):
        for model in SECTION41_MODELS:
            for task in TASKS:
                for M in DEMO_COUNTS:
                    name = f"{bank}/{model}/{task}_M{M}_J1.jsonl"
                    files.append((f"pools/single_target/{name}", final_sweeps / name, "jsonl"))
        for model in CANVAS_MODELS:
            for task in TASKS:
                for J in TARGET_COUNTS:
                    name = f"{bank}/{model}/{task}_M4_J{J}_il.jsonl"
                    files.append((f"pools/single_target/{name}", final_sweeps / name, "jsonl"))
    for model in SECTION41_MODELS:
        for task in TASKS:
            for M in DEMO_COUNTS:
                name = f"{model}/{task}_M{M}_J1.jsonl"
                files.append((f"pools/single_target/gold/{name}", gold_probe / name, "jsonl"))
    for model in CANVAS_MODELS:
        for M in GRID_DEMO_COUNTS:
            for J in TARGET_COUNTS:
                name = f"{model}/list_easy_fam_M{M}_J{J}.jsonl"
                files.append((f"pools/multi_target/{name}", mj_grid / name, "jsonl"))
        for task in TASKS:
            name = f"{model}/{task}_M4_J8_il.jsonl"
            files.append((f"pools/multi_target/crossbench_gold/{name}", gold_probe / name, "jsonl"))
    for model, (backbone, run) in HEAD_RUNS.items():
        dest = f"pools/length_allocation/{model}"
        ds_run = length_project / "ds_run" / "pools" / "report" / model
        for J in TARGET_COUNTS:
            files.append((f"{dest}/list_easy_fam_M4_J{J}_oracle.jsonl", final_sweeps / "report" / model / f"list_easy_fam_M4_J{J}_il.jsonl", "jsonl"))
            files.append((f"{dest}/list_easy_fam_M4_J{J}_learned.jsonl", ds_run / f"list_easy_fam_M4_J{J}_pc_learned_{run}.jsonl", "jsonl"))
        offsets = OFFSETS + ("0",) if model == "dqwen" else OFFSETS
        for J in (1, 2):
            for delta in offsets:
                files.append((f"{dest}/list_easy_fam_M4_J{J}_delta{delta}.jsonl", ds_run / f"list_easy_fam_M4_J{J}_pc_delta{delta}.jsonl", "jsonl"))
        files.append((f"{dest}/length_head_report.npz", length_project / "eval" / f"{run}_listbench_report.npz", "copy"))
        files.append((f"{dest}/length_head_report.jsonl", length_project / "data" / f"listbench_report_{backbone}.jsonl", "jsonl"))
    for name in JAVA_POOLS:
        files.append((f"java/pools/pools_{name}.jsonl", java_run / "generation" / "report" / f"pools_{name}.jsonl", "jsonl"))
    for name in ("verdicts.jsonl", "packet_index.jsonl"):
        files.append((f"java/judging/{name}", java_run / "judging" / name, "jsonl"))
        files.append((f"java/zero_demo/judging/{name}", java_zero_demo / "judging" / name, "jsonl"))
    files.append(("java/judging/calibration.csv", java_run / "judging" / "calibration.csv", "copy"))
    files.append(("java/judging/calibration/controls.jsonl", java_run / "judging" / "calibration" / "controls.jsonl", "jsonl"))
    for head in JAVA_LENGTH_HEADS:
        files.append((f"java/length/{head}_s5final_G4.npz", java_length / "eval" / f"{head}_s5final_G4.npz", "copy"))
    files.append(("java/length/fin_llada_m70_100k_s5final_G4.npz", length_project / "eval" / "fin_llada_m70_100k_s5final_G4.npz", "copy"))
    files.append(("java/length/prior_s5h_train.json", java_run / "length" / "prior_s5h_train.json", "copy"))
    for name in ZERO_DEMO_POOLS:
        files.append((f"java/zero_demo/pools/pools_{name}.jsonl", java_zero_demo / "generation" / "report" / f"pools_{name}.jsonl", "jsonl"))
    for backbone, run in HEAD_RUNS.values():
        files.append((f"heads/{backbone}.pt", length_project / "runs" / run / "final.pt", "head"))
        files.append((f"head_training/configs/{backbone}.json", length_project / "runs" / run / "config.json", "config"))
        files.append((f"head_training/fsp_{backbone}_train.jsonl", length_project / "data" / f"fsp_{backbone}_train.jsonl", "fsp"))
        for name in (f"edit_{backbone}_train", f"listbench_dev_{backbone}", f"edit_{backbone}_dev"):
            files.append((f"head_training/{name}.jsonl", length_project / "data" / f"{name}.jsonl", "jsonl"))
    files.append(("head_training/fsp_logical_train.jsonl", length_project / "data" / "fsp_logical_train.jsonl", "fsp"))
    files.append(("heads/java_head_C.pt", java_length / "runs" / "s5hc_C" / "final.pt", "head"))
    files.append(("head_training/configs/java_head_C.json", java_length / "runs" / "s5hc_C" / "config.json", "config"))
    for corpus in HEAD_C_CORPUS:
        for split in ("train", "dev"):
            files.append((f"head_training/java_head_C/{corpus}_{split}_manifest.jsonl", java_length / "data" / f"{corpus}_llada_{split}.jsonl", "manifest"))
    files.append(("head_training/java_head_C/corpus_audit.json", java_length / "corpus_audit.json", "copy"))
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


def fsp_jsonl(source):
    """The bytes of an FSP training file without the rows of EXCLUDED_FSP_SOURCES, the rows kept and the rows dropped.

    A row is dropped by its own "source" field, which the logical file and the three rendered files carry row for row
    (same ids, same order). Every other line, blank ones included, is kept byte for byte and in its order, so the kept
    rows keep their token ids, labels and n_tokens, and the file keeps its final newline.
    """
    lines = []
    kept = dropped = 0
    for line in jsonl_lines(source.read_bytes()):
        if not line.strip():
            lines.append(line)
        elif json.loads(line)["source"] in EXCLUDED_FSP_SOURCES:
            dropped += 1
        else:
            lines.append(line)
            kept += 1
    return "\n".join(lines).encode("utf-8"), kept, dropped


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


def private_hits(text):
    """The distinct private strings of `text` (leakcheck.find_private and LOCAL_PRIVATE), sorted; empty if clean."""
    local = {match.group() for match in LOCAL_PRIVATE.finditer(text)}
    return sorted(set(find_private(text)) | local)


def resaved_head(source, dest):
    """Write the head checkpoint `source` to `dest` without HEAD_CONFIG_DROPPED in its training config.

    The checkpoint is {"branch": state dict, "config": training config} (lengths.LengthHead.load). It is saved under
    its final file name, which torch.save writes into the archive, so the bytes depend only on the content. The new
    file must load with weights_only=True, hold the same tensors byte for byte and the same config minus
    HEAD_CONFIG_DROPPED, and no string of the config may hold private text (private_hits). Returns True if `dest` was
    written.
    """
    original = torch.load(source, map_location="cpu", weights_only=True)
    config = {key: value for key, value in original["config"].items() if key not in HEAD_CONFIG_DROPPED}
    if private_hits(json.dumps(config)):
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
    """Write every planned file into `dest`, copy the data card, and remove files that are no longer planned.

    Leftover temporary files (*.tmp, and staging directories of an interrupted head re-save) are removed first.
    """
    for path in sorted(dest.rglob("*"), reverse=True) if dest.is_dir() else []:
        relative = path.relative_to(dest).as_posix()
        if relative.startswith(".cache/") or relative == ".cache":
            continue
        if path.is_file() and path.name.endswith(".tmp"):
            print(f"removed (temporary file): {relative}")
            path.unlink()
        elif path.is_dir() and path.parent.name == "heads" and path.name.startswith("tmp"):
            shutil.rmtree(path)
    written = 0
    for path, source, kind in files:
        target = dest / path
        if kind == "copy":
            written += copy_file(source, target)
        elif kind == "jsonl":
            data = cleaned_jsonl(source)
            written += copy_file(source, target) if data is None else write_bytes(target, data)
        elif kind == "fsp":
            data, kept, dropped = fsp_jsonl(source)
            print(f"{path}: kept {kept} rows, dropped {dropped} (source {' or '.join(EXCLUDED_FSP_SOURCES)})")
            written += write_bytes(target, data)
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
    """Every file of the tree but SHA256SUMS itself and .cache/ (upload_large_folder's state, never uploaded)."""
    paths = []
    for path in dest.rglob("*"):
        relative = path.relative_to(dest).as_posix()
        if relative.startswith(".cache/") or relative == "SHA256SUMS":
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


def sums_text(dest, paths):
    """SHA256SUMS of the tree, in the format of sha256sum: '<hex>  <path>' per file, sorted by path."""
    return "".join(f"{sha256_file(dest / path)}  {path}\n" for path in paths)


def write_sums(dest, text):
    """SHA256SUMS in the tree and at the repository root."""
    write_bytes(dest / "SHA256SUMS", text.encode("utf-8"))
    write_bytes(ROOT / "SHA256SUMS", text.encode("utf-8"))


def sampled_matches(data):
    """The sampled candidate texts of a JSONL file that hold private text, or None if any match lies elsewhere.

    A model's sample can contain a path it made up (a Nemotron pool of Figure 6 holds a made-up home-directory path);
    such text is model output, which the analysis grades, and stays. Any match outside candidates' texts fails.
    """
    samples = []
    for line in jsonl_lines(data):
        if not private_hits(line):
            continue
        record = json.loads(line)
        candidates = record.get("candidates") or []
        texts = [text for candidate in candidates for text in candidate.get("texts", [])]
        rest = dict(record, candidates=[{key: value for key, value in candidate.items() if key != "texts"} for candidate in candidates])
        if private_hits(json.dumps(rest)):
            return None
        samples.extend(text for text in texts if private_hits(text))
    return samples


def check(dest, paths):
    """The checks of the module doc, on text and binary files alike; returns the list of failures."""
    failures = []
    fsp_ids = {}  # FSP file -> its row ids in file order
    for path in paths:
        data = (dest / path).read_bytes()
        hits = private_hits(data.decode("latin-1"))  # latin-1: one character per byte, binary files too
        samples = sampled_matches(data) if hits and path.endswith(".jsonl") else None
        if hits and samples:
            print(f"note: {path}: {hits} only inside sampled candidate texts (model output, kept): {samples}")
        elif hits:
            failures.append(f"{path}: private text {hits}")
        if not path.endswith(".jsonl"):
            continue
        fsp = path.startswith(FSP_PREFIX)
        if fsp:
            fsp_ids[path] = []
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
            if fsp:
                if record.get("source") in EXCLUDED_FSP_SOURCES:
                    failures.append(f"{path}:{number}: a row of {record['source']}, which the FSP files omit")
                    break
                fsp_ids[path].append(record.get("id"))
    if len({tuple(ids) for ids in fsp_ids.values()}) > 1:
        failures.append(f"the FSP files {sorted(fsp_ids)} do not list the same row ids in the same order")
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dest", type=Path, default=None, help="the upload directory (default: upload_dir of the sources)")
    parser.add_argument("--check", action="store_true", help="only hash and check the existing tree; writes nothing")
    args = parser.parse_args()
    sources = load_sources(required=("upload_dir",) if args.check else SOURCE_KEYS)  # --check reads only the tree
    print(f"warning: tools/assemble_artifacts.py reads the authors' private run directories ({SOURCES_FILE.name})", file=sys.stderr)
    args.dest = (args.dest or sources["upload_dir"]).resolve()
    files = plan({key: sources.get(key, Path()) for key in SOURCE_KEYS})  # --check only needs the planned names
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
    text = sums_text(args.dest, paths)
    failures = []
    if args.check:
        for manifest in (ROOT / "SHA256SUMS", args.dest / "SHA256SUMS"):
            if not manifest.is_file() or manifest.read_text(encoding="utf-8") != text:
                failures.append(f"{manifest} does not match the tree (run without --check to rewrite it)")
    else:
        write_sums(args.dest, text)
        link_top_level(args.dest)
    sums_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    failures += check(args.dest, paths)
    total = summary(args.dest, paths, sums_sha)
    if total > MAX_BYTES:
        failures.append(f"tree is {total / 1e9:.1f} GB, more than {MAX_BYTES / 1e9:.0f} GB")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print("CHECKS_FAILED" if failures else "CHECKS_OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
