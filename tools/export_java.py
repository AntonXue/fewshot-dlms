"""Provenance of data/java/: how the 127 helper-propagation episodes were exported from the authors' run.

One-off script, kept for the record; the repository works without it. It reads the package of the
authors' Section 5 run (the manifests, the episode reconstruction library and the frozen prompt
hashes) and writes

  <dest>/episodes.jsonl            the 127 episodes of manifest B (field `in_manifest_B`), every
                                   manifest field unchanged, plus what the canvas needs and the
                                   manifest only referred to: `helper_text`, and for every site
                                   `method_before` (the enclosing method before the commit) with the
                                   line range [i1, i2) of the changed span
  <dest>/episode_cells.csv         the 158 candidate episodes with their flags and population
  <dest>/exclusion_ledger.csv      the stage at which each episode left a population, and why
  <dest>/reference_validity.csv    the source-validity flags and the adjudication of each episode
  <dest>/source_groups.csv         the 338 helper-extraction groups the census started from

The CSV files are copied byte for byte. Before an episode is written, it is rebuilt with the
authors' library (`fr_lib.episode`) and rendered with their renderer; the script stops unless the
prompt hashes of the oracle canvas and of the one-mask canvas equal the hashes recorded in the
manifest, and unless the context of every site is identical before and after the change (so that
`method_after`, which holds the reference answer of a target, need not be shipped).

Usage:
    python tools/export_java.py --src <package directory of the Section 5 run>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from fewshot_dlms.leakcheck import find_private

DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "java"

COPIED_CSV = ["episode_cells.csv", "exclusion_ledger.csv", "reference_validity.csv", "source_groups.csv"]
# Keys of a record that describe the machine and the moment of a run, not the episode.
RUN_METADATA = ("host", "gpu_env", "wall_time")


def read_jsonl(path: Path) -> list[dict]:
    """Read a newline-delimited JSON file into a list of records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def site_context(site: dict) -> dict:
    """The fields of a rebuilt site that the canvas needs: the old method and the changed line range.

    The canvas shows the lines of the method before and after the changed span. They are the same lines
    in the method before and in the method after the commit, because the span is the smallest range of
    lines that differs; this is checked here, so only `method_before` is kept.
    """
    before = site["method_before"].split("\n")
    after = site["method_after"].split("\n")
    i1, i2, j1, j2 = site["i1"], site["i2"], site["j1"], site["j2"]
    assert before[:i1] == after[:j1], f"{site['sid']}: lines before the span differ"
    assert before[i2:] == after[j2:], f"{site['sid']}: lines after the span differ"
    assert "\n".join(before[i1:i2]) == site["span_old"], f"{site['sid']}: old span"
    assert "\n".join(after[j1:j2]) == site["span_new"], f"{site['sid']}: new span"
    return {"i1": i1, "i2": i2, "method_before": site["method_before"]}


def export_sites(manifest_sites: list[dict], rebuilt_sites: list[dict]) -> list[dict]:
    """Manifest sites with the canvas fields added; each must be the site the library rebuilt."""
    assert len(manifest_sites) == len(rebuilt_sites)
    exported = []
    for manifest_site, rebuilt in zip(manifest_sites, rebuilt_sites, strict=True):
        assert manifest_site["site_id"] == rebuilt["sid"]
        assert manifest_site["span_old"] == rebuilt["span_old"]
        assert (manifest_site["j1"], manifest_site["j2"]) == (rebuilt["j1"], rebuilt["j2"])
        exported.append({**manifest_site, **site_context(rebuilt)})
    return exported


def export_episode(record: dict, fr_lib, tokenizer) -> dict:
    """One shipped episode: the manifest record plus the helper text and the site contexts."""
    library = fr_lib.SX
    episode = fr_lib.episode(tokenizer, record)
    keys = [target["key"] for target in episode["targets"]]

    oracle = dict(zip(keys, record["oracle_lengths"], strict=True))
    oracle_canvas = library.render(tokenizer, episode, keys, oracle)
    assert library.sha(oracle_canvas["ids"]) == record["prompt_sha_oracle"], record["episode_id"]
    assert oracle_canvas["n_tokens"] == record["oracle_canvas_tokens"], record["episode_id"]

    compact_canvas = library.render(tokenizer, episode, keys, {key: 1 for key in keys})
    assert library.sha(compact_canvas["ids"]) == record["prompt_sha_compact"], record["episode_id"]
    assert compact_canvas["n_tokens"] == record["compact_tokens"], record["episode_id"]

    shipped = {}
    for key, value in record.items():
        if key in RUN_METADATA:
            continue
        shipped[key] = value
        if key == "helper_text_sha":
            shipped["helper_text"] = episode["helper_text"]
    shipped["demos"] = export_sites(record["demos"], episode["demos"])
    shipped["targets_visible"] = export_sites(record["targets_visible"], episode["targets"])
    return shipped


def check_no_private_strings(path: Path) -> None:
    """Stop if an exported file contains a machine path or machine metadata (fewshot_dlms.leakcheck)."""
    hits = find_private(path.read_text(encoding="utf-8"))
    assert not hits, f"{hits} in {path.name}"


def export(src: Path, dest: Path) -> None:
    """Export the admitted episodes and copy the ledgers from `src` into `dest`."""
    sys.path.insert(0, str(src / "scratch"))
    import fr_lib  # the authors' episode reconstruction; reads the public SWE-Refactor archive

    tokenizer = fr_lib.SX.TokOnly()
    records = [r for r in read_jsonl(src / "manifests" / "episodes.jsonl") if r["in_manifest_B"]]
    episodes = [export_episode(record, fr_lib, tokenizer) for record in records]

    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    for name in COPIED_CSV:
        shutil.copyfile(src / "manifests" / name, dest / name)

    commits = {episode["commit"] for episode in episodes}
    repositories = {episode["repository"] for episode in episodes}
    print(f"episodes.jsonl: {len(episodes)} episodes, {len(commits)} commits, {len(repositories)} repositories")
    for name in ["episodes.jsonl"] + COPIED_CSV:
        check_no_private_strings(dest / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="package directory of the authors' Section 5 run (its manifests and episode library); not shipped",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="output directory (default: data/java)",
    )
    args = parser.parse_args()
    if args.src is None:
        parser.error("--src is required: the package directory of the authors' Section 5 run")
    export(args.src, args.dest)


if __name__ == "__main__":
    main()
