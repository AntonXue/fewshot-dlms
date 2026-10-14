"""Rebuild the SST-2 and MIT Movie benchmark families from their public sources, and verify them.

These two benchmarks ship without text. data/benchmarks/<bank>/<task>.ids.jsonl holds the frozen
families with the `input` and `output` of every demonstration and target set to null and a
`source_ref` naming the upstream record. This script downloads the public corpora, fills the text
back in exactly as the authors' pipeline produced it, checks every family against the SHA-256
recorded in data/benchmarks/sha256.json, and caches the result in
data/benchmarks/_built/<bank>/<task>.jsonl, which fewshot_dlms.data.load_families reads.

    pip install -e ".[data]"
    python data/benchmarks/build.py
    python -c "from data.benchmarks import build; build.verify_all()"

Sources (pinned; a different copy fails the checksums):

  sst2_fam       GLUE SST-2 training split (Hugging Face dataset nyu-mll/glue, config sst2,
                 revision GLUE_REVISION). source_ref "train:N" is row N, counted from 0. The input
                 is the sentence with surrounding whitespace stripped; the output is the family's
                 code for the row's label (meta.codes maps the class names in meta.classes; label 0
                 is negative, 1 is positive).

  movie_dir_fam  MIT Movie trivia10k13 queries as packaged in
                 github.com/tonyzhaozh/few-shot-learning, file data/slot-movies/train at commit
                 MOVIE_COMMIT. source_ref "movies_train:N" is line N of that file, counted from 0
                 with blank lines included. The input is the words of the line joined by single
                 spaces; the output is the words of its Director span.
"""

import hashlib
import json
import urllib.request

from fewshot_dlms.data import (
    BANKS,
    BENCHMARK_DIR,
    BENCHMARKS,
    family_sha,
    load_families,
    read_jsonl,
)

GLUE_REPO = "nyu-mll/glue"
GLUE_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"

MOVIE_COMMIT = "e04d8643be91c2cce63f33e07760ff75d5aa3ad0"
MOVIE_URL = (
    "https://raw.githubusercontent.com/tonyzhaozh/few-shot-learning/"
    f"{MOVIE_COMMIT}/data/slot-movies/train"
)
MOVIE_SHA256 = "d99b0d79f6f149646b48dbc55d0bb3c21becf0f04ff0dc43d7f41e9742bca660"
MOVIE_LINE_SUFFIX = " <=> <NULL>"
MOVIE_FIELD = "Director"


class IntegrityError(Exception):
    """A family, or a downloaded source file, differs from what the authors froze."""


# ---------------------------------------------------------------------------- verification


def load_recorded_shas() -> dict[str, str]:
    """Return {"<bank>/<task>/<episode_id>": sha256} from data/benchmarks/sha256.json."""
    text = (BENCHMARK_DIR / "sha256.json").read_text(encoding="utf-8")
    return json.loads(text)


def verify(families: list[dict], bank: str, task: str) -> None:
    """Raise IntegrityError for the first family whose SHA-256 differs from the recorded one."""
    recorded = load_recorded_shas()
    for family in families:
        key = f"{bank}/{task}/{family['episode_id']}"
        if key not in recorded:
            raise IntegrityError(f"{key}: no sha recorded")
        if family_sha(family) != recorded[key]:
            raise IntegrityError(f"{key}: sha mismatch")


def verify_all() -> None:
    """Check every family of every bank and benchmark against sha256.json, then print the count."""
    recorded = load_recorded_shas()
    verified = set()
    for bank in BANKS:
        for task in BENCHMARKS:
            families = load_families(task, bank)
            verify(families, bank, task)
            verified.update(f"{bank}/{task}/{family['episode_id']}" for family in families)

    missing = sorted(set(recorded) - verified)
    if missing:
        raise IntegrityError(f"{len(missing)} recorded families not found (first: {missing[0]})")
    print(f"{len(verified)}/{len(recorded)} families OK")


# ---------------------------------------------------------------------------- sources


def fetch_sst2_train() -> list[tuple[str, int]]:
    """Download the GLUE SST-2 training split; return (sentence, label) per row, in row order."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        message = 'downloading SST-2 needs the optional dependency: pip install -e ".[data]"'
        raise ImportError(message) from error

    train = load_dataset(GLUE_REPO, "sst2", split="train", revision=GLUE_REVISION)
    return list(zip(train["sentence"], train["label"], strict=True))


def fetch_movie_lines() -> list[str]:
    """Download slot-movies/train at the pinned commit; line k is record movies_train:k."""
    with urllib.request.urlopen(MOVIE_URL, timeout=120) as response:
        raw = response.read()

    digest = hashlib.sha256(raw).hexdigest()
    if digest != MOVIE_SHA256:
        raise IntegrityError(f"slot-movies/train: expected sha256 {MOVIE_SHA256}, got {digest}")
    return raw.decode("utf-8").split("\n")


# ---------------------------------------------------------------------------- rebuilding


def row_number(source_ref: str, prefix: str) -> int:
    """Return N for a source_ref of the form '<prefix>:N'."""
    name, _, number = source_ref.partition(":")
    if name != prefix or not number.isdigit():
        raise ValueError(f"expected a '{prefix}:<row>' source_ref, got {source_ref!r}")
    return int(number)


def director_span(tags: list[str]) -> tuple[int, int]:
    """Return (start, end) of the single contiguous B-Director I-Director* span among the tags."""
    spans = []
    for position, tag in enumerate(tags):
        if tag == f"B-{MOVIE_FIELD}":
            spans.append([position, position + 1])
        elif tag == f"I-{MOVIE_FIELD}" and spans and spans[-1][1] == position:
            spans[-1][1] = position + 1
        elif tag == f"I-{MOVIE_FIELD}":
            raise ValueError(f"I-{MOVIE_FIELD} without a preceding B-{MOVIE_FIELD}")

    if len(spans) != 1:
        raise ValueError(f"expected exactly one {MOVIE_FIELD} span, found {len(spans)}")
    start, end = spans[0]
    return start, end


def parse_movie_line(line: str) -> tuple[str, str]:
    """Return (utterance, Director answer) of one slot-movies line.

    A line is a sequence of word:TAG tokens followed by ' <=> <NULL>', for example
    'what:O film:O by:O steven:B-Director spielberg:I-Director <=> <NULL>'.
    """
    if not line.endswith(MOVIE_LINE_SUFFIX):
        raise ValueError(f"unexpected slot-movies line ending: {line[-30:]!r}")

    words = []
    tags = []
    for token in line[: -len(MOVIE_LINE_SUFFIX)].split(" "):
        word, _, tag = token.rpartition(":")
        words.append(word)
        tags.append(tag)

    start, end = director_span(tags)
    return " ".join(words), " ".join(words[start:end])


def fill_text(family: dict, lookup) -> dict:
    """Return the family with input/output restored: lookup(source_ref) gives (input, output)."""
    filled = dict(family)
    for role in ("demos", "targets"):
        filled[role] = []
        for entry in family[role]:
            text, output = lookup(entry["source_ref"])
            filled[role].append({"id": entry["id"], "input": text, "output": output})
    return filled


def rebuild_sst2(family: dict, train: list[tuple[str, int]]) -> dict:
    """Restore the text of an SST-2 family from the GLUE training rows."""
    classes = family["meta"]["classes"]
    codes = family["meta"]["codes"]

    def lookup(source_ref):
        sentence, label = train[row_number(source_ref, "train")]
        return sentence.strip(), codes[classes[label]]

    return fill_text(family, lookup)


def rebuild_movie(family: dict, lines: list[str]) -> dict:
    """Restore the text of a MIT Movie family from the slot-movies lines."""

    def lookup(source_ref):
        return parse_movie_line(lines[row_number(source_ref, "movies_train")])

    return fill_text(family, lookup)


def write_built(bank: str, task: str, families: list[dict]) -> None:
    """Cache the rebuilt families where load_families looks for them."""
    path = BENCHMARK_DIR / "_built" / bank / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        for family in families:
            handle.write(json.dumps(family, ensure_ascii=False) + "\n")
    partial.replace(path)


# task -> (download the source, rebuild one family from it)
TEXT_FREE_TASKS = {
    "sst2_fam": (fetch_sst2_train, rebuild_sst2),
    "movie_dir_fam": (fetch_movie_lines, rebuild_movie),
}


def main() -> None:
    """Rebuild, verify and cache SST-2 and MIT Movie for both banks."""
    for task, (fetch, rebuild) in TEXT_FREE_TASKS.items():
        source = fetch()
        for bank in BANKS:
            ids_file = BENCHMARK_DIR / bank / f"{task}.ids.jsonl"
            families = [rebuild(family, source) for family in read_jsonl(ids_file)]
            verify(families, bank, task)
            write_built(bank, task, families)
            print(f"{bank}/{task}: {len(families)} families rebuilt and verified")


if __name__ == "__main__":
    main()
