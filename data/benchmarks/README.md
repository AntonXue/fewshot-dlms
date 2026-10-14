# Benchmark families

The four benchmarks of the paper, exactly as the authors froze them before running any model. A
*family* is one episode specification: an instruction, 16 ordered demonstrations and 8 ordered
targets (24 distinct inputs). Experiments use the prefixes `demos[:M]` and `targets[:J]`.

| benchmark | task | dev | report | shipped as |
|---|---|---|---|---|
| `sst2_fam` | SST-2 sentiment with opaque output codes | 32 | 64 | ids only, text rebuilt |
| `list_easy_fam` | List Functions, easy subset (BIG-bench) | 21 | 64 | verbatim |
| `salt_rep_fam` | SALT, repetition categories 5-6 | 32 | 64 | verbatim |
| `movie_dir_fam` | MIT Movie, director extraction | 32 | 64 | ids only, text rebuilt |

373 families in total. The dev bank is used to select beta; every reported number comes from the
report bank. List Functions has 85 easy functions, one family each: 21 form the dev bank and 64 the
report bank.

## Use

```python
from fewshot_dlms.data import load_families

families = load_families("list_easy_fam", "report")  # 64 dicts; bank defaults to "report"
```

A record has the keys `episode_id, bank, source, rule_id, category, split, instruction, demos,
targets, max_M, max_J, grader, meta`, where `demos` and `targets` are lists of
`{id, input, output}`. For SST-2 and MIT Movie, build the text once first (below); otherwise
`load_families` raises `FileNotFoundError: run python data/benchmarks/build.py`.

## Layout

```
data/benchmarks/
  dev/  report/                one directory per bank
    list_easy_fam.jsonl        verbatim copy of the frozen manifest
    salt_rep_fam.jsonl         verbatim copy of the frozen manifest
    sst2_fam.ids.jsonl         frozen manifest without text, plus `source_ref`
    movie_dir_fam.ids.jsonl    frozen manifest without text, plus `source_ref`
  sha256.json                  SHA-256 of every full-text family
  build.py                     rebuilds and verifies the two text-free benchmarks
  _built/                      written by build.py, git-ignored
```

## Rebuilding SST-2 and MIT Movie

These two benchmarks are stored without text so that the corpora are fetched from their own
distributions. In `*.ids.jsonl` the `input` and `output` of every demonstration and target are
`null`, and `source_ref` names the upstream record. `build.py` downloads the sources, restores the
text, and checks each family against `sha256.json` before caching it in `_built/`:

```bash
pip install -e ".[data]"                                             # from a checkout
python data/benchmarks/build.py                                      # needs network
python -c "from data.benchmarks import build; build.verify_all()"   # prints 373/373 families OK
```

| | `sst2_fam` | `movie_dir_fam` |
|---|---|---|
| source | GLUE SST-2, `train` split, Hugging Face `nyu-mll/glue` (config `sst2`) at revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | `data/slot-movies/train` of `github.com/tonyzhaozh/few-shot-learning` at commit `e04d8643be91c2cce63f33e07760ff75d5aa3ad0` |
| `source_ref` | `train:N`, row N counted from 0 | `movies_train:N`, line N counted from 0, blank lines included |
| `input` | the sentence, stripped of surrounding whitespace | the words of the line joined by single spaces |
| `output` | the family's code for the row's label (`meta.codes`; label 0 is negative, 1 is positive) | the words of the line's single `Director` span |

Both downloads are pinned (the movie file also by SHA-256, in `build.py`), so the rebuilt
families are the frozen ones.

The entry ids of SST-2 are positional (`d00`..`d15`, `t0`..`t7`); the same record ids are also
listed in `meta.demo_record_ids` and `meta.target_record_ids`.

The MIT Movie queries are the *trivia10k13* corpus of the MIT Spoken Language Systems group
(<http://sls.csail.mit.edu/downloads/movie/>), not the simpler `engtrain.bio` corpus. The
`few-shot-learning` repository packages its 7,816 training sentences as a shuffled 7,034-line
`train` file plus a 782-line `valid` file; `source_ref` indexes the shuffled `train` file, which is
why that exact file is pinned by commit and checksum.

## Sources and licences

| benchmark | source | version used | licence |
|---|---|---|---|
| `sst2_fam` | SST-2 (Socher et al., 2013, <https://aclanthology.org/D13-1170/>), as distributed in GLUE (Wang et al., 2018, <https://aclanthology.org/W18-5446/>) | Hugging Face `nyu-mll/glue`, config `sst2`, `train` split, revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | as published by GLUE / SST: see source; the text is not redistributed here |
| `list_easy_fam` | List Functions (Rule, 2020, <https://dspace.mit.edu/handle/1721.1/129232>), the BIG-bench task `list_functions` (Srivastava et al., 2022, <https://arxiv.org/abs/2206.04615>) | `github.com/google/BIG-bench` at commit `092b196c1f8f14a54bbc62f24759d43bde46dd3b`: 250 functions with 32 input/output pairs each | Apache-2.0 (the BIG-bench repository) |
| `salt_rep_fam` | SALT (Zheng et al., 2025, "The Curse of CoT", <https://arxiv.org/abs/2504.05081>) | `salt/salt.json` of `github.com/HKUST-KnowComp/CoT-ICL-Eval` at commit `3af5a9b0bc28b0059f0289d3790dc8b2a96cf06f` | MIT (the repository's `LICENSE`) |
| `movie_dir_fam` | MIT Movie, trivia10k13 corpus (Liu et al., 2012, "A Conversational Movie Search System Based on Conditional Random Fields"; MIT Spoken Language Systems group) | the copy in `github.com/tonyzhaozh/few-shot-learning` (Zhao et al., 2021, "Calibrate Before Use") at commit `e04d8643be91c2cce63f33e07760ff75d5aa3ad0` | see source (that repository is Apache-2.0; the corpus keeps its own terms); the text is not redistributed here |

The selections are the authors': the easy subset of List Functions (85 of the 250 functions, chosen by a pattern
over the upstream rule descriptions, `meta.easy_subset`), and the SALT records of repetition categories 5 and 6.
Each SALT family keeps its record's 4 released demonstrations and released query; its other 12 demonstrations and 7
targets were generated by the authors from the record's own dictionary and rule (`meta.demo_origin`,
`meta.target_origin`, `meta.generator`; ids with `_g`). They extend SALT and are not released SALT data.

## Integrity and provenance

- `sha256.json` maps `"<bank>/<task>/<episode_id>"` to the SHA-256 of
  `json.dumps(family, sort_keys=True)` for the full-text family. `build.verify(families, bank,
  task)` raises `IntegrityError` naming the first family that differs.
- `tools/export_benchmarks.py` is the one-off script that produced everything in this directory
  from the authors' frozen run: it copies the List Functions and SALT manifests, strips the text of
  the other two, and writes `sha256.json`. It needs that run, which is not part of this repository.
- Each source keeps its own terms. The text of SST-2 and MIT Movie is therefore not redistributed
  here.
