---
license: other
license_name: fewshot-dlms-per-group
pretty_name: "Multi-Target Few-Shot Prompting with Diffusion Language Models: frozen artifacts"
size_categories:
- 100K<n<1M
viewer: false
tags:
- diffusion-language-models
- few-shot
- reproducibility
---

# fewshot-dlms: frozen artifacts

The stored candidate pools, length heads, head-training data and judge verdicts behind the paper *Multi-Target
Few-Shot Prompting with Diffusion Language Models*. They belong to the code repository `fewshot-dlms`, whose
`experiments/*/analyze.py` scripts recompute every table and figure of the paper from these files on a CPU.

Download them with the code repository's fetcher, which pins the revision of this dataset and checks every file
against `SHA256SUMS`:

    python -m fewshot_dlms.fetch pools java        # what the analyses need (about 0.4 GB)
    python -m fewshot_dlms.fetch heads head_training

The files land under `artifacts/` with the layout below. To check a download by hand, run `sha256sum -c SHA256SUMS`
in that directory. This card lists what each group holds, where it came from, how it was changed on the way here,
and its license.

## Layout

| group | content | files | size |
|---|---|---|---|
| `pools/fig4` | Section 4.1 and cross-benchmark pools (K = 64), development and report banks, and the scored reference answers | 336 | 260 MB |
| `pools/fig5` | Section 4.2 M × J grid (K = 16) and the reference answers of the J = 8 cross-benchmark cells | 60 | 42 MB |
| `pools/fig6` | Section 4.3 pools (oracle, learned and offset lengths) and the frozen heads' predictions | 56 | 74 MB |
| `java` | Section 5 pools, judge verdicts and packet index, calibration controls, length predictions, zero-demonstration ablation | 21 | 10 MB |
| `heads` | the four length heads: LLaDA-1.5, Nemotron and dQwen3.5 benchmark heads, Java head C | 4 | 3.47 GB |
| `head_training` | rendered training prompts of the three benchmark heads, the four heads' training configs, head C's corpus manifests | 22 | 1.28 GB |

`SHA256SUMS` lists every file (except itself) as `<sha256>  <path>`, the format of `sha256sum`.

```
pools/fig4/{report,dev}/<model>/<task>_M{M}_J1.jsonl        Section 4.1: llada, nemotron_ar, dqwen, qwen; M = 1, 2, 4, 8, 16
pools/fig4/{report,dev}/<model>/<task>_M4_J{J}_il.jsonl     cross-benchmark: llada, nemotron_pt, dqwen; J = 1, 2, 4, 8
pools/fig4/gold/<model>/<task>_M{M}_J1.jsonl                reference answers scored under P1 and P0 (report bank)
pools/fig5/<model>/list_easy_fam_M{M}_J{J}.jsonl            M = 2, 4, 8, 16; J = 1, 2, 4, 8; llada, nemotron_pt, dqwen
pools/fig5/crossbench_gold/<model>/<task>_M4_J8_il.jsonl    reference answers of the J = 8 cross-benchmark cells, per target
pools/fig6/<model>/list_easy_fam_M4_J{J}_oracle.jsonl       oracle lengths, J = 1, 2, 4, 8
pools/fig6/<model>/list_easy_fam_M4_J{J}_learned.jsonl      lengths sampled from the frozen head, K = 64 / J
pools/fig6/<model>/list_easy_fam_M4_J{J}_delta{Δ}.jsonl     oracle length + Δ, Δ = -2, -1, +1, +2 (and 0 for dqwen), J = 1, 2
pools/fig6/<model>/length_head_report.{npz,jsonl}           the frozen head's distributions at every report target, its prompts
java/pools/pools_*.jsonl                                    main run: one record per (episode, regime), K = 16
java/judging/{verdicts,packet_index}.jsonl, calibration.csv, calibration/controls.jsonl
java/length/<head>_s5final_G4.npz, prior_s5h_train.json    head C (s5hc_C) and two comparator heads; the label prior
java/zero_demo/pools/pools_*.jsonl, java/zero_demo/judging/{verdicts,packet_index}.jsonl
heads/{llada,nemotron,dqwen,java_head_C}.pt
head_training/{fsp,edit}_<backbone>_train.jsonl, listbench_dev_<backbone>.jsonl, edit_<backbone>_dev.jsonl, fsp_logical_train.jsonl
head_training/configs/<head>.json, head_training/java_head_C/{s5g,s5h}_{train,dev}_manifest.jsonl, corpus_audit.json
```

`<task>` is `sst2_fam`, `list_easy_fam`, `salt_rep_fam` or `movie_dir_fam` (SST-2 with opaque output codes, List
Functions–Easy, SALT repetition categories 5–6, MIT Movie director extraction). `<model>` names the interface:
`llada` (LLaDA-1.5 canvas), `nemotron_pt` (Nemotron: causal prefix, bidirectional tail from the first target),
`nemotron_ar` (Nemotron causal suffix), `dqwen` (dQwen3.5 canvas), `qwen` (Qwen3-8B, free-length autoregressive). The
benchmark families themselves are in the code repository (`data/benchmarks/`, verified by `sha256.json`); the Java
episodes are in `data/java/`.

Every experiment README of the code repository (`experiments/fig{4,5,6,7}_*/README.md`) describes the records it
reads. In short, a pool record is one family (or Java episode) of one cell: its K candidates with their token ids,
texts, ell1 = log P1 (demonstration outputs visible) and ell0 = log P0 (demonstration outputs masked), per-target
terms where the cell has several targets, the reference answer and its token lengths, the model revision and the
run's own checks.

## Provenance

All files come from the authors' runs on H100 NVL GPUs (96 GB), torch 2.7.1+cu128, transformers 5.13.0, with the
models at the revisions below. Section numbers refer to the paper.

| model | Hugging Face id | revision |
|---|---|---|
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | `84346fd91ba60252d260022201ad6fc5a3468fb2` |
| Nemotron | `nvidia/Nemotron-Labs-Diffusion-8B` | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| dQwen3.5 | `UT-IFML/dQwen3.5-9B-Base` | `f193fa20e341504f8d9e2b4b8342048c466e6590` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

- **`pools/fig4`** (Sections 4.1, 4.4, Appendix B). The four-benchmark oracle-length sweep: development bank (32 / 21 /
  32 / 32 families for SST-2 / List / SALT / Movie) and report bank (64 per benchmark), 16 ordered demonstrations and 8
  ordered targets per family, nested prefixes. Families and placements were frozen before any inference with the seed
  strings `mtfsp-20261003-final-v1` (source partitions), `mtfsp-20261003-dev-v1`, `mtfsp-20261003-report-v1` and
  `mtfsp-20261003-placement-v1` (the 12-slot master layout of the cross-benchmark cells). The K = 64 answers of a
  family share one torch generator seeded with the first 16 hex digits of SHA-256(`algo_investigation|proposal|<model>|
  <episode>|<M>|<J>`) mod 2^31; cross-benchmark cells append `|('placement', 'master')`. `gold/` holds the reference
  answers of the Section 4.1 report cells, teacher-forced and scored under P1 and P0 (Figure 4's gold row).
  The development bank fixed each model's β before any report table was computed (`data/tables/fig4/beta_locks.json`
  in the code repository).
- **`pools/fig5`** (Section 4.2, Appendix A). List Functions–Easy, the 64 report families, M ∈ {2, 4, 8, 16} × J ∈
  {1, 2, 4, 8}, K = 16, three DLMs. Protocol hash `98fe06a36f6b6503` (every record's `protocol_hash`); proposal seed =
  first 16 hex digits of SHA-256(`sec42_mj|proposal|<model>|<episode>|<M>|<J>|master24`) mod 2^31; bootstrap seed
  string `mtfsp-20261011-mj-bootstrap-v1`. Two files hold one family twice (LLaDA-1.5 M = 16, J = 8; dQwen3.5 M = 16,
  J = 4); the analysis keeps the first complete record. `crossbench_gold/` holds the reference answers of the J = 8
  cross-benchmark cells of `pools/fig4`, scored target by target (Figure 8).
- **`pools/fig6`** (Section 4.3, Appendix B.5). The oracle pools are the List Functions–Easy cross-benchmark report pools
  of `pools/fig4` (`pools/fig4/report/<model>/list_easy_fam_M4_J{J}_il.jsonl`): the same bytes, shipped twice so that
  each figure's directory is complete. The learned and offset pools sample K = 64 / J candidates per family, each on
  its own canvas with the lengths drawn from the frozen head (seed of the allocations: (bank, backbone, family, J),
  `lengths.section43_seed`) or the oracle lengths shifted by Δ; content draws use the cross-benchmark seed above.
  Families with an oracle length + Δ below 1 are skipped. dQwen3.5's `delta0` is a same-session rerun of the oracle
  arm (its kernels are not reproducible across processes). The heads were frozen by a two-stage record (SHA-256
  `ec7b4ebd950a4a3dc0af8f8b898fe9d204b2818f40371ce7929cde453622597d`, published without machine paths as
  `experiments/fig6_lengths/frozen_config.json` in the code repository).
- **`java`** (Section 5, Appendix C). 127 helper-propagation episodes from 14 public Java repositories (the code
  repository's `data/java/`), LLaDA-1.5, K = 16, oracle and learned (head C) length regimes. Protocol hash
  `0e1d3c9040b09598`; the ablation's `c2a07e4a701f0a41`. The candidates sharing one allocation are drawn with the seed
  key12(`0e1d3c9040b09598`, `s5final`, regime, episode, allocation) mod 2^31, an ablation arm with
  key12(`c2a07e4a701f0a41`, `demo_ablation`, anchor, target sites, M) mod 2^31, where key12 is the first 12 hex digits
  of SHA-256 of the parts joined by `|` (`experiments/fig7_java/run.py`). The 8 pool files are the run's worker files as written; 23
  units appear in two of them with identical bytes. The verdicts were produced by Claude Sonnet 5.5 running as Claude
  Code sub-agents with rubric r1 (`experiments/fig7_java/rubric_r1.md`, SHA-256 `7a893ac4a9d60950…`): 2,089 report
  packets, 46 calibration controls (`calibration/controls.jsonl`, the judge's verdicts on them in `calibration.csv`) and
  698 ablation packets. The packet texts are not shipped; `judge.py` rebuilds them byte for byte. The three `npz` files
  hold the predicted length distributions (float32, `P [214, 256]`) of head C and of two comparator heads at the 214
  target sites; the comparator heads themselves are not shipped.
- **`heads`**. The checkpoints used for the paper's pools, re-saved without the run directory that was pickled in
  their training configuration (see the changes below): the tensors are identical to the authors' files, byte for
  byte. The stored pools and predictions keep recording the original files' SHA-256 (`head_ckpt_sha` in the Java pools,
  `ckpt_sha` in the `npz` files, the freeze record); the table below maps them to the published files. Each is a
  trainable fp32 copy of the
  backbone's last transformer block and final norm plus a categorical head over the lengths 1..256; the backbone stays
  frozen. `lengths.LengthHead.load` of the code repository loads them. The benchmark heads were trained on 100,000
  prompts (70 % structured few-shot / 30 % code edits, 1,563 updates, seed 0); head C on 20,000 draws of its Java
  corpus (50 % s5g / 50 % s5h, 313 updates, seed 0). This group has four files: the spec's "five heads" counted head C's
  comparator, whose predictions ship as `java/length/s5g_B_s5final_G4.npz` instead of a checkpoint.

  | file | SHA-256 of the authors' file (recorded in pools and predictions) | SHA-256 of the published file |
  |---|---|---|
  | `heads/llada.pt` | `0de13a8ac6b881493e368d281531326b84fe7f5d2c50b1dccba6491dd1dec1d8` | `a49355771567e3b091e8a5d79b09aadd460b677af0f850af6d4afe27360e1452` |
  | `heads/nemotron.pt` | `df46105dab89104d0d1ccdb34795343a69fadda5528d610fd725bf414fb62e62` | `c0509fba53347491ed927f1ee2d55f8ae9ce5859eeabd4ef8a245d215943842a` |
  | `heads/dqwen.pt` | `83126cce639d3aa28cee0ef13cc526cd63684b8c94c6ff51ddad511ba206232f` | `3241752d59aabb53769fa5e3353c04142fc03bbbeb3cc9116098f1cbbdcf927d` |
  | `heads/java_head_C.pt` | `ea54c8b92110f4c5c4dbe838721d23961775082712247f8099b74e8d7d312309` | `b2efefded5ac382b5fc67ace4ee6d9f0ebfa15b820f2431b6f67cdb09721cebf` |
- **`head_training`**. The authors' rendered training prompts, one file per backbone tokenizer (`train_head.py` of the
  code repository reads them):
  - `fsp_<backbone>_train.jsonl`: 100,000 structured few-shot prompts (List Functions tasks outside List
    Functions–Easy, SALT categories other than the evaluated ones, MIT Movie fields other than Director, ATIS fields
    other than the evaluated ones; seeds `mtfsp-20261004-structured-v1` (splits) and
    `mtfsp-20261007-lengthproject-fsp-v1` (episodes)). `fsp_logical_train.jsonl` holds the same episodes before
    rendering.
  - `edit_<backbone>_train.jsonl` (22,128 prompts) and `edit_<backbone>_dev.jsonl` (2,874): Java code edits mined from
    the repositories listed under Licenses, split by repository with seed `mtfsp-20261007-lengthproject-edit-v1`; each
    row names its repository, commit and file.
  - `listbench_dev_<backbone>.jsonl`: the List Functions–Easy development bank, M = 4, J ∈ {1, 2, 4, 8}.
  - `configs/<head>.json`: each head's training configuration.
  - `java_head_C/`: head C's training corpora as manifests (every field of a row except the token ids: repository,
    commit, file, unit, cell, labels, lengths) and the corpus audit. The rendered prompts of head C are not shipped.

### Changes from the authors' files

Everything is copied byte for byte except:

- Pool records lose the keys `host`, `hostname`, `gpu_env`, `gpu`, `device`, `pid`, `cwd`, `root`, `wall_time`,
  `pool_file` and `lengths_file` (the machine, the moment and the file locations of a run; no analysis reads them). A
  line without such a key is kept byte for byte.
- SST-2 and MIT Movie records of `pools/fig4` lose the benchmark text, as in the code repository's test fixtures:
  `target_inputs` and the decoded prompts `checks.p1_prompt_text` / `checks.p0_prompt_text` are replaced by their
  SHA-256 digests under the same key + `_sha256`. The reference answers and the sampled candidates stay. The code
  repository rebuilds the text from the original sources (`data/benchmarks/build.py`).
- The training configurations lose the run directory (`root`); head C's corpora are reduced to manifests (above).
- The four head checkpoints are loaded and saved again (`torch.save`, loadable with `weights_only=True`) with the same
  `root` entry removed from the training configuration pickled in them; nothing else in the configuration changes,
  and every tensor is identical to the original, byte for byte. Their SHA-256 therefore differ from the ones the
  pools record (table above).

The tables of the paper are recomputed from these files exactly (`make test-parity` in the code repository).

## Licenses

| files | license |
|---|---|
| pools, predictions, verdicts, manifests, configs and the rendered training prompts generated by the authors (`pools/`, `java/` outside the excerpts below, `head_training/`) | CC BY 4.0, except for third-party content as listed below |
| `heads/llada.pt`, `heads/java_head_C.pt` | CC BY 4.0; each holds a trained copy of a block of LLaDA-1.5 (MIT license) |
| `heads/dqwen.pt` | CC BY 4.0; holds a trained copy of a block of dQwen3.5-9B-Base (Apache-2.0) |
| `heads/nemotron.pt` | NVIDIA Nemotron Open Model License (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/): it holds a trained copy of a block of `nvidia/Nemotron-Labs-Diffusion-8B`. Licensed by NVIDIA Corporation under the NVIDIA Nemotron Open Model License. |

Third-party content keeps its own terms:

- **Benchmarks.** The pools of List Functions (BIG-bench) and SALT carry their inputs and outputs; the SST-2 and MIT
  Movie pools carry only reference answers (SST-2's are opaque letter codes, Movie's are director names) and the
  models' samples. The sources and their pinned versions are listed in the code repository's
  `data/benchmarks/README.md`.
- **Structured few-shot training prompts** (`head_training/fsp_*`). They contain MIT Movie queries (fields other than
  Director) and ATIS queries, taken from `github.com/tonyzhaozh/few-shot-learning` at commit
  `e04d8643be91c2cce63f33e07760ff75d5aa3ad0` (`data/slot-movies/train`, `data/atis/atis.train.pkl`; that repository is
  Apache-2.0), in plain text in `fsp_logical_train.jsonl` and as token ids in the rendered files. The MIT Movie corpus
  (MIT Spoken Language Systems group) and ATIS keep their own terms.
- **Java, Section 5** (`java/`: the calibration controls and the candidates' code). Excerpts of 14 repositories under
  Apache-2.0, LGPL-2.1, LGPL-3.0 or Apache-2.0, EPL-1.0, EPL-2.0 and a BSD-style license; the per-repository table is
  `data/java/LICENSES.md` of the code repository.
- **Java code edits** (`head_training/edit_*` and the manifests in `head_training/java_head_C/`). 28 repositories were
  mined; the length project's manifests record each repository and its split but not its license. The license below
  was read from the license file of each repository's clone (its default branch when mined), not per commit. The
  shipped files hold the 21 training and 4 development repositories; the 3 held-out confirmation repositories appear
  in no shipped file.

| repository | split | license |
|---|---|---|
| apache/commons-beanutils, commons-cli, commons-collections, commons-compress, commons-configuration, commons-csv, commons-imaging, commons-jexl, commons-net, commons-numbers, commons-rng, commons-validator, commons-vfs | train | Apache-2.0 |
| apache/curator, httpcomponents-client, httpcomponents-core, opennlp | train | Apache-2.0 |
| FasterXML/jackson-core, JodaOrg/joda-time, dropwizard/metrics | train | Apache-2.0 |
| jhy/jsoup | train | MIT |
| apache/commons-codec, commons-math, commons-pool; mybatis/mybatis-3 | dev | Apache-2.0 |
| apache/commons-dbcp; square/javapoet, square/retrofit | confirmation (not shipped) | Apache-2.0 |

None of these 28 repositories is a Section 5 evaluation repository.

## Integrity

`SHA256SUMS` is also committed at the root of the code repository, and its `ARTIFACTS.md` records the dataset
revision the fetcher pins, the SHA-256 of `SHA256SUMS`, and the library versions of the runs.
