# Figure 5: one target, example guidance (Section 4.1)

The single-target study of Section 4.1, the authors' development-bank choice of β, and the studies that reuse its
pools. Numbers refer to the arXiv version of the paper (v1). The code and the records write `M` for the paper's D;
in the tables the β = 0 rows (and keys such as `dev_gain_vs_ordinary_pp`, where `ordinary` means unguided) are
unguided sampling, and `gold` names the reference answer.

| Paper (arXiv v1) | Content | Output of `plot.py` | Table in `data/tables/` |
|---|---|---|---|
| Figure 5 (Section 4.1) | unguided accuracy and its paired gain at β = 0.5 against D, four benchmarks, four model interfaces | `single_target.pdf` | `single_target/sec41_accuracy.csv` |
| Appendix B.2, Single-Target Experiments | the reference answer's example reward per target token, (log P1 − log P0) / T (in the text of arXiv v1) | `reference_reward_row.pdf` | `single_target/gold_reward_cells.csv` |
| Figure 11 (Appendix B.5, Guidance-Strength Sensitivity) | A_all gain and the effective fraction exp(H)/K against β | `guidance_strength.pdf` | `single_target/beta_sweep.csv` |
| Figure 10 (Appendix B.3, Multi-Target Experiments) | cross-benchmark study, D = 4, J ∈ {1, 2, 4, 8}: drawn by `experiments/multi_target/plot.py` | — | `multi_target/cells_crossbench.csv` |
| Appendix B.6, Likelihood Sharpening | single-target residuals: guidance against power sampling at the matched exponent (LLaDA-1.5) | — | `single_target/sec44_power.csv` |
| not in arXiv v1 (authors' record) | the development-only β of every model | — | `single_target/beta_locks.json`, `single_target/dev_sec41.csv`, `single_target/dev_sec42.csv` |

`analyze.py --crossbench` writes `cells_crossbench.csv` (the authors' `sec42_all.csv`), which Figure 10 reads from
`data/tables/multi_target/`.

## Protocol

- **Benchmarks.** SST-2 with opaque output codes, List Functions–Easy, SALT (repetition), Movie Director: the frozen
  families of `data/benchmarks/` (report bank for every number, development bank for β only), graded by `grading.py`.
- **Episodes.** Section 4.1: the first D ∈ {1, 2, 4, 8, 16} demonstrations (`--M`), then the target T1 (J = 1, target
  last). Cross-benchmark study: D = 4 and the first J ∈ {1, 2, 4, 8} targets, placed in each family's frozen 12-slot master
  layout (`canvas.master_layout`; J = 1 keeps T1 at its master position).
- **Model interfaces** (Appendix B.1, Experimental Setup). `llada` and `dqwen`: one canvas, the target masked with its
  oracle length (the token count of `" " + answer`). `nemotron_ar`: Nemotron's causal suffix, exactly the oracle
  length, left to right. `qwen`: Qwen3-8B, free length, stopping at the first token that holds a newline or at
  `<|im_end|>` (200-token budget; graded on its first line). The cross-benchmark study runs the three canvas models
  `llada`, `nemotron_pt` (causal prefix, bidirectional tail) and `dqwen`.
- **Sampling.** K = 64 samples per family from P1 at temperature 1, all in one call: fixed-order decoding (Lemma 3.1,
  ascending canvas position) for canvases, left-to-right decoding for the causal interfaces. Seed of a family: the first
  16 hex digits of SHA-256(`algo_investigation|proposal|<model>|<episode_id>|<M>|<J>`) mod 2^31; the cross-benchmark
  cells append `|('placement', 'master')` (`run.proposal_seed`).
- **Scoring.** ell0 = log P0 of the distinct answers: canvases mask the demonstration outputs in place, causal interfaces
  delete them from the prompt (the `Output:` labels stay). Batches (the authors'; bf16 logits depend on them): canvases
  32 per forward pass while sampling and scoring; `nemotron_ar` 32 while sampling, 16 for P0; `qwen` 32 and 32.
- **Reference reward** (`--score-gold`). The reference answer scored under P1 and P0 in the same way; for Qwen3-8B also the
  probability of stopping right after it (any pure terminator), so its T counts the stop. Per family R*/T =
  (log P1 − log P0)/T; per cell the mean with a 95 % family-bootstrap interval.
- **Guidance.** Pool weights softmax(β · (ell1 − ell0)); β = 0 is unguided sampling, β = 0.5 the common strength of
  Figure 5. `sec41_accuracy.csv` holds the grid β ∈ {0, 0.025, 0.05, 0.125, 0.25, 0.5, 1, 2}.
- **Development-only β** (the authors' record; arXiv v1 reports β = 0.5 throughout). Per model, the mean A_target over
  its development cells (Section 4.1: D ∈ {2, 4, 8, 16};
  cross-benchmark: J ∈ {1, 2, 4, 8}; `nemotron_ar` and `qwen` Section 4.1 only, `nemotron_pt` cross-benchmark only), equal
  cell weights; the best positive β, ties within 0.25 pp to the smallest. LLaDA-1.5 0.5, dQwen3.5 0.5, Nemotron causal 1,
  Nemotron prefix/tail 1, Qwen3-8B 1. LLaDA-1.5's lock was frozen first, from the 28 development cells complete at the
  time (no J = 8 cell; its J = 1 point was then the Section 4.1 D = 4 pool); `analyze.py` reproduces it from those
  cells and also applies the rule to all 32 complete cells (`all_complete_cells` in `beta_locks.json`): same β. Each
  lock in `beta_locks.json` names its `cell_set`; LLaDA-1.5's `partial_cells_excluded` is historical (what its lock
  file recorded), and `partial_cells_excluded_derived` is the same field from today's complete pools (empty).
- **Intervals.** Family bootstrap, 10,000 resamples, percentile 95 %, families in pool-file order:
  - cells: seed string `mtfsp-20261003-bootstrap-v1`, cell key `<model>|<task>` (`bootstrap.cell_indices`), one index
    matrix per model and task, shared by every β, D and J (gains are paired);
  - `sec44_power.csv`: the same seed string, cell key `llada|<task>|44|J<J>`;
  - reference-reward rows: `bootstrap.boot_sd` with seed string `4.1|<model>|<task>|<M>|rpt`.
- **Appendix B.6, Likelihood Sharpening** (single-target residuals). LLaDA-1.5, D = 4, J ∈ {1, 4, 8} (J = 1: the
  Section 4.1 pool; J = 4, 8: cross-benchmark pools). Guidance at (β, τ) has total exponent 1/τ + β on P1;
  `gain_vs_matched_power` compares it with unguided power sampling at τ_eff = 1 / (1/τ + β).
- **Appendix B.5, Guidance-Strength Sensitivity.** List Functions–Easy cross-benchmark pools (D = 4, K = 64), J ∈ {1, 4},
  three DLMs, β ∈ {0, .25, .5, 1, 2}. Per family w = softmax(β R), A_all = Σ w·all_correct, H = −Σ w log w,
  D_eff = exp(H)/K; cell means, gains over β = 0 paired with the cell's index matrix (`<model>|list_easy_fam`).

## Commands

    python experiments/single_target/analyze.py --pools artifacts/pools/single_target --crossbench --beta-sweep \
        --compare data/tables/single_target data/tables/multi_target
    python experiments/single_target/plot.py                     # PDFs in experiments/single_target/out/

`analyze.py` needs the frozen pools (`python -m fewshot_dlms.fetch pools java`):

    artifacts/pools/single_target/{dev,report}/<model>/<task>_M{M}_J1.jsonl        Section 4.1: llada, nemotron_ar, dqwen, qwen; M in 1,2,4,8,16
    artifacts/pools/single_target/{dev,report}/<model>/<task>_M4_J{J}_il.jsonl     cross-benchmark: llada, nemotron_pt, dqwen; J in 1,2,4,8
    artifacts/pools/single_target/gold/<model>/<task>_M{M}_J1.jsonl                reference answers scored (report bank, Section 4.1)

It writes `sec41_accuracy.csv`, `gold_reward_cells.csv`, `gold_families.csv`, `dev_sec41.csv`, `dev_sec42.csv`,
`beta_locks.json`, `sec44_power.csv`, `diagnostics.csv` and, with the flags, `cells_crossbench.csv` and `beta_sweep.csv`,
in about 25 s on a CPU. With `--compare` it prints the largest difference per table against the paper's tables and, per
Section 4.1 cell, whether the paper's accuracy lies inside the 95 % CI of the pools analysed, at β = 0 and at β = 0.5.
`plot.py` needs only `data/tables/single_target/`.

GPU, one model per process (`CUDA_VISIBLE_DEVICES` picks the GPU):

    python experiments/single_target/run.py --model llada --tasks sst2_fam,list_easy_fam,salt_rep_fam,movie_dir_fam \
        --M 1,2,4,8,16 --K 64 --bank report --score-gold --out artifacts/reruns/single_target
    python experiments/single_target/run.py --model nemotron_pt --crossbench --J 1,2,4,8 --out artifacts/reruns/single_target
    make rerun-single-target        # LLaDA-1.5, SALT, D = 4, first 8 families, then analyze.py --compare

`--families N` keeps the first N families. Records are appended to `<out>/<bank>/<model>/<task>_M{M}_J{J}[_il].jsonl` as
families finish. Each carries its cell's `protocol_hash`, which depends on that cell alone; an interrupted run resumes
after the families with that hash and, before loading the model, refuses a file holding records of another protocol
(exit status 2, naming the file).
`<out>/<bank>/<model>/manifest_<hash>.json` holds every cell's configuration and hash, every seed, and per session (a
resumed run appends its own) the library and GPU versions and the command. A failed family is counted and the run
exits non-zero. SST-2 and Movie Director need their text rebuilt first (`python data/benchmarks/build.py`).

## Runtime

Sampling and scoring 64 report families at K = 64 took, on one H100 NVL in the authors' run (`seconds_per_family` in
`diagnostics.csv`): Section 4.1, all 20 cells of a model, 0.75 GPU-h (`llada`), 0.99 (`nemotron_ar`), 1.16 (`dqwen`),
0.64 (`qwen`); cross-benchmark, all 16 cells, 6.9 (`llada`), 11.9 (`nemotron_pt`), 10.6 (`dqwen`), dominated by J = 8
(71–121 s per family).

## Fidelity

- **Analysis.** On the frozen pools `analyze.py` reproduces `sec41_accuracy.csv`, `dev_sec41.csv`, `dev_sec42.csv`,
  `sec44_power.csv`, `cells_crossbench.csv` and `beta_sweep.csv` exactly (maximum difference 0), `gold_reward_cells.csv`
  to 4.4e-16, and every frozen β lock with its development statistic (`make test-parity`).
- **Figures.** Rendered with `pdftoppm -r 150`, `single_target.pdf` and `guidance_strength.pdf` are pixel-identical to
  Figures 5 and 11 of arXiv v1 (`reference_reward_row.pdf` has no figure there).
- **Reruns.** On an H100 NVL with torch 2.7.1+cu128, `run.py --K 64` reproduces the frozen records of LLaDA-1.5,
  Nemotron (causal suffix and prefix/tail) and Qwen3-8B bit for bit (tokens, ell1, ell0, per-target values; with
  `--score-gold` the frozen reference-answer scores too), checked on the first families of SST-2 and List
  Functions–Easy at D = 4 and of the cross-benchmark cell J = 2. dQwen3.5 runs autotuned Triton kernels whose numerics
  vary from process to process, so its reruns are statistical. Other GPUs or K ≠ 64 give other draws: judge such a rerun by `--compare`.
- One frozen pool file (LLaDA-1.5 List Functions–Easy, J = 8) holds one family twice; like the paper's analysis,
  `analyze.py` keeps the first complete record of every family.

## Resampling check

GPU sampling is statistical in general, so a rerun is judged by whether the frozen mean lies inside the rerun's
95 % CI, not by equality. One full Section 4.1 cell was rerun on a GPU and analysed with `--compare`:

    CUDA_VISIBLE_DEVICES=2 python experiments/single_target/run.py --model llada --tasks sst2_fam --M 4 --K 64 --families 64 --bank report --out artifacts/reruns/single_target
    python experiments/single_target/analyze.py --pools artifacts/reruns/single_target --out artifacts/reruns/single_target/analysis --compare data/tables/single_target

- **Cell.** LLaDA-1.5 (`GSAI-ML/LLaDA-1.5` at revision `84346fd91ba60252d260022201ad6fc5a3468fb2`), SST-2 with
  opaque codes (text rebuilt with `data/benchmarks/build.py`), D = 4, all 64 report families, K = 64.
- **Platform and time.** One H100 NVL 96 GB, torch 2.7.1+cu128: 64 s wall-clock (57 s load the model, 5 s sample
  and score the cell).
- **Numbers.** Frozen mean (`data/tables/single_target/sec41_accuracy.csv`) against the rerun's mean and 95 % CI
  (`sec41_accuracy.csv` of the rerun). `--compare` prints the β = 0 and β = 0.5 accuracy verdicts, one line each;
  the paired-gain row is read from the rerun's `paired_gain` CI columns.

  | β | quantity | frozen | rerun [95 % CI] | verdict |
  |---|---|---|---|---|
  | 0 | accuracy | 0.445 | 0.445 [0.383, 0.510] | inside |
  | 0.5 | accuracy | 0.589 | 0.589 [0.514, 0.665] | inside |
  | 0.5 | paired gain | +0.144 | +0.144 [+0.109, +0.180] | inside |

- **Note.** On this platform the rerun reproduced the frozen pool bit for bit (tokens, ell1, ell0 of all 64 × 64
  answers), so `--compare` reports a largest difference of 0 in `sec41_accuracy.csv` and `sec44_power.csv` and the
  rerun's mean equals the frozen one. On another GPU the draws differ and the verdict is the check. On this stack
  the "inside" verdicts therefore hold trivially; `tests/test_resampling_compare.py` checks that `--compare` also
  reports "outside" when a rerun's mean is far from the frozen one (on Figure 6's `analyze.py`, which prints the
  same verdict rule).
