# Figure 4: one target, example guidance (Section 4.1)

The single-target study of Section 4.1, the development-bank choice of β (Appendix B, Experimental Setup, "Guidance
selection"), and the studies that reuse its pools:

| Paper | Content | Output of `plot.py` | Table in `data/tables/` |
|---|---|---|---|
| Figure 4 (Section 4.1) | ordinary accuracy and its paired gain at β = 0.5 against M, four benchmarks, four model interfaces | `fig4.pdf` | `fig4/sec41_accuracy.csv` |
| Appendix B, Single-Target Prompting (gold row) | gold example reward per target token, (log P1 − log P0) / T of the reference answer | `gold_reward_row.pdf` | `fig4/gold_reward_cells.csv` |
| Guidance-strength sweep figure (Fig. 10, arXiv version only; Appendix B, Guidance-Strength Sensitivity) | guidance-strength sweep: A_all gain and effective support exp(H)/K against β | `fig11_beta_sweep.pdf` | `fig4/beta_sweep.csv` |
| Cross-benchmark figure (Fig. 9, both builds; Appendix B, Multi-Target Prompting) | cross-benchmark study, M = 4, J ∈ {1, 2, 4, 8}: drawn by `experiments/fig5_multi_target/plot.py` | — | `fig5/cells_crossbench.csv` |
| Appendix B, Experimental Setup (Guidance selection) | the development-only β of every model | — | `fig4/beta_locks.json`, `fig4/dev_sec41.csv`, `fig4/dev_sec42.csv` |
| Appendix B, Likelihood Sharpening | sharpening residuals: guidance against power sampling at the matched exponent (LLaDA-1.5) | — | `fig4/sec44_power.csv` |

`analyze.py --crossbench` writes `cells_crossbench.csv` (the paper's `sec42_all.csv`), which the cross-benchmark figure
(Fig. 9, both builds) reads from `data/tables/fig5/`.

## Protocol

- **Benchmarks.** SST-2 with opaque output codes, List Functions–Easy, SALT (repetition), Movie Director: the frozen
  families of `data/benchmarks/` (report bank for every number, development bank for β only), graded by `grading.py`.
- **Episodes.** Section 4.1: the first M ∈ {1, 2, 4, 8, 16} demonstrations, then the target T1 (J = 1, target last).
  Cross-benchmark study: M = 4 and the first J ∈ {1, 2, 4, 8} targets, placed in each family's frozen 12-slot master
  layout (`canvas.master_layout`; J = 1 keeps T1 at its master position).
- **Model interfaces** (Appendix B, Experimental Setup). `llada` and `dqwen`: one canvas, the target masked with its
  oracle length (the token count of `" " + answer`). `nemotron_ar`: Nemotron's causal suffix, exactly the oracle
  length, left to right. `qwen`: Qwen3-8B, free length, stopping at the first token that holds a newline or at
  `<|im_end|>` (200-token budget; graded on its first line). The cross-benchmark study runs the three canvas models
  `llada`, `nemotron_pt` (causal prefix, bidirectional tail) and `dqwen`.
- **Sampling.** K = 64 answers per family from P1 at temperature 1, all in one call: fixed-order decoding (Lemma 3.1,
  ascending canvas position) for canvases, left-to-right decoding for the causal interfaces. Seed of a family: the first
  16 hex digits of SHA-256(`algo_investigation|proposal|<model>|<episode_id>|<M>|<J>`) mod 2^31; the cross-benchmark
  cells append `|('placement', 'master')` (`run.proposal_seed`).
- **Scoring.** ell0 = log P0 of the distinct answers: canvases mask the demonstration outputs in place, causal interfaces
  delete them from the prompt (the `Output:` labels stay). Batches (the authors'; bf16 logits depend on them): canvases
  32 per forward pass while sampling and scoring; `nemotron_ar` 32 while sampling, 16 for P0; `qwen` 32 and 32.
- **Gold reward** (`--score-gold`). The reference answer scored under P1 and P0 in the same way; for Qwen3-8B also the
  probability of stopping right after it (any pure terminator), so its T counts the stop. Per family R*/T =
  (log P1 − log P0)/T; per cell the mean with a 95 % family-bootstrap interval.
- **Guidance.** Pool weights softmax(β · (ell1 − ell0)); β = 0 is ordinary sampling, β = 0.5 the common strength of
  Figure 4. `sec41_accuracy.csv` holds the grid β ∈ {0, 0.025, 0.05, 0.125, 0.25, 0.5, 1, 2}.
- **Development-only β.** Per model, the mean A_target over its development cells (Section 4.1: M ∈ {2, 4, 8, 16};
  cross-benchmark: J ∈ {1, 2, 4, 8}; `nemotron_ar` and `qwen` Section 4.1 only, `nemotron_pt` cross-benchmark only), equal
  cell weights; the best positive β, ties within 0.25 pp to the smallest. LLaDA-1.5 0.5, dQwen3.5 0.5, Nemotron causal 1,
  Nemotron prefix/tail 1, Qwen3-8B 1. LLaDA-1.5's lock was frozen first, from the 28 development cells complete at the
  time (no J = 8 cell; its J = 1 point was then the Section 4.1 M = 4 pool); `analyze.py` reproduces it from those
  cells and also applies the rule to all 32 complete cells (`all_complete_cells` in `beta_locks.json`): same β. Each
  lock in `beta_locks.json` names its `cell_set`; LLaDA-1.5's `partial_cells_excluded` is historical (what its lock
  file recorded), and `partial_cells_excluded_derived` is the same field from today's complete pools (empty).
- **Intervals.** Family bootstrap, 10,000 resamples, percentile 95 %, families in pool-file order:
  - cells: seed string `mtfsp-20261003-bootstrap-v1`, cell key `<model>|<task>` (`bootstrap.cell_indices`), one index
    matrix per model and task, shared by every β, M and J (gains are paired);
  - `sec44_power.csv`: the same seed string, cell key `llada|<task>|44|J<J>`;
  - gold rows: `bootstrap.boot_sd` with seed string `4.1|<model>|<task>|<M>|rpt`.
- **Appendix B, Likelihood Sharpening** (sharpening residuals). LLaDA-1.5, M = 4, J ∈ {1, 4, 8} (J = 1: the Section
  4.1 pool; J = 4, 8: cross-benchmark pools). Guidance at (β, τ) has total exponent 1/τ + β on P1;
  `gain_vs_matched_power` compares it with ordinary power sampling at τ_eff = 1 / (1/τ + β).
- **Appendix B, Guidance-Strength Sensitivity.** List Functions–Easy cross-benchmark pools (M = 4, K = 64), J ∈ {1, 4},
  three DLMs, β ∈ {0, .25, .5, 1, 2}. Per family w = softmax(β R), A_all = Σ w·all_correct, H = −Σ w log w,
  D_eff = exp(H)/K; cell means, gains over β = 0 paired with the cell's index matrix (`<model>|list_easy_fam`).

## Commands

    python experiments/fig4_single_target/analyze.py --pools artifacts/pools/fig4 --crossbench --beta-sweep \
        --compare data/tables/fig4 data/tables/fig5
    python experiments/fig4_single_target/plot.py                     # PDFs in experiments/fig4_single_target/out/

`analyze.py` needs the frozen pools (`python -m fewshot_dlms.fetch pools java`):

    artifacts/pools/fig4/{dev,report}/<model>/<task>_M{M}_J1.jsonl        Section 4.1: llada, nemotron_ar, dqwen, qwen; M in 1,2,4,8,16
    artifacts/pools/fig4/{dev,report}/<model>/<task>_M4_J{J}_il.jsonl     cross-benchmark: llada, nemotron_pt, dqwen; J in 1,2,4,8
    artifacts/pools/fig4/gold/<model>/<task>_M{M}_J1.jsonl                reference answers scored (report bank, Section 4.1)

It writes `sec41_accuracy.csv`, `gold_reward_cells.csv`, `gold_families.csv`, `dev_sec41.csv`, `dev_sec42.csv`,
`beta_locks.json`, `sec44_power.csv`, `diagnostics.csv` and, with the flags, `cells_crossbench.csv` and `beta_sweep.csv`,
in about 25 s on a CPU. With `--compare` it prints the largest difference per table against the paper's tables and, per
Section 4.1 cell, whether the paper's accuracy lies inside the 95 % CI of the pools analysed, at β = 0 and at β = 0.5.
`plot.py` needs only `data/tables/fig4/`.

GPU, one model per process (`CUDA_VISIBLE_DEVICES` picks the GPU):

    python experiments/fig4_single_target/run.py --model llada --tasks sst2_fam,list_easy_fam,salt_rep_fam,movie_dir_fam \
        --M 1,2,4,8,16 --K 64 --bank report --score-gold --out artifacts/reruns/fig4
    python experiments/fig4_single_target/run.py --model nemotron_pt --crossbench --J 1,2,4,8 --out artifacts/reruns/fig4
    make rerun-fig4        # LLaDA-1.5, SALT, M = 4, first 8 families, then analyze.py --compare

`--families N` keeps the first N families. Records are appended to `<out>/<bank>/<model>/<task>_M{M}_J{J}[_il].jsonl` as
families finish. Each carries its cell's `protocol_hash`, which depends on that cell alone; an interrupted run resumes
after the families with that hash and, before loading the model, refuses a file holding records of another protocol
(exit status 2, naming the file).
`<out>/<bank>/<model>/manifest_<hash>.json` holds every cell's configuration and hash, every seed, and the library and
GPU versions. A failed family is counted and the run exits non-zero. SST-2 and Movie Director need their text rebuilt
first (`python data/benchmarks/build.py`).

## Runtime

Sampling and scoring 64 report families at K = 64 took, on one H100 NVL in the authors' run (`seconds_per_family` in
`diagnostics.csv`): Section 4.1, all 20 cells of a model, 0.75 GPU-h (`llada`), 0.99 (`nemotron_ar`), 1.16 (`dqwen`),
0.64 (`qwen`); cross-benchmark, all 16 cells, 6.9 (`llada`), 11.9 (`nemotron_pt`), 10.6 (`dqwen`), dominated by J = 8
(71–121 s per family).

## Fidelity

- **Analysis.** On the frozen pools `analyze.py` reproduces `sec41_accuracy.csv`, `dev_sec41.csv`, `dev_sec42.csv`,
  `sec44_power.csv`, `cells_crossbench.csv` and `beta_sweep.csv` exactly (maximum difference 0), `gold_reward_cells.csv`
  to 4.4e-16, and every frozen β lock with its development statistic (`make test-parity`).
- **Figures.** Rendered with `pdftoppm -r 150`, the three PDFs are pixel-identical to the paper's figures.
- **Reruns.** On an H100 NVL with torch 2.7.1+cu128, `run.py --K 64` reproduces the frozen records of LLaDA-1.5,
  Nemotron (causal suffix and prefix/tail) and Qwen3-8B bit for bit (tokens, ell1, ell0, per-target values; with
  `--score-gold` the frozen gold scores too), checked on the first families of SST-2 and List Functions–Easy at M = 4 and
  of the cross-benchmark cell J = 2. dQwen3.5 runs autotuned Triton kernels whose numerics vary from process to process,
  so its reruns are statistical. Other GPUs or K ≠ 64 give other draws: judge such a rerun by `--compare`.
- One frozen pool file (LLaDA-1.5 List Functions–Easy, J = 8) holds one family twice; like the paper's analysis,
  `analyze.py` keeps the first complete record of every family.

## Resampling check

GPU sampling is statistical in general, so a rerun is judged by whether the frozen mean lies inside the rerun's
95 % CI, not by equality. One full Section 4.1 cell was rerun on a GPU and analysed with `--compare`:

    CUDA_VISIBLE_DEVICES=2 python experiments/fig4_single_target/run.py --model llada --tasks sst2_fam --M 4 --K 64 --families 64 --bank report --out artifacts/reruns/fig4
    python experiments/fig4_single_target/analyze.py --pools artifacts/reruns/fig4 --out artifacts/reruns/fig4/analysis --compare data/tables/fig4

- **Cell.** LLaDA-1.5 (`GSAI-ML/LLaDA-1.5` at revision `84346fd91ba60252d260022201ad6fc5a3468fb2`), SST-2 with
  opaque codes (text rebuilt with `data/benchmarks/build.py`), M = 4, all 64 report families, K = 64.
- **Platform and time.** One H100 NVL 96 GB, torch 2.7.1+cu128: 64 s wall-clock (57 s load the model, 5 s sample
  and score the cell).
- **Numbers.** Frozen mean (`data/tables/fig4/sec41_accuracy.csv`) against the rerun's mean and 95 % CI
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
  reports "outside" when a rerun's mean is far from the frozen one (on Figure 5's `analyze.py`, which prints the
  same verdict rule).
