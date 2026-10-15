# Figure 6: several targets on one canvas (Section 4.2)

The D × J grid of Section 4.2 and three appendix figures that use the same machinery. Numbers refer to the arXiv
version of the paper (v1). The code and the records write `M` for the paper's D; `gold` is the reference answer.

| Paper (arXiv v1) | Content | Output of `plot.py` | Table in `data/tables/multi_target/` |
|---|---|---|---|
| Figure 6 (Section 4.2) | unguided A_all, unguided A_avg, guidance gain in A_all at β = 0.5 | `multi_target.pdf` | `cells.csv` |
| Figure 9 (Appendix B.3, Multi-Target Experiments) | reference NLL per token by reveal position, P1 vs P0 (cross-benchmark study, J = 8) | `multi_target_mechanism.pdf` | `mechanism.csv` |
| Figure 10 (Appendix B.3) | cross-benchmark study (D = 4, K = 64): unguided A_all and its gain at β = 0.5 against J | `crossbench_d4.pdf` | `cells_crossbench.csv` |
| Figure 12 (Appendix B.6, Likelihood Sharpening) | sharpening control (target P1^{3/2}) and example guidance minus it, D × J grid | `sharpening.pdf` | `sharpening.csv` |

`gold.csv` and `diagnostics.csv` hold the reference answers' support and the pool diagnostics of every cell
(coverage, reward spread, effective sample size). In `cells.csv` the rows with β = 0 are unguided sampling.

## Protocol

- **Task.** List Functions–Easy, the 64 families of the report bank (`data/benchmarks/report/list_easy_fam.jsonl`),
  graded by the integer-list grader of `grading.py`.
- **Grid.** Models `llada`, `nemotron_pt` (causal prefix, bidirectional tail), `dqwen`; D ∈ {2, 4, 8, 16}
  demonstrations (`--M`); J ∈ {1, 2, 4, 8} targets: 48 cells × 64 families.
- **Canvas.** One chat turn with demonstrations and targets interleaved in the family's 24-slot master layout
  (`canvas.master_layout`, seed `mtfsp-20261011-mj-placement-v1`); cell (D, J) keeps the first D demonstrations and
  T1..TJ.
  Every target gets its oracle length: the token count of `" " + answer` under the model's tokenizer.
- **Sampling.** K = 16 infills per family from P1 by fixed-order decoding (ascending canvas position, one token
  per step, Lemma 3.1), all 16 in one reveal. The seed of a family's pool is the first 16 hex digits of
  SHA-256(`sec42_mj|proposal|<model>|<episode_id>|<M>|<J>|master24`) mod 2^31 (`run.proposal_seed`).
- **Scoring.** ell0 = log P0 with the demonstration outputs masked, for the distinct infills and the reference
  infill together; the reference infill is also scored alone under P1. Both are teacher-forced in the same
  reveal order.
- **Guidance.** Pool weights softmax(β · (ell1 − ell0)); β = 0 is unguided sampling and β = 0.5 the common
  strength of the paper (β = 0.1, 0.25 and 1 are in `cells.csv` as sensitivity values). The sharpening control
  has the same total exponent on P1 without P0: weights `pool_weights(ell1, ell0, 0, tau=2/3)` ∝ P1^{1/2}.
- **Intervals.** Paired family bootstrap, 10,000 resamples, percentile 95 %:
  - `cells.csv`, `gold.csv`: seed string `mtfsp-20261011-mj-bootstrap-v1`, cell key `<model>|<M>|<J>`
    (`bootstrap.cell_indices`);
  - `sharpening.csv`: seed SHA-256(`agentC-appA-20261011-mj-v1|<model>|<M>|<J>`), first 16 hex digits mod 2^32
    (`analyze.sharpening_indices`);
  - `mechanism.csv`: seed string `gold|<model>|<task>|8` (`bootstrap.boot_sd`).
- **Cross-benchmark study (Figures 9 and 10).** A separate study of Section 4.1's pipeline: SST-2, List
  Functions–Easy, SALT and Movie Director, D = 4, J ∈ {1, 2, 4, 8}, K = 64, three DLMs. Its pools are not mixed with
  the K = 16 grid. Figure 10 plots its cell table (`cells_crossbench.csv`, written by
  `single_target/analyze.py --crossbench`); Figure 9 uses its reference answers scored target by target under P1
  and P0.

## Commands

    python experiments/multi_target/analyze.py --pools artifacts/pools/multi_target --compare data/tables/multi_target
    python experiments/multi_target/plot.py                      # PDFs in experiments/multi_target/out/

`analyze.py` needs the frozen pools (`python -m fewshot_dlms.fetch pools java`):

    artifacts/pools/multi_target/<model>/list_easy_fam_M{M}_J{J}.jsonl            one record per family (the grid)
    artifacts/pools/multi_target/crossbench_gold/<model>/<task>_M4_J8_il.jsonl    reference answers scored (Figure 9)

It writes `families.csv`, `cells.csv`, `gold.csv`, `diagnostics.csv`, `sharpening.csv`, `mechanism.csv`,
`candidates.csv.gz` and `coverage.json`, takes about 10 s on a CPU, and with `--compare` prints the largest
difference per column against the paper's tables and, per cell, whether the paper's A_all lies inside the 95 % CI
of the pools analysed, at β = 0 and at β = 0.5. `plot.py` needs only `data/tables/multi_target/`.

GPU, one model per process:

    python experiments/multi_target/run.py --model llada --M 2,4,8,16 --J 1,2,4,8 --K 16 --out artifacts/reruns/multi_target
    make rerun-multi-target        # LLaDA-1.5, cell D = 4, J = 2, first 8 families, then analyze.py --compare

`--families N` keeps the first N families. Records are appended to
`<out>/<model>/list_easy_fam_M{M}_J{J}.jsonl` as families finish. Each carries its cell's `protocol_hash`, which
depends on that cell alone; an interrupted run resumes after the families with that hash and, before loading the
model, refuses a file holding records of another protocol (exit status 2, naming the file).
`<out>/<model>/manifest_<hash>.json` holds the run's configuration and hash, every pool's hash, every seed, and the
library and GPU versions. A failed family is counted and the run exits non-zero.

## Runtime

Sampling and scoring one cell (64 families, K = 16) took, on one H100 NVL in the authors' run:

| model | J = 1 | J = 2 | J = 4 | J = 8 | whole grid |
|---|---|---|---|---|---|
| `llada` | 3–5 min | 7–13 min | 19–37 min | 73–111 min | 8.3 GPU-h |
| `nemotron_pt` | 6–11 min | 8–14 min | 25–45 min | 91–147 min | 10.9 GPU-h |
| `dqwen` | 7–11 min | 9–16 min | 26–46 min | 81–132 min | 11.0 GPU-h |

(ranges over D; `seconds_per_family` in `diagnostics.csv`).

## Fidelity

- **Analysis.** On the frozen pools `analyze.py` reproduces `cells.csv`, `gold.csv`, `diagnostics.csv`,
  `sharpening.csv` and `mechanism.csv` exactly: every column, maximum difference 0 (`make test-parity`).
- **Figures.** Rendered with `pdftoppm -r 150`, the four PDFs are pixel-identical to Figures 6, 9, 10 and 12 of
  arXiv v1.
- **Reruns.** On an H100 NVL with torch 2.7.1+cu128, `run.py --K 16` reproduces the frozen records of LLaDA-1.5
  and Nemotron bit for bit (tokens, ell1, ell0 and the reference scores; checked on two families of the cells
  (4, 2), (4, 8), (16, 2), (16, 8) and (4, 2), (4, 8) respectively). dQwen3.5 runs autotuned Triton kernels whose
  numerics vary from process to process, so its reruns are statistical. Other GPUs or K ≠ 16 give other draws:
  judge such a rerun by `--compare`, which asks whether the frozen mean lies inside the rerun's 95 % CI.
- Two frozen pool files, LLaDA-1.5 (16, 8) and dQwen3.5 (16, 4), hold one family twice; like the paper's
  analysis, `analyze.py` keeps the first complete record of every family.

## Resampling check

GPU sampling is statistical in general, so a rerun is judged by whether the frozen mean lies inside the rerun's
95 % CI, not by equality. One full cell was rerun on a GPU and analysed with `--compare`:

    CUDA_VISIBLE_DEVICES=2 python experiments/multi_target/run.py --model llada --M 4 --J 2 --K 16 --families 64 --out artifacts/reruns/multi_target
    python experiments/multi_target/analyze.py --pools artifacts/reruns/multi_target --out artifacts/reruns/multi_target/analysis --compare data/tables/multi_target

- **Cell.** LLaDA-1.5 (`GSAI-ML/LLaDA-1.5` at revision `84346fd91ba60252d260022201ad6fc5a3468fb2`), D = 4, J = 2,
  all 64 report families, K = 16.
- **Platform and time.** One H100 NVL 96 GB, torch 2.7.1+cu128: 7.4 min wall-clock (444 s, of which 57 s load the
  model).
- **Numbers.** Frozen mean (`data/tables/multi_target/cells.csv`) against the rerun's mean and 95 % CI (`cells.csv` of the
  rerun). `--compare` prints the β = 0 and β = 0.5 A_all verdicts, one line each; the gain row is read from the
  rerun's `gain_all` CI columns.

  | β | quantity | frozen | rerun [95 % CI] | verdict |
  |---|---|---|---|---|
  | 0 | A_all | 0.516 | 0.516 [0.406, 0.623] | inside |
  | 0.5 | A_all | 0.564 | 0.564 [0.456, 0.671] | inside |
  | 0.5 | gain in A_all | +0.049 | +0.049 [+0.025, +0.077] | inside |

- **Note.** On this platform the rerun reproduced the frozen pool bit for bit (tokens, ell1, ell0 of all 64 × 16
  candidates), so `--compare` reports a largest difference of 0 in `cells.csv`, `gold.csv` and `sharpening.csv`
  (only the timing column of `diagnostics.csv` differs) and the rerun's mean equals the frozen one. On another GPU
  the draws differ and the verdict is the check. On this stack the "inside" verdicts therefore hold trivially;
  `tests/test_resampling_compare.py` checks that `--compare` also reports "outside" when a rerun's mean is far from
  the frozen one.
