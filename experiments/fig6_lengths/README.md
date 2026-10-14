# Figure 6: target lengths (Section 4.3)

A masked target holds exactly as many tokens as it has masks, so every target needs a length before the model fills
it. Section 4.3 compares the reference length L* (the oracle setting of Sections 4.1 and 4.2) with lengths drawn from
a learned length head, and measures how much accuracy a forced offset L = L* + Δ costs.

| Paper | Content | Output of `plot.py` | Table in `data/tables/fig6/` |
|---|---|---|---|
| Figure 6 (Section 4.3), top row | A_avg with sampled learned lengths vs oracle lengths, against J | `fig6.pdf` | `sec43_downstream.csv` |
| Figure 6 (Section 4.3), bottom row | A_avg with L = L* + Δ at every target, J = 1, 2 | `fig6.pdf` | `sec43_length_perturbation.csv` |
| Appendix B.5, fidelity row | the head's mass on the exact length and within 1 and 2 tokens | `fidelity_row.pdf` | `sec43_length_metrics.csv` |

A_avg is the per-target accuracy (`A_target` in the tables). `sec43_length_metrics.csv` also holds the
visible-demonstration prior (the empirical distribution of the M demonstration output lengths), the reference the head
is compared with.

## Protocol

- **Task.** List Functions–Easy, the 64 families of the report bank, graded by the integer-list grader of `grading.py`.
  M = 4 demonstrations and J ∈ {1, 2, 4, 8} targets in the family's published 12-slot layout; ordinary sampling
  (β = 0: every candidate has weight 1/K). Models `llada`, `nemotron_pt` (causal prefix, bidirectional tail), `dqwen`.
- **Allocations** (`lengths.allocate`). Each of the K candidates of a family gets its own target lengths:
  - *oracle*: L_j = L*_j, the token count of `" " + answer` under the model's tokenizer;
  - *learned*: the frozen head reads the compact prompt once (`lengths.compact_prompt`: the canvas with one mask per
    target) and candidate k, target j draws L = F_j⁻¹(u_kj), u_kj = SHA-256(`lpx-alloc-v1|report|<backbone>|<family>|<J>|<k>|<j>`)
    / 2⁶⁴ (`lengths.sample_allocations`, seed `lengths.section43_seed`). The head's probabilities are rounded to float32
    before drawing, as they were stored in the authors' evaluation file;
  - *offset*: L_j = L*_j + Δ at every target, Δ ∈ {−2, −1, +1, +2}, J ∈ {1, 2}. A family with some L*_j + Δ < 1 is
    skipped and counted, never clipped: 3 families at J = 1 and 6 at J = 2 for Δ < 0. (`lengths.allocate("offset")`
    clips; `run.py` skips first.)
- **K.** K = 64 / J candidates (64, 32, 16, 8), so every cell samples 64 targets per family. The oracle arm is the
  K = 64 pool of the same families from the Section 4.1/4.2 sweep (the cross-benchmark study of Figure 9), which
  `run.py --mode oracle --K 64` reproduces token for token.
- **Sampling.** Candidates with the same allocation share a canvas, and candidate k is revealed on the canvas of its
  own allocation (`adapters.base.reveal` with one context per distinct allocation; Lemma 3.1, ascending positions, one
  token per step, ordinary support, chunks of 32 canvases). All K draws of a family come from one torch Generator
  seeded with the first 16 hex digits of SHA-256(`algo_investigation|proposal|<model>|<family>|4|<J>|('placement', 'master')`)
  mod 2³¹ (`run.proposal_seed`), the seed of the oracle pools: the three modes share their content randomness. Only
  log P1 is computed; `ell0` is NaN in the records.
- **The Δ = 0 arm.** For LLaDA-1.5 and Nemotron it is the oracle pool. For dQwen3.5 it is a rerun of the oracle
  allocation in the same session as its offsets (`delta0`, K = 64 / J), because dQwen's autotuned kernels drift
  between sessions (`tests/gpu_util.py`); dQwen's offsets are therefore not paired with its oracle pool
  (`diff_vs_oracle` is empty for them).
- **Metrics** (`analyze.py`). Per family: A_target = mean over candidates and targets of c_kj; A_all = share of
  candidates with every target correct; length_exact = share of allocated lengths equal to L*; Pr(L ≠ L* | correct);
  generated target tokens per family; format failures (answers that are not an integer list). Learned − oracle is
  paired over families. The head's fidelity: P_r = mass within r tokens of L* (r = 0, 1, 2), NLL of L*, E|L − L*|,
  mode exact and the ECE of the mode, per site, averaged within a family, then over families
  (`lengths.length_mass`).
- **Intervals.** 10,000 family resamples, percentile 95 %, `numpy.random.default_rng(seed)` with seed = the first 8 hex
  digits of SHA-256(`lpx-bootstrap-v1`) plus J (A_target), 100 + J (learned − oracle) or 200 + J (P_r); the
  offset rows on the families common to every Δ (the curves of Figure 6) use seed 1000 + J. Families are resampled in
  pool-file order (A_target, offsets) or sorted by id (paired differences, P_r), as in the authors' analysis.
- **The heads** are the authors' final checkpoints, loaded unchanged (`lengths.LengthHead.load`); `frozen_config.json`
  is their freeze record (training configuration, checkpoint SHA-256s, protocol) with machine paths removed.

## Commands

    python experiments/fig6_lengths/analyze.py --pools artifacts/pools/fig6 --compare data/tables/fig6
    python experiments/fig6_lengths/plot.py                      # PDFs in experiments/fig6_lengths/out/

`analyze.py` needs the frozen pools (`python -m fewshot_dlms.fetch pools java`):

    artifacts/pools/fig6/<model>/list_easy_fam_M4_J{J}_oracle.jsonl     oracle lengths, K = 64, J = 1, 2, 4, 8
    artifacts/pools/fig6/<model>/list_easy_fam_M4_J{J}_learned.jsonl    sampled learned lengths, K = 64 / J
    artifacts/pools/fig6/<model>/list_easy_fam_M4_J{J}_delta{Δ}.jsonl   Δ = -2, -1, +1, +2 (and 0 for dqwen), J = 1, 2
    artifacts/pools/fig6/<model>/length_head_report.npz                 the frozen head's distributions at every target
                                                                        site of the report prompts (P [960, 256] float32,
                                                                        ids "<family>|<J>|<j>", y = L*)
    artifacts/pools/fig6/<model>/length_head_report.jsonl               per (family, J): oracle lengths, demonstration
                                                                        lengths demo_L, and the compact prompt

`<model>` is `llada`, `nemotron_pt` or `dqwen`. One record per family; each candidate keeps its tokens, the texts of its
J targets, ell1 and its allocation `alloc_L`. The pool files are the authors' Section 4.3 pools (the per-candidate run
of the length project) and, for the oracle arm, their Section 4.1/4.2 pools; the npz files are their head evaluation.
`analyze.py` writes `sec43_downstream.csv`, `sec43_length_perturbation.csv`, `sec43_length_metrics.csv`,
`families.csv` and `coverage.json` in about 5 s on a CPU, and with `--compare` prints the largest difference per column
against the paper's tables and, per downstream cell, whether the paper's A_avg lies inside the 95 % CI of the pools
analysed. `plot.py` needs only `data/tables/fig6/`.

GPU, one model per process (`--delta` takes a list, written with `=` because it starts with a minus sign):

    python experiments/fig6_lengths/run.py --model llada --mode learned --J 1,2,4,8 --out artifacts/reruns/fig6
    python experiments/fig6_lengths/run.py --model llada --mode offset --delta=-2,-1,1,2 --J 1,2 --out artifacts/reruns/fig6
    python experiments/fig6_lengths/run.py --model llada --mode oracle --J 1,2,4,8 --K 64 --out artifacts/reruns/fig6
    make rerun-fig6        # LLaDA-1.5, J = 1, first 8 families, oracle and learned, then analyze.py --compare

`--K N` replaces K = 64 / J at every J; `--families N` keeps the first N families. The learned mode loads the head
from `artifacts/heads/<backbone>.pt` (`llada`, `nemotron`, `dqwen`; `--head` to override) and needs flash-attn for
LLaDA-1.5 (`pip install -e .[heads]`; see `pyproject.toml`). Records are appended to
`<out>/<model>/list_easy_fam_M4_J{J}_<condition>.jsonl` as families finish, so an interrupted run resumes;
`<out>/<model>/manifest_<hash>.json` holds the configuration, its hash (the records' `protocol_hash`), every seed, and
the library and GPU versions.

## Runtime

Sampling one cell (64 families) took, on one H100 NVL in the authors' run (LLaDA-1.5): learned lengths 13 min (J = 1),
31 min (J = 2), 54 min (J = 4), 76 min (J = 8); an offset 6–13 min (J = 1) and 10–20 min (J = 2). Nemotron and
dQwen3.5 took 0.5 to 1.4 times as long per cell. Distinct allocations rarely share a canvas, which is why K is 64 / J.

## Fidelity

- **Analysis.** On the frozen pools `analyze.py` reproduces the three tables: largest difference 4.4e-16 in
  `sec43_length_metrics.csv`, 9.7e-17 in `sec43_downstream.csv` and 1.1e-13 in `sec43_length_perturbation.csv`
  (`make test-parity`; the authors' table step re-read its inputs from CSV, which moves the last digit). Every offset
  pool holds exactly the families `run.py` keeps.
- **Figures.** Rendered with `pdftoppm -r 150` (and `-r 300`), `fig6.pdf` is pixel-identical to the paper's Figure 6 and
  `fidelity_row.pdf` to the authors' fidelity row.
- **Reruns.** On an H100 NVL with torch 2.7.1+cu128, `run.py` at the paper's K reproduces the frozen LLaDA-1.5 records
  bit for bit: allocations, tokens, texts and ell1 of every candidate (checked on the first two families of learned
  J = 1, offset +1 at J = 2, and oracle J = 2 at K = 64). The ported heads give the frozen distributions to about 1e-6 in
  total variation for LLaDA-1.5 (`tests/test_gpu_head.py`) and exactly for Nemotron; dQwen3.5's head
  and pools vary from process to process (autotuned Triton kernels), so its reruns are statistical. Other GPUs or
  other K give other draws: judge such a rerun by `--compare`, which asks whether the frozen mean lies inside the
  rerun's 95 % CI.

## Training a length head (optional)

The paper's heads ship frozen; `train_head.py` retrains one (Appendix B.5). Restated with `frozen_config.json`:

- **Architecture.** The backbone stays frozen. A trainable fp32 copy of its last transformer block and final norm
  (LLaDA-1.5: block 31 and ln_f; Nemotron: layer 33 and encoder.norm; dQwen3.5: layer 31 and model.norm) reads the
  states entering that block, and a linear head over the lengths 1..256 (standard deviation 10⁻³, zero bias, seed 0)
  reads the normalised state at each placeholder (`lengths.Branch`; `lengths.fresh_head` builds the untrained head).
  219M trainable parameters for LLaDA-1.5 and Nemotron, 211M for dQwen3.5.
- **Data.** Two sources of compact prompts, each rendered once per backbone tokenizer, labels = exact token counts:
  *structured few-shot prompts* (100,000 training prompts: List Functions tasks outside List Functions–Easy with 40
  held back, SALT categories other than 5, 6 and 12, MIT Movie fields other than Director, ATIS fields other than the
  evaluation ones; J uniform on {1, 2, 4, 8}, M ∈ {2, 4, 8} with probabilities 0.2 / 0.6 / 0.2; seeds
  `mtfsp-20261004-structured-v1` (splits) and `mtfsp-20261007-lengthproject-fsp-v1` (episodes)) and *Java code edits*
  (about 22,000 training prompts from 28 repositories disjoint from Section 5: the pre-commit file with every hunk of a
  related edit group as `<old>…</old><new>…</new>`, one placeholder per target block, cropped to 2,048 tokens under
  every tokenizer; splits by repository, seed `mtfsp-20261007-lengthproject-edit-v1`). Prompts over 2,048 tokens or with
  a target over 256 tokens are dropped at load (45 or 40 code-edit prompts per backbone).
- **Schedule.** 100,000 prompts = 1,563 updates of 64 prompts, 70 % structured few-shot / 30 % code edits: each
  update draws from one source (1,094 and 469 updates, in a seeded order), each source through an epoch-wise shuffled
  iterator. LLaDA-1.5 packs prompts into microbatches of at most 24,576 tokens; the other backbones run one prompt per
  forward pass.
- **Loss and optimiser.** Prompt-normalised cross-entropy (mean over prompts of the mean site NLL). AdamW, β = (0.9,
  0.999), weight decay 0.01 on matrices, learning rates 3·10⁻⁵ (block and norm) and 3·10⁻⁴ (head), 5 % linear warm-up
  then cosine decay, gradient clipping at 1.0, bf16 autocast with fp32 master weights. The final checkpoint is the
  model; development data (List Functions–Easy dev bank, 1,500 held-out code edits) is only reported.
- **Compute.** 1.8 h (LLaDA-1.5), 2.1 h (Nemotron) and 2.7 h (dQwen3.5) on one H100 NVL.

The rendered training files ship with the artifacts (`artifacts/head_training/`, see the docstring of `train_head.py`):

    python experiments/fig6_lengths/train_head.py --backbone llada --data artifacts/head_training --out runs/llada_head

`--steps N` stops after N updates of the same schedule. `--check-render N` re-renders the first N structured few-shot
episodes from `fsp_logical_train.jsonl` and compares them with the shipped prompts token for token. The checkpoint
(`final.pt`) loads with `lengths.LengthHead.load`; for LLaDA-1.5 the script ends by comparing the new head's
exact-length mass with the frozen head's on the ten fixture canvases of `data/fixtures/fig6/`.

**Retraining check** (LLaDA-1.5, H100 NVL). `train_head.py` follows the authors' run update for update: the same source,
learning rate and epoch counters at every logged update, and a training loss within 2·10⁻³ of theirs over the first 200
updates (bf16 kernels are not deterministic, so the runs drift apart slowly). A full retrain (1,563 updates, 54 min)
reached a List Functions–Easy dev P0 of 0.379 (theirs: 0.373) and an exact-length mass of 0.497 on the ten fixture
canvases, against 0.518 for the frozen head. After 200 updates (`--steps 200`) that mass is still 0.234.
