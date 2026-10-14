# fewshot-dlms

Code for the paper *Multi-Target Few-Shot Prompting with Diffusion Language Models*.

A diffusion language model fills masked positions anywhere in a sequence, so a prompt can hold several
few-shot demonstrations and several masked targets at once, and the model answers all targets in one
pass. The paper studies how to prompt such a model well: how to use the demonstrations as evidence
(example guidance), how to place several targets on one canvas, and how many tokens each target should
get. This repository reproduces the paper's figures from frozen artifacts on a CPU in minutes, and
re-runs the experiments on GPUs with the same seeds.

> Status: work in progress. Built so far: benchmark families, scoring, bootstrap, grading and the
> canvas for List Functions. Parts marked *coming* are built in later milestones.

## The three ideas

Notation: a *canvas* holds M demonstrations and J masked targets. An *infill* is one token string for
every masked target. P1 is the model's distribution over infills with the demonstration outputs
visible; P0 is the same model with the demonstration outputs masked (the positions `E` of the canvas).

**1. Example guidance.** The reward of an infill is how much the demonstration outputs raise its likelihood.
Sample K infills from P1 and reweight them:

    R = log P1 − log P0                    example reward
    Q_β ∝ P1 · e^{βR} = P1^{1+β} P0^{−β}  β = 0 is ordinary sampling
    Q_0.5 ∝ P1^{3/2} P0^{−1/2}             β = 0.5, the paper's common strength

The sharpening control raises the same total power on P1 without using P0: `Q ∝ P1^{3/2}` (temperature 2/3).
Implemented in `src/fewshot_dlms/scoring.py`: `example_reward`, `pool_weights(ell1, ell0, beta, tau)`,
`family_metrics` (A_all, A_target, effective sample size).

**2. One canvas, many targets.** Demonstrations and J masked targets are interleaved in one sequence and
all targets are filled in one pass, in a fixed reveal order. Sampling and the two log-likelihoods
(P1 with `E` visible, P0 with `E` masked) are in `src/fewshot_dlms/adapters/base.py` (*coming*); the
canvas itself, with its target sites and `E`, is `src/fewshot_dlms/canvas.py`.

**3. Target lengths.** A masked target needs a token count. Using the length of the reference answer is
the oracle setting; Section 4.3 predicts it with a small head on the frozen model. Implemented in
`src/fewshot_dlms/lengths.py` (*coming*).

## Paper section → directory

| Paper | Content | Directory |
|---|---|---|
| Section 4.1 | single target, example guidance | `experiments/fig4_single_target` (*coming*) |
| Section 4.2 | several targets, M × J grid | `experiments/fig5_multi_target` (*coming*) |
| Section 4.3 | target lengths | `experiments/fig6_lengths` (*coming*) |
| Section 5 | Java refactoring | `experiments/fig7_java` (*coming*) |

Frozen benchmark families are in `data/benchmarks/` (see its README).

## Reproduce in three commands

    pip install -e .
    python -m fewshot_dlms.fetch pools java      # downloads the frozen pools (coming)
    make figures                                 # Figures 4–7 from the pools, on CPU (coming)

Today `make test` runs the CPU unit tests, which need no downloads.

## Three levels of fidelity

1. **Replay (GPU, exact tokens).** The frozen pools store each candidate's token ids and both
   log-likelihoods. The ported canvas and adapter must reproduce them for the stored tokens within
   bf16 tolerance. This checks canvas, reveal order and scoring together (`make test-gpu`).
2. **Analysis parity (CPU, exact).** Each `analyze.py` reproduces the paper's tables from the frozen
   pools to 1e-9 (`make test-parity`); a two-family fixture is checked to 1e-12 on every `make test`.
3. **Resampling (GPU, statistical).** `make rerun-fig5` samples a cell again from scratch. A rerun is
   not bit-identical to the paper: it passes when the frozen mean lies inside the rerun's 95 % CI.

## The Java judge

The paper's Java verdicts (Section 5) were produced by Claude Sonnet 5.5 running through Claude Code
sub-agents. The public `judge.py` (*coming*) calls the Anthropic API with the same rubric. The original
verdicts ship with the pools, so a replacement judge can be validated against them.
