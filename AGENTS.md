# fewshot-dlms agent guide

Code for the paper "Multi-Target Few-Shot Prompting with Diffusion Language Models". Read README.md
first; this is the short version for coding agents.

## Conventions

- Pool weights are `softmax(β · (ℓ1 − ℓ0))` with ℓ1 = log P1 (demonstration outputs visible) and
  ℓ0 = log P0 (demonstration outputs masked, positions `E`). β = 0 is ordinary sampling; β = 0.5 is
  the paper's common strength, Q ∝ P1^{3/2} P0^{−1/2}. The sharpening control is
  `pool_weights(ell1, ell0, 0.0, tau=2/3)`, which has the same total power 3/2 on P1.
- Equations live in `src/fewshot_dlms/scoring.py`; resampling intervals in `bootstrap.py`; answer
  extraction and graders in `grading.py`; the prompt (target sites, `E`) in `canvas.py`.
- A *family* is one frozen episode specification (`data.load_families(task, bank)`); an experiment cell
  (M, J) uses its first M demonstrations and first J targets.

## Commands

- `make test`: CPU unit tests and fixtures; needs no downloads, no GPU. Run before every commit.
- `make test-parity`: CPU, compares `analyze.py` output with the paper's tables; needs
  `python -m fewshot_dlms.fetch pools java` first.
- `make test-gpu`: replay of stored log-likelihoods; needs a CUDA GPU and the model weights.

## CPU and GPU

- CPU: everything in `analyze.py`, `plot.py`, `scoring.py`, `bootstrap.py`, `grading.py`, `canvas.py`.
- GPU: `run.py` and `adapters/` (sampling, log-likelihoods). One process per GPU.
- GPU resampling is statistical, not bit-identical: judge a rerun by whether the frozen mean lies
  inside the rerun's 95 % CI, never by exact equality.

## Rules

- `data/` and `artifacts/` are inputs, never edited by hand. `data/` is frozen and written by the
  scripts in `tools/`; `artifacts/` is fetched.
- The paper's Java verdicts came from Claude Sonnet 5.5 through Claude Code sub-agents. The public
  `judge.py` calls the Anthropic API with the same rubric; the original verdicts ship with the data.
- Code is plain on purpose: one statement per line, docstrings that name the paper quantity, no
  clever abstractions. New code follows `src/fewshot_dlms/scoring.py` and `canvas.py`.
