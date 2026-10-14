# fewshot-dlms agent guide

Code for the paper "Multi-Target Few-Shot Prompting with Diffusion Language Models". README.md is the full
introduction; this is the short version for coding agents.

## Where things are

- `src/fewshot_dlms/`: `scoring.py` (the equations), `bootstrap.py` (intervals), `grading.py` (answer extraction,
  graders), `canvas.py` (the prompt: target sites, `E`), `adapters/` (the only code that touches a model),
  `lengths.py` (allocation, length head), `data.py` (frozen families and Java episodes), `fetch.py` (artifacts).
- `experiments/fig{4_single_target,5_multi_target,6_lengths,7_java}/`: one pipeline per paper figure, `run.py` (GPU),
  `analyze.py` and `plot.py` (CPU); each README maps paper sections, tables and commands.
- `data/`: `benchmarks/` and `java/` (inputs), `tables/` (the paper's tables, parity targets), `fixtures/` (CPU tests).
- A *family* is one frozen episode specification (`data.load_families(task, bank)`); a cell (M, J) uses its first M
  demonstrations and first J targets.

## Conventions

- Pool weights are `softmax(β · (ℓ1 − ℓ0))` with ℓ1 = log P1 (demonstration outputs visible) and ℓ0 = log P0
  (demonstration outputs masked, positions `E`). β = 0 is ordinary sampling; β = 0.5 is the paper's common strength,
  Q ∝ P1^{3/2} P0^{−1/2}. The sharpening control is `pool_weights(ell1, ell0, 0.0, tau=2/3)`, target P1^{3/2}.
- Code is plain on purpose: one statement per line, docstrings that name the paper quantity, no clever abstractions.
  New code follows `scoring.py` and `canvas.py`.

## Artifacts

`python -m fewshot_dlms.fetch <group> ...` downloads the dataset `UT-IFML/fewshot-dlms` at the revision pinned in
ARTIFACTS.md into `artifacts/` and checks `SHA256SUMS`. Groups: `pools` (`pools/fig4`, `pools/fig5`, `pools/fig6`)
and `java` (about 390 MB together; what the analyses need), `heads` (3.5 GB) and `head_training` (1.3 GB).

## Tests

- `make test`: CPU unit tests and fixtures; no downloads, no GPU. Run it before every commit.
- `make test-parity`: CPU; recomputes every paper table from the pools; needs `fetch pools java` (skips without it).
- `make test-gpu`: replay on a CUDA GPU; needs the pinned checkpoints, flash-attn, `fetch java heads`.
- `make test-api`: calls the Anthropic API and spends credits; needs `ANTHROPIC_API_KEY`. Never run it unasked.
- SST-2 / MIT Movie text: `python data/benchmarks/build.py` (needs network and the `data` extra); otherwise the
  tests that need it skip.

## Rules

- Never edit the frozen tables under `data/tables/`: they are the paper's numbers. If `analyze.py` disagrees with
  them, the code is wrong. `data/` is otherwise written only by the scripts in `tools/`; `artifacts/` is fetched.
- No private paths, user names, host names or machine metadata in any committed file. `tests/test_java_data.py`
  (`PRIVATE_PATTERNS`) holds the generic patterns; keep new files clean of them too.
- GPU: one process per GPU. A fresh GPU run is statistical, not bit-identical: judge it by whether the paper's mean
  lies inside the rerun's 95 % CI (`analyze.py --compare`), never by exact equality.
- The paper's Java verdicts came from Claude Sonnet 5.5 running as Claude Code sub-agents (rubric
  `experiments/fig7_java/rubric_r1.md`); they ship with the artifacts. `judge.py` re-implements the judge with the
  Anthropic API; re-judging is approximate and costs credits, so do not call it in tests or loops.

## Adding a model adapter

Subclass `Adapter` in `src/fewshot_dlms/adapters/base.py`: set `name`, `model_id`, a pinned `revision`, `mask_id`
and `kind` (`"dlm"` or `"ar"`), keep the tokenizer in `tok` with `V` and `control_ids`, and implement `logits` and
the four chat methods (`chat_header`, `chat_user_open`, `chat_user_close_assistant_open`, `chat_end`). Override
`begin_reveal` only if attention depends on where the targets start (as Nemotron's prefix/tail mode does). The
reveal loop (`sample_fixed_order`, `logprob_fixed_order`) is model-independent. Register the adapter in
`adapters/__init__.py` (`NAMES`, `adapter_class`) and add a replay test in the style of `tests/test_gpu_replay.py`.
