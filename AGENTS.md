# fewshot-dlms agent guide

Code for "Multi-Target Few-Shot Prompting with Diffusion Language Models". README.md is the full introduction.

## Where things are

- `src/fewshot_dlms/`: `scoring.py` (the equations), `bootstrap.py` (intervals), `grading.py` (answers, graders),
  `canvas.py` (target sites, `E`), `adapters/` (the only code that touches a model), `lengths.py` (allocation,
  length head = the paper's length predictor), `data.py` (families, Java episodes), `fetch.py` (artifacts).
- `experiments/{single_target,multi_target,length_allocation,java_refactoring}/`: one pipeline per study, `run.py`
  (GPU), `analyze.py` and `plot.py` (CPU); each README maps paper sections, figures, tables and commands.
- `data/`: `benchmarks/` and `java/` (inputs), `tables/` (the paper's tables, parity targets), `fixtures/` (CPU tests).
- A *family* is one frozen episode (`data.load_families(task, bank)`); cell (M, J) uses its first M demos, J targets.
  The code and the stored records write `M` for the paper's D (demonstrations); keep `M` in code, flags and records.
- A ported file's first line, `# ported from editdiff/<path> @ <commit>`, names its source in the authors' research
  repository at a fixed commit. Keep it when editing the file.

## Conventions

- Pool weights are `softmax(β · (ℓ1 − ℓ0))`, ℓ1 = log P1 (demonstration outputs visible), ℓ0 = log P0 (outputs
  masked at `E`). β = 0 is unguided sampling; β = 0.5 is the paper's common strength, Q ∝ P1^{3/2} P0^{−1/2}. The
  sharpening control is `pool_weights(ell1, ell0, 0.0, tau=2/3)`, target P1^{3/2}.
- Figure, section and appendix numbers are those of the paper's arXiv version (v1), e.g. Figure 12 (Appendix B.6,
  Likelihood Sharpening). Name files, directories and tests by content, never by figure number.
- Prose follows the paper: unguided (β = 0), example guidance, samples, reference answer, Java predictor, D. Tables,
  records, flags and hashes keep the authors' names (`ordinary`, `tilting`, `gold`, `proposal`, `--score-gold`, `M`).
- Code is plain on purpose: one statement per line, docstrings naming the paper quantity; follow `scoring.py`.

## Artifacts and tests

- `python -m fewshot_dlms.fetch <group> ...` downloads `UT-IFML/fewshot-dlms` at the revision pinned in ARTIFACTS.md
  into `artifacts/` and checks `SHA256SUMS`. Groups: `pools` and `java` (about 390 MB together, what the analyses
  need), `heads` (3.5 GB), `head_training` (1.0 GB).
- `make test`: CPU unit tests and fixtures; no downloads, no GPU. Run it before every commit.
- `make test-parity` and `make reproduce`: CPU; every paper table from the pools; need `fetch pools java`.
- `make test-gpu`: replay on a CUDA GPU; needs the pinned checkpoints, flash-attn, `fetch java heads`.
- `make test-api`: calls the Anthropic API, spends credits; needs `fetch java`, `ANTHROPIC_API_KEY`. Never unasked.
- SST-2 / MIT Movie text: `python data/benchmarks/build.py` (network, `data` extra); tests that need it skip without.

## Rules

- Never edit the frozen tables under `data/tables/`: they are the paper's numbers; if `analyze.py` disagrees, the
  code is wrong. `data/` is otherwise written only by `tools/`; `artifacts/` is fetched.
- No private paths, user names, host names or machine metadata in a committed file; `src/fewshot_dlms/leakcheck.py`
  holds the generic patterns (`find_private`), which the leak tests and `tools/assemble_artifacts.py` share.
- GPU: one process per GPU. On the authors' stack (H100 NVL, torch 2.7.1+cu128) reruns of LLaDA-1.5, Nemotron and
  Qwen3-8B at the paper's K reproduce the frozen records bit for bit. Across GPUs, torch builds or K, and for
  dQwen3.5, judge a rerun by `analyze.py --compare` (paper's mean inside the rerun's 95 % CI), never by equality.
- The paper's Java verdicts came from Claude Sonnet 5.5 running as Claude Code sub-agents (rubric
  `experiments/java_refactoring/rubric_r1.md`) and ship with the artifacts. `judge.py` re-implements the judge with the
  Anthropic API; re-judging is approximate and costs credits, so never call it in tests or loops.

## Adding a model adapter

Subclass `Adapter` in `src/fewshot_dlms/adapters/base.py`: a pinned `model_id`/`revision`, `name`, `mask_id`, `kind`,
the tokenizer in `tok` with `V` and `control_ids`, `logits` and the four chat methods. A canvas model (`kind = "dlm"`)
runs in the model-independent reveal loop (`sample_fixed_order`, `logprob_fixed_order`); override `begin_reveal`
only if attention depends on where the targets start (Nemotron prefix/tail). A left-to-right model (`kind = "ar"`)
returns next-token logits and runs in `sample_ar` / `logprob_ar` / `decode_ar` on `canvas.ar_prompt_ids`; set
`default_support`, `sample_chunk`, `p0_chunk` and, for free-length stopping, `stop_ids()` (see `qwen_ar.py`). Then
register it in `adapters/__init__.py` (`NAMES`, `adapter_class`), add it to the pipelines' model lists
(`INTERFACE` and `MODELS41`/`MODELS42` in `single_target/run.py`, `MODELS` in the `multi_target` and
`length_allocation` `run.py`), give it entries in `TOLERANCES` and `BATCHED_TOLERANCES` of `tests/gpu_util.py`, and add
a replay test like `tests/test_gpu_replay.py`. CPU tests need no checkpoint: `tests/fakes.py` (`FakeAdapter`, a
byte-level stand-in) and `tests/test_adapters_cpu.py` (fake tokenizer and model classes with `monkeypatch`) are the
harness to copy.
