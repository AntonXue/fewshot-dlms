# fewshot-dlms

Code and frozen artifacts for *Multi-Target Few-Shot Prompting with Diffusion Language Models* (under review, 2026).
A diffusion language model (DLM) fills masked positions anywhere, so one prompt, the *canvas*, holds D demonstrations
and J masked targets filled jointly. This repository recomputes every table of the paper and its experimental
figures (Figures 5–12) on a CPU from stored sample pools, and reruns them on a GPU. Figure, section and appendix
numbers refer to the arXiv version of the paper (v1). The code and the stored records write `M` for the paper's D,
and the tables and records keep the authors' names: the column or rule `ordinary` is the unguided row (β = 0),
`tilting` is example guidance, `gold` the reference answer and `proposal` (in seed strings) a sample.

## Reproduce in three commands

    pip install -e .                             # CPU only: add --extra-index-url https://download.pytorch.org/whl/cpu
    python -m fewshot_dlms.fetch pools java      # stored pools and judge verdicts, about 390 MB, into artifacts/
    make reproduce                               # every table from the pools, then every figure; CPU, about 1 min

`make reproduce` writes `artifacts/reproduce/` and prints each table's difference to the paper's (`data/tables/`,
which `make figures` plots offline into `experiments/*/out/`, 10 PDFs). Its "inside the CI" verdicts are trivially
true, since the pools analysed are the paper's own; they test reruns (`make rerun-*`). The fetcher pins the revision in
[`ARTIFACTS.md`](ARTIFACTS.md) and checks SHA-256s; while the dataset is private it needs a Hugging Face login. Tests
(`pip install -e '.[dev]'` first):

    make test          # CPU unit and fixture tests; no downloads, about 30 s
    make test-parity   # CPU; every paper table from the pools; needs `fetch pools java`
    make test-gpu      # GPU replay; needs the checkpoints, flash-attn, `fetch java heads`
    make test-api      # the judge via the Anthropic API; needs `fetch java` and ANTHROPIC_API_KEY; spends credits

`make test` skips the tests that read the SST-2 / MIT Movie text until `pip install -e '.[data]'` and
`python data/benchmarks/build.py` (network) build it.

## The three ideas

**Example guidance.** P1 is the model's distribution over infills (a token string per target) with the
demonstration outputs visible, P0 with them masked (positions `E`); ℓ1 = log P1 (`ell1`), ℓ0 = log P0 (`ell0`).
Draw K samples from P1, score each under P0, and resample one with weights softmax(βR):

    R     = log P1 − log P0 = ℓ1 − ℓ0       example reward
    Q_β   ∝ P1 · e^{βR} = P1^{1+β} P0^{−β}  β = 0 is unguided sampling
    Q_0.5 ∝ P1^{3/2} P0^{−1/2}              β = 0.5, the paper's common strength

The matched sharpening control has the same power 3/2 on P1, without P0: Q ∝ P1^{3/2} (τ = 2/3).
Code: `scoring.py` (`example_reward`, `pool_weights(ell1, ell0, beta, tau)`, `family_metrics`).

**Multi-target canvases.** Demonstrations and J targets are interleaved in one sequence. A fixed reveal order
(ascending position, one token per step) makes the DLM's predictions a normalised distribution over complete infills
(Lemma 3.1), sampled jointly and scored exactly under P1 and P0. A_all needs every target correct; A_avg
(`A_target`) averages over targets. Code: `canvas.py`, `adapters/base.py`.

**Length allocation.** A target holds exactly as many tokens as it has masks. *Oracle* lengths are the reference
answers' token counts; a learned *length predictor* (the code's length head: a trainable copy of the DLM's last block,
categorical over 1..256) lets each sample draw its lengths; *offsets* L_j = L_j* + Δ measure what a wrong length costs.
The predictor is shared by P1 and P0, so it cancels from R. Code: `lengths.py`.

## Paper → directory

| Paper (arXiv v1) | Content | Directory | `make figures` output |
|---|---|---|---|
| Section 4.1, Figure 5; Appendix B.2 | one target, D demonstrations, four model interfaces; the reference answer's reward | [`single_target`](experiments/single_target/README.md) | `single_target.pdf`, `reference_reward_row.pdf` |
| Section 4.2, Figure 6; Appendix B.3, Figures 9–10 | D × J grid; reference NLL by reveal position; cross-benchmark study (D = 4) | [`multi_target`](experiments/multi_target/README.md) | `multi_target.pdf`, `multi_target_mechanism.pdf`, `crossbench_d4.pdf` |
| Section 4.3, Figure 7; Appendix B.4 | oracle, learned and offset lengths; the length predictor and its trainer | [`length_allocation`](experiments/length_allocation/README.md) | `length_allocation.pdf`, `length_fidelity.pdf` |
| Section 5, Figure 8; Appendix C, Table 1 | Java helper refactoring, LLM-judged; strata, zero-demonstration ablation, Java predictor | [`java_refactoring`](experiments/java_refactoring/README.md) | `java_refactoring.pdf` |
| Appendix B.1 | model interfaces and decoding (and the authors' development-only β record) | `single_target` | — |
| Appendix B.5, Figure 11 | guidance strength β ∈ {0, .25, .5, 1, 2} on the stored pools | `single_target` (`--beta-sweep`) | `guidance_strength.pdf` |
| Appendix B.6, Figure 12 | likelihood sharpening at the matched exponent: D × J grid; single-target residuals | `multi_target`; `single_target` | `sharpening.pdf` (in `multi_target`) |

`make figures` writes each PDF to `experiments/<directory>/out/`. `reference_reward_row.pdf` and
`length_fidelity.pdf` plot quantities that arXiv v1 reports only in its text (Appendices B.2 and B.4). Figure 8's
verdicts come from an LLM judge (Claude Sonnet 5.5 as Claude Code sub-agents, rubric
`experiments/java_refactoring/rubric_r1.md`); they ship with the artifacts, and `judge.py` re-judges through the
Anthropic API, approximately and at a cost. `python examples/walkthrough.py --cached` walks one pool from the canvas
to ℓ1, ℓ0, R, weights and picks.

## Three levels of fidelity

- **Replay (GPU).** In the authors' batches on their stack (one H100 NVL, torch 2.7.1+cu128) the stored ℓ1 and ℓ0
  are reproduced bit for bit for LLaDA-1.5, Nemotron and Qwen3-8B; dQwen3.5's autotuned kernels vary between
  processes, so it matches within a tolerance (`make test-gpu`).
- **Analysis (CPU).** Every table is reproduced from the stored pools to at most 1e-9, and Figures 5–12 drawn from
  the tables are pixel-identical to the arXiv figures at 150 dpi (`make test-parity`, `make reproduce`).
- **Resampling (GPU).** On that stack `run.py` at the paper's K reproduces the frozen records of LLaDA-1.5, Nemotron
  and Qwen3-8B bit for bit. Elsewhere, with another K, or for dQwen3.5 a rerun passes when the paper's mean lies in
  its 95 % CI (`analyze.py --compare`; `make rerun-single-target` … `rerun-java`; each README's "Resampling check").

## Requirements

Python 3.12 and the versions pinned in `pyproject.toml`. The authors ran every GPU step on one H100 NVL (96 GB) per
process. The adapters load these checkpoints in bf16 from the Hugging Face cache (`HF_HUB_OFFLINE=1` for offline use):

| adapter | Hugging Face model | pinned revision |
|---|---|---|
| `llada` | `GSAI-ML/LLaDA-1.5` | `84346fd91ba60252d260022201ad6fc5a3468fb2` |
| `nemotron_pt`, `nemotron_ar` | `nvidia/Nemotron-Labs-Diffusion-8B` | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| `dqwen` | `UT-IFML/dQwen3.5-9B-Base` | `f193fa20e341504f8d9e2b4b8342048c466e6590` |
| `qwen` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

dQwen3.5 needs `pip install -e '.[gpu]'`, LLaDA-1.5's length predictor flash-attn (install:
[`experiments/length_allocation/README.md`](experiments/length_allocation/README.md)).

## Layout and size

    README.md AGENTS.md THIRD_PARTY.md LICENSE ARTIFACTS.md SHA256SUMS Makefile pyproject.toml
    src/fewshot_dlms/   3,038   the library
    examples/             271   walkthrough.py
    experiments/        7,168   one pipeline per study: run.py, analyze.py, plot.py, README
    tests/              5,578   CPU, parity, GPU and API tests
    tools/              1,183   scripts that cut data/ and the artifacts from the authors' runs
    data/  artifacts_upload/  artifacts/   inputs and the paper's tables; the data card; downloads (not in git)

Lines of Python (`wc -l` over tracked `*.py`). A reader needs `src/` and `examples/` (3,309 lines, kept under 5,000);
parity tests hold the ports in `experiments/` to the paper. A port's first line,
`# ported from editdiff/<path> @ <commit>`, names its source in the authors' research repository.

## License and citation

The code is MIT ([`LICENSE`](LICENSE)). Fonts, benchmark data, Java excerpts, checkpoints and artifacts keep their
own terms: see [`THIRD_PARTY.md`](THIRD_PARTY.md) and the data card,
[huggingface.co/datasets/UT-IFML/fewshot-dlms](https://huggingface.co/datasets/UT-IFML/fewshot-dlms) (private until
the public release; text in [`artifacts_upload/README.md`](artifacts_upload/README.md)).

    @misc{fewshotdlms2026, title = {Multi-Target Few-Shot Prompting with Diffusion Language Models},
      year = {2026}, note = {Under review}, url = {https://github.com/AntonXue/fewshot-dlms}}
