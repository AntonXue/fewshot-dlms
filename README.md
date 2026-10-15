# fewshot-dlms

Code and frozen artifacts for *Multi-Target Few-Shot Prompting with Diffusion Language Models* (under review, 2026).
A diffusion language model (DLM) fills masked positions anywhere, so one prompt, the *canvas*, holds M demonstrations
and J masked targets filled jointly. This repository recomputes every table of the paper and its experimental
figures (Figures 4–7 and the Appendix B figures) on a CPU from stored candidate pools, and reruns them on a GPU.

## Reproduce in three commands

    pip install -e .                             # CPU only: add --extra-index-url https://download.pytorch.org/whl/cpu
    python -m fewshot_dlms.fetch pools java      # stored pools and judge verdicts, about 390 MB, into artifacts/
    make reproduce                               # every table from the pools, then every figure; CPU, about 1 min

`make reproduce` writes `artifacts/reproduce/` and prints each table's difference to the paper's (`data/tables/`,
which `make figures` plots offline into `experiments/fig*/out/`, 10 PDFs). Its "inside the CI" verdicts are trivially
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
Draw K infills from P1, score each under P0, and resample one with weights softmax(βR):

    R     = log P1 − log P0 = ℓ1 − ℓ0       example reward
    Q_β   ∝ P1 · e^{βR} = P1^{1+β} P0^{−β}  β = 0 is ordinary sampling
    Q_0.5 ∝ P1^{3/2} P0^{−1/2}              β = 0.5, the paper's common strength

The matched sharpening control has the same power 3/2 on P1, without P0: Q ∝ P1^{3/2} (τ = 2/3).
Code: `scoring.py` (`example_reward`, `pool_weights(ell1, ell0, beta, tau)`, `family_metrics`).

**Multi-target canvases.** Demonstrations and J targets are interleaved in one sequence. A fixed reveal order
(ascending position, one token per step) makes the DLM's predictions a normalised distribution over complete infills
(Lemma 3.1), sampled jointly and scored exactly under P1 and P0. A_all needs every target correct; A_avg
(`A_target`) averages over targets. Code: `canvas.py`, `adapters/base.py`.

**Length allocation.** A target holds exactly as many tokens as it has masks. *Oracle* lengths are the reference
answers' token counts; a *learned* head (a trainable copy of the DLM's last block, categorical over 1..256) lets each
candidate draw its lengths; *offsets* L = L* + Δ measure what a wrong length costs. The head is shared by P1 and P0,
so it cancels from R. Code: `lengths.py`.

## Paper → directory

| Paper | Content | Directory |
|---|---|---|
| Section 4.1, Figure 4 | one target, M demonstrations, four model interfaces | [`experiments/single_target`](experiments/single_target/README.md) |
| Section 4.2, Figure 5 | M × J grid of demonstrations and targets | [`experiments/multi_target`](experiments/multi_target/README.md) |
| Section 4.3, Figure 6; Appendix B, Length Allocation | oracle, learned and offset lengths; the length head and its trainer | [`experiments/length_allocation`](experiments/length_allocation/README.md) |
| Section 5, Figure 7; Appendix C, Table 1 | Java helper refactoring, LLM-judged; strata, zero-demonstration ablation | [`experiments/java_refactoring`](experiments/java_refactoring/README.md) |
| Appendix B, Experimental Setup (Guidance selection); Single-Target Prompting | model interfaces, development-only β; gold reward row | `single_target` |
| Appendix B, Multi-Target Prompting | mechanism figure (Fig. 8, both builds); cross-benchmark figure (Fig. 9, both builds) | `multi_target` |
| Appendix B, Likelihood Sharpening | M × J sharpening figure (Fig. 10 ACL / Fig. 11 arXiv); matched-power residuals | `multi_target`; `single_target` |
| Appendix B, Guidance-Strength Sensitivity (arXiv only) | guidance-strength sweep figure (Fig. 10 arXiv) | `single_target/analyze.py --beta-sweep` |

Appendix material is named by topic: its numbers differ between the ACL and arXiv builds. The *ACL version* is the
submitted manuscript; the *arXiv version* is the longer build with the guidance-strength appendix.
`make figures` writes these files (directories under `experiments/`):

| `make figures` output | paper figure |
|---|---|
| `single_target/out/single_target.pdf` | Fig. 4 (Section 4.1) |
| `multi_target/out/multi_target.pdf` | Fig. 5 (Section 4.2) |
| `length_allocation/out/length_allocation.pdf` | Fig. 6 (Section 4.3) |
| `java_refactoring/out/java_refactoring.pdf` | Fig. 7 (Section 5) |
| `multi_target/out/multi_target_mechanism.pdf`, `crossbench_d4.pdf`, `sharpening.pdf` | Appendix B: mechanism (Fig. 8, both builds), cross-benchmark (Fig. 9, both builds), M × J sharpening (Fig. 10 ACL / Fig. 11 arXiv) |
| `single_target/out/reference_reward_row.pdf`, `guidance_strength.pdf`; `length_allocation/out/length_fidelity.pdf` | Appendix B: gold-reward row (Single-Target Prompting), guidance-strength sweep (arXiv version only, Fig. 10 there), length-head fidelity row (Length Allocation) |

`python examples/walkthrough.py --cached` walks one pool from the canvas to ℓ1, ℓ0, R, weights and picks.

## Three levels of fidelity

- **Replay (GPU).** In the authors' batches on their stack (one H100 NVL, torch 2.7.1+cu128) the stored ℓ1 and ℓ0
  are reproduced bit for bit for LLaDA-1.5, Nemotron and Qwen3-8B; dQwen3.5's autotuned kernels vary between
  processes, so it matches within a tolerance (`make test-gpu`).
- **Analysis (CPU).** Every table is reproduced from the stored pools to at most 1e-9, and the figures drawn from
  the tables are pixel-identical to the paper's at 150 dpi (`make test-parity`, `make reproduce`).
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

dQwen3.5 needs `pip install -e '.[gpu]'`, LLaDA-1.5's length heads flash-attn (install:
[`experiments/length_allocation/README.md`](experiments/length_allocation/README.md)).

## The Java judge

Figure 7's verdicts come from an LLM judge, Claude Sonnet 5.5 running as Claude Code sub-agents with the rubric
`experiments/java_refactoring/rubric_r1.md`. They ship with the artifacts, so Figure 7 needs no API access; `judge.py`
re-judges through the Anthropic API, approximately and at a cost.

## Layout and size

    README.md AGENTS.md THIRD_PARTY.md LICENSE ARTIFACTS.md SHA256SUMS Makefile pyproject.toml
    src/fewshot_dlms/   3,036   the library
    examples/             271   walkthrough.py
    experiments/        7,160   one pipeline per figure: run.py, analyze.py, plot.py, README
    tests/              5,527   CPU, parity, GPU and API tests
    tools/              1,134   scripts that cut data/ and the artifacts from the authors' runs
    data/  artifacts_upload/  artifacts/   inputs and the paper's tables; the data card; downloads (not in git)

Lines of Python (`wc -l` over tracked `*.py`). A reader needs `src/` and `examples/` (3,307 lines, kept under 5,000);
parity tests hold the ports in `experiments/` to the paper. A port's first line,
`# ported from editdiff/<path> @ <commit>`, names its source in the authors' research repository.

## License and citation

The code is MIT ([`LICENSE`](LICENSE)). Fonts, benchmark data, Java excerpts, checkpoints and artifacts keep their
own terms: see [`THIRD_PARTY.md`](THIRD_PARTY.md) and the data card,
[huggingface.co/datasets/UT-IFML/fewshot-dlms](https://huggingface.co/datasets/UT-IFML/fewshot-dlms) (private until
the public release; text in [`artifacts_upload/README.md`](artifacts_upload/README.md)).

    @misc{fewshotdlms2026, title = {Multi-Target Few-Shot Prompting with Diffusion Language Models},
      year = {2026}, note = {Under review}, url = {https://github.com/AntonXue/fewshot-dlms}}
