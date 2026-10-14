# fewshot-dlms

Code and frozen artifacts for the paper *Multi-Target Few-Shot Prompting with Diffusion Language Models* (under
review, 2026).

A diffusion language model (DLM) fills masked positions anywhere in a sequence, so one prompt, the *canvas*, can hold
M few-shot demonstrations and J masked targets, and the model fills all J targets jointly. The paper uses the
demonstrations as evidence for the infill (example guidance), places several targets on one canvas, and allocates a
length to every target. This repository recomputes every table of the paper and its experimental figures (Figures
4–7 and the Appendix B figures) on a CPU from the stored candidate pools, replays the stored candidates on a GPU, and
samples new pools with the paper's seeds.

## Reproduce in three commands

    pip install -e .
    python -m fewshot_dlms.fetch pools java      # the stored pools and judge verdicts, about 390 MB, into artifacts/
    make reproduce                               # every table from the pools, then every figure, CPU, about 1 min

`make reproduce` recomputes every table from `artifacts/` into `artifacts/reproduce/tables/` (printing its largest
difference to the paper's tables) and draws the figures from them into `artifacts/reproduce/figures/`; `make
figures` draws them from the paper's tables in `data/tables/`, with no download. The fetcher pins the dataset
`UT-IFML/fewshot-dlms` at the revision in [`ARTIFACTS.md`](ARTIFACTS.md), checks every file against `SHA256SUMS`,
and, while the dataset is private, needs a Hugging Face login with access. The groups `heads` (3.5 GB) and
`head_training` (1.3 GB) serve learned lengths on a GPU and head retraining. Without a GPU, install CPU torch:

    pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu

Tests (`pip install -e '.[dev]'` first):

    make test          # CPU unit tests and fixtures; no downloads, about 30 s
    make test-parity   # CPU; every paper table from the pools; needs `fetch pools java` (skipped without it)
    make test-gpu      # GPU replay; needs a CUDA GPU, the checkpoints, flash-attn, `fetch java heads`
    make test-api      # the judge through the Anthropic API; needs `fetch java` and ANTHROPIC_API_KEY, spends credits

SST-2 and MIT Movie ship without their text; GPU runs on them and a few tests need it rebuilt once
([`data/benchmarks/README.md`](data/benchmarks/README.md)).

## The three ideas

Notation: an *infill* u puts one token string at every masked target. P1 is the model's distribution over infills
with the demonstration outputs visible, P0 the same model with the demonstration outputs masked in place (the
positions `E` of the canvas). In the code ℓ1 = log P1(u) is `ell1` and ℓ0 = log P0(u) is `ell0`.

**1. Example guidance.** The example reward of an infill is how much the demonstration outputs raise its likelihood.
Draw K infills from P1, score each under P0, and resample one with weights softmax(βR):

    R     = log P1 − log P0 = ℓ1 − ℓ0       example reward
    Q_β   ∝ P1 · e^{βR} = P1^{1+β} P0^{−β}  β = 0 is ordinary sampling
    Q_0.5 ∝ P1^{3/2} P0^{−1/2}              β = 0.5, the paper's common strength

The matched sharpening control puts the same total power 3/2 on P1 without P0: Q ∝ P1^{3/2}, temperature τ = 2/3.
Code: `src/fewshot_dlms/scoring.py` (`example_reward`, `pool_weights(ell1, ell0, beta, tau)`, `family_metrics`).

**2. Multi-target canvases.** The M demonstrations and the J masked targets are interleaved in one sequence. Fixing
the reveal order (ascending canvas position, one token per step) turns the DLM's token predictions into a normalised
distribution over complete infills (fixed-order decoding, Lemma 3.1): the J targets are sampled jointly, a token
revealed at one target informs the others, and any infill is scored exactly under P1 and P0. A_all is the accuracy
with every target correct, A_avg the mean per-target accuracy (`A_target` in the code). Code: `canvas.py` (target
sites, `E`) and `adapters/base.py` (`sample_fixed_order`, `logprob_fixed_order`).

**3. Length allocation.** A masked target holds exactly as many tokens as it has masks, so every target needs a
length L_j before it is filled. *Oracle* lengths are the token counts of the reference answers (Sections 4.1, 4.2
and one regime of Section 5). A *learned* length predictor (a trainable copy of the frozen DLM's last block and a
categorical head over 1..256) reads a compact prompt with one placeholder per target, and each candidate draws its
own allocation. Forced *offsets* L = L* + Δ measure what a wrong length costs. The predictor is shared by P1 and P0,
so it cancels from R. Code: `lengths.py` (`allocate`, `LengthHead`, `sample_allocations`).

## Paper → directory

| Paper | Content | Directory |
|---|---|---|
| Section 4.1, Figure 4 | one target, M demonstrations, four model interfaces | [`experiments/fig4_single_target`](experiments/fig4_single_target/README.md) |
| Section 4.2, Figure 5 | M × J grid of demonstrations and targets on one canvas | [`experiments/fig5_multi_target`](experiments/fig5_multi_target/README.md) |
| Section 4.3, Figure 6; Appendix B, Length Allocation | oracle, learned and offset lengths; the length head and its trainer | [`experiments/fig6_lengths`](experiments/fig6_lengths/README.md) |
| Section 5, Figure 7; Appendix C, Table 1 | Java helper refactoring, LLM-judged; strata, β sensitivity, zero-demonstration ablation | [`experiments/fig7_java`](experiments/fig7_java/README.md) |
| Appendix B, Experimental Setup (Guidance selection); Single-Target Prompting | model interfaces, development-only β; gold example reward row | `fig4_single_target` |
| Appendix B, Multi-Target Prompting | mechanism figure (Fig. 8, both builds); cross-benchmark figure (Fig. 9, both builds) | `fig5_multi_target` |
| Appendix B, Likelihood Sharpening | M × J sharpening figure (Fig. 10 ACL / Fig. 11 arXiv); residuals against matched power sampling | `fig5_multi_target`; `fig4_single_target` |
| Appendix B, Guidance-Strength Sensitivity (arXiv version only) | guidance-strength sweep figure (Fig. 10 arXiv) | `fig4_single_target/analyze.py --beta-sweep` |

Appendix subsections and figures are named by topic: their numbers differ between the ACL and arXiv builds. Each
experiment README gives its protocol, seeds, commands, runtime and fidelity. `examples/walkthrough.py` follows one
pool (M = 2, J = 2, K = 8) from the canvas to ℓ1, ℓ0, R, the weights and the ordinary and guided picks; `--cached`
runs it on a CPU.

## Three levels of fidelity

1. **Replay (GPU, exact tokens).** The pools store each candidate's token ids, ℓ1 and ℓ0. Scored in the batches of
   the authors' runs on their stack (one H100 NVL, torch 2.7.1+cu128), the ported canvases and adapters reproduce
   them bit for bit for LLaDA-1.5, Nemotron (both interfaces) and Qwen3-8B; dQwen3.5's autotuned Triton kernels vary
   from process to process, so it agrees within a tolerance (`make test-gpu`; `tests/gpu_util.py`).
2. **Analysis (CPU, exact).** From the stored pools, each `analyze.py` reproduces every table of the paper to at most
   1e-9 (`make test-parity`, `make reproduce`), and the figures drawn from the tables are pixel-identical to the
   paper's at 150 dpi. On every `make test`, small fixture pools reproduce their stored expected metrics.
3. **Resampling (GPU).** On the authors' stack, `run.py` at the paper's K reproduces the frozen records of
   LLaDA-1.5, Nemotron and Qwen3-8B bit for bit. On another GPU or torch build, with another K, or for dQwen3.5 the
   draws differ and a rerun is statistical: it passes when the paper's mean lies inside its 95 % CI (`analyze.py
   --compare`). `make rerun-fig4` … `make rerun-fig7` rerun the first 8 families or episodes of one cell per
   pipeline; each experiment README's "Resampling check" reports a full cell rerun on a GPU.

## Requirements

Python 3.12 with the versions pinned in `pyproject.toml` (torch 2.7.1, transformers 5.13.0); the CPU steps need
nothing else. The authors ran every GPU step (`run.py`, `make test-gpu`, the live walkthrough) on one H100 NVL
(96 GB) per process; the Java runs of LLaDA-1.5 need about 16 GiB (`experiments/fig7_java/README.md`). The adapters
in `src/fewshot_dlms/adapters/` load these checkpoints at pinned revisions in bf16 (15–17 GB each; the three
diffusion models run their own modelling code, `trust_remote_code`):

| adapter | Hugging Face model | revision |
|---|---|---|
| `llada` | `GSAI-ML/LLaDA-1.5` | `84346fd91ba60252d260022201ad6fc5a3468fb2` |
| `nemotron_pt`, `nemotron_ar` | `nvidia/Nemotron-Labs-Diffusion-8B` (prefix/tail canvas; causal suffix) | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| `dqwen` | `UT-IFML/dQwen3.5-9B-Base` | `f193fa20e341504f8d9e2b4b8342048c466e6590` |
| `qwen` | `Qwen/Qwen3-8B` (free-length left-to-right continuation) | `b968826d9c46dd6066d109eabc6255188de91218` |

They are taken from the Hugging Face cache (`$HF_HOME`) and downloaded on first use; without network access, fill
the cache beforehand and set `HF_HUB_OFFLINE=1`. **dQwen3.5** needs the `gpu` extra, `pip install -e '.[gpu]'`
(flash-linear-attention 0.5.1, the kernels of the paper's runs); do not install causal-conv1d, or transformers
switches kernels and the numbers drift. **LLaDA-1.5's length heads** (learned lengths in Figures 6 and 7,
`train_head.py --backbone llada`, the head tests) need flash-attn, which compiles against the installed torch:

    pip install -e .
    pip install "flash-attn==2.8.3" --no-build-isolation

## The Java judge

Figure 7's verdicts come from an LLM judge, Claude Sonnet 5.5 running as Claude Code sub-agents with the rubric
`experiments/fig7_java/rubric_r1.md`. They ship with the artifacts, so reproducing Figure 7 needs no API access.
`experiments/fig7_java/judge.py` re-implements the judge with the Anthropic API (same packets and rubric, model
`claude-sonnet-5-5`); re-judging is approximate, needs `pip install -e '.[judge]'` and `ANTHROPIC_API_KEY`, and
spends API credits.

## Layout and size

    README.md  AGENTS.md  THIRD_PARTY.md  LICENSE  Makefile  pyproject.toml  ARTIFACTS.md  SHA256SUMS
    src/fewshot_dlms/      2,923 lines   the library: canvas, adapters, scoring, bootstrap, grading, lengths, fetch
    examples/                241 lines   walkthrough.py
    experiments/figN_*/    6,926 lines   one pipeline per figure: run.py (GPU), analyze.py, plot.py (CPU), README
    tests/                 4,833 lines   CPU unit and fixture tests, parity (fetched pools), GPU replay, API
    tools/                 1,125 lines   the scripts that cut data/ and assembled the artifacts from the authors' runs
    data/                                benchmarks/, java/ (episodes, licenses), tables/ (frozen), fixtures/
    artifacts_upload/                    the artifacts' data card (the dataset's README.md)
    artifacts/                           written by the fetcher and make reproduce, not in git

Counts are lines of Python. The library a reader needs is `src/` and `examples/` (3,164 lines, kept under 5,000);
the pipelines in `experiments/` are ports of the authors' scripts, held to the paper's numbers by the parity tests.
A ported file's first line, `# ported from editdiff/<path> @ <commit>`, names its source in the authors' research
repository at a fixed commit.

## Licenses

The code is MIT ([`LICENSE`](LICENSE)). The bundled fonts, the benchmark data, the Java excerpts, the model
checkpoints and the artifacts keep their own terms: see [`THIRD_PARTY.md`](THIRD_PARTY.md) and the artifacts' data
card, [huggingface.co/datasets/UT-IFML/fewshot-dlms](https://huggingface.co/datasets/UT-IFML/fewshot-dlms) (private
until the public release; its text is [`artifacts_upload/README.md`](artifacts_upload/README.md)).

## Citation

    @misc{fewshotdlms2026,
      title  = {Multi-Target Few-Shot Prompting with Diffusion Language Models},
      year   = {2026},
      note   = {Under review},
      url    = {https://github.com/AntonXue/fewshot-dlms}
    }
