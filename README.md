# fewshot-dlms

Code and frozen artifacts for the paper *Multi-Target Few-Shot Prompting with Diffusion Language Models* (under
review, 2026).

A diffusion language model (DLM) fills masked positions anywhere in a sequence, so one prompt, the *canvas*, can hold
M few-shot demonstrations and J masked targets, and the model fills all J targets jointly. The paper uses the
demonstrations as evidence for the infill (example guidance), places several targets on one canvas, and allocates a
length to every target. This repository recomputes every table and figure of the paper on a CPU from the stored
candidate pools, replays the stored candidates on a GPU, and samples new pools with the paper's seeds.

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
| Section 4.3, Figure 6 | oracle, learned and offset target lengths | [`experiments/fig6_lengths`](experiments/fig6_lengths/README.md) |
| Section 5, Figure 7 | Java helper refactoring, LLM-judged | [`experiments/fig7_java`](experiments/fig7_java/README.md) |
| Appendix B.1, B.2, gold row | model interfaces, development-only β, gold example reward | `experiments/fig4_single_target` |
| Appendix B.3, B.4 (Figures 8–10) | cross-benchmark study, mechanism, sharpening control | `experiments/fig5_multi_target` |
| Appendix B.5 | length head: fidelity row, trainer | `experiments/fig6_lengths` |
| Appendix B.6 (Figure 11) | guidance-strength sweep | `experiments/fig4_single_target/analyze.py --beta-sweep` |
| Appendix C, Table 1 | Java strata, β sensitivity, zero-demonstration ablation, pool classes | `experiments/fig7_java` |

Each experiment README gives its protocol, seeds, commands, runtime and fidelity. `examples/walkthrough.py` follows
one pool end to end (a List Functions–Easy family, M = 2, J = 2, K = 8): the canvas, then ℓ1, ℓ0, R and the weight of
every candidate, then the ordinary and the guided pick. `python examples/walkthrough.py --cached` runs it on a CPU
from a stored pool; without `--cached` it samples and scores the pool on one GPU.

## Reproduce in three commands

    pip install -e .
    python -m fewshot_dlms.fetch pools java      # the stored pools and judge verdicts, about 390 MB, into artifacts/
    make figures                                 # every figure, PDFs in experiments/*/out/, CPU, seconds

`make figures` draws Figures 4–11 and two appendix rows from the paper's tables in `data/tables/`. Those tables are
computed from the pools: each `analyze.py` recomputes its tables from `artifacts/`, `make test-parity` checks all of
them against `data/tables/`, and `plot.py --tables` draws from a recomputed table, for example:

    python experiments/fig6_lengths/analyze.py --pools artifacts/pools/fig6 --compare data/tables/fig6
    python experiments/fig6_lengths/plot.py --tables experiments/fig6_lengths/out

The fetcher downloads the dataset `UT-IFML/fewshot-dlms` at the revision pinned in [`ARTIFACTS.md`](ARTIFACTS.md)
and checks every file against `SHA256SUMS`. The groups `heads` (3.5 GB) and `head_training` (1.3 GB) are fetched
separately, only for learned lengths on a GPU or to retrain a head. On a machine without a GPU,
`pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu` installs the CPU build of torch.

Tests (`pip install -e '.[dev]'` first):

    make test          # CPU unit tests and fixtures; no downloads, about 30 s
    make test-parity   # CPU; every paper table from the pools; needs `fetch pools java` (skipped without it)
    make test-gpu      # GPU replay; needs a CUDA GPU, the checkpoints, flash-attn, `fetch java heads`
    make test-api      # calls the Anthropic API (the judge); needs ANTHROPIC_API_KEY, spends credits

SST-2 and MIT Movie ship without their text. Runs on these two benchmarks, four CPU tests and some GPU tests need it
rebuilt from the public sources once: `pip install -e '.[data]'` and `python data/benchmarks/build.py`
(see [`data/benchmarks/README.md`](data/benchmarks/README.md)).

## Three levels of fidelity

1. **Replay (GPU, exact tokens).** The pools store each candidate's token ids, ℓ1 and ℓ0. Scored in the batches of
   the authors' runs on an H100 NVL with torch 2.7.1+cu128, the ported canvases and adapters reproduce the stored
   records of LLaDA-1.5, Nemotron (all interfaces) and Qwen3-8B bit for bit. dQwen3.5 runs autotuned Triton kernels
   whose numerics vary from process to process, so its replay is statistical (`make test-gpu`; tolerances in
   `tests/gpu_util.py`).
2. **Analysis (CPU, exact).** From the stored pools, each `analyze.py` reproduces every table of the paper to at most
   1e-9 (`make test-parity`), and every figure drawn from the tables is pixel-identical to the paper's at 150 dpi
   (`make figures`). Small fixture pools are checked the same way on every `make test`.
3. **Resampling (GPU, statistical).** A fresh run on another GPU, with another K or for dQwen3.5 draws other
   candidates. It is judged by whether the paper's mean lies inside the rerun's 95 % CI (`analyze.py --compare`;
   `make rerun-fig4` … `make rerun-fig7` rerun the first 8 families or episodes of one setting). The experiment READMEs report such reruns
   under "Resampling check".

## Requirements

Python 3.12 with the versions pinned in `pyproject.toml` (torch 2.7.1, transformers 5.13.0). The CPU steps need
nothing else.

GPU steps (`run.py`, `make test-gpu`, the live walkthrough) need one H100-class GPU with 80–96 GB; the paper's runs
used one H100 NVL (96 GB) per process. The adapters in `src/fewshot_dlms/adapters/` load these checkpoints at pinned
revisions in bf16 (15–17 GB each; the three diffusion models run their own modelling code, `trust_remote_code`):

| adapter | Hugging Face model | revision |
|---|---|---|
| `llada` | `GSAI-ML/LLaDA-1.5` | `84346fd91ba60252d260022201ad6fc5a3468fb2` |
| `nemotron_pt`, `nemotron_ar` | `nvidia/Nemotron-Labs-Diffusion-8B` (prefix/tail canvas; causal suffix) | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| `dqwen` | `UT-IFML/dQwen3.5-9B-Base` | `f193fa20e341504f8d9e2b4b8342048c466e6590` |
| `qwen` | `Qwen/Qwen3-8B` (free-length left-to-right continuation) | `b968826d9c46dd6066d109eabc6255188de91218` |

They are taken from the Hugging Face cache (`$HF_HOME`) and downloaded on first use. Without network access, fill
the cache beforehand and set `HF_HUB_OFFLINE=1`.

- **dQwen3.5** needs the `gpu` extra, `pip install -e '.[gpu]'` (flash-linear-attention 0.5.1, the kernels of the
  paper's runs). Do not install causal-conv1d: transformers would switch to another kernel and the numbers drift.
- **LLaDA-1.5's length heads** (learned lengths in Figures 6 and 7, `train_head.py --backbone llada`, the head tests)
  need flash-attn, which compiles against the installed torch, so install it after the package:

      pip install -e .
      pip install "flash-attn==2.8.3" --no-build-isolation

## The Java judge

Figure 7's correctness verdicts come from an LLM judge: Claude Sonnet 5.5 running as Claude Code sub-agents with
the rubric `experiments/fig7_java/rubric_r1.md`. The paper's verdicts ship with the artifacts, and `analyze.py`
reads them, so no API access is needed to reproduce Figure 7. `experiments/fig7_java/judge.py` re-implements the
judge with the Anthropic API (same packets, same rubric, model `claude-sonnet-5-5`) for new candidates; re-judging
is approximate, needs `pip install -e '.[judge]'` and `ANTHROPIC_API_KEY`, and spends API credits. `make test-api`
judges two calibration controls through the API; no other test calls it.

## Layout and size

    src/fewshot_dlms/     2,721 lines   the library: canvas, adapters, scoring, bootstrap, grading, lengths, fetch
    examples/               241 lines   walkthrough.py
    experiments/figN_*/   6,904 lines   one pipeline per figure: run.py (GPU), analyze.py, plot.py (CPU), README
    tests/                4,574 lines   CPU unit and fixture tests, parity (fetched pools), GPU replay, API
    tools/                1,096 lines   one-off scripts that cut data/ and the artifacts from the authors' runs
    data/benchmarks/                    the four frozen benchmarks (dev and report banks)
    data/java/                          the 127 Java episodes, ledgers, provenance, upstream licenses
    data/tables/                        the paper's tables (parity targets, frozen)
    data/fixtures/                      small pools for the CPU tests and the cached walkthrough
    artifacts/                          written by the fetcher, not in git

Counts are lines of Python. The library a reader needs to understand the method is `src/` and `examples/`
(about 3,000 lines). The pipelines in `experiments/` are faithful transliterations of the authors' scripts, held to
the paper's numbers by the parity tests.

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
