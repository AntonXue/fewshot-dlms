# Artifacts

The frozen pools, length heads, head-training data and judge verdicts live on the Hugging Face dataset
[`UT-IFML/fewshot-dlms`](https://huggingface.co/datasets/UT-IFML/fewshot-dlms) (5.14 GB). Its README is the data card:
provenance, seeds, protocol hashes and licenses per group. `python -m fewshot_dlms.fetch` downloads it at the pinned
revision into `artifacts/` and checks every file against `SHA256SUMS`.

| | |
|---|---|
| dataset | `UT-IFML/fewshot-dlms` (repo type `dataset`) |
| pinned revision | `60a9ff00abcea4132e660fd65177a1fe9a2d576f` (also in `data/artifacts_revision.txt`) |
| file list | [`SHA256SUMS`](SHA256SUMS): `<sha256>  <path>` for each of the 500 files, the format of `sha256sum` |
| SHA-256 of `SHA256SUMS` | `00ef78a61a0fb13e2889a2367d3fb41e1f7215734dad55a004855e649e5f044e` |
| built by | `tools/assemble_artifacts.py` (data card: `artifacts_upload/README.md`) |

The dataset holds `SHA256SUMS` too; the copy here is the one the fetcher trusts. To check a download by hand:

    cd artifacts && sha256sum --check --ignore-missing ../SHA256SUMS

## Groups

| group | files | bytes | what reads it |
|---|---|---|---|
| `pools/fig4` | 336 | 260,397,761 | `experiments/fig4_single_target/analyze.py` (also Figure 9 and Appendix B.6) |
| `pools/fig5` | 60 | 41,915,679 | `experiments/fig5_multi_target/analyze.py` (Figures 5, 8, 10) |
| `pools/fig6` | 56 | 73,747,644 | `experiments/fig6_lengths/analyze.py` |
| `java` | 21 | 10,060,127 | `experiments/fig7_java/analyze.py`, `judge.py` |
| `heads` | 4 | 3,473,108,322 | `run.py --mode learned` (Figure 6), `fig7_java/run.py --regime learned`, `make test-gpu` |
| `head_training` | 22 | 1,278,322,098 | `experiments/fig6_lengths/train_head.py` |
| `README.md` | 1 | 16,207 | the data card |
| total | 500 | 5,137,567,838 | |

`pools/fig6/<model>/list_easy_fam_M4_J{J}_oracle.jsonl` are the same bytes as
`pools/fig4/report/<model>/list_easy_fam_M4_J{J}_il.jsonl`, shipped in both groups.

The four heads, by SHA-256 (the values the freeze record, the fixtures and the Java pools refer to):

| file | SHA-256 |
|---|---|
| `heads/llada.pt` | `0de13a8ac6b881493e368d281531326b84fe7f5d2c50b1dccba6491dd1dec1d8` |
| `heads/nemotron.pt` | `df46105dab89104d0d1ccdb34795343a69fadda5528d610fd725bf414fb62e62` |
| `heads/dqwen.pt` | `83126cce639d3aa28cee0ef13cc526cd63684b8c94c6ff51ddad511ba206232f` |
| `heads/java_head_C.pt` | `ea54c8b92110f4c5c4dbe838721d23961775082712247f8099b74e8d7d312309` |

## Models

| model | Hugging Face id | revision |
|---|---|---|
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | `84346fd91ba60252d260022201ad6fc5a3468fb2` |
| Nemotron | `nvidia/Nemotron-Labs-Diffusion-8B` | `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| dQwen3.5 | `UT-IFML/dQwen3.5-9B-Base` | `f193fa20e341504f8d9e2b4b8342048c466e6590` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

The adapters in `src/fewshot_dlms/adapters/` load exactly these revisions.

## Environment of the runs

- GPU: NVIDIA H100 NVL (96 GB), one process per GPU.
- Python 3.12, torch 2.7.1+cu128, transformers 5.13.0, numpy 1.26.4, pandas 2.3.3, matplotlib 3.10.7
  (`pyproject.toml` pins the same versions).
- dQwen3.5: flash-linear-attention 0.5.1 (the `gpu` extra), without causal-conv1d.
- LLaDA-1.5 length heads: flash-attn 2.8.3.
- The upload used huggingface_hub 1.23.0 (`upload_large_folder`).
