# Artifacts

The frozen pools, length heads, head-training data and judge verdicts live on the Hugging Face dataset
[`UT-IFML/fewshot-dlms`](https://huggingface.co/datasets/UT-IFML/fewshot-dlms) (4.89 GB). Its README is the data card:
provenance, seeds, protocol hashes and licenses per group. `python -m fewshot_dlms.fetch` downloads groups of it at the
pinned revision into `artifacts/` and checks every file against `SHA256SUMS`.

## Fetching

Run from the repository root:

    python -m fewshot_dlms.fetch pools java       # make test-parity, make reproduce, every analyze.py (386 MB)
    python -m fewshot_dlms.fetch heads            # the length heads for the GPU runs (3.47 GB)
    python -m fewshot_dlms.fetch head_training    # the heads' training data (1.03 GB)
    python -m fewshot_dlms.fetch --check          # re-check what is under artifacts/, no download
    python -m fewshot_dlms.fetch                  # the group table; downloads nothing

Each group arrives with `SHA256SUMS` and the data card `README.md`. The fetcher checks the downloaded `SHA256SUMS`
against the SHA-256 below, then every file present against it, and fails on a mismatch (delete the listed files and
run the fetch again) or on a missing file of a requested group. `--dest` overrides `artifacts/`. `--revision`
downloads, or with `--check` checks, another dataset revision: the fetcher warns that the check of `SHA256SUMS`
against the pinned SHA-256 below is skipped, and checks the files against that revision's `SHA256SUMS` only. While
the dataset is private, the download needs a Hugging Face login with access (`hf auth login` or `HF_TOKEN`); a
refused login or an unreachable Hub ends the fetch with one line naming the cause and the remedy (exit status 1).

| | |
|---|---|
| dataset | `UT-IFML/fewshot-dlms` (repo type `dataset`) |
| pinned revision | `f65650d34e80892b94a69083371fd939ed64edcc` (also in `data/artifacts_revision.txt`) |
| file list | [`SHA256SUMS`](SHA256SUMS): `<sha256>  <path>` for each of the 500 files, the format of `sha256sum` |
| SHA-256 of `SHA256SUMS` | `752eeb64450508902125753bc3476dffe3f3164df84c8f95ad734fac2d573680` |
| built by | `tools/assemble_artifacts.py` (data card: `artifacts_upload/README.md`) |

The dataset holds `SHA256SUMS` too; the copy here is the one the fetcher trusts. To check a download by hand:

    cd artifacts && sha256sum --check --ignore-missing ../SHA256SUMS

## Groups

| group | files | bytes | what reads it |
|---|---|---|---|
| `pools/single_target` | 336 | 260,397,761 | `experiments/single_target/analyze.py` (also the cross-benchmark and guidance-strength sweep figures) |
| `pools/multi_target` | 60 | 41,915,679 | `experiments/multi_target/analyze.py` (Figure 5, the mechanism and M × J sharpening figures) |
| `pools/length_allocation` | 56 | 73,747,644 | `experiments/length_allocation/analyze.py` |
| `java` | 21 | 10,060,127 | `experiments/java_refactoring/analyze.py`, `judge.py` |
| `heads` | 4 | 3,473,109,316 | `experiments/length_allocation/run.py --mode learned`, `experiments/java_refactoring/run.py --regime learned`, `make test-gpu` |
| `head_training` | 22 | 1,025,900,426 | `experiments/length_allocation/train_head.py` |
| `README.md` | 1 | 21,334 | the data card |
| total | 500 | 4,885,152,287 | |

`pools/length_allocation/<model>/list_easy_fam_M4_J{J}_oracle.jsonl` are the same bytes as
`pools/single_target/report/<model>/list_easy_fam_M4_J{J}_il.jsonl`, shipped in both groups.

`head_training/fsp_*` hold 66,499 of the authors' 100,000 structured few-shot training prompts: the 33,501 built
from MIT Movie and ATIS are not shipped (data card, `head_training`). The frozen heads were trained on all
100,000.

### Heads re-saved

The four heads are the authors' checkpoints re-saved with one entry removed from the training configuration pickled
in them: `root`, the authors' run directory (no other key changed). The weights are unchanged: every tensor was
checked equal to the original (`torch.equal` and the raw bytes), and each file loads with `torch.load(...,
weights_only=True)` and `lengths.LengthHead.load`. The re-saved files therefore have new SHA-256s. The stored records
keep the original values as provenance: `ckpt_sha` in the `npz` predictions, `head_ckpt_sha` in the Java pools, and
`checkpoint_sha256` in `experiments/length_allocation/frozen_config.json` (the freeze record). The length_allocation head fixture
holds both (`checkpoint_sha256`, `trained_checkpoint_sha256`), and `experiments/java_refactoring/run.py` accepts both for
head C.

| file | original SHA-256 (recorded in pools and predictions) | published SHA-256 |
|---|---|---|
| `heads/llada.pt` | `0de13a8ac6b881493e368d281531326b84fe7f5d2c50b1dccba6491dd1dec1d8` | `a49355771567e3b091e8a5d79b09aadd460b677af0f850af6d4afe27360e1452` |
| `heads/nemotron.pt` | `df46105dab89104d0d1ccdb34795343a69fadda5528d610fd725bf414fb62e62` | `c0509fba53347491ed927f1ee2d55f8ae9ce5859eeabd4ef8a245d215943842a` |
| `heads/dqwen.pt` | `83126cce639d3aa28cee0ef13cc526cd63684b8c94c6ff51ddad511ba206232f` | `3241752d59aabb53769fa5e3353c04142fc03bbbeb3cc9116098f1cbbdcf927d` |
| `heads/java_head_C.pt` | `ea54c8b92110f4c5c4dbe838721d23961775082712247f8099b74e8d7d312309` | `b2efefded5ac382b5fc67ace4ee6d9f0ebfa15b820f2431b6f67cdb09721cebf` |

The training configurations in `head_training/configs/` likewise lack `root` and the GPU index `gpu`.

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
- The upload used huggingface_hub 1.23.0 (`upload_large_folder`; the re-saved heads, configs, card and
  `SHA256SUMS` were then replaced with `upload_folder`, later the four structured few-shot files, card and
  `SHA256SUMS` once more). The dataset's history was then squashed into the single pinned commit.
