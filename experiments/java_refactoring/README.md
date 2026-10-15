# Figure 8: Java helper refactoring (Section 5, Appendix C)

A developer extracted a helper method and rewrote some call sites to use it. LLaDA-1.5 sees the helper and D
accepted rewrites (the demonstrations) and must rewrite J other sites of the same file (the targets). The 127
episodes, their provenance and licenses are in `data/java/`. Numbers refer to the arXiv version of the paper (v1).
The code, the records and the cell names (`M1J2`) write `M` for the paper's D; in the tables the rule `ordinary` is
unguided sampling (β = 0) and `tilting` example guidance (β = 0.5); the Java predictor is length head C (`s5hc_C`,
`java_head_C.pt`).

| Paper (arXiv v1) | Content | Output | Table in `data/tables/java_refactoring/` |
|---|---|---|---|
| Figure 8, left; Appendix C.2 | length mass of the Java predictor within ±2 tokens of the reference length, vs the training-label prior | `plot.py` → `java_refactoring.pdf` | `length_fidelity_by_bin.csv` |
| Figure 8, right; Table 1 (Appendix C.3) | joint judged success per cell, oracle and learned lengths, β = 0 and 0.5 | `plot.py` → `java_refactoring.pdf` | `summary_by_cell.csv`, `paired_intervals.csv` |
| Appendix C.3 | strata (development exposure, reference visible, literal copy, test code, …) | | `strata.csv` |
| Appendix C.2 | length fidelity over all 80 sites, the Java predictor and two comparator predictors | | `length_fidelity.csv` |
| Appendix C.2 | zero-demonstration ablation (D = 0 vs D = 1) | | `zero_demo_main_table.csv` |
| Appendix C.3 | pool classes: estimates on pools with both outcomes | | `pool_taxonomy_estimates.csv` |
| Section 5.2; Appendix C.3 | the Checkstyle pool where guidance moves weight to a wrong draw | | `checkstyle_case.json` |
| not in arXiv v1 | guidance strength β ∈ {0, 0.1, 0.25, 0.5, 1} | | `beta_sensitivity.csv` |

`metrics_by_commit.csv` holds the commit means behind every interval.

## Protocol

- **Episodes.** The 127 episodes of population B (`data/java/episodes.jsonl`, 50 commits, 14 repositories) in
  four cells (D, J) ∈ {1, 2} × {1, 2}: 52, 28, 28 and 19 episodes. The canvas is `canvas.build_java_canvas`
  with the instruction `JAVA_INSTRUCTION`; `E` are the demonstrations' accepted edits.
- **Lengths.** *Oracle*: every target gets the token count of its reference replacement. *Learned*: each of the
  K candidates draws its own lengths from the Java predictor (length head C, `artifacts/heads/java_head_C.pt`, read
  by `lengths.LengthHead`, on the episode's compact prompt). Candidate k, target j takes the inverse CDF of the
  predictor's distribution at u = SHA-256(`s5-final-alloc-v1|0e1d3c9040b09598|<episode>|<J>|<k>|<j>|<r>`)[:16] / 2^64,
  r = 0; a draw whose canvas would exceed 2,048 tokens is redrawn with r + 1 (no draw of the paper needed it).
- **Sampling.** K = 16 infills per episode and regime from P1 by fixed-order decoding (ascending positions,
  temperature 1), one reveal per distinct allocation. Seed: the first 12 hex digits of
  SHA-256(`0e1d3c9040b09598|s5final|<regime>|<episode>|<allocation tuple>`) mod 2^31 (`run.generation_seed`).
  The support is the ordinary token support plus LLaDA-1.5's 115 logit rows beyond its tokenizer, which the Java
  runs' adapter did not exclude (`run.java_support`; without them the log-likelihoods differ by about 1e-5).
- **Scoring.** The distinct infills of a canvas are teacher-forced under P1 (`logp1`) and under P0, the canvas
  with `E` masked (`logp0`), in the same batches as sampling; R = logp1 − logp0.
- **Judge.** Every distinct candidate text is judged once, blind (`judge.py`, rubric r1 in `rubric_r1.md`): a
  draw counts 1 when every target is judged correct; "incorrect" and "uncertain" count 0 (`U_commit_upper_uncertain`
  counts "uncertain" as 1).
- **Estimand.** U_β = Σ_k softmax(βR)_k y_k per (episode, regime); a cell's value is the mean over commits of the
  mean of the commit's episodes. β = 0 is unguided sampling (`rule` = `ordinary` in the tables), β = 0.5 example
  guidance (`rule` = `tilting`).
- **Intervals.** Commit bootstrap, 10,000 resamples, `random.Random(0)`, one `randrange` per commit in order of first
  appearance, percentiles at sorted indices int(0.025 n) and int(0.975 n); paired contrasts resample the
  per-commit differences (`analyze.commit_bootstrap`). Length fidelity resamples commits with all their sites
  (`analyze.site_bootstrap`); the zero-demonstration ablation sorts the commits and takes the upper index
  int(0.975 n) − 1, as its own analysis did (`analyze.zero_demo_bootstrap`).
- **Zero-demonstration ablation** (Appendix C.2). Each D = 1 episode (80 anchors) is sampled without its
  demonstration (D = 0), with it (D = 1) and, for the 47 anchors whose targets also form a D = 2 episode, with both
  demonstrations of that episode (D = 2), under the neutral instruction `JAVA_ZERO_DEMO_INSTRUCTION`, oracle lengths, K = 16,
  β = 0. Seed: SHA-256(`c2a07e4a701f0a41|demo_ablation|<anchor>|<target sites joined by +>|<M>`)[:12] mod 2^31.
  Every arm was judged with the anchor's demonstration in the packet. The ablation ships the paper's verdicts
  (`java/zero_demo/judging/verdicts.jsonl`); re-judging it is not supported, and `judge.py` exits with status 2 on
  its pool files. A rerun of the ablation (`run.py --study zero_demo`) is graded by the shipped verdicts wherever it
  draws texts the paper drew.

## The judge

The paper's verdicts were produced by **Claude Sonnet 5.5 (`claude-sonnet-5-5`) running as Claude Code (version
2.1.288) general-purpose sub-agents** with the Read and Write tools, each call judging up to five packets of different
episodes, with extended thinking enabled (its content was not retained) and no temperature or max-token settings. For
the main run's 2,089 packets, the package's own records count 272 batched calls (329 in total); a later audit of the
session transcripts counts 420 batched calls of at most five packets (477 in total); the verdicts themselves are
unaffected. The ablation's 698 packets reused 93 verdicts of the main run (byte-identical packets) and judged 605 in
calls of up to five packets of different anchors. Before the main run, 46 calibration controls built from the source
of 12 episodes (references, no-ops, partial edits, swapped arguments, flipped conditions, wrong receivers, harmless
comments) were judged one packet per call: 46/46 matched their expected verdicts (45/46 when the ablation judged them
again).
The original verdicts ship with the artifacts (`java/judging/verdicts.jsonl`, `java/zero_demo/judging/verdicts.jsonl`)
and `analyze.py` reads them.

`judge.py` re-implements the instrument with the Anthropic API: the same packets (byte for byte; checked against
all 2,089 + 698 + 46 frozen packets in `make test-parity`), rubric r1 verbatim as the system prompt, one packet per
call, model `claude-sonnet-5-5`, the verdict JSON parsed as in the paper's run. **Re-judging is approximate**: an API
call is not a Claude Code sub-agent, and an LLM's verdicts vary between calls. It needs `pip install -e '.[judge]'` and
`ANTHROPIC_API_KEY` (without it, it exits with status 2 before judging anything), and it costs API credits (a
packet with the rubric is several thousand input tokens). A reply that is not a valid verdict, or a declined
request, leaves its packet unjudged for the next run; any other API error stops the run, which resumes from `--out`.

    python experiments/java_refactoring/judge.py --calibrate --limit 5          # first 5 calibration controls vs expected and the paper's judge
    python experiments/java_refactoring/judge.py --pools artifacts/reruns/java_refactoring/pools   # every new packet of a rerun

A packet's key is the first 20 hex digits of SHA-256(`<rubric sha16>|<episode>|<demo sites>|<texts as sorted JSON>`),
so a rerun that draws texts the paper already judged is graded with the paper's verdicts (`analyze.py --verdicts`).

One test calls the API (two calibration controls; marker `api`). `make test` and `make test-parity` never run it;
run it on purpose with `make test-api` (or `pytest -m api`), with `ANTHROPIC_API_KEY` set and the artifacts fetched.

## Commands

    python experiments/java_refactoring/analyze.py --pools artifacts/java/pools --compare data/tables/java_refactoring
    python experiments/java_refactoring/plot.py                      # java_refactoring.pdf in experiments/java_refactoring/out/

`analyze.py` needs the frozen artifacts (`python -m fewshot_dlms.fetch pools java`), next to each other:

    artifacts/java/pools/pools_*.jsonl                    main run: one record per (episode, regime), K = 16 candidates
    artifacts/java/judging/verdicts.jsonl                 the judge's verdicts (also packet_index.jsonl, calibration/)
    artifacts/java/length/<head>_s5final_G4.npz           predictions of the Java predictor (s5hc_C) and two comparators
    artifacts/java/length/prior_s5h_train.json            the training-label length prior
    artifacts/java/zero_demo/pools/pools_*.jsonl          the zero-demonstration ablation
    artifacts/java/zero_demo/judging/verdicts.jsonl       its verdicts

It writes every table above plus `metrics_by_episode.csv` in about 10 s on a CPU; `--compare` prints the largest
difference per column against the paper's tables and, per cell, whether the paper's value lies inside the 95 % CI
of the pools analysed. `plot.py` needs only `data/tables/java_refactoring/`.

GPU (LLaDA-1.5, about 16 GiB):

    python experiments/java_refactoring/run.py --regime oracle  --M 1,2 --J 1,2 --K 16 --out artifacts/reruns/java_refactoring/pools
    python experiments/java_refactoring/run.py --regime learned --head artifacts/heads/java_head_C.pt --out artifacts/reruns/java_refactoring/pools
    python experiments/java_refactoring/run.py --study zero_demo --M 0,1,2 --J 1,2 --out artifacts/reruns/java_refactoring/zero_demo/pools
    make rerun-java        # oracle regime, first 8 episodes, then analyze.py with the paper's verdicts and --compare

`--families N` keeps the first N episodes (anchors). `--head-probs artifacts/java/length/s5hc_C_s5final_G4.npz` draws
the learned lengths from the stored predictions instead of running the Java predictor. The two learned runs hash different
checkpoint identities, the stored predictions' `ckpt_sha` (the original checkpoint's SHA-256) against the published
file's SHA-256 (ARTIFACTS.md), so each refuses the other's pool: give them separate `--out` directories. Records are
appended to `<out>/pools_<regime>.jsonl` (or `pools_zero_demo.jsonl`) as episodes finish. Each carries its pool's
`config_hash` (model, revision, regime, K, chunk, head; not the cells selected); an interrupted run resumes after the
episodes with that hash and, before loading the model, refuses a file holding records of another configuration or the
frozen pools (exit status 2, naming the file). `<out>/manifest_<hash>.json` holds the configuration, the pool's hash,
every seed and the library and GPU versions. A failed family is counted and the run exits non-zero.

## Runtime

On one H100 NVL in the authors' run: the main run took 2.9 GPU-h (oracle 0.55 h, learned 2.33 h; a learned episode has
up to 16 canvases), the ablation 0.3 GPU-h. The judge's sub-agent calls for the main run: the package's own records
count 272 batched calls for the 2,089 report packets (329 in total, with 46 calibration controls, one retry of a
malformed reply and ten agreement repeats); a later audit of the session transcripts counts 420 batched calls of at most
five packets (477 in total); the verdicts themselves are unaffected. The ablation judged 605 fresh report packets in
calls of up to five packets, and 79 packets more (the 46 controls again, 6 repeats of them, 7 malformed replies, 20
agreement repeats); its records do not give the number of calls.

## Fidelity

- **Analysis.** On the frozen artifacts `analyze.py` reproduces all nine tables to at most 2.2e-16 (the bootstrap
  intervals included) and `checkstyle_case.json` (`make test-parity`). The parity tests also rebuild the 2,089
  report, 698 ablation and 46 calibration judge packets byte for byte, redraw all 2,032 learned allocations from
  the stored predictions of the Java predictor, and recompute the ablation's 207 units and seeds.
- **Figure.** Rendered with `pdftoppm -r 150` (and 300), `java_refactoring.pdf` is pixel-identical to Figure 8 of
  arXiv v1 (whose right panel, like this one, has no y label).
- **Reruns.** On an H100 NVL with torch 2.7.1+cu128, `run.py --K 16` reproduces the frozen records bit for bit
  (tokens, texts, logp1, logp0, every record field): oracle and learned regimes (the Java predictor evaluated live) on
  the first two episodes, and the ablation's arms D = 0, 1, 2 of the first two anchors. `make test-gpu` replays two stored
  pools (an oracle M2J2 pool and a learned M1J2 pool with three canvases) exactly and checks the Java predictor
  against the stored predictions. Other GPUs give other draws; judge such a rerun by `--compare`.

## Resampling check

GPU sampling is statistical in general, so a rerun is judged by whether the frozen mean lies inside the rerun's
95 % CI, not by equality. One full cell was rerun on a GPU:

    CUDA_VISIBLE_DEVICES=2 python experiments/java_refactoring/run.py --regime oracle --M 1 --J 1 --K 16 --out artifacts/reruns/java_refactoring/pools
    python experiments/java_refactoring/analyze.py --pools artifacts/reruns/java_refactoring/pools --out artifacts/reruns/java_refactoring/analysis

- **Cell.** LLaDA-1.5 (`GSAI-ML/LLaDA-1.5` at revision `84346fd91ba60252d260022201ad6fc5a3468fb2`), oracle lengths,
  (D, J) = (1, 1): all 52 episodes of the cell (50 commits), K = 16.
- **Platform and time.** One H100 NVL 96 GB, torch 2.7.1+cu128: 9.7 min wall-clock (581 s, of which 56 s load the
  model).
- **Without a judge.** U needs a verdict for every draw, and new texts need `judge.py` (an API key). Without one,
  `analyze.py` counts every draw as unjudged (U = 0), and the check is restricted to the pool statistics that need no
  verdict: the columns `n_unique_text`, `modal_share`, `R_range`, `ESS0.5` and `maxw0.5` of `metrics_by_episode.csv`,
  averaged over the 52 episodes. Unguided sampling (β = 0) has weight 1/16 on every draw (ESS 16) by construction.

  | pool statistic | rerun, mean | rerun, median | paper pools, mean |
  |---|---|---|---|
  | distinct texts per pool | 3.12 | 1 | 3.12 |
  | share of the most frequent text | 0.853 | 1 | 0.853 |
  | range of R = logp1 − logp0 in a pool | 1.80 | 0 | 1.80 |
  | ESS, unguided (β = 0) / guided (β = 0.5) | 16 / 14.08 | 16 / 16 | 16 / 14.08 |
  | largest weight, unguided / guided | 0.0625 / 0.120 | 0.0625 / 0.0625 | 0.0625 / 0.120 |

  The last column is `analyze.py` on the downloaded paper pools (`artifacts/java/pools`): the same columns of its
  `metrics_by_episode.csv`, averaged over the 52 oracle episodes of `M1J1`. No API key was used for this check.

- **With the paper's verdicts.** On this platform the rerun reproduced the frozen pools bit for bit (tokens, texts,
  logp1 and logp0 of all 52 × 16 draws, every episode's statistic above identical), so the shipped verdicts grade every
  draw through the packet key (0 of 832 draws unjudged), and the CI verdict can be taken:

      python experiments/java_refactoring/analyze.py --pools artifacts/reruns/java_refactoring/pools --verdicts artifacts/java/judging/verdicts.jsonl \
          --out artifacts/reruns/java_refactoring/analysis_paper_verdicts --compare data/tables/java_refactoring

  | rule | U, frozen | U, rerun [95 % CI] | verdict |
  |---|---|---|---|
  | unguided (`ordinary`, β = 0) | 0.761 | 0.761 [0.652, 0.860] | inside |
  | guidance (`tilting`, β = 0.5) | 0.729 | 0.729 [0.610, 0.841] | inside |

  The `M1J1` rows are the check; the `all` rows that `--compare` also prints set the frozen value over all 127
  episodes against this 52-episode rerun and do not apply.

- **Note.** On this stack the "inside" verdicts above hold trivially, because the rerun is bit-identical to the frozen
  pools; `tests/test_resampling_compare.py` checks that `--compare` also reports "outside" when a rerun's mean is far
  from the frozen one.
