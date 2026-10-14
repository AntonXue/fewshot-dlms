# Figure 7: Java helper refactoring (Section 5, Appendix C)

A developer extracted a helper method and rewrote some call sites to use it. LLaDA-1.5 sees the helper and M
accepted rewrites (the demonstrations) and must rewrite J other sites of the same file (the targets). The 127
episodes, their provenance and licenses are in `data/java/`.

| Paper | Content | Output | Table in `data/tables/fig7/` |
|---|---|---|---|
| Figure 7, left | length mass of head C within ±2 tokens of the reference length, vs the training-label prior | `plot.py` → `fig7.pdf` | `length_fidelity_by_bin.csv` |
| Figure 7, right; Table 1 (Appendix C) | joint judged success per cell, oracle and learned lengths, β = 0 and 0.5 | `plot.py` → `fig7.pdf` | `summary_by_cell.csv`, `paired_intervals.csv` |
| Appendix C | strata (development exposure, reference visible, literal copy, test code, …) | | `strata.csv` |
| Appendix C | guidance strength β ∈ {0, 0.1, 0.25, 0.5, 1} | | `beta_sensitivity.csv` |
| Appendix C | length fidelity over all 80 sites, head C and two comparator heads | | `length_fidelity.csv` |
| Appendix C | zero-demonstration ablation (M = 0 vs M = 1) | | `zero_demo_main_table.csv` |
| Appendix C | pool classes: estimates on pools with both outcomes | | `pool_taxonomy_estimates.csv` |
| Section 5 | the Checkstyle pool where guidance moves weight to a wrong draw | | `checkstyle_case.json` |

`metrics_by_commit.csv` holds the commit means behind every interval.

## Protocol

- **Episodes.** The 127 episodes of population B (`data/java/episodes.jsonl`, 50 commits, 14 repositories) in
  four cells (M, J) ∈ {1, 2} × {1, 2}: 52, 28, 28 and 19 episodes. The canvas is `canvas.build_java_canvas`
  with the instruction `JAVA_INSTRUCTION`; `E` are the demonstrations' accepted edits.
- **Lengths.** *Oracle*: every target gets the token count of its reference replacement. *Learned*: each of the
  K candidates draws its own lengths from length head C (`artifacts/heads/java_head_C.pt`, read by
  `lengths.LengthHead`, on the episode's compact prompt). Candidate k, target j takes the inverse CDF of the
  head's distribution at u = SHA-256(`s5-final-alloc-v1|0e1d3c9040b09598|<episode>|<J>|<k>|<j>|<r>`)[:16] / 2^64,
  r = 0; a draw whose canvas would exceed 2,048 tokens is redrawn with r + 1 (no draw of the paper needed it).
- **Sampling.** K = 16 infills per episode and regime from P1 by fixed-order decoding (ascending positions,
  temperature 1), one reveal per distinct allocation. Seed: the first 12 hex digits of
  SHA-256(`0e1d3c9040b09598|s5final|<regime>|<episode>|<allocation tuple>`) mod 2^31 (`run.generation_seed`).
  The support is the ordinary support plus LLaDA-1.5's 115 logit rows beyond its tokenizer, which the Java
  runs' adapter did not exclude (`run.java_support`; without them the log-likelihoods differ by about 1e-5).
- **Scoring.** The distinct infills of a canvas are teacher-forced under P1 (`logp1`) and under P0, the canvas
  with `E` masked (`logp0`), in the same batches as sampling; R = logp1 − logp0.
- **Judge.** Every distinct candidate text is judged once, blind (`judge.py`, rubric r1 in `rubric_r1.md`): a
  draw counts 1 when every target is judged correct; "incorrect" and "uncertain" count 0 (`U_commit_upper_uncertain`
  counts "uncertain" as 1).
- **Estimand.** U_β = Σ_k softmax(βR)_k y_k per (episode, regime); a cell's value is the mean over commits of the
  mean of the commit's episodes. β = 0 is ordinary sampling, β = 0.5 example guidance (`rule` = `tilting` in the tables).
- **Intervals.** Commit bootstrap, 10,000 resamples, `random.Random(0)`, one `randrange` per commit in order of first
  appearance, percentiles at sorted indices int(0.025 n) and int(0.975 n); paired contrasts resample the
  per-commit differences (`analyze.commit_bootstrap`). Length fidelity resamples commits with all their sites
  (`analyze.site_bootstrap`); the zero-demonstration ablation sorts the commits and takes the upper index
  int(0.975 n) − 1, as its own analysis did (`analyze.zero_demo_bootstrap`).
- **Zero-demonstration ablation.** Each M = 1 episode (80 anchors) is sampled without its demonstration (M = 0),
  with it (M = 1) and, for the 47 anchors whose targets also form an M = 2 episode, with both demonstrations of
  that episode (M = 2), under the neutral instruction `JAVA_ZERO_DEMO_INSTRUCTION`, oracle lengths, K = 16,
  β = 0. Seed: SHA-256(`c2a07e4a701f0a41|demo_ablation|<anchor>|<target sites joined by +>|<M>`)[:12] mod 2^31.
  Every arm was judged with the anchor's demonstration in the packet.

## The judge

The paper's verdicts were produced by **Claude Sonnet 5.5 running as Claude Code sub-agents** (five packets of
different episodes per call; 2,089 packets for the main run, 698 for the ablation). Before the run, 46 calibration
controls built from the source of 12 episodes (references, no-ops, partial edits, swapped arguments, flipped
conditions, wrong receivers, harmless comments) were judged one per call: 46/46 matched their expected verdicts.
The original verdicts ship with the artifacts (`java/judging/verdicts.jsonl`, `java/zero_demo/judging/verdicts.jsonl`)
and `analyze.py` reads them.

`judge.py` re-implements the instrument with the Anthropic API: the same packets (byte for byte; checked against
all 2,089 + 698 + 46 frozen packets in `make test-parity`), rubric r1 verbatim as the system prompt, one packet per
call, model `claude-sonnet-5-5`, the verdict JSON parsed as in the paper's run. **Re-judging is approximate**: an API
call is not a Claude Code sub-agent, and an LLM's verdicts vary between calls. It needs `pip install -e '.[judge]'` and
`ANTHROPIC_API_KEY`, and it costs API credits (a packet with the rubric is several thousand input tokens).

    python experiments/fig7_java/judge.py --calibrate --limit 5          # first 5 calibration controls vs expected and the paper's judge
    python experiments/fig7_java/judge.py --pools artifacts/reruns/fig7/pools   # every new packet of a rerun

A packet's key is the first 20 hex digits of SHA-256(`<rubric sha16>|<episode>|<demo sites>|<texts as sorted JSON>`),
so a rerun that draws texts the paper already judged is graded with the paper's verdicts (`analyze.py --verdicts`).

## Commands

    python experiments/fig7_java/analyze.py --pools artifacts/java/pools --compare data/tables/fig7
    python experiments/fig7_java/plot.py                      # fig7.pdf in experiments/fig7_java/out/

`analyze.py` needs the frozen artifacts (`python -m fewshot_dlms.fetch pools java`), next to each other:

    artifacts/java/pools/pools_*.jsonl                    main run: one record per (episode, regime), K = 16 candidates
    artifacts/java/judging/verdicts.jsonl                 the judge's verdicts (also packet_index.jsonl, calibration/)
    artifacts/java/length/<head>_s5final_G4.npz           predictions of head C (s5hc_C) and two comparator heads
    artifacts/java/length/prior_s5h_train.json            the training-label length prior
    artifacts/java/zero_demo/pools/pools_*.jsonl          the zero-demonstration ablation
    artifacts/java/zero_demo/judging/verdicts.jsonl       its verdicts

It writes every table above plus `metrics_by_episode.csv` in about 10 s on a CPU; `--compare` prints the largest
difference per column against the paper's tables and, per cell, whether the paper's value lies inside the 95 % CI
of the pools analysed. `plot.py` needs only `data/tables/fig7/`.

GPU (LLaDA-1.5, about 16 GiB):

    python experiments/fig7_java/run.py --regime oracle  --M 1,2 --J 1,2 --K 16 --out artifacts/reruns/fig7/pools
    python experiments/fig7_java/run.py --regime learned --head artifacts/heads/java_head_C.pt --out artifacts/reruns/fig7/pools
    python experiments/fig7_java/run.py --study zero_demo --M 0,1,2 --J 1,2 --out artifacts/reruns/fig7/zero_demo/pools
    make rerun-fig7        # oracle regime, first 8 episodes, then analyze.py with the paper's verdicts and --compare

`--families N` keeps the first N episodes (anchors). `--head-probs artifacts/java/length/s5hc_C_s5final_G4.npz`
draws the learned lengths from the stored predictions instead of running head C. Records are appended to
`<out>/pools_<regime>.jsonl` (or `pools_zero_demo.jsonl`) as episodes finish, so an interrupted run resumes;
`<out>/manifest_<hash>.json` holds the configuration, every seed and the library and GPU versions.

## Runtime

On one H100 NVL in the authors' run: the main run took 2.9 GPU-h (oracle 0.55 h, learned 2.33 h; a learned
episode has up to 16 canvases), the ablation 0.3 GPU-h, the judge 329 + 684 sub-agent calls.

## Fidelity

- **Analysis.** On the frozen artifacts `analyze.py` reproduces all nine tables to at most 2.2e-16 (the bootstrap
  intervals included) and `checkstyle_case.json` (`make test-parity`). The parity tests also rebuild the 2,089
  report, 698 ablation and 46 calibration judge packets byte for byte, redraw all 2,032 learned allocations from
  the stored head-C predictions, and recompute the ablation's 207 units and seeds.
- **Figure.** Rendered with `pdftoppm -r 150` (and 300), `fig7.pdf` is pixel-identical to the paper's figure.
- **Reruns.** On an H100 NVL with torch 2.7.1+cu128, `run.py --K 16` reproduces the frozen records bit for bit
  (tokens, texts, logp1, logp0, every record field): oracle and learned regimes (head C evaluated live) on the first
  two episodes, and the ablation's arms M = 0, 1, 2 of the first two anchors. `make test-gpu` replays two stored
  pools (an oracle M2J2 pool and a learned M1J2 pool with three canvases) exactly and checks head C against the
  stored predictions. Other GPUs give other draws; judge such a rerun by `--compare`.
