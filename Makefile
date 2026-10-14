PY ?= python
# Extra pytest arguments, e.g. make test-gpu PYTEST_ARGS="-k replay"
PYTEST_ARGS ?=
# The figures from the paper's tables in data/tables/ (no download needed).
figures: ; $(PY) experiments/fig4_single_target/plot.py && $(PY) experiments/fig5_multi_target/plot.py && $(PY) experiments/fig6_lengths/plot.py && $(PY) experiments/fig7_java/plot.py
# The same figures from the download (python -m fewshot_dlms.fetch pools java): every analyze.py recomputes its tables
# from artifacts/ into $(REPRO)/tables/<fig>/ and prints its difference to data/tables/, then every plot.py draws from
# those tables into $(REPRO)/figures/. The cross-benchmark figure (fig5's plot.py) reads the table that fig4's
# analyze.py --crossbench writes.
REPRO ?= artifacts/reproduce
reproduce:
	$(PY) experiments/fig4_single_target/analyze.py --pools artifacts/pools/fig4 --crossbench --beta-sweep --out $(REPRO)/tables/fig4 --compare data/tables/fig4 data/tables/fig5
	$(PY) experiments/fig5_multi_target/analyze.py --pools artifacts/pools/fig5 --out $(REPRO)/tables/fig5 --compare data/tables/fig5
	cp $(REPRO)/tables/fig4/cells_crossbench.csv $(REPRO)/tables/fig5/
	$(PY) experiments/fig6_lengths/analyze.py --pools artifacts/pools/fig6 --out $(REPRO)/tables/fig6 --compare data/tables/fig6
	$(PY) experiments/fig7_java/analyze.py --pools artifacts/java/pools --out $(REPRO)/tables/fig7 --compare data/tables/fig7
	$(PY) experiments/fig4_single_target/plot.py --tables $(REPRO)/tables/fig4 --out $(REPRO)/figures
	$(PY) experiments/fig5_multi_target/plot.py --tables $(REPRO)/tables/fig5 --out $(REPRO)/figures
	$(PY) experiments/fig6_lengths/plot.py --tables $(REPRO)/tables/fig6 --out $(REPRO)/figures
	$(PY) experiments/fig7_java/plot.py --tables $(REPRO)/tables/fig7 --out $(REPRO)/figures
test: ; $(PY) -m pytest -q tests -m "not gpu and not parity and not api" $(PYTEST_ARGS)
test-parity: ; $(PY) -m pytest -q tests -m parity $(PYTEST_ARGS)
test-gpu: ; $(PY) -m pytest -q tests -m gpu $(PYTEST_ARGS)
# Calls the Anthropic API (judge.py on two calibration controls); spends credits; needs ANTHROPIC_API_KEY and the
# fetched java group (python -m fewshot_dlms.fetch java).
test-api: ; $(PY) -m pytest -q tests -m api $(PYTEST_ARGS)
rerun-fig4: ; $(PY) experiments/fig4_single_target/run.py --model llada --tasks salt_rep_fam --M 4 --families 8 --score-gold --out artifacts/reruns/fig4 && $(PY) experiments/fig4_single_target/analyze.py --pools artifacts/reruns/fig4 --out artifacts/reruns/fig4/analysis --compare data/tables/fig4
rerun-fig5: ; $(PY) experiments/fig5_multi_target/run.py --model llada --M 4 --J 2 --families 8 --out artifacts/reruns/fig5 && $(PY) experiments/fig5_multi_target/analyze.py --pools artifacts/reruns/fig5 --out artifacts/reruns/fig5/analysis --compare data/tables/fig5
# rerun-fig6's learned step needs the LLaDA-1.5 length head: python -m fewshot_dlms.fetch heads
rerun-fig6: ; $(PY) experiments/fig6_lengths/run.py --model llada --mode oracle --J 1 --K 64 --families 8 --out artifacts/reruns/fig6 && $(PY) experiments/fig6_lengths/run.py --model llada --mode learned --J 1 --families 8 --out artifacts/reruns/fig6 && $(PY) experiments/fig6_lengths/analyze.py --pools artifacts/reruns/fig6 --out artifacts/reruns/fig6/analysis --compare data/tables/fig6
rerun-fig7: ; $(PY) experiments/fig7_java/run.py --regime oracle --families 8 --K 16 --out artifacts/reruns/fig7/pools && $(PY) experiments/fig7_java/analyze.py --pools artifacts/reruns/fig7/pools --verdicts artifacts/java/judging/verdicts.jsonl --out artifacts/reruns/fig7/analysis --compare data/tables/fig7
.PHONY: figures reproduce test test-parity test-gpu test-api rerun-fig4 rerun-fig5 rerun-fig6 rerun-fig7
