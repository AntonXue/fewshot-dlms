PY ?= python
# Extra pytest arguments, e.g. make test-gpu PYTEST_ARGS="-k replay"
PYTEST_ARGS ?=
# The figures from the paper's tables in data/tables/ (no download needed).
figures: ; $(PY) experiments/single_target/plot.py && $(PY) experiments/multi_target/plot.py && $(PY) experiments/length_allocation/plot.py && $(PY) experiments/java_refactoring/plot.py
# The same figures from the download (python -m fewshot_dlms.fetch pools java): every analyze.py recomputes its tables
# from artifacts/ into $(REPRO)/tables/<fig>/ and prints its difference to data/tables/, then every plot.py draws from
# those tables into $(REPRO)/figures/. The cross-benchmark figure (multi_target's plot.py) reads the table that single_target's
# analyze.py --crossbench writes. The "inside the CI" verdicts that --compare prints here are trivially true: the pools
# analysed are the ones that produced data/tables/, so every frozen mean meets its own interval. They test a rerun's
# pools (make rerun-*); here the column differences are the check.
REPRO ?= artifacts/reproduce
reproduce:
	$(PY) experiments/single_target/analyze.py --pools artifacts/pools/single_target --crossbench --beta-sweep --out $(REPRO)/tables/single_target --compare data/tables/single_target data/tables/multi_target
	$(PY) experiments/multi_target/analyze.py --pools artifacts/pools/multi_target --out $(REPRO)/tables/multi_target --compare data/tables/multi_target
	cp $(REPRO)/tables/single_target/cells_crossbench.csv $(REPRO)/tables/multi_target/
	$(PY) experiments/length_allocation/analyze.py --pools artifacts/pools/length_allocation --out $(REPRO)/tables/length_allocation --compare data/tables/length_allocation
	$(PY) experiments/java_refactoring/analyze.py --pools artifacts/java/pools --out $(REPRO)/tables/java_refactoring --compare data/tables/java_refactoring
	$(PY) experiments/single_target/plot.py --tables $(REPRO)/tables/single_target --out $(REPRO)/figures
	$(PY) experiments/multi_target/plot.py --tables $(REPRO)/tables/multi_target --out $(REPRO)/figures
	$(PY) experiments/length_allocation/plot.py --tables $(REPRO)/tables/length_allocation --out $(REPRO)/figures
	$(PY) experiments/java_refactoring/plot.py --tables $(REPRO)/tables/java_refactoring --out $(REPRO)/figures
	@echo REPRODUCE_DONE: tables in $(REPRO)/tables, figures in $(REPRO)/figures
test: ; $(PY) -m pytest -q tests -m "not gpu and not parity and not api" $(PYTEST_ARGS)
test-parity: ; $(PY) -m pytest -q tests -m parity $(PYTEST_ARGS)
test-gpu: ; $(PY) -m pytest -q tests -m gpu $(PYTEST_ARGS)
# Calls the Anthropic API (judge.py on two calibration controls); spends credits; needs ANTHROPIC_API_KEY and the
# fetched java group (python -m fewshot_dlms.fetch java).
test-api: ; $(PY) -m pytest -q tests -m api $(PYTEST_ARGS)
rerun-single-target: ; $(PY) experiments/single_target/run.py --model llada --tasks salt_rep_fam --M 4 --families 8 --score-gold --out artifacts/reruns/single_target && $(PY) experiments/single_target/analyze.py --pools artifacts/reruns/single_target --out artifacts/reruns/single_target/analysis --compare data/tables/single_target
rerun-multi-target: ; $(PY) experiments/multi_target/run.py --model llada --M 4 --J 2 --families 8 --out artifacts/reruns/multi_target && $(PY) experiments/multi_target/analyze.py --pools artifacts/reruns/multi_target --out artifacts/reruns/multi_target/analysis --compare data/tables/multi_target
# rerun-length-allocation's learned step needs the LLaDA-1.5 length head: python -m fewshot_dlms.fetch heads
rerun-length-allocation: ; $(PY) experiments/length_allocation/run.py --model llada --mode oracle --J 1 --K 64 --families 8 --out artifacts/reruns/length_allocation && $(PY) experiments/length_allocation/run.py --model llada --mode learned --J 1 --families 8 --out artifacts/reruns/length_allocation && $(PY) experiments/length_allocation/analyze.py --pools artifacts/reruns/length_allocation --out artifacts/reruns/length_allocation/analysis --compare data/tables/length_allocation
rerun-java: ; $(PY) experiments/java_refactoring/run.py --regime oracle --families 8 --K 16 --out artifacts/reruns/java_refactoring/pools && $(PY) experiments/java_refactoring/analyze.py --pools artifacts/reruns/java_refactoring/pools --verdicts artifacts/java/judging/verdicts.jsonl --out artifacts/reruns/java_refactoring/analysis --compare data/tables/java_refactoring
.PHONY: figures reproduce test test-parity test-gpu test-api rerun-single-target rerun-multi-target rerun-length-allocation rerun-java
