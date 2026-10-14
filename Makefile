PY ?= python
# Extra pytest arguments, e.g. make test-gpu PYTEST_ARGS="-k replay"
PYTEST_ARGS ?=
figures: ; $(PY) experiments/fig4_single_target/plot.py && $(PY) experiments/fig5_multi_target/plot.py && $(PY) experiments/fig6_lengths/plot.py && $(PY) experiments/fig7_java/plot.py
test: ; $(PY) -m pytest -q tests -m "not gpu and not parity and not api" $(PYTEST_ARGS)
test-parity: ; $(PY) -m pytest -q tests -m parity $(PYTEST_ARGS)
test-gpu: ; $(PY) -m pytest -q tests -m gpu $(PYTEST_ARGS)
# Calls the Anthropic API (judge.py on two calibration controls); spends credits, needs ANTHROPIC_API_KEY.
test-api: ; $(PY) -m pytest -q tests -m api $(PYTEST_ARGS)
rerun-fig5: ; $(PY) experiments/fig5_multi_target/run.py --model llada --M 4 --J 2 --families 8 --out artifacts/reruns/fig5 && $(PY) experiments/fig5_multi_target/analyze.py --pools artifacts/reruns/fig5 --out artifacts/reruns/fig5/analysis --compare data/tables/fig5
rerun-fig7: ; $(PY) experiments/fig7_java/run.py --regime oracle --families 8 --K 16 --out artifacts/reruns/fig7/pools && $(PY) experiments/fig7_java/analyze.py --pools artifacts/reruns/fig7/pools --verdicts artifacts/java/judging/verdicts.jsonl --out artifacts/reruns/fig7/analysis --compare data/tables/fig7
.PHONY: figures test test-parity test-gpu test-api rerun-fig5 rerun-fig7
