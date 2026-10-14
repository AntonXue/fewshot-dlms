PY ?= python
# Extra pytest arguments, e.g. make test-gpu PYTEST_ARGS="-k replay"
PYTEST_ARGS ?=
figures: ; $(PY) experiments/fig4_single_target/plot.py && $(PY) experiments/fig5_multi_target/plot.py && $(PY) experiments/fig6_lengths/plot.py && $(PY) experiments/fig7_java/plot.py
test: ; $(PY) -m pytest -q tests -m "not gpu and not parity" $(PYTEST_ARGS)
test-parity: ; $(PY) -m pytest -q tests -m parity $(PYTEST_ARGS)
test-gpu: ; $(PY) -m pytest -q tests -m gpu $(PYTEST_ARGS)
rerun-fig5: ; $(PY) experiments/fig5_multi_target/run.py --model llada --M 4 --J 2 --families 8 --out artifacts/reruns/fig5 && $(PY) experiments/fig5_multi_target/analyze.py --pools artifacts/reruns/fig5 --out artifacts/reruns/fig5/analysis --compare data/tables/fig5
rerun-fig6: ; $(PY) experiments/fig6_lengths/run.py --model llada --mode oracle --J 1 --K 64 --families 8 --out artifacts/reruns/fig6 && $(PY) experiments/fig6_lengths/run.py --model llada --mode learned --J 1 --families 8 --out artifacts/reruns/fig6 && $(PY) experiments/fig6_lengths/analyze.py --pools artifacts/reruns/fig6 --out artifacts/reruns/fig6/analysis --compare data/tables/fig6
.PHONY: figures test test-parity test-gpu rerun-fig5 rerun-fig6
