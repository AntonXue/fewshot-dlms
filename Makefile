PY ?= python
# Extra pytest arguments, e.g. make test-gpu PYTEST_ARGS="-k replay"
PYTEST_ARGS ?=
figures: ; $(PY) experiments/fig4_single_target/plot.py && $(PY) experiments/fig5_multi_target/plot.py && $(PY) experiments/fig6_lengths/plot.py && $(PY) experiments/fig7_java/plot.py
test: ; $(PY) -m pytest -q tests -m "not gpu and not parity" $(PYTEST_ARGS)
test-parity: ; $(PY) -m pytest -q tests -m parity $(PYTEST_ARGS)
test-gpu: ; $(PY) -m pytest -q tests -m gpu $(PYTEST_ARGS)
rerun-fig4: ; $(PY) experiments/fig4_single_target/run.py --model llada --tasks salt_rep_fam --M 4 --families 8 --score-gold --out artifacts/reruns/fig4 && $(PY) experiments/fig4_single_target/analyze.py --pools artifacts/reruns/fig4 --out artifacts/reruns/fig4/analysis --compare data/tables/fig4
rerun-fig5: ; $(PY) experiments/fig5_multi_target/run.py --model llada --M 4 --J 2 --families 8 --out artifacts/reruns/fig5 && $(PY) experiments/fig5_multi_target/analyze.py --pools artifacts/reruns/fig5 --out artifacts/reruns/fig5/analysis --compare data/tables/fig5
.PHONY: figures test test-parity test-gpu rerun-fig4 rerun-fig5
