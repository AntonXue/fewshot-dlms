PY ?= python
figures: ; $(PY) experiments/fig4_single_target/plot.py && $(PY) experiments/fig5_multi_target/plot.py && $(PY) experiments/fig6_lengths/plot.py && $(PY) experiments/fig7_java/plot.py
test: ; $(PY) -m pytest -q tests -m "not gpu and not parity"
test-parity: ; $(PY) -m pytest -q tests -m parity
test-gpu: ; $(PY) -m pytest -q tests -m gpu
rerun-fig5: ; $(PY) experiments/fig5_multi_target/run.py --model llada --M 4 --J 2 --families 8 --out artifacts/reruns/fig5
.PHONY: figures test test-parity test-gpu rerun-fig5
