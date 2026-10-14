import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def artifacts():
    """artifacts/ with the fetched pools; skips the test otherwise. The check is on artifacts/pools, not artifacts/:
    `make rerun-*` and `make reproduce` write into artifacts/ too. Tests of the Java group check artifacts/java on top
    (the `java` fixtures)."""
    if not (ARTIFACTS / "pools").is_dir():
        pytest.skip("artifacts not fetched: run `python -m fewshot_dlms.fetch pools java`")
    return ARTIFACTS


@pytest.fixture(scope="session")
def artifact_root():
    """artifacts/ without a check, for tests that need other groups than the pools (java, heads) and check the files
    they read themselves (the GPU tests)."""
    return ARTIFACTS
