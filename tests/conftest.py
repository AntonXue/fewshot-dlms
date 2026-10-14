import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def artifacts():
    if not ARTIFACTS.exists():
        pytest.skip("artifacts not fetched: run `python -m fewshot_dlms.fetch pools java`")
    return ARTIFACTS
