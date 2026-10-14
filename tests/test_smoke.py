"""The package imports, and its version has one source: pyproject.toml and fewshot_dlms.__version__ agree, and so does
the installed distribution's metadata when it was installed from this checkout."""
import importlib.metadata
import json
import pathlib
import tomllib
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest

import fewshot_dlms

ROOT = pathlib.Path(__file__).resolve().parents[1]


def installed_from_this_checkout(distribution):
    """Whether the installed distribution is the package imported here: an editable install of this checkout, or a
    regular install whose package directory is the one imported."""
    text = distribution.read_text("direct_url.json")
    if text and json.loads(text).get("dir_info", {}).get("editable"):
        project = pathlib.Path(url2pathname(urlparse(json.loads(text)["url"]).path)).resolve()
        return project == ROOT
    installed = pathlib.Path(distribution.locate_file("fewshot_dlms")).resolve()
    return installed == pathlib.Path(fewshot_dlms.__file__).resolve().parent


def test_pyproject_version_is_the_package_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == fewshot_dlms.__version__


def test_installed_version_is_the_package_version():
    try:
        distribution = importlib.metadata.distribution("fewshot-dlms")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("fewshot-dlms is not installed (pip install -e .)")
    if not installed_from_this_checkout(distribution):
        pytest.skip("the installed fewshot-dlms comes from another checkout or copy; pip install -e . here to check it")
    assert importlib.metadata.version("fewshot-dlms") == fewshot_dlms.__version__
