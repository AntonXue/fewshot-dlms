"""fewshot_dlms.leakcheck: the one definition of the generic private-text patterns (machine paths, machine metadata).

The test strings are assembled from pieces, so that this file holds no machine path itself.
"""
import re

from fewshot_dlms.leakcheck import PRIVATE_PATTERN
from fewshot_dlms.leakcheck import find_private


def test_find_private_names_every_generic_pattern():
    paths = ["/" + "hdd" + "1/run", "/" + "SSD" + "2/x", "/" + "home/someone/x", "/" + "mnt/data"]
    keys = ['"' + key + '": "x"' for key in ("host", "gpu_env", "hostname", "cwd")]
    for text in paths + keys:
        assert find_private(text), text
    text = " ".join(paths + keys)
    assert find_private(text) == sorted(set(find_private(text)))
    assert len(find_private(text.lower())) == 8


def test_find_private_passes_clean_text():
    for text in ("artifacts/pools/fig5/llada", "/ssd", "home/x", "hostname", '"host_id": 3', "/hdd/1", ""):
        assert find_private(text) == [], text


def test_binary_data_is_checked_through_latin1():
    data = b"PK\x03\x04\xff\x00" + ("/" + "mnt/x").encode() + b"\x80\x81"
    assert find_private(data.decode("latin-1")) == ["/" + "mnt/"]
    assert PRIVATE_PATTERN.flags & re.IGNORECASE
