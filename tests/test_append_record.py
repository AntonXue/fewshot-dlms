"""append_record of every pipeline that appends JSON lines (the four run.py and fig7's judge.py) after a killed write.

A run killed in the middle of a write can leave a pool file whose last line is cut off, without its newline. The next
record must start on a line of its own: appended to the cut-off line it would make both unreadable. The resume reader
of each pipeline skips the cut-off line, so that family is sampled again, and keeps every complete record. CPU only.
"""
import importlib.util
import json

import pytest

# script -> (key that identifies a unit, hash field of the records)
SCRIPTS = {
    "fig4_single_target/run.py": ("episode_id", "protocol_hash"),
    "fig5_multi_target/run.py": ("episode_id", "protocol_hash"),
    "fig6_lengths/run.py": ("episode_id", "protocol_hash"),
    "fig7_java/run.py": ("episode_id", "config_hash"),
    "fig7_java/judge.py": ("key", None),
}
DIGEST = "0123456789abcdef"


def load_script(root, script):
    spec = importlib.util.spec_from_file_location("append_" + script.replace("/", "_")[:-3], root / "experiments" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def done(module, script, path):
    """The units that the script's resume reader finds complete in `path`."""
    if script == "fig7_java/judge.py":
        return module.judged_keys(path)
    if script == "fig7_java/run.py":
        return module.done_units(path, "episode_id", DIGEST)
    return module.done_episodes(path, DIGEST)


def record(script, name):
    key, hash_field = SCRIPTS[script]
    entry = {key: name}
    if hash_field:
        entry[hash_field] = DIGEST
    return entry


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_append_after_a_cut_off_line_starts_a_new_line(root, tmp_path, script):
    module = load_script(root, script)
    path = tmp_path / "pool.jsonl"
    good = [json.dumps(record(script, name)) for name in ("a", "b")]
    cut = json.dumps(record(script, "c"))[:9]
    path.write_text("\n".join(good) + "\n" + cut, encoding="utf-8")
    assert done(module, script, path) == {"a", "b"}

    module.append_record(path, record(script, "d"))
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines == good + [cut, json.dumps(record(script, "d")), ""]
    assert done(module, script, path) == {"a", "b", "d"}


@pytest.mark.parametrize("script", list(SCRIPTS))
def test_append_to_a_new_or_complete_file_adds_no_blank_line(root, tmp_path, script):
    module = load_script(root, script)
    path = tmp_path / "pool.jsonl"
    module.append_record(path, record(script, "a"))
    module.append_record(path, record(script, "b"))
    expected = "".join(json.dumps(record(script, name)) + "\n" for name in ("a", "b"))
    assert path.read_text(encoding="utf-8") == expected
