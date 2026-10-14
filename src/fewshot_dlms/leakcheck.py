"""The generic private-text check behind AGENTS.md's rule: no machine path or machine metadata in a committed or shipped
file.

PRIVATE_PATTERN matches absolute paths of the authors' kind of machine (/hdd<n>, /ssd<n>, /home/, /mnt/) and the
record keys that name a machine or a working directory ("host":, "gpu_env":, "hostname":, "cwd":), in any case. It
holds no user or host name. Each caller adds the patterns of its own site (the vocabulary of internal notes, the
research repository's name) next to it. The callers: tests/test_fixture_leaks.py (data/fixtures),
tests/test_java_data.py (data/java), tests/test_artifacts_manifest.py (the manifest files) and
tools/assemble_artifacts.py (every file of the Hugging Face dataset).
"""
import re

PRIVATE_PATTERNS = (
    r"/(?:hdd|ssd)\d",
    r"/home/",
    r"/mnt/",
    r'"host":',
    r'"gpu_env":',
    r'"hostname":',
    r'"cwd":',
)
PRIVATE_PATTERN = re.compile("|".join(PRIVATE_PATTERNS), re.IGNORECASE)


def find_private(text: str) -> list[str]:
    """The distinct strings of `text` that PRIVATE_PATTERN matches, sorted; empty if the text is clean.

    Binary data can be checked as data.decode("latin-1"), which maps every byte to one character.
    """
    return sorted({match.group() for match in PRIVATE_PATTERN.finditer(text)})
