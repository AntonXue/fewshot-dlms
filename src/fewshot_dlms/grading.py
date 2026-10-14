# ported from editdiff/experiments/algo_investigation/dx_common.py @ d276fa6
"""Answer extraction and the fixed graders that decide correctness for all four benchmarks.

A pool record carries one raw text per target. `answer` turns it into the model's answer string and `grade`
compares that string with the record's gold under the record's `grader`:

  int_list      List Functions: the answer must be a bracketed integer list; correct if its integers equal the gold's.
  int_row       1D-ARC: the answer must be whitespace-separated single digits; correct if the tokens equal the gold's.
  anything else exact string match (the pools name it `exact_string`): SST-2 letter codes, SALT words, Movie Director names.

Outer whitespace is trimmed, nothing else (exact match is case-sensitive). `grade` returns (correct, parse_ok):
parse_ok says the answer was well formed for the grader (non-empty for exact match), correct says it is also right.

Model-name convention: `answer` takes the model name stored in the record ("llada", "qwen", "dqwen", ...). The
fs_analyze_lib loader (Figures 4 and 5, Qwen per-hole pools) first maps every name starting with "qwen" to "qwen";
callers that follow it must do the same before calling `answer`.
"""
import re

LIST_RX = re.compile(r"^\[\s*(-?\d+(\s*,\s*-?\d+)*)?\s*\]$")


def answer(text, model):
    """Answer string from one raw candidate text.

    The chat end-of-turn token becomes a newline; Qwen (a free-length causal model) keeps only the first line; a single
    leading space (the separator after the prompt marker) is dropped.
    """
    t = text.replace("<|im_end|>", "\n")
    if model == "qwen":
        t = t.split("\n")[0]
    return t[1:] if t.startswith(" ") else t


def _ints(text):
    """All integers in a text, in order, as a list of ints."""
    return [int(x) for x in re.findall(r"-?\d+", text)]


def grade(grader, ans, gold):
    """(canonical_correct, parse_ok): the fixed graders of the runs. Outer whitespace is trimmed, nothing else."""
    s = ans.strip()
    if grader == "int_list":
        ok = bool(LIST_RX.match(s))
        correct = ok and _ints(s) == _ints(gold)
        return correct, ok
    if grader == "int_row":
        toks = s.split()
        ok = len(toks) > 0 and all(re.match(r"^\d$", t) for t in toks)
        correct = ok and toks == gold.split()
        return correct, ok
    correct = s == gold.strip() and s != ""
    return correct, s != ""
