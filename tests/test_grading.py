"""Unit tests for grading: answer extraction and the fixed graders of the runs.

Every case below is a candidate text copied from a frozen pool record, together with that record's `gold`
and `grader` fields. Cases marked "derived" are a one-character variation of a real record that exercises a
normalisation rule the real data happens not to hit.
"""
import pytest

from fewshot_dlms.grading import answer
from fewshot_dlms.grading import grade

# (id, grader, raw candidate text, gold, model, expected (correct, parse_ok))
CASES = [
    # int_list: List Functions (list_easy_fam), LLaDA
    ("int_list-correct", "int_list", " [6, 4]", "[6, 4]", "llada", (True, True)),
    ("int_list-wrong-value", "int_list", " [0, 7]", "[6, 4]", "llada", (False, True)),
    ("int_list-correct-long", "int_list", " [3, 1, 4, 0, 2, 7, 6]", "[3, 1, 4, 0, 2, 7, 6]", "llada", (True, True)),
    ("int_list-wrong-element", "int_list", " [88, 1, 51, 4]", "[88, 1, 91, 4]", "llada", (False, True)),
    ("int_list-empty-answer", "int_list", " ", "[]", "llada", (False, False)),
    ("int_list-trailing-junk", "int_list", " [6, 26, 8, 46, 4, 94, 18, 3]\n\n```", "[6, 26, 8, 46, 4, 3, 94, 18, 3]", "llada", (False, False)),
    ("int_list-run-on", "int_list", " [61]\n\nInput: [1, 13, 15]\nOutput: [13, 15, 1, 1]", "[61, 61, 61, 61, 61, 61, 61, 61, 61]", "llada", (False, False)),
    ("int_list-spacing-is-ignored (derived)", "int_list", " [6,4]", "[6, 4]", "llada", (True, True)),
    ("int_list-empty-list (derived)", "int_list", " []", "[]", "llada", (True, True)),
    ("int_list-qwen", "int_list", " [6, 4]<|im_end|>", "[6, 4]", "qwen", (True, True)),
    # int_row: 1D-ARC rows, LLaDA
    ("int_row-correct", "int_row", " 0 0 0 0 0 0 0 0 0 0 9 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0", "0 0 0 0 0 0 0 0 0 0 9 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0", "llada", (True, True)),
    ("int_row-wrong-row", "int_row", " 0 5 0 0 7 7 7 7 0", "0 0 7 7 7 7 5 0 0", "llada", (False, True)),
    ("int_row-run-on", "int_row", " 7 7 7 7 7 0 0 7 0 0 0 0 0 7 7 7 7 7 0 0 7 0 0 0 0 0 0 7 7 7 7 7 0 0 7 0 0 0 0 0\nOutput: ", "0 0 7 7 7 7 7 0 0 0 0 0 0 0 0 0 7 7 7 7 7 0 0 0 0 0 0 0 0 0 7 7 7 7 7 0 0 0 0 0 0 0", "llada", (False, False)),
    ("int_row-empty-answer (derived)", "int_row", " ", "0 0 7", "llada", (False, False)),
    ("int_row-multi-digit-token (derived)", "int_row", " 0 10 7", "0 10 7", "llada", (False, False)),
    # exact_string: SST-2 letter codes (sst2_fam), LLaDA and Qwen
    ("sst2-correct", "exact_string", " H", "H", "llada", (True, True)),
    ("sst2-wrong-code", "exact_string", " D", "H", "llada", (False, True)),
    ("sst2-qwen-correct", "exact_string", " J<|im_end|>", "J", "qwen", (True, True)),
    ("sst2-qwen-wrong-code", "exact_string", " L<|im_end|>", "G", "qwen", (False, True)),
    # exact_string: SALT repetition (salt_rep_fam), LLaDA and Qwen
    ("salt-correct", "exact_string", " egn egn xjd sbc", "egn egn xjd sbc", "llada", (True, True)),
    ("salt-swapped-words", "exact_string", " egn egn sbc xjd", "egn egn xjd sbc", "llada", (False, True)),
    ("salt-wrong-word", "exact_string", " egn egn vch sbc", "egn egn xjd sbc", "llada", (False, True)),
    ("salt-qwen-trailing-newlines", "exact_string", " egn egn xjd sbc\n\n", "egn egn xjd sbc", "qwen", (True, True)),
    # exact_string: Movie Director names (movie_dir_fam), LLaDA
    ("movie-correct", "exact_string", " james cameron", "james cameron", "llada", (True, True)),
    ("movie-case-sensitive", "exact_string", " James Cameron\n", "james cameron", "llada", (False, True)),
    ("movie-different-name", "exact_string", " lost cloud", "disney", "llada", (False, True)),
    ("movie-misspelt-reference", "exact_string", " quentin tarantino", "quentin tarentino", "llada", (False, True)),
    ("movie-misspelt-reference-matched", "exact_string", " quentin tarentino", "quentin tarentino", "llada", (True, True)),
    ("movie-qwen-correct", "exact_string", " david cronenberg<|im_end|>", "david cronenberg", "qwen", (True, True)),
    # exact_string: an empty answer never counts, even against an empty gold (derived)
    ("exact-empty-answer (derived)", "exact_string", " ", "x", "llada", (False, False)),
    ("exact-empty-gold (derived)", "exact_string", "", "", "llada", (False, False)),
]


@pytest.mark.parametrize("grader,text,gold,model,expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_grade_on_frozen_record_texts(grader, text, gold, model, expected):
    assert grade(grader, answer(text, model), gold) == expected


def test_grade_returns_plain_bools():
    for _, grader, text, gold, model, _ in CASES:
        correct, parse_ok = grade(grader, answer(text, model), gold)
        assert type(correct) is bool
        assert type(parse_ok) is bool


def test_answer_drops_exactly_one_leading_space():
    assert answer(" H", "llada") == "H"
    assert answer("  H", "llada") == " H"
    assert answer("H", "llada") == "H"


def test_answer_turns_end_of_turn_into_a_newline():
    assert answer(" J<|im_end|>", "llada") == "J\n"
    assert answer(" J<|im_end|>", "dqwen") == "J\n"


def test_answer_keeps_only_the_first_line_for_qwen():
    run_on = " [61]\n\nInput: [1, 13, 15]\nOutput: [13, 15, 1, 1]"
    assert answer(run_on, "qwen") == "[61]"
    assert answer(run_on, "llada") == run_on[1:]
    assert answer(" J<|im_end|>", "qwen") == "J"


def test_first_line_rule_changes_the_verdict_on_run_on_text():
    run_on = " [61]\n\nInput: [1, 13, 15]\nOutput: [13, 15, 1, 1]"
    gold = "[61]"
    assert grade("int_list", answer(run_on, "qwen"), gold) == (True, True)
    assert grade("int_list", answer(run_on, "llada"), gold) == (False, False)


def test_outer_whitespace_is_trimmed_on_both_sides_of_exact_match():
    assert grade("exact_string", "  james cameron \n", " james cameron\n") == (True, True)
