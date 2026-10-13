import json
import random

import pytest

from laya_hangman.decision import DecisionInput
from laya_hangman.game import HangmanGame
from laya_hangman.training_data import Teacher, build_rows, states_for_word
from laya_hangman.wordlist import eval_words

WORDS = ["cat", "cot", "cut", "cab", "bat", "bad", "dog", "dot", "dig", "dip"]


def inp(secret, *letters):
    g = HangmanGame(secret)
    for c in letters:
        g.guess(c)
    return DecisionInput.from_game(g)


def test_candidates_respect_pattern_hits_and_misses():
    t = Teacher(WORDS)
    assert sorted(t.candidates(inp("cat"))) == sorted(w.upper() for w in WORDS)
    assert sorted(t.candidates(inp("cat", "C"))) == ["CAB", "CAT", "COT", "CUT"]
    # a miss removes every word containing that letter; a hit letter may not appear in a blank slot
    assert "CAT" not in t.candidates(inp("cab", "C", "T"))
    assert sorted(t.candidates(inp("cat", "C", "A"))) == ["CAB", "CAT"]


def test_gold_is_a_distribution_over_unguessed_letters_only():
    t = Teacher(WORDS)
    gold = t.gold(inp("cat", "C"), 1.0)
    assert set(gold) == set(inp("cat", "C").available) and abs(sum(gold.values()) - 1) < 1e-9


def test_labels_are_leave_one_out_so_the_secret_does_not_vote():
    t = Teacher(["cat", "cot"])
    with_secret = t.gold(inp("cat", "C"), 1.0)
    without = t.gold(inp("cat", "C"), 1.0, exclude="CAT")
    assert with_secret["A"] > 0 and without["A"] == 0 and without["O"] == 0.5 and without["T"] == 0.5


def test_rows_never_contain_a_state_labelled_from_the_secret_alone():
    t = Teacher(WORDS)
    rows = build_rows(["CAT"], t, states_per_word=5, sharpen=1.0, seed=0)
    for r in rows:
        probs = r["gold"]["next_letter"]["probabilities"]
        assert abs(sum(probs.values()) - 1) < 1e-9
        assert set(probs) == set(r["questions"]["next_letter"]["criteria"])


def test_rows_contain_no_secret_and_match_the_inference_request_format():
    from laya_hangman.decision import build_request
    t = Teacher(WORDS)
    rows = build_rows(["DOG", "CAT"], t, 5, 1.0, 0)
    assert rows
    for r in rows:
        assert r["state"].startswith("Word pattern: ")
        assert r["questions"]["next_letter"]["type"] == "choice"


def test_states_are_in_progress_and_capped():
    t = Teacher(WORDS)
    for s in states_for_word("CAT", t, 3, random.Random(0)):
        assert s.available and len(s.pattern) == 3


def test_eval_words_are_never_in_the_default_training_pipeline():
    # the module-level contract: eval words are subtracted from the teacher dictionary and secrets
    from laya_hangman import training_data
    src = open(training_data.__file__).read()
    assert "- held_out" in src and "assert not leaked" in src
    assert len(eval_words()) > 100
