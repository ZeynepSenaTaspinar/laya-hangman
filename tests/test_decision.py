import dataclasses
import inspect
import json
import math
import random

import pytest

from conftest import laya_style_result
from laya_hangman.decision import (DecisionInput, InvalidModelResponse, build_request, parse_laya_result,
                                   GuessOutcome)
from laya_hangman.game import ALPHABET, HangmanGame


def request_for(game):
    return build_request(DecisionInput.from_game(game))


def test_request_contains_all_26_letters_as_options_on_the_first_turn():
    inp = DecisionInput.from_game(HangmanGame("zebra"))
    req = build_request(inp)
    assert req.questions["next_letter"]["criteria"] == list(ALPHABET)
    assert req.questions["next_letter"]["type"] == "choice"


def test_options_are_exactly_the_unguessed_letters():
    g = HangmanGame("zebra")
    g.guess("E"); g.guess("Q")
    options = build_request(DecisionInput.from_game(g)).questions["next_letter"]["criteria"]
    assert "E" not in options and "Q" not in options and len(options) == 24


def test_request_never_contains_the_secret_word_or_its_hidden_letters_in_the_state():
    secret = "PLUMBER"
    g = HangmanGame(secret)
    g.guess("E"); g.guess("X")
    req = request_for(g)
    blob = json.dumps(req.to_dict())
    assert secret not in blob.upper()
    # the visible pattern shows only revealed letters: the hidden P, L, U, M, B, R never appear in it
    pattern = req.state.split("Word pattern: ")[1].split(" (")[0]
    assert pattern == "_ _ _ _ _ E _"


def test_requests_are_identical_for_secrets_that_look_the_same_so_far():
    """Non-interference: whatever differs between two secrets must not influence the request."""
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        length = rng.randint(3, 8)
        a = "".join(rng.choice("ETAOINSHR") for _ in range(length))
        b = "".join(rng.choice("ETAOINSHR") for _ in range(length))
        guesses = rng.sample("ETAOIN", rng.randint(1, 4))
        ga, gb = HangmanGame(a), HangmanGame(b)
        for c in guesses:
            if ga.status != "in_progress" or gb.status != "in_progress":
                break
            ga.guess(c); gb.guess(c)
        if ga.pattern == gb.pattern and ga.incorrect_count == gb.incorrect_count and \
                [r.hit for r in ga.guesses] == [r.hit for r in gb.guesses] and a != b:
            assert request_for(ga).to_dict() == request_for(gb).to_dict()
            checked += 1
    assert checked > 5, "the property test did not exercise enough pairs"


def test_decision_input_has_only_the_whitelisted_fields():
    fields = {f.name for f in dataclasses.fields(DecisionInput)}
    assert fields == {"pattern", "guesses", "incorrect_remaining", "max_incorrect", "available"}
    assert {f.name for f in dataclasses.fields(GuessOutcome)} == {"letter", "hit"}


def test_build_request_takes_only_the_decision_input():
    params = list(inspect.signature(build_request).parameters)
    assert params == ["inp", "max_len", "head_max_len"]


def test_instructions_are_english_and_brief():
    req = request_for(HangmanGame("cat"))
    ins = req.questions["next_letter"]["instructions"]
    assert "Hangman" in ins and len(ins.split()) < 30


# ---- response validation -------------------------------------------------------------------

def inp_after(*letters, secret="BANANA"):
    g = HangmanGame(secret)
    for c in letters:
        g.guess(c)
    return DecisionInput.from_game(g)


def parse(result, inp=None):
    inp = inp or inp_after("A")
    return parse_laya_result(result, inp, build_request(inp), 12.5)


def test_a_valid_laya_result_is_parsed():
    inp = inp_after("A")
    d = parse(laya_style_result(inp, choice="N"), inp)
    assert d.letter == "N" and d.source == "laya" and d.latency_ms == 12.5
    assert set(d.probabilities) == set(inp.available) and d.confidence == 0.1 and d.answer_confidence == 0.2


@pytest.mark.parametrize("mutate", [
    lambda r: None,
    lambda r: [],
    lambda r: {},
    lambda r: {"answers": {}},
    lambda r: {"answers": {"next_letter": "N"}},
    lambda r: r["answers"]["next_letter"].update(type="score") or r,
    lambda r: r["answers"]["next_letter"].update(choice="A") or r,          # already guessed
    lambda r: r["answers"]["next_letter"].update(choice="n") or r,          # not an offered label
    lambda r: r["answers"]["next_letter"].update(choice=None) or r,
    lambda r: r["answers"]["next_letter"].update(choice=7) or r,
    lambda r: r["answers"]["next_letter"].update(probabilities=None) or r,
    lambda r: r["answers"]["next_letter"]["probabilities"].pop("B") or r,    # missing letter
    lambda r: r["answers"]["next_letter"]["probabilities"].update(A=0.5) or r,  # guessed letter present
    lambda r: r["answers"]["next_letter"]["probabilities"].update(B=-0.1) or r,
    lambda r: r["answers"]["next_letter"]["probabilities"].update(B=float("nan")) or r,
    lambda r: r["answers"]["next_letter"]["probabilities"].update(B=float("inf")) or r,
    lambda r: r["answers"]["next_letter"]["probabilities"].update(B="0.5") or r,
    lambda r: r["answers"]["next_letter"]["probabilities"].update(B=True) or r,
    lambda r: r["answers"]["next_letter"]["probabilities"].update(B=0.9) or r,  # no longer sums to 1
    lambda r: r["usage"].update(truncated=True) or r,
    lambda r: r["usage"].update(options={"next_letter": {"total": 26, "distinct": 20}}) or r,
])
def test_invalid_laya_results_raise_instead_of_producing_a_guess(mutate):
    inp = inp_after("A")
    result = mutate(laya_style_result(inp, choice="N"))
    with pytest.raises(InvalidModelResponse):
        parse(result, inp)


def test_laya_rounding_of_probabilities_is_tolerated():
    inp = DecisionInput.from_game(HangmanGame("zebra"))
    probs = {c: 0.0385 for c in inp.available}  # 26 * 0.0385 = 1.001 (typical 4-decimal rounding noise)
    probs["E"] = 0.0375
    d = parse(laya_style_result(inp, choice="E", probs=probs), inp)
    assert d.letter == "E"
