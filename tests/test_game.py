import pytest

from laya_hangman.game import (GameOver, HangmanGame, InvalidGuess, InvalidWord, LOST, WON, IN_PROGRESS,
                               normalize_word)


def test_repeated_letters_are_all_revealed():
    g = HangmanGame("banana")
    rec = g.guess("a")
    assert rec.hit and rec.occurrences == 3
    assert "".join(g.pattern) == "_A_A_A"
    g.guess("N")
    assert "".join(g.pattern) == "_ANANA"
    assert g.incorrect_count == 0


def test_win_is_detected_immediately():
    g = HangmanGame("cat")
    for c in "CA":
        g.guess(c)
    assert g.status == IN_PROGRESS
    g.guess("T")
    assert g.status == WON
    assert g.reveal() == "CAT"
    with pytest.raises(GameOver):
        g.guess("Z")


def test_loss_after_exactly_six_incorrect_guesses():
    g = HangmanGame("cat")
    for i, c in enumerate("BDEFGH", 1):
        assert g.status == IN_PROGRESS
        g.guess(c)
        assert g.incorrect_count == i
    assert g.status == LOST and g.remaining_incorrect == 0
    assert g.reveal() == "CAT"
    with pytest.raises(GameOver):
        g.guess("C")


def test_five_misses_is_not_a_loss():
    g = HangmanGame("cat")
    for c in "BDEFG":
        g.guess(c)
    assert g.status == IN_PROGRESS and g.remaining_incorrect == 1


def test_a_hit_does_not_cost_a_life():
    g = HangmanGame("moon")
    g.guess("O")
    assert g.remaining_incorrect == 6


@pytest.mark.parametrize("bad", ["", " ", "ab", "1", "é", "!", None, 5, "AB"])
def test_invalid_or_malformed_guesses_are_rejected_without_changing_state(bad):
    g = HangmanGame("cat")
    g.guess("C")
    before = (g.pattern, g.guessed_letters, g.incorrect_count)
    with pytest.raises(InvalidGuess):
        g.guess(bad)
    assert (g.pattern, g.guessed_letters, g.incorrect_count) == before


def test_repeated_guess_is_rejected_and_does_not_cost_a_life():
    g = HangmanGame("cat")
    g.guess("Z")
    with pytest.raises(InvalidGuess, match="already"):
        g.guess("z")
    assert g.incorrect_count == 1 and g.guessed_letters == ("Z",)


@pytest.mark.parametrize("raw", ["", "   ", "two words", "abc1", "café", "naïve", "a-b", "x", "a" * 31, None, 3])
def test_invalid_words_are_rejected_with_a_message(raw):
    with pytest.raises(InvalidWord) as exc:
        normalize_word(raw)
    assert str(exc.value)


def test_words_are_normalised_to_upper_case():
    assert normalize_word("  PlUm ") == "PLUM"


def test_secret_is_hidden_until_the_game_ends():
    g = HangmanGame("secret")
    with pytest.raises(GameOver):
        g.reveal()
    assert g.available_letters and len(g.available_letters) == 26


def test_available_letters_exclude_guessed_and_are_empty_when_over():
    g = HangmanGame("ab")
    g.guess("A")
    assert "A" not in g.available_letters and len(g.available_letters) == 25
    g.guess("B")
    assert g.available_letters == ()
