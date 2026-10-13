"""Hangman rules. This module owns the secret word; nothing else may read it.

The only way information about a game leaves this module is through the public properties
(`pattern`, `guesses`, `available_letters`, ...) and `reveal()`, which refuses to answer until the
game is over. Deciders never receive a `HangmanGame`; they receive a `DecisionInput`
(see `decision.py`) that is built from the public properties alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_INCORRECT = 6
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 30

IN_PROGRESS = "in_progress"
WON = "won"
LOST = "lost"


class InvalidWord(ValueError):
    """The secret word is not acceptable."""


class InvalidGuess(ValueError):
    """The guess is not a single, not-yet-guessed English letter."""


class GameOver(InvalidGuess):
    """A guess was attempted after the game had ended."""


def normalize_word(raw: object) -> str:
    """Return the upper-case secret word, or raise `InvalidWord` with a message for the user."""
    if not isinstance(raw, str):
        raise InvalidWord("The secret word must be text.")
    word = raw.strip()
    if not word:
        raise InvalidWord("Enter a secret word.")
    bad = sorted({c for c in word if not ("A" <= c.upper() <= "Z" and c.isascii())})
    if bad:
        shown = " ".join(repr(c) for c in bad[:5])
        raise InvalidWord(
            f"Only the English letters A-Z are allowed (no spaces, digits, accents or punctuation). "
            f"Not allowed: {shown}."
        )
    if not (MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH):
        raise InvalidWord(f"The word must be {MIN_WORD_LENGTH}-{MAX_WORD_LENGTH} letters long.")
    return word.upper()


def normalize_letter(raw: object) -> str:
    if not isinstance(raw, str) or len(raw.strip()) != 1 or not re.fullmatch(r"[A-Za-z]", raw.strip()):
        raise InvalidGuess("A guess must be a single English letter A-Z.")
    return raw.strip().upper()


@dataclass(frozen=True)
class GuessRecord:
    letter: str
    hit: bool
    occurrences: int  # how many positions the letter revealed (0 for a miss)


class HangmanGame:
    def __init__(self, secret: str, max_incorrect: int = MAX_INCORRECT) -> None:
        self._secret = normalize_word(secret)
        if max_incorrect < 1:
            raise ValueError("max_incorrect must be at least 1")
        self.max_incorrect = max_incorrect
        self._guesses: list[GuessRecord] = []

    # ---- public, secret-free view -------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self._secret)

    @property
    def guesses(self) -> Tuple[GuessRecord, ...]:
        return tuple(self._guesses)

    @property
    def guessed_letters(self) -> Tuple[str, ...]:
        return tuple(g.letter for g in self._guesses)

    @property
    def incorrect_count(self) -> int:
        return sum(1 for g in self._guesses if not g.hit)

    @property
    def remaining_incorrect(self) -> int:
        return self.max_incorrect - self.incorrect_count

    @property
    def pattern(self) -> Tuple[str, ...]:
        """One entry per position: the letter if revealed, otherwise '_'."""
        hit = {g.letter for g in self._guesses if g.hit}
        return tuple(c if c in hit else "_" for c in self._secret)

    @property
    def status(self) -> str:
        if "_" not in self.pattern:
            return WON
        if self.incorrect_count >= self.max_incorrect:
            return LOST
        return IN_PROGRESS

    @property
    def available_letters(self) -> Tuple[str, ...]:
        """Letters not yet guessed. Empty once the game has ended."""
        if self.status != IN_PROGRESS:
            return ()
        taken = set(self.guessed_letters)
        return tuple(c for c in ALPHABET if c not in taken)

    def reveal(self) -> str:
        """The secret word, available only after the game has ended."""
        if self.status == IN_PROGRESS:
            raise GameOver("The secret word is hidden until the game ends.")
        return self._secret

    # ---- moves ----------------------------------------------------------------------------
    def guess(self, raw_letter: object) -> GuessRecord:
        if self.status != IN_PROGRESS:
            raise GameOver("The game is already over.")
        letter = normalize_letter(raw_letter)
        if letter in self.guessed_letters:
            raise InvalidGuess(f"The letter {letter} was already guessed.")
        count = self._secret.count(letter)
        record = GuessRecord(letter=letter, hit=count > 0, occurrences=count)
        self._guesses.append(record)
        return record
