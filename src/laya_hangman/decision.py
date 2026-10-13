"""The narrow object a decider is allowed to see, and the Laya request/response contract.

`DecisionInput` holds exactly the information a human player would have: the visible pattern, the
guesses so far with their outcomes, the remaining incorrect-guess allowance and the letters that
have not been guessed. It is built from the public properties of `HangmanGame` only.

Everything Laya receives is produced by `build_request`, which takes a `DecisionInput` and nothing
else, so the secret word cannot reach the model by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .game import HangmanGame

QUESTION_ID = "next_letter"

# Brief Hangman instructions shown to the model as the question text. Keep this short: the
# instruction competes with the 26 letter options for Laya's option/head token budget.
INSTRUCTIONS = (
    "Hangman: guess the hidden English word one letter at a time. "
    "Choose the unguessed letter most likely to appear in the word."
)

# Laya English checkpoint defaults (laya 0.4.6: max_len=512, head_max_len=192). They are passed
# explicitly on every call so the budget that was verified is the budget that is used.
DEFAULT_MAX_LEN = 512
DEFAULT_HEAD_MAX_LEN = 192


class InvalidModelResponse(RuntimeError):
    """Laya returned something that cannot be used as a legal guess."""


@dataclass(frozen=True)
class GuessOutcome:
    letter: str
    hit: bool


@dataclass(frozen=True)
class DecisionInput:
    pattern: Tuple[str, ...]
    guesses: Tuple[GuessOutcome, ...]
    incorrect_remaining: int
    max_incorrect: int
    available: Tuple[str, ...]

    @classmethod
    def from_game(cls, game: HangmanGame) -> "DecisionInput":
        return cls(
            pattern=game.pattern,
            guesses=tuple(GuessOutcome(g.letter, g.hit) for g in game.guesses),
            incorrect_remaining=game.remaining_incorrect,
            max_incorrect=game.max_incorrect,
            available=game.available_letters,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": "".join(self.pattern),
            "guesses": [{"letter": g.letter, "outcome": "hit" if g.hit else "miss"} for g in self.guesses],
            "incorrect_remaining": self.incorrect_remaining,
            "max_incorrect": self.max_incorrect,
            "available_letters": list(self.available),
        }


@dataclass(frozen=True)
class LayaRequest:
    """Exactly what is passed to `laya.Agent.predict(state, questions, ...)`."""

    state: str
    questions: Dict[str, Dict[str, Any]]
    max_len: int = DEFAULT_MAX_LEN
    head_max_len: int = DEFAULT_HEAD_MAX_LEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "questions": self.questions,
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
        }


def render_state(inp: DecisionInput) -> str:
    pattern = " ".join(inp.pattern)
    if inp.guesses:
        guesses = ", ".join(f"{g.letter} {'hit' if g.hit else 'miss'}" for g in inp.guesses)
    else:
        guesses = "none"
    return (
        f"Word pattern: {pattern} ({len(inp.pattern)} letters). "
        f"Incorrect guesses left: {inp.incorrect_remaining} of {inp.max_incorrect}. "
        f"Guesses so far: {guesses}."
    )


def build_request(inp: DecisionInput, max_len: int = DEFAULT_MAX_LEN,
                  head_max_len: int = DEFAULT_HEAD_MAX_LEN) -> LayaRequest:
    if not inp.available:
        raise ValueError("There are no unguessed letters to choose from.")
    question = {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        # A list of bare labels: each option is rendered as just the letter, which keeps all 26
        # options inside the option budget (see README, "Token budget").
        "criteria": list(inp.available),
    }
    return LayaRequest(state=render_state(inp), questions={QUESTION_ID: question},
                       max_len=max_len, head_max_len=head_max_len)


@dataclass(frozen=True)
class Decision:
    letter: str
    source: str  # "laya" or "baseline"
    latency_ms: float
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None  # Laya "confidence" field (normalised-entropy based)
    answer_confidence: Optional[float] = None  # Laya "answer_confidence" field (calibrated max-prob)
    request: Optional[Dict[str, Any]] = None  # the exact request, for the UI
    usage: Optional[Dict[str, Any]] = field(default=None)


def parse_laya_result(result: Any, inp: DecisionInput, request: LayaRequest, latency_ms: float) -> Decision:
    """Validate a raw `Agent.predict` result and turn it into a `Decision`.

    Anything unexpected raises `InvalidModelResponse`. There is deliberately no fallback to a
    random or heuristic letter: a bad response is an error the user sees.
    """
    bad = InvalidModelResponse
    if not isinstance(result, dict):
        raise bad(f"Laya returned {type(result).__name__}, expected a dict.")
    answers = result.get("answers")
    answer = answers.get(QUESTION_ID) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise bad(f"Laya's result has no answer for {QUESTION_ID!r}.")
    if answer.get("type") != "choice":
        raise bad(f"Expected a 'choice' answer, got {answer.get('type')!r}.")
    letter = answer.get("choice")
    if not isinstance(letter, str) or letter not in inp.available:
        raise bad(f"Laya chose {letter!r}, which is not one of the available letters.")
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(inp.available):
        raise bad("Laya's probabilities do not cover exactly the available letters.")
    clean: Dict[str, float] = {}
    for key, value in probs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value < 0 or value > 1:
            raise bad(f"Invalid probability for {key!r}: {value!r}.")
        clean[key] = float(value)
    # Laya rounds each probability to 4 decimals, so allow up to 0.5e-4 of rounding error per option.
    if abs(sum(clean.values()) - 1.0) > 0.5e-4 * len(clean) + 1e-6:
        raise bad(f"Probabilities sum to {sum(clean.values()):.4f}, not 1.")
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    if usage.get("truncated"):
        raise bad("Laya reported that the game state was truncated; raise max_len (see README).")
    if usage.get("options"):
        raise bad("Laya reported that letter options collapsed in the option budget; "
                  "raise head_max_len (see README).")

    def _num(name: str) -> Optional[float]:
        v = answer.get(name)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    return Decision(
        letter=letter, source="laya", latency_ms=latency_ms, probabilities=clean,
        confidence=_num("confidence"), answer_confidence=_num("answer_confidence"),
        request=request.to_dict(), usage=dict(usage),
    )
