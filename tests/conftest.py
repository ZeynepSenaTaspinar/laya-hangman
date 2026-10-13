import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # ordinary tests never touch the network
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laya_hangman.decision import Decision, DecisionInput  # noqa: E402


class ScriptedDecider:
    """Mock inference: plays a fixed sequence of letters, skipping any already guessed."""

    name = "mock"

    def __init__(self, order="ETAOINSHRDLCUMWFGYPBVKJXQZ", source="laya"):
        self.order, self.source, self.calls = order, source, []

    def decide(self, inp: DecisionInput) -> Decision:
        self.calls.append(inp)
        letter = next(c for c in self.order if c in inp.available)
        return Decision(letter=letter, source=self.source, latency_ms=1.5,
                        probabilities={c: 1 / len(inp.available) for c in inp.available},
                        confidence=0.5, answer_confidence=0.5, request={"mock": True})


class GatedDecider(ScriptedDecider):
    """Blocks inside `decide` until released, to hold a turn in flight."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.entered, self.release = threading.Event(), threading.Event()

    def decide(self, inp):
        self.entered.set()
        assert self.release.wait(5), "test never released the decider"
        return super().decide(inp)


def laya_style_result(inp, choice=None, probs=None, usage=None):
    """A dict shaped like `laya.Agent.predict` output (verified against laya 0.4.6 source)."""
    letters = list(inp.available)
    p = probs or {c: round(1 / len(letters), 4) for c in letters}
    if probs is None:  # make the rounded values sum to ~1 by giving the remainder to the first letter
        p[letters[0]] = round(p[letters[0]] + 1 - sum(p.values()), 4)
    best = choice or letters[0]
    return {"model": "laya-rl-agent",
            "answers": {"next_letter": {"type": "choice", "choice": best, "probabilities": p,
                                        "confidence": 0.1, "answer_confidence": 0.2,
                                        "action": {"act_probability": 1.0}}},
            "usage": usage if usage is not None else {"input_tokens": 120, "output_tokens": 0,
                                                       "state_tokens": 60, "state_tokens_dropped": 0,
                                                       "truncated": False, "truncated_questions": []}}


@pytest.fixture
def scripted():
    return ScriptedDecider()
