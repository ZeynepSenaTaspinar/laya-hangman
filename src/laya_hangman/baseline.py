"""Deterministic frequency baseline. It is NOT Laya and is always labelled as the baseline.

The order is the classic "ETAOIN SHRDLU"-style ranking of letters by frequency in English text,
fixed here as a plain constant. It was not derived from, or adjusted to, the word lists or the
evaluation set in this repository.
"""
from __future__ import annotations

import time

from .decision import Decision, DecisionInput

FREQUENCY_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
assert sorted(FREQUENCY_ORDER) == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class FrequencyBaseline:
    name = "baseline"

    def decide(self, inp: DecisionInput) -> Decision:
        started = time.perf_counter()
        for letter in FREQUENCY_ORDER:
            if letter in inp.available:
                return Decision(letter=letter, source="baseline",
                                latency_ms=(time.perf_counter() - started) * 1000.0)
        raise ValueError("There are no unguessed letters to choose from.")
