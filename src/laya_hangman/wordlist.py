"""The bundled word list and the fixed evaluation split.

`data/words.txt` is an original, hand-curated list of common English words (see README, "Word list").
The split is a pure function of each word's SHA-256, so it never changes unless the list does:
words whose hash is divisible by 5 form the evaluation set; everything else is the development pool.
Tune prompts or settings on the pool, never on the evaluation set.
"""
from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from importlib import resources
from typing import List, Optional

EVAL_MODULUS = 5


@lru_cache(maxsize=1)
def all_words() -> tuple:
    text = resources.files("laya_hangman").joinpath("data/words.txt").read_text(encoding="utf-8")
    words = sorted({w.strip().upper() for w in text.splitlines() if w.strip()})
    return tuple(words)


def is_eval_word(word: str) -> bool:
    return int(hashlib.sha256(word.lower().encode()).hexdigest(), 16) % EVAL_MODULUS == 0


def eval_words() -> List[str]:
    return [w for w in all_words() if is_eval_word(w)]


def dev_words() -> List[str]:
    return [w for w in all_words() if not is_eval_word(w)]


def random_word(rng: Optional[random.Random] = None) -> str:
    """A random word from the whole list (the app is not an evaluation)."""
    return (rng or random.SystemRandom()).choice(all_words())
