"""Build a fine-tuning set that teaches Laya Hangman. TRAINING-TIME ONLY.

    python -m laya_hangman.training_data --out train.jsonl

What this is: a *teacher* turns each partial game state into soft labels (for every unguessed letter,
how many dictionary words that are still consistent with the visible state contain it). Laya is then
fine-tuned on (state, labels) pairs with `laya-train`, like any supervised decision task.

What this is not: at play time nothing changes. Laya still receives only `DecisionInput` (pattern,
guesses, allowance, unguessed letters) and answers in one forward pass. No dictionary, candidate list
or answer is ever part of an inference request; the app and the evaluation harness are untouched.
The teacher is disclosed as such in the README: a fine-tuned Laya has *learned* letter statistics from
this teacher, and should be compared with the baseline on that understanding.

Leakage rules enforced here (and tested):
  * no word of the evaluation split is a training secret, and none is in the teacher dictionary;
  * the teacher dictionary is ENABLE (public domain) plus this repo's word list, minus the eval words;
  * labels are leave-one-out: the secret word is removed from the candidates when its states are labelled,
    so a label never encodes the secret's own letters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .baseline import FREQUENCY_ORDER
from .decision import QUESTION_ID, DecisionInput, build_request
from .game import HangmanGame
from .wordlist import all_words, dev_words, eval_words

ENABLE_URL = "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"
MIN_LEN, MAX_LEN = 4, 12


def load_enable(path: Optional[str]) -> Tuple[List[str], str]:
    """The ENABLE word list (lower-case a-z words only) and the sha256 of the raw file."""
    if path and Path(path).exists():
        raw = Path(path).read_bytes()
    else:
        with urllib.request.urlopen(ENABLE_URL, timeout=60) as resp:  # noqa: S310 - fixed https URL
            raw = resp.read()
        if path:
            Path(path).write_bytes(raw)
    words = [w for w in raw.decode().split() if re.fullmatch(r"[a-z]+", w)]
    return words, hashlib.sha256(raw).hexdigest()


class Teacher:
    """Counts, for each letter, the consistent dictionary words that contain it."""

    def __init__(self, words: Iterable[str]) -> None:
        by_len: Dict[int, List[str]] = {}
        for w in words:
            by_len.setdefault(len(w), []).append(w.upper())
        self._blob = {n: "\n" + "\n".join(ws) + "\n" for n, ws in by_len.items()}
        self.size = sum(len(v) for v in by_len.values())

    def candidates(self, inp: DecisionInput, exclude: Optional[str] = None) -> List[str]:
        n = len(inp.pattern)
        if n not in self._blob:
            return []
        guessed = "".join(g.letter for g in inp.guesses)
        blank = f"[^{guessed}\\n]" if guessed else "[^\\n]"
        regex = "^" + "".join(c if c != "_" else blank for c in inp.pattern) + "$"
        found = re.findall(regex, self._blob[n], flags=re.MULTILINE)
        return [w for w in found if w != exclude] if exclude else found

    def letter_counts(self, inp: DecisionInput, exclude: Optional[str] = None) -> Counter:
        counts: Counter = Counter()
        for w in self.candidates(inp, exclude):
            for c in set(w):
                if c in inp.available:
                    counts[c] += 1
        return counts

    def gold(self, inp: DecisionInput, sharpen: float, exclude: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Soft labels over the unguessed letters. `exclude` is the secret itself (leave-one-out): the
        label then reflects what the *other* words say, as it would for a word the teacher has not seen."""
        counts = self.letter_counts(inp, exclude)
        if not counts:
            return None
        weights = {c: float(counts.get(c, 0)) ** sharpen for c in inp.available}
        total = sum(weights.values())
        return {c: w / total for c, w in weights.items()}


def _policy_pick(policy: str, inp: DecisionInput, teacher: Teacher, rng: random.Random) -> str:
    avail = list(inp.available)
    if policy == "random":
        return rng.choice(avail)
    if policy == "freq":  # frequency order with some noise
        return rng.choice(avail) if rng.random() < 0.3 else next(c for c in FREQUENCY_ORDER if c in avail)
    counts = teacher.letter_counts(inp)  # "teacher": greedy, with some noise
    if not counts or rng.random() < 0.2:
        return rng.choice(avail)
    return max(avail, key=lambda c: (counts.get(c, 0), -avail.index(c)))


def states_for_word(word: str, teacher: Teacher, k: int, rng: random.Random) -> List[DecisionInput]:
    """Play one game with a randomly chosen policy and return up to `k` of its in-progress states."""
    policy = rng.choice(["random", "freq", "teacher", "teacher"])
    game = HangmanGame(word)
    seen: List[DecisionInput] = []
    while game.status == "in_progress":
        inp = DecisionInput.from_game(game)
        seen.append(inp)
        game.guess(_policy_pick(policy, inp, teacher, rng))
    rng.shuffle(seen)
    return seen[:k]


def build_rows(secrets: Sequence[str], teacher: Teacher, states_per_word: int, sharpen: float,
               seed: int) -> List[dict]:
    rng = random.Random(seed)
    rows, seen_states = [], set()
    for word in secrets:
        for inp in states_for_word(word, teacher, states_per_word, rng):
            req = build_request(inp)
            if req.state in seen_states:
                continue
            gold = teacher.gold(inp, sharpen, exclude=word)
            if gold is None:
                continue
            seen_states.add(req.state)
            rows.append({"state": req.state, "questions": req.questions,
                         "gold": {QUESTION_ID: {"probabilities": gold}}})
    rng.shuffle(rows)
    return rows


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m laya_hangman.training_data")
    p.add_argument("--out", default="train.jsonl")
    p.add_argument("--enable", default="enable1.txt", help="path to enable1.txt (downloaded if missing)")
    p.add_argument("--enable-words", type=int, default=3000, help="extra random ENABLE words used as secrets")
    p.add_argument("--dev-states", type=int, default=10, help="states per dev-list word")
    p.add_argument("--enable-states", type=int, default=4, help="states per extra ENABLE word")
    p.add_argument("--sharpen", type=float, default=3.0, help="exponent applied to letter counts before normalising")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    held_out = {w.lower() for w in eval_words()}
    enable, enable_sha = load_enable(args.enable)
    enable = [w for w in enable if MIN_LEN <= len(w) <= MAX_LEN]
    dictionary = sorted(({w for w in enable} | {w.lower() for w in all_words()}) - held_out)
    teacher = Teacher(dictionary)
    rng = random.Random(args.seed)
    extra = rng.sample([w for w in dictionary if w.upper() not in set(dev_words())], args.enable_words)
    dev = [w for w in dev_words()]

    rows = build_rows([w.upper() for w in dev], teacher, args.dev_states, args.sharpen, args.seed)
    rows += build_rows([w.upper() for w in extra], teacher, args.enable_states, args.sharpen, args.seed + 1)
    rng.shuffle(rows)

    leaked = [w for w in held_out if w in {x.lower() for x in dev} | set(extra) or w in set(dictionary)]
    assert not leaked, f"evaluation words leaked into training: {leaked[:5]}"
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    manifest = {
        "rows": len(rows), "teacher_dictionary_words": teacher.size, "enable_sha256": enable_sha,
        "enable_source": ENABLE_URL, "eval_words_excluded": len(held_out), "sharpen": args.sharpen,
        "dev_words_as_secrets": len(dev), "extra_enable_secrets": len(extra), "seed": args.seed,
        "eval_words_sha256": hashlib.sha256("\n".join(eval_words()).encode()).hexdigest(),
    }
    Path(args.out + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

