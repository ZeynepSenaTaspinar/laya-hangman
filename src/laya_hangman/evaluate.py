"""Reproducible evaluation: the same games, rules and fixed word set for Laya and the baseline.

    python -m laya_hangman.evaluate --split eval --deciders baseline laya --out results/eval.json

Deciders only ever receive a `DecisionInput` (visible pattern, guesses, allowance, unguessed letters),
exactly like the app. The evaluation answers are never part of any request. If Laya cannot load or
returns an unusable response the run aborts with an error; no result is invented or substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .baseline import FREQUENCY_ORDER, FrequencyBaseline
from .decision import DecisionInput
from .game import IN_PROGRESS, MAX_INCORRECT, WON, HangmanGame
from .sessions import Decider
from .wordlist import EVAL_MODULUS, all_words, dev_words, eval_words


@dataclass
class GameResult:
    word: str
    won: bool
    incorrect: int
    total: int
    latencies_ms: List[float]
    guesses: str


def play_game(word: str, decider: Decider) -> GameResult:
    game = HangmanGame(word, MAX_INCORRECT)
    latencies: List[float] = []
    while game.status == IN_PROGRESS:
        decision = decider.decide(DecisionInput.from_game(game))
        latencies.append(decision.latency_ms)
        game.guess(decision.letter)
    return GameResult(word=word, won=game.status == WON, incorrect=game.incorrect_count,
                      total=len(game.guesses), latencies_ms=latencies, guesses="".join(game.guessed_letters))


def summarise(results: Sequence[GameResult], with_latency: bool) -> Dict[str, Any]:
    n = len(results)
    out: Dict[str, Any] = {
        "games": n,
        "wins": sum(r.won for r in results),
        "win_rate": sum(r.won for r in results) / n if n else None,
        "mean_incorrect_guesses": statistics.fmean(r.incorrect for r in results) if n else None,
        "mean_total_guesses": statistics.fmean(r.total for r in results) if n else None,
    }
    if with_latency:
        lat = sorted(x for r in results for x in r.latencies_ms)
        out.update(
            inferences=len(lat),
            mean_inference_ms=statistics.fmean(lat) if lat else None,
            median_inference_ms=statistics.median(lat) if lat else None,
            p95_inference_ms=lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None)
    return out


def _versions() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                            "laya_hangman": __version__}
    for name in ("torch", "transformers", "laya", "huggingface_hub", "numpy"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            info[name] = f"not importable ({type(exc).__name__})"
    return info


def _words_for(split: str) -> List[str]:
    return {"eval": eval_words, "dev": dev_words, "all": lambda: list(all_words())}[split]()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    words = _words_for(args.split)
    if args.limit:
        words = words[: args.limit]
    report: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rules": {"max_incorrect": MAX_INCORRECT, "guess_type": "single letters A-Z, no repeats"},
        "word_set": {
            "split": args.split, "count": len(words),
            "selection": f"sha256(word) mod {EVAL_MODULUS} == 0 -> eval; otherwise dev. Fixed by the word list.",
            "sha256_of_word_sequence": hashlib.sha256("\n".join(words).encode()).hexdigest(),
            "limited_to_first": args.limit or None,
        },
        "environment": _versions(),
        "protocol_note": ("Prompt, token budgets and decoding were fixed before any evaluation and were not "
                          "tuned on the evaluation split."),
        "deciders": {},
    }
    for name in args.deciders:
        if name == "baseline":
            decider: Decider = FrequencyBaseline()
            config: Dict[str, Any] = {"letter_order": FREQUENCY_ORDER}
        else:
            from .model import LayaDecider, ModelManager
            manager = ModelManager()
            manager.start()
            if manager.wait() != "ready":
                raise SystemExit(f"Laya did not load; no Laya results were produced.\n{manager.error}")
            decider = LayaDecider(manager)
            config = {k: v for k, v in manager.status().items()
                      if k in ("model_id", "requested_revision", "resolved_revision", "device", "device_note",
                               "hardware", "load_seconds", "max_len", "head_max_len", "budget")}
        results: List[GameResult] = []
        t0 = time.perf_counter()
        for i, word in enumerate(words, 1):
            results.append(play_game(word, decider))
            if args.progress and i % 20 == 0:
                print(f"  {name}: {i}/{len(words)} games", file=sys.stderr, flush=True)
        report["deciders"][name] = {
            "config": config, "summary": summarise(results, with_latency=(name == "laya")),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "games": [r.__dict__ for r in results] if args.per_game else "omitted (use --per-game)",
        }
    return report


def render_markdown(report: Dict[str, Any]) -> str:
    ws = report["word_set"]
    lines = [f"Word set: `{ws['split']}`, {ws['count']} words (sequence sha256 `{ws['sha256_of_word_sequence'][:16]}…`). "
             f"Rules: {report['rules']['max_incorrect']} incorrect guesses allowed.", "",
             "| Decider | Games | Win rate | Mean incorrect | Mean total guesses | Mean inference (ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, data in report["deciders"].items():
        s = data["summary"]
        lat = f"{s['mean_inference_ms']:.1f}" if s.get("mean_inference_ms") is not None else "n/a"
        label = "Laya" if name == "laya" else "Frequency baseline"
        lines.append(f"| {label} | {s['games']} | {s['win_rate'] * 100:.1f}% ({s['wins']}/{s['games']}) | "
                     f"{s['mean_incorrect_guesses']:.2f} | {s['mean_total_guesses']:.2f} | {lat} |")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m laya_hangman.evaluate")
    p.add_argument("--split", choices=["eval", "dev", "all"], default="eval")
    p.add_argument("--deciders", nargs="+", choices=["baseline", "laya"], default=["baseline", "laya"])
    p.add_argument("--limit", type=int, default=0, help="only the first N words of the split (smoke runs)")
    p.add_argument("--out", help="write the full JSON report here")
    p.add_argument("--per-game", action="store_true", help="include every game in the JSON report")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)
    report = run(args)
    print(render_markdown(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
