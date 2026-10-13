"""REAL-MODEL smoke test. Skipped unless LAYA_REAL_MODEL=1.

    LAYA_REAL_MODEL=1 python -m pytest -m real_model -v -s

It downloads the Laya checkpoint (first run), loads it exactly as the app does, runs the startup
budget check (real predictions for all 26 options) and plays one complete game with real inference.
It asserts only structural properties (legal letters, valid probabilities, no truncation): it does
not assert that Laya plays well.
"""
import os

import pytest

pytestmark = [pytest.mark.real_model,
              pytest.mark.skipif(os.environ.get("LAYA_REAL_MODEL") != "1",
                                 reason="set LAYA_REAL_MODEL=1 to run against the real Laya checkpoint")]

from laya_hangman.decision import DecisionInput  # noqa: E402
from laya_hangman.game import HangmanGame  # noqa: E402
from laya_hangman.model import ModelManager, READY  # noqa: E402


@pytest.fixture(scope="module")
def manager():
    os.environ.pop("HF_HUB_OFFLINE", None)   # the real test may download
    m = ModelManager()
    m.start()
    state = m.wait(3600)
    assert state == READY, f"Laya did not load: {m.error}"
    return m


def test_budget_check_passed_on_the_real_tokenizer(manager):
    cases = manager.status()["budget"]["cases"]
    assert cases[0]["head"]["options"] == 26 and cases[0]["head"]["options_distinct"] == 26
    assert not any(c["truncated"] for c in cases)


def test_one_full_game_with_real_inference(manager):
    game = HangmanGame("elephant")
    latencies = []
    while game.status == "in_progress":
        d = manager.decide(DecisionInput.from_game(game))
        assert d.source == "laya" and d.letter in game.available_letters
        assert abs(sum(d.probabilities.values()) - 1) < 0.01 and d.latency_ms > 0
        latencies.append(d.latency_ms)
        game.guess(d.letter)
    assert game.status in ("won", "lost")
    print(f"\ndevice={manager.device} guesses={len(game.guesses)} status={game.status} "
          f"mean_ms={sum(latencies) / len(latencies):.1f}")
