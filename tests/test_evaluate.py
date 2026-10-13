import hashlib
import json
from pathlib import Path

import pytest

from conftest import ScriptedDecider
from laya_hangman import evaluate
from laya_hangman.baseline import FREQUENCY_ORDER, FrequencyBaseline
from laya_hangman.decision import DecisionInput
from laya_hangman.game import HangmanGame
from laya_hangman.wordlist import all_words, dev_words, eval_words, is_eval_word


def test_baseline_skips_guessed_letters_and_follows_the_documented_order():
    g = HangmanGame("quiz")
    d = FrequencyBaseline()
    seen = []
    while g.status == "in_progress":
        letter = d.decide(DecisionInput.from_game(g)).letter
        g.guess(letter)
        seen.append(letter)
    assert len(set(seen)) == len(seen)
    assert seen == [c for c in FREQUENCY_ORDER if c in seen]       # strictly the fixed order
    assert d.decide(DecisionInput.from_game(HangmanGame("quiz"))).letter == "E"


def test_baseline_is_deterministic():
    runs = [evaluate.play_game("PLANET", FrequencyBaseline()).guesses for _ in range(3)]
    assert len(set(runs)) == 1


def test_play_game_scores_wins_and_losses_with_the_same_rules():
    win = evaluate.play_game("CAT", ScriptedDecider(order="CAT"))
    assert win.won and win.incorrect == 0 and win.total == 3
    loss = evaluate.play_game("CAT", ScriptedDecider(order="BDEFGHCAT"))
    assert not loss.won and loss.incorrect == 6 and loss.total == 6


def test_deciders_receive_only_decision_inputs_never_the_answer():
    spy = ScriptedDecider()
    evaluate.play_game("ZEBRA", spy)
    assert spy.calls and all(isinstance(c, DecisionInput) for c in spy.calls)
    assert all("ZEBRA" not in repr(c) for c in spy.calls)


def test_summary_statistics():
    rs = [evaluate.GameResult("A", True, 1, 4, [10.0, 20.0], ""), evaluate.GameResult("B", False, 6, 9, [30.0], "")]
    s = evaluate.summarise(rs, with_latency=True)
    assert s["win_rate"] == 0.5 and s["mean_incorrect_guesses"] == 3.5 and s["mean_total_guesses"] == 6.5
    assert s["mean_inference_ms"] == pytest.approx(20.0) and s["inferences"] == 3


def test_word_split_is_disjoint_stable_and_matches_the_committed_file():
    ev, dev = set(eval_words()), set(dev_words())
    assert ev and dev and not (ev & dev) and ev | dev == set(all_words())
    assert 0.15 < len(ev) / len(all_words()) < 0.25
    assert all(is_eval_word(w) for w in ev) and not any(is_eval_word(w) for w in dev)
    committed = (Path(__file__).resolve().parents[1] / "evaluation" / "eval_words.txt").read_text().split()
    assert committed == eval_words()


def test_word_list_is_clean():
    for w in all_words():
        assert w.isascii() and w.isalpha() and w.isupper() and 4 <= len(w) <= 12


def test_cli_baseline_report(tmp_path, capsys):
    out = tmp_path / "r.json"
    evaluate.main(["--split", "eval", "--deciders", "baseline", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["deciders"]["baseline"]["summary"]["games"] == len(eval_words())
    assert report["word_set"]["sha256_of_word_sequence"] == hashlib.sha256("\n".join(eval_words()).encode()).hexdigest()
    assert "Frequency baseline" in capsys.readouterr().out


def test_laya_run_aborts_instead_of_inventing_results_when_the_model_cannot_load(monkeypatch):
    from laya_hangman import model

    def failing_loader(manager, device):
        raise OSError("cannot reach huggingface.co")

    real_cls = model.ModelManager
    monkeypatch.setattr(model, "ModelManager", lambda: real_cls(loader=failing_loader))

    class Args:
        split, limit, deciders, per_game, progress = "eval", 3, ["laya"], False, False

    with pytest.raises(SystemExit) as exc:
        evaluate.run(Args)
    assert "no Laya results were produced" in str(exc.value) and "cannot reach" in str(exc.value)
