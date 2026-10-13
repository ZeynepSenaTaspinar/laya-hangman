import threading

import pytest

from conftest import GatedDecider, ScriptedDecider
from laya_hangman.decision import Decision
from laya_hangman.game import InvalidGuess
from laya_hangman.sessions import (GameFinished, SessionManager, StaleTurn, TurnInProgress, UnknownSession)


def run_in_thread(fn, *args):
    box = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    return t, box


def test_a_turn_applies_the_guess_and_records_it():
    sm = SessionManager()
    s = sm.create("cat")
    turn = sm.play_turn(s.id, ScriptedDecider(order="CAT"))
    assert turn.letter == "C" and turn.hit and turn.pattern_after == "C__"
    assert s.public_view()["turns"][0]["source"] == "laya"


def test_overlapping_turns_on_one_game_are_rejected():
    sm = SessionManager()
    s = sm.create("planet")
    slow = GatedDecider()
    t, box = run_in_thread(sm.play_turn, s.id, slow)
    assert slow.entered.wait(5)
    with pytest.raises(TurnInProgress):
        sm.play_turn(s.id, ScriptedDecider())
    slow.release.set(); t.join(5)
    assert "error" not in box
    assert len(s.game.guesses) == 1 and not s.busy       # exactly one guess was applied


def test_a_response_for_a_replaced_game_is_discarded_and_cannot_touch_the_new_game():
    sm = SessionManager()
    old = sm.create("planet")
    slow = GatedDecider()
    t, box = run_in_thread(sm.play_turn, old.id, slow)
    assert slow.entered.wait(5)
    new = sm.create("cat", previous_id=old.id)           # New Game while inference is running
    slow.release.set(); t.join(5)
    assert isinstance(box.get("error"), StaleTurn)
    assert old.game.guesses == ()                          # old game untouched
    assert new.game.guesses == () and new.version == 0     # new game untouched
    with pytest.raises(UnknownSession):
        sm.get(old.id)


def test_closing_a_game_mid_flight_discards_the_result():
    sm = SessionManager()
    s = sm.create("planet")
    slow = GatedDecider()
    t, box = run_in_thread(sm.play_turn, s.id, slow)
    assert slow.entered.wait(5)
    sm.close(s.id)
    slow.release.set(); t.join(5)
    assert isinstance(box.get("error"), StaleTurn) and s.game.guesses == ()


def test_a_failing_decider_changes_nothing_and_the_turn_can_be_retried():
    sm = SessionManager()
    s = sm.create("cat")

    class Boom:
        name = "boom"

        def decide(self, inp):
            raise RuntimeError("inference exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        sm.play_turn(s.id, Boom())
    assert s.game.guesses == () and not s.busy and s.version == 0
    assert sm.play_turn(s.id, ScriptedDecider(order="C")).letter == "C"


def test_an_illegal_letter_from_a_decider_is_an_error_not_a_repair():
    sm = SessionManager()
    s = sm.create("cat")
    sm.play_turn(s.id, ScriptedDecider(order="C"))

    class Repeats:
        name = "repeats"

        def decide(self, inp):
            return Decision(letter="C", source="laya", latency_ms=1.0)

    with pytest.raises(InvalidGuess):
        sm.play_turn(s.id, Repeats())
    assert s.game.guessed_letters == ("C",) and not s.busy


def test_no_turns_after_the_game_is_over():
    sm = SessionManager()
    s = sm.create("ab")
    d = ScriptedDecider(order="AB")
    sm.play_turn(s.id, d); sm.play_turn(s.id, d)
    with pytest.raises(GameFinished):
        sm.play_turn(s.id, d)
    assert s.public_view()["secret"] == "AB" and s.public_view()["status"] == "won"


def test_the_secret_is_never_in_the_public_view_while_playing():
    sm = SessionManager()
    s = sm.create("hidden")
    sm.play_turn(s.id, ScriptedDecider(order="E"))
    view = s.public_view()
    assert "secret" not in view and "HIDDEN" not in str(view).upper().replace("_", "")


def test_independent_games_run_concurrently():
    sm = SessionManager()
    a, b = sm.create("alpha"), sm.create("bravo")
    slow = GatedDecider()
    t, box = run_in_thread(sm.play_turn, a.id, slow)
    assert slow.entered.wait(5)
    assert sm.play_turn(b.id, ScriptedDecider(order="B")).letter == "B"   # not blocked by game a
    slow.release.set(); t.join(5)
    assert "error" not in box


def test_session_cap_closes_the_oldest():
    sm = SessionManager(max_sessions=2)
    a = sm.create("one"); sm.create("two"); sm.create("three")
    assert a.closed
    with pytest.raises(UnknownSession):
        sm.get(a.id)
