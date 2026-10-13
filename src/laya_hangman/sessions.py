"""Server-side game sessions with protection against overlapping and stale turns.

A turn has three phases:
  1. under the session lock: check the game can take a turn, mark it busy, snapshot `version`;
  2. outside the lock: ask the decider (slow, may take a while, may fail);
  3. under the lock again: apply the guess *only if* the session is still open and `version` is
     unchanged; otherwise the result is discarded (`StaleTurn`).

`busy` makes a second concurrent turn on the same session fail fast (`TurnInProgress`), so a guess is
never decided twice from the same position. Closing a session (New Game) flips `closed`, so an
in-flight response for the old game can never touch anything, and it can never reach a new game
because every game gets its own session object.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .decision import Decision, DecisionInput
from .game import IN_PROGRESS, HangmanGame, InvalidGuess

MAX_SESSIONS = 50


class Decider(Protocol):
    name: str

    def decide(self, inp: DecisionInput) -> Decision: ...


class SessionError(Exception):
    pass


class UnknownSession(SessionError):
    pass


class TurnInProgress(SessionError):
    pass


class GameFinished(SessionError):
    pass


class StaleTurn(SessionError):
    """The game was replaced or closed while the decider was working; the result was discarded."""


@dataclass
class TurnRecord:
    number: int
    letter: str
    hit: bool
    occurrences: int
    source: str
    latency_ms: float
    probabilities: Optional[Dict[str, float]]
    confidence: Optional[float]
    answer_confidence: Optional[float]
    request: Optional[Dict[str, Any]]
    pattern_after: str
    incorrect_after: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Session:
    id: str
    game: HangmanGame
    lock: threading.Lock = field(default_factory=threading.Lock)
    busy: bool = False
    closed: bool = False
    version: int = 0
    turns: List[TurnRecord] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def public_view(self) -> Dict[str, Any]:
        g = self.game
        view: Dict[str, Any] = {
            "id": self.id,
            "status": g.status,
            "pattern": list(g.pattern),
            "length": g.length,
            "max_incorrect": g.max_incorrect,
            "incorrect_count": g.incorrect_count,
            "remaining_incorrect": g.remaining_incorrect,
            "guesses": [{"letter": r.letter, "hit": r.hit, "occurrences": r.occurrences} for r in g.guesses],
            "available_letters": list(g.available_letters),
            "turns": [t.to_dict() for t in self.turns],
            "busy": self.busy,
            "version": self.version,
        }
        if g.status != IN_PROGRESS:
            view["secret"] = g.reveal()  # only after the game has ended
        return view


class SessionManager:
    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self, secret: str, previous_id: Optional[str] = None) -> Session:
        if previous_id:
            self.close(previous_id)
        session = Session(id=uuid.uuid4().hex, game=HangmanGame(secret))
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > self._max:
                _, old = self._sessions.popitem(last=False)
                old.closed = True
        return session

    def get(self, sid: str) -> Session:
        with self._lock:
            session = self._sessions.get(sid)
        if session is None or session.closed:
            raise UnknownSession("Unknown or closed game.")
        return session

    def close(self, sid: str) -> None:
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is not None:
            with session.lock:
                session.closed = True

    def play_turn(self, sid: str, decider: Decider) -> TurnRecord:
        session = self.get(sid)
        with session.lock:
            if session.closed:
                raise UnknownSession("Unknown or closed game.")
            if session.game.status != IN_PROGRESS:
                raise GameFinished("The game is already over.")
            if session.busy:
                raise TurnInProgress("A guess is already being made for this game.")
            session.busy = True
            version = session.version
            inp = DecisionInput.from_game(session.game)
        try:
            decision = decider.decide(inp)  # slow; deliberately outside the lock
        except BaseException:
            with session.lock:
                session.busy = False
            raise
        with session.lock:
            session.busy = False
            if session.closed or session.version != version:
                raise StaleTurn("The game changed while the guess was being made; the result was discarded.")
            try:
                record = session.game.guess(decision.letter)
            except InvalidGuess:
                # A decider returned an illegal letter. Surface it; never repair it silently.
                raise
            session.version += 1
            turn = TurnRecord(
                number=len(session.turns) + 1, letter=record.letter, hit=record.hit,
                occurrences=record.occurrences, source=decision.source, latency_ms=round(decision.latency_ms, 2),
                probabilities=decision.probabilities, confidence=decision.confidence,
                answer_confidence=decision.answer_confidence, request=decision.request,
                pattern_after="".join(session.game.pattern), incorrect_after=session.game.incorrect_count)
            session.turns.append(turn)
            return turn

    def decision_input(self, sid: str) -> DecisionInput:
        session = self.get(sid)
        with session.lock:
            return DecisionInput.from_game(session.game)
