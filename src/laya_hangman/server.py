"""FastAPI application: JSON API plus the static browser UI. Binds to localhost by default."""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .baseline import FREQUENCY_ORDER, FrequencyBaseline
from .decision import INSTRUCTIONS, InvalidModelResponse, build_request
from .game import (MAX_INCORRECT, MAX_WORD_LENGTH, MIN_WORD_LENGTH, GameOver, InvalidGuess, InvalidWord,
                   normalize_word)
from .model import LayaDecider, ModelManager, ModelNotReady
from .sessions import (GameFinished, SessionManager, StaleTurn, TurnInProgress, UnknownSession)
from .wordlist import all_words, eval_words, random_word

STATIC = resources.files("laya_hangman").joinpath("static")


class NewGame(BaseModel):
    word: Optional[str] = None
    random: bool = False
    previous_id: Optional[str] = None


class Turn(BaseModel):
    engine: str = "laya"  # "laya" or "baseline"


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status,
                        headers={"Cache-Control": "no-store"})


def create_app(manager: Optional[ModelManager] = None, autostart: bool = True,
               allowed_hosts: Optional[list] = None) -> FastAPI:
    manager = manager or ModelManager()
    sessions = SessionManager()
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="turn")
    baseline = FrequencyBaseline()
    laya_decider = LayaDecider(manager)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if autostart:
            manager.start()
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Laya Hangman", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=allowed_hosts or ["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/status")
    async def status() -> Dict[str, Any]:
        words = all_words()
        return {
            "app_version": __version__,
            "model": manager.status(),
            "rules": {"max_incorrect": MAX_INCORRECT, "min_word_length": MIN_WORD_LENGTH,
                      "max_word_length": MAX_WORD_LENGTH},
            "baseline": {"label": "Frequency baseline (not Laya)", "order": FREQUENCY_ORDER},
            "instructions": INSTRUCTIONS,
            "word_list": {"count": len(words), "eval_count": len(eval_words())},
        }

    @app.post("/api/model/load")
    async def load_model() -> Dict[str, Any]:
        started = manager.start()
        return {"started": started, "model": manager.status()}

    @app.post("/api/games")
    async def new_game(body: NewGame):
        try:
            if body.random:
                word = random_word()
            else:
                word = normalize_word(body.word)
        except InvalidWord as exc:
            return _error(422, "invalid_word", str(exc))
        session = sessions.create(word, previous_id=body.previous_id)
        return session.public_view()

    @app.get("/api/games/{sid}")
    async def get_game(sid: str):
        try:
            session = sessions.get(sid)
        except UnknownSession as exc:
            return _error(404, "unknown_game", str(exc))
        with session.lock:
            return session.public_view()

    @app.delete("/api/games/{sid}")
    async def close_game(sid: str):
        sessions.close(sid)
        return {"closed": True}

    @app.get("/api/games/{sid}/decision-input")
    async def decision_input(sid: str):
        """The exact request Laya would receive for the next guess. Contains no secret."""
        try:
            inp = sessions.decision_input(sid)
        except UnknownSession as exc:
            return _error(404, "unknown_game", str(exc))
        if not inp.available:
            return {"game_over": True, "decision_input": inp.to_dict(), "laya_request": None}
        req = build_request(inp, manager.config.max_len, manager.config.head_max_len)
        return {"game_over": False, "decision_input": inp.to_dict(), "laya_request": req.to_dict()}

    @app.post("/api/games/{sid}/turn")
    async def take_turn(sid: str, body: Turn):
        if body.engine not in ("laya", "baseline"):
            return _error(422, "bad_engine", "engine must be 'laya' or 'baseline'.")
        decider = laya_decider if body.engine == "laya" else baseline
        loop = asyncio.get_running_loop()
        try:
            turn = await loop.run_in_executor(executor, sessions.play_turn, sid, decider)
        except UnknownSession as exc:
            return _error(404, "unknown_game", str(exc))
        except TurnInProgress as exc:
            return _error(409, "turn_in_progress", str(exc))
        except GameFinished as exc:
            return _error(409, "game_finished", str(exc))
        except StaleTurn as exc:
            return _error(409, "stale_turn", str(exc))
        except ModelNotReady as exc:
            return _error(503, "model_not_ready", str(exc), model_state=manager.state)
        except InvalidModelResponse as exc:
            return _error(502, "invalid_model_response", str(exc))
        except (GameOver, InvalidGuess) as exc:
            return _error(502, "illegal_guess", str(exc))
        except Exception as exc:  # noqa: BLE001 - real inference failure: show it, do not fake a guess
            return _error(502, "inference_failed", f"{type(exc).__name__}: {exc}")
        try:
            session = sessions.get(sid)
        except UnknownSession as exc:
            return _error(409, "stale_turn", str(exc))
        with session.lock:
            return {"turn": turn.to_dict(), "game": session.public_view()}

    @app.get("/")
    async def index():
        return FileResponse(str(STATIC.joinpath("index.html")), media_type="text/html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    app.state.manager = manager
    app.state.sessions = sessions
    return app


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: Optional[list] = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="laya-hangman", description="Local Hangman where Laya guesses the letters.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--no-autoload", action="store_true", help="do not start loading Laya at startup")
    args = parser.parse_args(argv)
    hosts = ["localhost", "127.0.0.1", "[::1]"]
    if not _is_loopback(args.host):
        print(f"WARNING: binding to {args.host} exposes the game (including secret words you type) "
              f"to your network. The default is 127.0.0.1.")
        hosts = ["*"]
    app = create_app(autostart=not args.no_autoload, allowed_hosts=hosts)
    print(f"Laya Hangman -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
