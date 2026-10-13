"""Loads Laya once and serialises inference. Reports honest lifecycle states.

States: idle -> (downloading ->) loading -> checking -> ready <-> inferring, or error.
`error` stays until `start()` is called again (the UI's Retry button).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .decision import (DEFAULT_HEAD_MAX_LEN, DEFAULT_MAX_LEN, QUESTION_ID, Decision, DecisionInput,
                       GuessOutcome, InvalidModelResponse, LayaRequest, build_request, parse_laya_result)
from .game import ALPHABET, MAX_WORD_LENGTH
from .hardware import choose_device, detect_hardware

MODEL_ID = "convaiinnovations/laya"  # English checkpoint recommended by the official README
# Reviewed commit published in laya 0.4.6 (laya/revisions.py: PINNED_REVISIONS).
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
_ALLOW = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]  # same set Agent downloads

IDLE, DOWNLOADING, LOADING, CHECKING, READY, INFERRING, ERROR = (
    "idle", "downloading", "loading", "checking", "ready", "inferring", "error")
_BUSY = (DOWNLOADING, LOADING, CHECKING, READY, INFERRING)


class ModelNotReady(RuntimeError):
    pass


@dataclass
class LoadConfig:
    model_id: str = MODEL_ID
    revision: Optional[str] = MODEL_REVISION
    device: str = "auto"
    max_len: int = DEFAULT_MAX_LEN
    head_max_len: int = DEFAULT_HEAD_MAX_LEN

    @classmethod
    def from_env(cls) -> "LoadConfig":
        env = os.environ
        return cls(
            model_id=env.get("LAYA_MODEL", MODEL_ID),
            revision=(env.get("LAYA_REVISION", MODEL_REVISION) or None),
            device=env.get("LAYA_DEVICE", "auto"),
            max_len=int(env.get("LAYA_MAX_LEN", DEFAULT_MAX_LEN)),
            head_max_len=int(env.get("LAYA_HEAD_MAX_LEN", DEFAULT_HEAD_MAX_LEN)),
        )


def worst_case_inputs() -> List[DecisionInput]:
    """Synthetic, non-secret inputs at the two extremes of the token budget."""
    letters = tuple(ALPHABET)
    first_turn = DecisionInput(tuple("_" * MAX_WORD_LENGTH), (), 6, 6, letters)  # all 26 options
    order = "ETAOINSHRDLCUMWFGYPB"  # 20 guesses: 5 misses + 15 hits, 6 letters left, 1 life left
    misses = set(order[:5])
    guesses = tuple(GuessOutcome(c, c not in misses) for c in order)
    hits = [c for c in order if c not in misses]
    pattern = tuple(hits[i % len(hits)] if i % 3 else "_" for i in range(MAX_WORD_LENGTH))
    late = DecisionInput(pattern, guesses, 1, 6, tuple(c for c in ALPHABET if c not in order))
    return [first_turn, late]


def measure_head(tok: Any, request: LayaRequest) -> Optional[Dict[str, int]]:
    """Best-effort token accounting using Laya's own head builder (a private helper).

    Returns None if the helper is not importable. The authoritative check is the `usage` block of a
    real prediction, which `ModelManager._check` also enforces via `parse_laya_result`.
    """
    try:
        from laya.common import build_head
    except Exception:  # pragma: no cover
        return None
    q = request.questions[QUESTION_ID]
    internal = {"t": "choice", "ins": q["instructions"], "crit": {k: None for k in q["criteria"]}}
    ids, _markers, stats = build_head(tok, internal, request.head_max_len)
    return {"head_tokens": len(ids), "options": stats["options"], "options_distinct": stats["options_distinct"],
            "state_room": request.max_len - len(ids) - 1, "tokens_per_option_cap": stats["tokens_per_option"] or 0}


class ModelManager:
    """Owns the single Laya agent. Inference is serialised by an internal lock."""

    def __init__(self, config: Optional[LoadConfig] = None,
                 loader: Optional[Callable[["ModelManager", str], Any]] = None) -> None:
        self.config = config or LoadConfig.from_env()
        self._loader = loader or default_loader
        self._lock = threading.RLock()       # guards status fields
        self._infer_lock = threading.Lock()  # one forward pass at a time
        self._thread: Optional[threading.Thread] = None
        self.agent: Any = None
        self.state = IDLE
        self.message = "Model not loaded yet."
        self.error: Optional[str] = None
        self.hardware: Dict[str, Any] = {}
        self.device: Optional[str] = None
        self.device_note: Optional[str] = None
        self.resolved_revision: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self.budget: Optional[Dict[str, Any]] = None
        self.inferences = 0
        self.is_test_double = False

    # ---- status ----------------------------------------------------------------------------
    def set_state(self, state: str, message: str) -> None:
        with self._lock:
            self.state, self.message = state, message
            if state != ERROR:
                self.error = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state, "message": self.message, "error": self.error,
                "model_id": self.config.model_id, "requested_revision": self.config.revision,
                "resolved_revision": self.resolved_revision, "device": self.device,
                "device_note": self.device_note, "hardware": self.hardware,
                "load_seconds": self.load_seconds, "budget": self.budget,
                "max_len": self.config.max_len, "head_max_len": self.config.head_max_len,
                "inferences": self.inferences, "is_test_double": self.is_test_double,
            }

    # ---- loading ---------------------------------------------------------------------------
    def start(self) -> bool:
        """Begin loading in the background. Returns False if loading or already loaded."""
        with self._lock:
            if self.state in _BUSY or (self._thread is not None and self._thread.is_alive()):
                return False
            self.state, self.message, self.error = LOADING, "Starting…", None
            self.device_note = None
            self._thread = threading.Thread(target=self._load, name="laya-loader", daemon=True)
            self._thread.start()
            return True

    def wait(self, timeout: Optional[float] = None) -> str:
        t = self._thread
        if t is not None:
            t.join(timeout)
        return self.state

    def _fail(self, exc: BaseException, context: str) -> None:
        with self._lock:
            self.agent = None
            self.state = ERROR
            self.error = f"{context}: {type(exc).__name__}: {exc}"
            hint = _hint_for(exc)
            if hint:
                self.error += f"\nHint: {hint}"
            self.message = "Laya is not available. Fix the problem described below, then press Retry."

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            self.hardware = detect_hardware()
            device = choose_device(self.hardware, self.config.device)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc, "Hardware detection failed")
            return
        try:
            agent, budget = self._load_and_check(device)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the user verbatim
            auto = self.config.device.strip().lower() == "auto"
            if device == "cpu" or not auto or isinstance(exc, InvalidModelResponse):
                self._fail(exc, "Loading Laya failed")
                return
            # Automatic device choice picked an accelerator and it did not work: say so, use CPU.
            reason = f"{type(exc).__name__}: {exc}"
            self.set_state(LOADING, f"{device} failed ({reason}); retrying on CPU…")
            try:
                agent, budget = self._load_and_check("cpu")
            except BaseException as exc2:  # noqa: BLE001
                self._fail(exc2, f"Loading Laya failed on {device} ({reason}) and again on cpu")
                return
            device = "cpu"
            self.device_note = f"Fell back to CPU because loading or checking failed on the accelerator: {reason}"
        with self._lock:
            self.agent, self.device, self.budget = agent, device, budget
            self.load_seconds = round(time.perf_counter() - started, 2)
            self.state, self.message = READY, f"Laya is ready on {device}."

    def _load_and_check(self, device: str) -> tuple:
        agent = self._loader(self, device)
        self.set_state(CHECKING, "Running a real test prediction and checking that all 26 letter options fit…")
        return agent, self._check(agent)

    def _check(self, agent: Any) -> Dict[str, Any]:
        """Real predictions on synthetic worst-case inputs; verifies compatibility and the token budget."""
        report: Dict[str, Any] = {"cases": []}
        names = ("all 26 letter options, 30-letter word", "longest guess history")
        for name, inp in zip(names, worst_case_inputs()):
            req = build_request(inp, self.config.max_len, self.config.head_max_len)
            tok = getattr(agent, "tok", None)
            head = measure_head(tok, req) if tok is not None else None
            if head and head["options_distinct"] < head["options"]:
                raise InvalidModelResponse(
                    f"{name}: only {head['options_distinct']} of {head['options']} letter options have a distinct "
                    f"token span. Raise LAYA_HEAD_MAX_LEN (now {self.config.head_max_len}).")
            t0 = time.perf_counter()
            result = agent.predict(req.state, req.questions, max_len=req.max_len, head_max_len=req.head_max_len)
            ms = (time.perf_counter() - t0) * 1000.0
            decision = parse_laya_result(result, inp, req, ms)  # raises on truncation / collapse / bad schema
            usage = decision.usage or {}
            report["cases"].append({
                "name": name, "options": len(inp.available), "head": head,
                "input_tokens": usage.get("input_tokens"), "state_tokens": usage.get("state_tokens"),
                "truncated": bool(usage.get("truncated")), "ms": round(ms, 1)})
        report["ok"] = True
        return report

    # ---- inference ---------------------------------------------------------------------------
    def decide(self, inp: DecisionInput) -> Decision:
        """Run one real Laya forward pass (blocking). Never substitutes another source on failure."""
        req = build_request(inp, self.config.max_len, self.config.head_max_len)
        with self._lock:
            if self.state not in (READY, INFERRING) or self.agent is None:
                raise ModelNotReady(f"Laya is not ready (state: {self.state}).")
            agent = self.agent
        with self._infer_lock:
            self.set_state(INFERRING, "Laya is choosing a letter…")
            try:
                t0 = time.perf_counter()
                result = agent.predict(req.state, req.questions, max_len=req.max_len,
                                       head_max_len=req.head_max_len)
                ms = (time.perf_counter() - t0) * 1000.0
                decision = parse_laya_result(result, inp, req, ms)
            finally:
                with self._lock:
                    if self.state == INFERRING:
                        self.state, self.message = READY, f"Laya is ready on {self.device}."
            with self._lock:
                self.inferences += 1
            return decision


def _hint_for(exc: BaseException) -> Optional[str]:
    text = f"{type(exc).__name__} {exc}".lower()
    if any(w in text for w in ("proxy", "connection", "timeout", "timed out", "resolve", "unreachable",
                               "huggingface", "offline", "401", "403", "404", "localentrynotfound")):
        return ("the checkpoint is downloaded from huggingface.co on first use (about 1-2 GB). Check network "
                "access to huggingface.co, or download the files elsewhere and point LAYA_MODEL at that "
                "folder. Once cached, the app also works offline (HF_HUB_OFFLINE=1).")
    if "out of memory" in text or "mps" in text:
        return "try LAYA_DEVICE=cpu to run on the CPU instead."
    return None


class LayaDecider:
    name = "laya"

    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager

    def decide(self, inp: DecisionInput) -> Decision:
        return self.manager.decide(inp)


def default_loader(manager: ModelManager, device: str) -> Any:
    """Download (if needed) and load the Laya checkpoint on `device`."""
    import laya
    from huggingface_hub import snapshot_download

    cfg = manager.config
    path = cfg.model_id
    if not os.path.isdir(path):
        try:
            path = snapshot_download(cfg.model_id, revision=cfg.revision, allow_patterns=_ALLOW,
                                     local_files_only=True)
        except Exception:  # not in the local cache (or offline cache incomplete): download it
            manager.set_state(DOWNLOADING, f"Downloading {cfg.model_id} from Hugging Face (first run only)…")
            path = snapshot_download(cfg.model_id, revision=cfg.revision, allow_patterns=_ALLOW)
        parts = os.path.normpath(path).split(os.sep)
        manager.resolved_revision = parts[-1] if "snapshots" in parts else cfg.revision
    manager.set_state(LOADING, f"Loading Laya onto {device}…")
    return laya.load(path, device=device)
