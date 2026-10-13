import threading
import time

import pytest

from conftest import laya_style_result
from laya_hangman.decision import DecisionInput, InvalidModelResponse
from laya_hangman.game import HangmanGame
from laya_hangman.hardware import choose_device
from laya_hangman.model import (ERROR, READY, LoadConfig, ModelManager, ModelNotReady, worst_case_inputs)


class FakeAgent:
    """Stands in for laya.Agent: same predict() contract, no weights."""

    def __init__(self, behaviour=None, delay=0.0):
        self.behaviour, self.delay = behaviour, delay
        self.calls, self.active, self.max_active = [], 0, 0
        self._lock = threading.Lock()

    def predict(self, state, questions, max_len=None, head_max_len=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            self.calls.append((state, questions, max_len, head_max_len))
            if self.behaviour:
                return self.behaviour(state, questions)
            options = questions["next_letter"]["criteria"]
            inp = DecisionInput(("_",), (), 6, 6, tuple(options))
            return laya_style_result(inp, choice=options[0])
        finally:
            with self._lock:
                self.active -= 1


def manager_with(loader, device="auto"):
    return ModelManager(LoadConfig(device=device), loader=loader)


def ready_manager(agent=None, device="cpu"):
    agent = agent or FakeAgent()
    m = manager_with(lambda mgr, dev: agent, device=device)
    m.start()
    assert m.wait(5) == READY
    return m, agent


def test_lifecycle_reaches_ready_and_reports_status():
    seen = []
    m = manager_with(lambda mgr, dev: (seen.append(mgr.state), FakeAgent())[1], device="cpu")
    m.start(); m.wait(5)
    st = m.status()
    assert st["state"] == READY and st["device"] == "cpu" and st["budget"]["ok"] and seen == ["loading"]
    assert st["model_id"] == "convaiinnovations/laya"


def test_the_startup_check_runs_real_predict_calls_for_26_options_and_the_longest_state():
    m, agent = ready_manager()
    assert len(agent.calls) == 2
    first, late = agent.calls
    assert len(first[1]["next_letter"]["criteria"]) == 26
    assert len(late[1]["next_letter"]["criteria"]) == 6
    assert all(c[2] == 512 and c[3] == 192 for c in agent.calls)   # budgets passed explicitly


def test_worst_case_inputs_are_legal():
    for inp in worst_case_inputs():
        assert len(inp.pattern) == 30 and inp.incorrect_remaining >= 1 and inp.available


def test_load_failure_is_an_error_state_with_the_message_and_retry_works():
    attempts = []

    def loader(mgr, dev):
        attempts.append(dev)
        if len(attempts) == 1:
            raise OSError("network is unreachable")
        return FakeAgent()

    m = manager_with(loader, device="cpu")
    m.start(); assert m.wait(5) == ERROR
    assert "network is unreachable" in m.status()["error"] and "Retry" in m.status()["message"]
    with pytest.raises(ModelNotReady):
        m.decide(DecisionInput.from_game(HangmanGame("cat")))
    assert m.start() is True and m.wait(5) == READY       # Retry


def test_start_is_idempotent_while_loading_or_ready():
    gate = threading.Event()
    m = manager_with(lambda mgr, dev: (gate.wait(5), FakeAgent())[1], device="cpu")
    assert m.start() is True and m.start() is False
    gate.set(); m.wait(5)
    assert m.start() is False


def test_accelerator_failure_falls_back_to_cpu_visibly(monkeypatch):
    monkeypatch.setattr("laya_hangman.model.detect_hardware", lambda: {"mps_available": True, "cuda_available": False})
    tried = []

    def loader(mgr, dev):
        tried.append(dev)
        if dev == "mps":
            raise RuntimeError("MPS op not implemented")
        return FakeAgent()

    m = manager_with(loader, device="auto")
    m.start(); assert m.wait(5) == READY
    assert tried == ["mps", "cpu"] and m.device == "cpu"
    assert "MPS op not implemented" in m.device_note


def test_explicitly_requested_device_is_never_silently_replaced(monkeypatch):
    monkeypatch.setattr("laya_hangman.model.detect_hardware", lambda: {"mps_available": True, "cuda_available": False})

    def loader(mgr, dev):
        raise RuntimeError("MPS broke")

    m = manager_with(loader, device="mps")
    m.start(); assert m.wait(5) == ERROR and "MPS broke" in m.error


def test_requesting_an_unavailable_accelerator_is_an_error():
    m = manager_with(lambda mgr, dev: FakeAgent(), device="mps")
    m.start()
    state = m.wait(5)
    import torch
    if not torch.backends.mps.is_available():
        assert state == ERROR and "not available" in m.error


@pytest.mark.parametrize("info,requested,expected", [
    ({"cuda_available": False, "mps_available": True}, "auto", "mps"),
    ({"cuda_available": True, "mps_available": True}, "auto", "cuda"),
    ({"cuda_available": False, "mps_available": False}, "auto", "cpu"),
    ({"cuda_available": False, "mps_available": True}, "cpu", "cpu"),
])
def test_choose_device(info, requested, expected):
    assert choose_device(info, requested) == expected


def test_choose_device_rejects_unavailable_or_unknown():
    with pytest.raises(RuntimeError):
        choose_device({"mps_available": False}, "mps")
    with pytest.raises(RuntimeError):
        choose_device({}, "tpu")


def test_budget_failure_makes_laya_unavailable_instead_of_guessing_with_a_truncated_state():
    def truncating(state, questions):
        options = questions["next_letter"]["criteria"]
        inp = DecisionInput(("_",), (), 6, 6, tuple(options))
        return laya_style_result(inp, usage={"truncated": True, "state_tokens": 900, "state_tokens_dropped": 400})

    m = manager_with(lambda mgr, dev: FakeAgent(truncating), device="cpu")
    m.start(); assert m.wait(5) == ERROR and "truncated" in m.error


def test_decide_calls_predict_with_the_documented_request_and_times_it():
    m, agent = ready_manager(FakeAgent(delay=0.02))
    agent.calls.clear()
    inp = DecisionInput.from_game(HangmanGame("zebra"))
    d = m.decide(inp)
    assert d.source == "laya" and d.letter == "A" and d.latency_ms >= 20
    state, questions, max_len, head = agent.calls[0]
    assert "zebra" not in state.lower() and (max_len, head) == (512, 192)
    assert m.status()["inferences"] == 1 and m.state == READY


def test_invalid_model_output_raises_and_the_manager_stays_usable():
    flaky = {"bad": False}

    def behaviour(state, questions):
        options = questions["next_letter"]["criteria"]
        inp = DecisionInput(("_",), (), 6, 6, tuple(options))
        return {"answers": {}} if flaky["bad"] else laya_style_result(inp)

    m, _ = ready_manager(FakeAgent(behaviour))
    flaky["bad"] = True
    inp = DecisionInput.from_game(HangmanGame("zebra"))
    with pytest.raises(InvalidModelResponse):
        m.decide(inp)
    assert m.state == READY
    flaky["bad"] = False
    assert m.decide(inp).letter == "A"


def test_an_exception_from_predict_propagates_and_state_recovers():
    calls = {"n": 0}

    def behaviour(state, questions):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("out of memory")
        options = questions["next_letter"]["criteria"]
        return laya_style_result(DecisionInput(("_",), (), 6, 6, tuple(options)))

    m, _ = ready_manager(FakeAgent(behaviour))
    with pytest.raises(RuntimeError, match="out of memory"):
        m.decide(DecisionInput.from_game(HangmanGame("zebra")))
    assert m.state == READY


def test_forward_passes_are_serialised():
    m, agent = ready_manager(FakeAgent(delay=0.03))
    inp = DecisionInput.from_game(HangmanGame("zebra"))
    threads = [threading.Thread(target=m.decide, args=(inp,)) for _ in range(4)]
    [t.start() for t in threads]; [t.join(5) for t in threads]
    assert agent.max_active == 1


def test_decide_before_loading_raises_model_not_ready():
    m = ModelManager(LoadConfig(), loader=lambda mgr, dev: FakeAgent())
    with pytest.raises(ModelNotReady):
        m.decide(DecisionInput.from_game(HangmanGame("cat")))
