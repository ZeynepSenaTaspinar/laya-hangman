import json

import pytest
from fastapi.testclient import TestClient

from laya_hangman.decision import DecisionInput
from laya_hangman.model import LoadConfig, ModelManager
from laya_hangman.server import create_app
from conftest import laya_style_result
from test_model import FakeAgent


def make_client(agent=None, load=True, state_ready=True):
    agent = agent or FakeAgent()
    manager = ModelManager(LoadConfig(device="cpu"), loader=lambda m, d: agent)
    manager.is_test_double = True
    app = create_app(manager=manager, autostart=False, allowed_hosts=["testserver"])
    client = TestClient(app)
    if load:
        manager.start(); manager.wait(5)
    return client, manager, agent


def create(client, word="planet", **kw):
    r = client.post("/api/games", json={"word": word, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_status_reports_model_state_and_rules():
    client, *_ = make_client()
    st = client.get("/api/status").json()
    assert st["model"]["state"] == "ready" and st["model"]["is_test_double"] is True
    assert st["rules"]["max_incorrect"] == 6 and st["baseline"]["label"].endswith("(not Laya)")


@pytest.mark.parametrize("word", ["", "two words", "abc1", "café", "x", "a" * 31])
def test_invalid_words_get_a_422_with_a_message(word):
    client, *_ = make_client()
    r = client.post("/api/games", json={"word": word})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_word" and r.json()["error"]["message"]


def test_missing_word_is_rejected():
    client, *_ = make_client()
    assert client.post("/api/games", json={}).status_code == 422


def test_secret_is_not_in_any_response_until_the_game_ends():
    client, *_ = make_client()
    g = create(client, "planet")
    sid = g["id"]
    bodies = [json.dumps(g)]
    bodies.append(client.get(f"/api/games/{sid}").text)
    bodies.append(client.get(f"/api/games/{sid}/decision-input").text)
    for _ in range(3):
        r = client.post(f"/api/games/{sid}/turn", json={"engine": "baseline"})
        bodies.append(r.text)
    for b in bodies:
        assert "PLANET" not in b.upper()
    # play to the end: the secret is revealed only then
    for _ in range(30):
        r = client.post(f"/api/games/{sid}/turn", json={"engine": "baseline"})
        if r.status_code != 200:
            break
        if r.json()["game"]["status"] != "in_progress":
            assert r.json()["game"]["secret"] == "PLANET"
            break


def test_random_word_game_hides_the_word_from_the_client():
    client, *_ = make_client()
    g = client.post("/api/games", json={"random": True}).json()
    assert "secret" not in g and g["length"] >= 4 and g["status"] == "in_progress"


def test_laya_turn_uses_the_model_and_returns_probabilities_and_latency():
    client, manager, agent = make_client(FakeAgent(delay=0.01))
    sid = create(client)["id"]
    agent.calls.clear()
    r = client.post(f"/api/games/{sid}/turn", json={"engine": "laya"}).json()
    t = r["turn"]
    assert t["source"] == "laya" and t["letter"] == "A" and t["latency_ms"] >= 10
    assert len(t["probabilities"]) == 26 and abs(sum(t["probabilities"].values()) - 1) < 0.002
    assert t["request"]["state"].startswith("Word pattern: _ _ _ _ _ _")
    assert len(agent.calls) == 1
    assert "PLANET" not in json.dumps(agent.calls).upper()


def test_baseline_turns_are_labelled_and_follow_the_fixed_order():
    client, *_ = make_client(load=False)    # the baseline needs no model at all
    sid = create(client)["id"]
    letters = []
    for _ in range(3):
        t = client.post(f"/api/games/{sid}/turn", json={"engine": "baseline"}).json()["turn"]
        assert t["source"] == "baseline" and t["probabilities"] is None
        letters.append(t["letter"])
    assert letters == ["E", "T", "A"]


def test_laya_turn_when_model_not_ready_is_a_503_and_no_guess_is_made():
    client, manager, _ = make_client(load=False)
    sid = create(client)["id"]
    r = client.post(f"/api/games/{sid}/turn", json={"engine": "laya"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_ready"
    assert client.get(f"/api/games/{sid}").json()["guesses"] == []


def test_bad_model_output_is_a_502_with_no_substitute_guess():
    good = FakeAgent()
    client, manager, _ = make_client(good)       # the startup check passes with the good behaviour

    def bad(state, questions):
        return {"answers": {"next_letter": {"type": "choice", "choice": "Q", "probabilities": {}}}, "usage": {}}

    good.behaviour = bad
    sid = create(client)["id"]
    r = client.post(f"/api/games/{sid}/turn", json={"engine": "laya"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "invalid_model_response"
    assert client.get(f"/api/games/{sid}").json()["guesses"] == []
    good.behaviour = None                         # retry succeeds
    assert client.post(f"/api/games/{sid}/turn", json={"engine": "laya"}).status_code == 200


def test_inference_exception_is_a_502_and_not_silently_replaced():
    good = FakeAgent()
    client, manager, _ = make_client(good)

    def boom(state, questions):
        raise RuntimeError("MPS backend out of memory")

    good.behaviour = boom
    sid = create(client)["id"]
    r = client.post(f"/api/games/{sid}/turn", json={"engine": "laya"})
    assert r.status_code == 502 and "out of memory" in r.json()["error"]["message"]
    assert client.get(f"/api/games/{sid}").json()["guesses"] == []


def test_turn_after_game_over_is_409_and_unknown_game_is_404():
    client, *_ = make_client(load=False)
    sid = create(client, "ab")["id"]
    for _ in range(40):
        r = client.post(f"/api/games/{sid}/turn", json={"engine": "baseline"})
        if r.status_code != 200 or r.json()["game"]["status"] != "in_progress":
            break
    r = client.post(f"/api/games/{sid}/turn", json={"engine": "baseline"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "game_finished"
    assert client.get("/api/games/nope").status_code == 404
    assert client.post("/api/games/nope/turn", json={"engine": "baseline"}).status_code == 404


def test_new_game_with_previous_id_closes_the_old_game():
    client, *_ = make_client(load=False)
    old = create(client)["id"]
    new = create(client, "moon", previous_id=old)["id"]
    assert client.get(f"/api/games/{old}").status_code == 404
    assert client.get(f"/api/games/{new}").status_code == 200
    assert client.delete(f"/api/games/{new}").status_code == 200
    assert client.get(f"/api/games/{new}").status_code == 404


def test_unknown_engine_is_rejected():
    client, *_ = make_client(load=False)
    sid = create(client)["id"]
    assert client.post(f"/api/games/{sid}/turn", json={"engine": "random"}).status_code == 422


def test_decision_input_endpoint_has_no_secret_and_lists_the_open_letters():
    client, *_ = make_client()
    sid = create(client, "planet")["id"]
    client.post(f"/api/games/{sid}/turn", json={"engine": "baseline"})
    shown = client.get(f"/api/games/{sid}/decision-input").json()
    assert shown["laya_request"]["questions"]["next_letter"]["criteria"] == shown["decision_input"]["available_letters"]
    assert "PLANET" not in json.dumps(shown).upper()


def test_decision_input_equals_the_real_request():
    client, manager, agent = make_client()
    sid = create(client, "planet")["id"]
    shown = client.get(f"/api/games/{sid}/decision-input").json()["laya_request"]
    agent.calls.clear()
    client.post(f"/api/games/{sid}/turn", json={"engine": "laya"})
    state, questions, max_len, head = agent.calls[0]
    assert shown == {"state": state, "questions": questions, "max_len": max_len, "head_max_len": head}


def test_wrong_host_header_is_rejected():
    manager = ModelManager(LoadConfig(device="cpu"), loader=lambda m, d: FakeAgent())
    app = create_app(manager=manager, autostart=False)   # default: loopback hosts only
    client = TestClient(app, base_url="http://evil.example")
    assert client.get("/api/status").status_code == 400
    assert TestClient(app, base_url="http://localhost").get("/api/status").status_code == 200


def test_static_ui_is_served_with_security_headers():
    client, *_ = make_client(load=False)
    r = client.get("/")
    assert r.status_code == 200 and "Laya Hangman" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200


def test_retry_endpoint_restarts_a_failed_load():
    attempts = {"n": 0}

    def loader(m, d):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("offline")
        return FakeAgent()

    manager = ModelManager(LoadConfig(device="cpu"), loader=loader)
    client = TestClient(create_app(manager=manager, autostart=False, allowed_hosts=["testserver"]))
    manager.start(); manager.wait(5)
    assert client.get("/api/status").json()["model"]["state"] == "error"
    assert client.post("/api/model/load").json()["started"] is True
    manager.wait(5)
    assert client.get("/api/status").json()["model"]["state"] == "ready"
