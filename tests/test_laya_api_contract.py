"""Runs this app against the REAL laya.Agent code path using a tiny random-weight model.

What this verifies: the request shape is accepted by Laya 0.4.6, the result schema is what
`parse_laya_result` expects, 26 letter options get distinct token spans within the default budget,
the state is not truncated at the worst case, and Laya's own `usage` block reports it.
What it does NOT verify: anything about the quality of the real checkpoint.
"""
import pytest

pytest.importorskip("laya")
pytest.importorskip("torch")

from laya_hangman.decision import DecisionInput, build_request, parse_laya_result
from laya_hangman.game import ALPHABET, HangmanGame
from laya_hangman.model import LoadConfig, ModelManager, READY, measure_head, worst_case_inputs
from tiny_laya import make_tiny_agent


@pytest.fixture(scope="module")
def agent():
    return make_tiny_agent()


def test_real_agent_accepts_the_request_and_returns_the_expected_schema(agent):
    inp = DecisionInput.from_game(HangmanGame("zebra"))
    req = build_request(inp)
    raw = agent.predict(req.state, req.questions, max_len=req.max_len, head_max_len=req.head_max_len)
    ans = raw["answers"]["next_letter"]
    assert ans["type"] == "choice" and ans["choice"] in ALPHABET
    assert set(ans["probabilities"]) == set(ALPHABET) and abs(sum(ans["probabilities"].values()) - 1) < 0.002
    assert {"confidence", "answer_confidence"} <= set(ans)
    assert raw["usage"]["truncated"] is False and "options" not in raw["usage"]
    d = parse_laya_result(raw, inp, req, 1.0)
    assert d.letter == ans["choice"] and d.confidence == ans["confidence"]


def test_all_26_options_have_distinct_token_spans_within_the_head_budget(agent):
    first = build_request(worst_case_inputs()[0])
    head = measure_head(agent.tok, first)
    assert head["options"] == 26 and head["options_distinct"] == 26
    assert head["tokens_per_option_cap"] == 0          # no per-option cut was needed
    assert head["head_tokens"] < first.head_max_len


def test_a_too_small_head_budget_is_detected_by_laya_itself(agent):
    inp = worst_case_inputs()[0]
    req = build_request(inp, max_len=512, head_max_len=40)
    head = measure_head(agent.tok, req)
    assert head["tokens_per_option_cap"] > 0           # Laya had to cut each option


def test_the_longest_state_is_not_truncated_by_laya(agent):
    for inp in worst_case_inputs():
        req = build_request(inp)
        raw = agent.predict(req.state, req.questions, max_len=req.max_len, head_max_len=req.head_max_len)
        assert raw["usage"]["truncated"] is False and raw["usage"]["state_tokens_dropped"] == 0


def test_a_budget_that_truncates_the_state_is_reported_as_an_error(agent):
    inp = worst_case_inputs()[1]
    req = build_request(inp, max_len=60, head_max_len=48)
    raw = agent.predict(req.state, req.questions, max_len=60, head_max_len=48)
    assert raw["usage"]["truncated"] is True
    with pytest.raises(Exception, match="truncated"):
        parse_laya_result(raw, inp, req, 1.0)


def test_manager_end_to_end_with_the_real_agent_class(agent):
    m = ModelManager(LoadConfig(device="cpu"), loader=lambda mgr, dev: agent)
    m.start()
    assert m.wait(30) == READY
    assert m.status()["budget"]["ok"] and m.status()["budget"]["cases"][0]["head"]["options_distinct"] == 26
    g = HangmanGame("zebra")
    seen = set()
    while g.status == "in_progress":
        d = m.decide(DecisionInput.from_game(g))
        assert d.letter not in seen
        seen.add(d.letter)
        g.guess(d.letter)
    assert g.status in ("won", "lost")
