"""Serve the real UI + API with a MOCK Laya (test double) for the browser flow test.

The status chip reads "TEST DOUBLE (not Laya)" so it can never be mistaken for the real model.
The mock agent returns laya-shaped output with probabilities biased toward frequent letters.
Run: python tests/e2e/serve_mock.py PORT [delay_seconds] [fail_after_n_calls]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import uvicorn  # noqa: E402

from laya_hangman.baseline import FREQUENCY_ORDER  # noqa: E402
from laya_hangman.model import LoadConfig, ModelManager  # noqa: E402
from laya_hangman.server import create_app  # noqa: E402


class MockAgent:
    def __init__(self, delay, fail_after):
        self.delay, self.fail_after, self.calls = delay, fail_after, 0

    def predict(self, state, questions, max_len=None, head_max_len=None):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail_after is not None and self.calls > 2 + self.fail_after:   # 2 startup-check calls
            raise RuntimeError("mock inference failure (test)")
        options = questions["next_letter"]["criteria"]
        weights = {c: 1.0 / (1 + FREQUENCY_ORDER.index(c)) for c in options}
        total = sum(weights.values())
        probs = {c: round(w / total, 4) for c, w in weights.items()}
        probs[options[0]] = round(probs[options[0]] + 1 - sum(probs.values()), 4)
        choice = max(probs, key=probs.get)
        return {"model": "mock", "answers": {"next_letter": {"type": "choice", "choice": choice, "probabilities": probs,
                                                               "confidence": 0.3, "answer_confidence": 0.4}},
                "usage": {"input_tokens": 100, "state_tokens": 40, "state_tokens_dropped": 0, "truncated": False}}


if __name__ == "__main__":
    port = int(sys.argv[1])
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    fail_after = int(sys.argv[3]) if len(sys.argv) > 3 else None
    manager = ModelManager(LoadConfig(device="cpu"), loader=lambda m, d: MockAgent(delay, fail_after))
    manager.is_test_double = True
    uvicorn.run(create_app(manager=manager, autostart=True), host="127.0.0.1", port=port, log_level="warning")
