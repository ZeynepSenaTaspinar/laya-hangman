# Laya Hangman

A local, English-language Hangman game in which the **[Laya](https://github.com/NandhaKishorM/laya)** decision model
([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), Apache-2.0) plays the letter-guessing agent.
You type a secret word (or draw a random one); Laya guesses one letter per turn. Everything runs on your machine:
no Jev, no paid API, no text-generating model.

> **Result in one line.** The app works end to end with real local Laya inference, but the stock `convaiinnovations/laya`
> checkpoint **plays Hangman badly**: 0 wins in 20 games, versus 2 wins for a fixed letter-frequency baseline (see
> [Measured results](#measured-results)). This is consistent with Laya's own documentation, which says the base checkpoints are
> near chance zero-shot and meant to be fine-tuned. The sample is small (20 games) and was run on one machine.

## What does what

| Laya (the model) | The game code (this repo) |
|---|---|
| Chooses the next letter from the unguessed ones, in one forward pass (`Agent.predict`, a `choice` question). Returns per-letter probabilities. | Owns the secret word, validates input (A-Z, 2-30 letters), reveals every occurrence, rejects repeats, ends on a win or after 6 incorrect guesses. |
| Sees only: visible pattern, guesses so far with hit/miss, incorrect guesses left, brief Hangman instructions, the unguessed letters as options. | Builds that input from public game properties only (`DecisionInput`), never passes the secret, checks every response, and refuses to substitute a guess if Laya fails. |

There is **no dictionary solver behind Laya**. Laya does not choose the word. The "Frequency baseline" is a separate, always-labelled
mode that guesses in the fixed order `ETAOINSHRDLCUMWFGYPBVKJXQZ` (a conventional English letter-frequency order, a constant in
`baseline.py`, not derived from this repo's word lists).

## Setup and start

Requires Python 3.10+ (tested on 3.13). Works on macOS (Apple Silicon or Intel), Linux and Windows.

```bash
git clone <this repo> && cd laya-hangman
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m laya_hangman          # http://127.0.0.1:8765
```

Open <http://127.0.0.1:8765>. On first start Laya downloads its checkpoint, then loads it **once** and reuses it for every turn.
The status chip shows: not loaded -> downloading -> loading -> checking -> ready <-> inference in progress, or error (with a Retry button).

**macOS.** On Apple Silicon the default PyTorch wheel includes the MPS (Metal) backend, and the app uses it automatically when
`torch.backends.mps.is_available()` is true. On **Intel Macs** PyTorch stops at 2.2.2, so use the stack Laya documents instead of
`requirements.txt`: `pip install -r requirements-intel-mac.txt` (this exact stack, plus `fastapi uvicorn`, was installed and run successfully on an Intel Mac, Python 3.11).

Options (environment variables or flags): `PORT`/`--port`, `HOST`/`--host` (default `127.0.0.1`; binding elsewhere prints a warning and
exposes secret words you type to your network), `--no-autoload`, `LAYA_DEVICE` (`auto`|`cpu`|`mps`|`cuda`), `LAYA_MODEL`
(repo id or local checkpoint folder), `LAYA_REVISION`, `LAYA_MAX_LEN`, `LAYA_HEAD_MAX_LEN`.

## Model, revision and resources

* Checkpoint: **`convaiinnovations/laya`** - the English checkpoint (ModernBERT-large encoder, ~421M parameters, 512-token context) that
  Laya's README recommends for English. Package: `laya==0.4.6` on PyPI.
* Revision: pinned to `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, the reviewed commit listed in `laya/revisions.py` (`PINNED_REVISIONS`) of
  laya 0.4.6. The app requests exactly this revision; it was downloaded successfully on the owner's machine.
  The app records the revision actually resolved in `/api/status` and in evaluation reports.
* Download size: `model.safetensors` was 843 MB in the owner's download (about 1 GB in total with tokenizer and encoder config). Memory use was not measured. Cache location: `HF_HUB_CACHE`. Weights are never committed (`.gitignore`).
  `rl_agent_config.json, model.safetensors, tokenizer/*, encoder/*`. Expect on the order of 1-2 GB to download (a rough estimate from the
  parameter count, **unverified**) and similar RAM. Cache location: `HF_HUB_CACHE`. Weights are never committed (`.gitignore`).
* Hardware: the app reports what PyTorch sees. Priority is CUDA -> Apple MPS -> CPU (Laya's own order). After loading it runs real test
  predictions on the chosen device; if an *automatically chosen* accelerator fails to load or pass the check, it falls back to CPU and says
  so in the UI. An explicitly requested `LAYA_DEVICE` is never silently replaced.

## Token budget: do all 26 letters fit?

Laya renders a `choice` question as `[CLS] question [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`; options share a `head_max_len`
budget (default 192 of 512 for the English checkpoint). When options do not fit, Laya cuts each one to `(head_max_len-16)//k` tokens
(about 6 for 26 options) and can make them indistinguishable (Laya README, "Honest limits"). This app avoids that by passing the options as
**bare letters** (`criteria: ["A", "B", ...]`, each rendered as just the letter), keeping the instruction short, and putting game facts in the
state. It also passes `max_len=512, head_max_len=192` explicitly and, **at every startup**, runs real predictions on two worst-case synthetic inputs
(26 options with a 30-letter word; a long guess history) and refuses to become "ready" if Laya's `usage` reports truncation or collapsed options, or
if `build_head` reports fewer than 26 distinct option spans. Raise `LAYA_HEAD_MAX_LEN` / `LAYA_MAX_LEN` if that check fails on your setup.
The check ran here only with a tiny stand-in tokenizer, **not** with the real ModernBERT tokenizer; the real result appears in `/api/status -> model.budget` once the model loads.

Laya also documents that option *position* biases English-checkpoint answers and that the base checkpoints are close to chance zero-shot on
typed decisions ("a fast base to specialise, not a zero-shot decision engine"). Expect Laya to play poorly until fine-tuned; this is an experiment, and
the evaluation below exists to measure that honestly. No option shuffling/averaging is applied (kept simple; see `option_order` in Laya's docs).

## Using the app

Enter a word (masked input, cleared after starting) or click **Random word**, choose who guesses (Laya or the baseline), then **Next Guess**, **Auto Play**,
**Pause** or **New Game**. Pause prevents any further turn from being scheduled; a guess already running finishes and is shown. The page shows the
pattern, a Hangman drawing, remaining mistakes, the selected letter, the measured inference time, Laya's per-letter probabilities and its
`confidence` / `answer_confidence` fields (model-reported scores, not guarantees), a turn history, and **What Laya sees**: the exact request passed
to `Agent.predict`, without the secret. No natural-language "reasoning" is shown, because Laya produces none.

Engineering guarantees: game state and secret live server-side; each game has its own session; a second overlapping turn on a game gets HTTP 409; a response
that arrives after New Game (or after the session is closed) is discarded server-side and ignored client-side; inference runs in worker threads, one forward
pass at a time; Auto Play stops on game end or any error; the server binds to localhost and rejects foreign `Host` headers.
Failures (model not ready 503, invalid model output / inference exception 502) show an explicit error with Retry and **never** produce a guessed letter.

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest                      # 126 tests, mocked inference + tiny random-weight real laya.Agent
.venv/bin/python tests/e2e/browser_flow.py      # Playwright/Chromium flow against a labelled MOCK Laya (set CHROMIUM=path if needed)
LAYA_REAL_MODEL=1 .venv/bin/python -m pytest -m real_model -v -s    # REAL checkpoint smoke test (downloads weights)
```

Covered: repeated-letter reveal, win/loss, invalid words/guesses, repeats, no secret in requests (including a non-interference property test: different secrets with the same visible
history give byte-identical requests), invalid/failed model responses, stale and concurrent turns, device fallback, token-budget failure handling, baseline determinism, evaluation split.

## Evaluation

```bash
.venv/bin/python -m laya_hangman.evaluate --split eval --deciders baseline laya --out results/eval.json --progress
```

Same games and rules for both deciders on the fixed **evaluation split** (232 words in `evaluation/eval_words.txt`: every word whose SHA-256 is divisible by 5). Deciders only receive a
`DecisionInput`; answers are never in a request. The report lists win rate, mean incorrect guesses, mean total guesses, Laya's mean/median/p95 inference latency, and the model/runtime configuration.
If Laya cannot load or returns an invalid response the run aborts; nothing is substituted. The prompt and budgets were fixed before any evaluation and **no tuning was done on the evaluation split**
(use the other 834 "dev" words, `--split dev`, for any tuning).

### Measured results

Baseline on the full evaluation split (232 words, 6 incorrect guesses allowed), measured in the dev sandbox (`results/baseline-eval.json`):
win rate 10.8% (25/232), mean incorrect 5.82, mean total guesses 9.27.

Side by side on the **first 20 words of the evaluation split**, real Laya inference (`convaiinnovations/laya`, laya 0.4.6, transformers 4.57.6,
torch 2.2.2, Python 3.11, **Intel Mac, CPU**, `max_len=512`, `head_max_len=192`), run by the repository owner:

| Decider | Games | Win rate | Mean incorrect | Mean total guesses | Mean inference (ms) |
|---|---:|---:|---:|---:|---:|
| Frequency baseline | 20 | 10.0% (2/20) | 5.80 | 9.15 | n/a |
| Laya | 20 | 0.0% (0/20) | 6.00 | 7.30 | 217.5 |

Laya lost every game and found about 1.3 correct letters per game. With only 20 games this is not a precise estimate, but it shows no sign
that Laya beats the baseline, and **no claim to the contrary is made**. The full 232-word Laya run was not done.

First-turn diagnostic (real model, 26 options): the probabilities are not uniform (top letter about 0.10 against 1/26 = 0.038), but the preferred
letters are N, R, G, V, Y, Z, J rather than E, T, A, O, I. For "cat", "elephant" and "extraordinary" the top letters were N, N and G. The model reacts to the input
but has not learned a useful Hangman strategy. Why is not established by this experiment. Plausible causes (unverified): the decision head was never trained for this task, and the
state is given as a text pattern with underscores. Laya also prints a warning on load that its `choice:11+` temperature is out of range, so its `confidence` values are uncalibrated for 11+ options.

Not tried: option-order averaging, per-letter descriptions, and fine-tuning on Hangman (Laya ships fine-tuning scripts; see its docs). Fine-tuning is the most likely way to make Laya play well.

## Word list

`src/laya_hangman/data/words.txt` is an **original, hand-written list of 1,066 common English words (4-12 letters)** created for this project and released under this repo's MIT license; no third-party list is
redistributed. Spelling of every word was checked once against the public-domain ENABLE word list (`enable1.txt`); five candidates not in ENABLE (names/compounds) were dropped. It is not a frequency-validated list:
"common" is the author's judgement. Random words are drawn from the whole list; evaluation uses the fixed split described above. Words typed by the user may be any A-Z word.

## Troubleshooting

* **State `error`: ProxyError / 403 / connection errors** - the checkpoint cannot be downloaded. Allow `huggingface.co`, or download the checkpoint elsewhere and set `LAYA_MODEL=/path/to/checkpoint`; then press Retry. Offline use works once cached (`HF_HUB_OFFLINE=1`). This is exactly what happens in the sandbox this was built in.
* **Fell back to CPU** - the status panel shows the accelerator error. Force CPU with `LAYA_DEVICE=cpu`, or try `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported MPS ops.
* **"options collapsed" / "truncated"** at startup - raise `LAYA_HEAD_MAX_LEN` (e.g. 256) and/or `LAYA_MAX_LEN` (e.g. 1024); the English checkpoint's native limit is 512, so values above that are unverified.
* **Out of memory** - close other apps or use a smaller `LAYA_MAX_LEN`.
* **Blank page / 400** - open it as `http://127.0.0.1:8765` or `http://localhost:8765` (other Host headers are rejected).

## Verification: what was and was not checked

| Item | Status |
|---|---|
| Game rules, session/concurrency logic, API, validation of Laya-shaped results, evaluation harness | **Passed with mocks** (126 automated tests) |
| Laya's real `Agent.predict` path, result schema (`answers.next_letter.choice/probabilities/confidence/answer_confidence`, `usage.truncated`), 26-option budget logic | **Passed with a tiny random-weight stand-in** using the installed laya 0.4.6 (not the real checkpoint/tokenizer) |
| Browser flow (invalid word, masked input, next guess, auto play to the end, pause, New Game with a late response, error + retry, baseline label, 375px layout) | **Passed against a labelled mock Laya** (a bug where Auto Play stopped itself during inference was found and fixed this way) |
| The real app started with the real loader | **Verified**: shows `error` with the 403 message and hint, Laya turns return HTTP 503, baseline turns work |
| Hardware detection (CUDA/MPS/CPU) | CPU path observed; **MPS/CUDA paths untested** (no such hardware) |
| Real checkpoint download, load and inference on an Intel Mac CPU (reported by the owner); 20-game comparison above | **Verified on the owner's machine** (not reproduced by the author of this code) |
| Real-tokenizer token-budget report (`/api/status -> model.budget`) | Startup check passed (the app reached `ready`); the numbers were not captured |
| Full 232-word Laya evaluation, Apple Silicon (MPS) path, real-model pytest smoke test | **Not run** |

## License

MIT for this repository. Laya is Apache-2.0 and is installed from PyPI, not vendored.
