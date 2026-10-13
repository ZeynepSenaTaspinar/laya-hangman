"""Browser flow check with Playwright + Chromium against the MOCK-Laya server (see serve_mock.py).

    python tests/e2e/browser_flow.py [screenshot_dir]

Everything here exercises the real UI, real HTTP API and real game logic; only the Laya model is a
labelled test double. Exits non-zero on the first failed expectation.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

HERE = Path(__file__).parent
SHOTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)
CHROMIUM = os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium")


def start_server(port, delay, fail_after=None):
    args = [sys.executable, str(HERE / "serve_mock.py"), str(port), str(delay)]
    if fail_after is not None:
        args.append(str(fail_after))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1)
            return proc
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("mock server did not start")


def wait_ready(page):
    expect(page.locator("#laya-chip-text")).to_contain_text("ready", timeout=15000)


def start_game(page, word):
    page.fill("#word", word)
    page.click("#start-word")
    expect(page.locator("#game")).to_be_visible()


def n_rows(page):
    return page.locator("#history tr").count()


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        raise SystemExit(1)


def main():
    p1, p2, p3 = start_server(18765, 0.15), start_server(18766, 1.0), start_server(18767, 0.1, fail_after=2)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto("http://127.0.0.1:18765/")
            wait_ready(page)
            check("status chip labels the test double", "TEST DOUBLE" in page.inner_text("#laya-chip-text"))

            # invalid input is rejected clearly and no game starts
            page.fill("#word", "ab1")
            page.click("#start-word")
            expect(page.locator("#word-error")).to_contain_text("Only the English letters", timeout=5000)
            check("invalid word shows an error and no game", not page.locator("#game").is_visible())

            # secret entry is masked and disappears after starting
            check("secret input is a password field", page.get_attribute("#word", "type") == "password")
            start_game(page, "banana")
            check("input cleared after start", page.input_value("#word") == "")
            check("secret not visible anywhere in the page", "BANANA" not in page.inner_text("body").upper())
            check("pattern shows blanks only", page.inner_text("#pattern").replace(" ", "") == "______")

            # one manual guess
            page.click("#next")
            expect(page.locator("#history tr")).to_have_count(1, timeout=5000)
            check("history source is Laya (mock)", "Laya" in page.inner_text("#history tr:first-child"))
            check("inference duration shown", "ms" in page.inner_text("#d-latency"))
            check("probabilities listed", page.locator("#probs .prob").count() >= 20)
            page.click("#sees summary")
            sees = page.inner_text("#sees-json")
            check("'What Laya sees' shows the request without the secret",
                  '"questions"' in sees and "BANANA" not in sees.upper())
            page.screenshot(path=str(SHOTS / "after_first_guess.png"), full_page=True)

            # auto play to the end, then it must stop on its own
            page.click("#auto")
            expect(page.locator("#outcome")).to_contain_text("The word was BANANA", timeout=30000)
            check("auto play stopped at game end",
                  page.locator("#pause").is_disabled() and not page.locator("#next").is_enabled())
            page.screenshot(path=str(SHOTS / "finished.png"), full_page=True)

            # pause prevents further turns
            page.click("#new")
            start_game(page, "xylophone")
            page.click("#auto")
            expect(page.locator("#history tr")).not_to_have_count(0, timeout=5000)
            page.click("#pause")
            time.sleep(0.6)  # let a possibly in-flight turn land
            rows = n_rows(page)
            time.sleep(1.5)
            check("pause schedules no further turns", n_rows(page) == rows)
            check("next guess works again after pause", page.locator("#next").is_enabled())

            # New Game while a slow inference is in flight: the late response must not touch the new game
            page2 = browser.new_page(viewport={"width": 1100, "height": 900})
            page2.goto("http://127.0.0.1:18766/")
            wait_ready(page2)
            start_game(page2, "planet")
            page2.click("#auto")
            time.sleep(0.3)
            page2.click("#new")
            start_game(page2, "moon")
            time.sleep(2.5)  # the old inference finishes meanwhile
            check("stale response ignored by the new game",
                  n_rows(page2) == 0 and "0 of 6" in page2.inner_text("#mistakes"))
            page2.close()

            # inference failure: Auto Play stops, explicit error + retry, nothing is faked
            page3 = browser.new_page(viewport={"width": 1100, "height": 900})
            page3.goto("http://127.0.0.1:18767/")
            wait_ready(page3)
            start_game(page3, "planet")
            page3.click("#auto")
            expect(page3.locator("#error")).to_be_visible(timeout=10000)
            check("error text is explicit", "mock inference failure" in page3.inner_text("#error-text"))
            rows = n_rows(page3)
            time.sleep(1.0)
            check("auto play stopped on error", n_rows(page3) == rows and page3.locator("#pause").is_disabled())
            check("retry button available", page3.locator("#error-retry").is_visible())
            page3.screenshot(path=str(SHOTS / "error.png"), full_page=True)
            page3.close()

            # baseline mode is separately labelled
            page.click("#new")
            page.check("input[value=baseline]")
            start_game(page, "quiz")
            page.click("#next")
            expect(page.locator("#history tr")).to_have_count(1, timeout=5000)
            check("baseline turns are labelled 'not Laya'", "not Laya" in page.inner_text("#history tr:first-child"))

            # phone-sized viewport: no horizontal page scroll
            page.set_viewport_size({"width": 375, "height": 800})
            check("no horizontal scroll at 375px",
                  page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))
            page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
            browser.close()
        print("ALL BROWSER CHECKS PASSED (mock Laya)")
    finally:
        for proc in (p1, p2, p3):
            proc.kill()


if __name__ == "__main__":
    main()
