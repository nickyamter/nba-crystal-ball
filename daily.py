# daily.py
# The once-a-day job: refresh stats, score yesterday's lines and picks, pull
# today's lines (inside the free limits), log today's picks for every weight
# combination, and rebuild dashboard.html.
#
#   .venv/bin/python daily.py           runs unless today's work is already finished
#   .venv/bin/python daily.py --force   runs again regardless
#
# Safe to start several times a day: each step only does what is still missing.

import fcntl
import sys
import traceback
from datetime import datetime

import budget
from common import BASE_DIR, load_state, save_state

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "daily.log"


class _Tee:
    """Send everything printed to both the terminal and logs/daily.log."""
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run(force: bool = False) -> int:
    today = budget.today_eastern().isoformat()
    if load_state().get("last_complete_date") == today and not force:
        print("Already finished today's run.")
        return 0

    import dashboard
    import model
    import BRscrape1_2 as scraper

    failed = False
    try:
        scraper.main()
        scored = model.score_experiments(today)
        print(f"Scored {scored} earlier exploration picks → WEIGHT_HITRATE.csv")
        logged = model.generate_experiments(model.build_players(), today)
        print(f"Logged {logged} exploration picks for {today}")
    except Exception:
        failed = True
        traceback.print_exc()

    try:
        print(f"Dashboard → {dashboard.build()}")
    except Exception:
        failed = True
        traceback.print_exc()

    # the day is finished once the odds step has nothing left to try
    odds_today = budget.load_usage()["odds"].get("today") or {}
    state = load_state()
    state["last_run_at"] = datetime.now(budget.EASTERN).isoformat(timespec="seconds")
    if not failed and odds_today.get("date") == today and odds_today.get("final"):
        state["last_complete_date"] = today
    save_state(state)
    return 1 if failed else 0


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    lock = open(LOG_DIR / "daily.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another daily run is in progress.")
        return 0

    with open(LOG_PATH, "a") as log:
        sys.stdout = sys.stderr = _Tee(sys.__stdout__, log)
        print(f"\n=== Run started {datetime.now(budget.EASTERN).strftime('%Y-%m-%d %H:%M:%S %Z')} ===")
        code = run(force="--force" in sys.argv)
        print("=== Run finished ===" if code == 0 else "=== Run failed ===")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    return code


if __name__ == "__main__":
    sys.exit(main())
