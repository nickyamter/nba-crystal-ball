# budget.py
# Hard spending limits for the two keyed APIs (The Odds API, RapidAPI injuries).
# Every call that can use up quota is cleared here first, and anything uncertain
# (no answer from the provider, missing quota headers, a second pull on the same
# day) is treated as "do not call".

import calendar
import json
import os
import random
from datetime import datetime, timedelta

import pytz
import requests

from common import data_path, ODDS_API_KEY, SGO_API_KEY

EASTERN = pytz.timezone("US/Eastern")
USAGE_PATH = data_path("API_USAGE.json")

# ----- LIMITS -----
ODDS_FREE_CREDITS = 500      # The Odds API free plan, per month
ODDS_RESERVE = 10            # credits that are never spent
ODDS_SPORTS_URL = "https://api.the-odds-api.com/v4/sports"  # free call, reports credits left

# SportsGameOdds free plan: one object per game returned, 2,500 per rolling 30 days.
# Nothing on this plan is billed by usage (past the limit, requests are refused),
# so these limits exist to keep well inside it and to notice if the plan changes.
SGO_FREE_OBJECTS = 2500
SGO_RESERVE = 500            # objects that are never used
SGO_FREE_TIER = "amateur"
SGO_MAX_PULLS_PER_DAY = 3    # retries while DraftKings has not posted every game's lines yet

# Eastern hour after which a day's lines are taken as they are: partial lines are
# accepted, and no lines at all sends the pull to the fallback service.
LINES_GIVE_UP_HOUR = 14
SGO_USAGE_URL = "https://api.sportsgameodds.com/v2/account/usage"  # free call

# The injury report is not used by the predictions, and RapidAPI is the one service
# here that can bill overages if a card is on file, so it stays off unless .env
# has INJURY_PULL_ENABLED=1.
INJURY_PULL_ENABLED = os.environ.get("INJURY_PULL_ENABLED", "0") == "1"
INJURY_RESERVE_SHARE = 0.10  # stop once only this share of the plan's requests is left
INJURY_RESERVE_MIN = 5


def today_eastern():
    return datetime.now(EASTERN).date()


# =========================================================
# LEDGER (CSVs/API_USAGE.json)
# =========================================================
def load_usage() -> dict:
    try:
        with open(USAGE_PATH) as f:
            usage = json.load(f)
    except Exception:
        usage = {}
    usage.setdefault("odds", {})
    usage["odds"].setdefault("months", {})
    usage.setdefault("injury", {})
    usage.setdefault("sgo", {})
    usage["sgo"].setdefault("days", {})
    return usage


def save_usage(usage: dict) -> None:
    tmp = str(USAGE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(usage, f, indent=2)
    os.replace(tmp, USAGE_PATH)


def _month_key(day) -> str:
    return day.strftime("%Y-%m")


def _to_int(val):
    try:
        return int(float(val))
    except Exception:
        return None


# =========================================================
# ODDS
# =========================================================
def odds_credits_remaining():
    """Ask The Odds API how many credits are left. This call itself is free.
    Returns None if the answer can't be read."""
    if not ODDS_API_KEY:
        return None
    try:
        resp = requests.get(ODDS_SPORTS_URL, params={"apiKey": ODDS_API_KEY}, timeout=15)
        if resp.status_code != 200:
            return None
        return _to_int(resp.headers.get("x-requests-remaining"))
    except Exception:
        return None


def plan_odds_pull(usage: dict, n_events: int, remaining, base_cost: int, extra_cost: int,
                   games_left_in_month=None, today=None):
    """What may be pulled today. Returns (games, games_with_extra, reason).

    Two independent locks: the provider's own count of credits left, and this
    project's own monthly tally. The smaller one wins, minus the reserve.

    Every game today gets the base stats (points, rebounds) if the credits allow.
    The extra stat (assists) is only added from credits that the base stats for
    the rest of the month's games will not need, spread evenly over those games.
    Without a schedule to count from, no extras are pulled."""
    today = today or today_eastern()
    odds = usage["odds"]

    if odds.get("last_pull_date") == today.isoformat():
        return 0, 0, "already pulled today"
    if n_events <= 0:
        return 0, 0, "no games today"
    if remaining is None:
        return 0, 0, "could not confirm credits left"

    spent = odds["months"].get(_month_key(today), {}).get("credits", 0)
    spendable = min(remaining, ODDS_FREE_CREDITS - spent) - ODDS_RESERVE
    if spendable < base_cost:
        return 0, 0, "monthly credit limit reached"

    games = min(n_events, spendable // base_cost)
    extra = 0
    if games_left_in_month is not None:
        games_left = max(games_left_in_month, n_events)
        spare = spendable - games_left * base_cost
        if spare > 0:
            share = min(1.0, spare / (games_left * extra_cost))
            extra = min(games, int(share * n_events + 1e-9))
    return int(games), int(extra), "ok"


def choose_events(events: list, n: int, today=None) -> list:
    """Which n of today's games to pull. Seeded by the date so a rerun picks the
    same games, and so the choice isn't always the earliest (East-coast) tip-offs."""
    today = today or today_eastern()
    if n >= len(events):
        return list(events)
    ordered = sorted(events, key=lambda e: str(e.get("id")))
    return random.Random(today.isoformat()).sample(ordered, n)


def can_spend_odds(usage: dict, cost: int, today=None) -> bool:
    """Checked again before every single odds request."""
    today = today or today_eastern()
    odds = usage["odds"]
    remaining = odds.get("remaining")
    if remaining is None:
        return False
    spent = odds["months"].get(_month_key(today), {}).get("credits", 0)
    return min(remaining, ODDS_FREE_CREDITS - spent) - ODDS_RESERVE >= cost


def record_odds_request(usage: dict, headers, max_cost: int, today=None) -> bool:
    """Book one odds request against the ledger. Returns False if the provider did
    not report its counters, in which case the full cost is assumed and the caller
    must stop pulling."""
    today = today or today_eastern()
    odds = usage["odds"]
    month = odds["months"].setdefault(_month_key(today), {"credits": 0, "requests": 0})

    cost = _to_int(headers.get("x-requests-last"))
    remaining = _to_int(headers.get("x-requests-remaining"))
    reported = cost is not None and remaining is not None
    if cost is None:
        cost = max_cost
    if remaining is None:
        remaining = (odds.get("remaining") or 0) - cost

    month["credits"] += cost
    month["requests"] += 1
    odds["remaining"] = remaining
    save_usage(usage)
    return reported


# =========================================================
# SPORTSGAMEODDS
# =========================================================
def sgo_account():
    """Ask SportsGameOdds for the plan and this month's usage. Free.
    Returns {"tier", "max", "used"} or None if the answer can't be read."""
    if not SGO_API_KEY:
        return None
    try:
        resp = requests.get(SGO_USAGE_URL, headers={"X-Api-Key": SGO_API_KEY}, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()["data"]
        month = data["rateLimits"]["per-month"]
        return {
            "tier": str(data.get("tier", "")).lower(),
            "max": _to_int(month.get("max-entities")),
            "used": _to_int(month.get("current-entities")),
        }
    except Exception:
        return None


def record_sgo_account(usage: dict, account) -> None:
    sgo = usage["sgo"]
    if account is not None:
        sgo["tier"] = account["tier"]
        sgo["max"] = account["max"]
        sgo["used"] = account["used"]
        if account["max"] is not None and account["used"] is not None:
            sgo["remaining"] = min(account["max"], SGO_FREE_OBJECTS) - account["used"]
        sgo["checked_at"] = datetime.now(EASTERN).isoformat(timespec="seconds")
    save_usage(usage)


def plan_sgo_pull(usage: dict, account, max_cost: int, today=None):
    """May SportsGameOdds be asked for today's lines? Returns (allowed, reason).
    Anything unclear is a no, and the caller falls back to The Odds API."""
    today = today or today_eastern()
    sgo = usage["sgo"]
    day = sgo["days"].get(today.isoformat(), {})

    if day.get("complete"):
        return False, "already pulled today"
    if day.get("pulls", 0) >= SGO_MAX_PULLS_PER_DAY:
        return False, "tried enough times today"
    if account is None:
        return False, "could not confirm the allowance left"
    if account["tier"] != SGO_FREE_TIER:
        return False, f"the account is on the '{account['tier']}' plan, not the free one"
    if account["max"] is None or account["used"] is None:
        return False, "the service did not report an allowance"

    # two locks: the provider's count, and this project's own tally of the last 30 days
    since = (today - timedelta(days=30)).isoformat()
    own_tally = sum(d.get("objects", 0) for date, d in sgo["days"].items() if date > since)
    allowance = min(account["max"], SGO_FREE_OBJECTS)
    left = allowance - max(account["used"], own_tally) - SGO_RESERVE
    if left < max_cost:
        return False, "monthly allowance reached"
    return True, "ok"


def record_sgo_pull(usage: dict, objects: int, complete: bool, today=None) -> None:
    today = today or today_eastern()
    sgo = usage["sgo"]
    day = sgo["days"].setdefault(today.isoformat(), {"pulls": 0, "objects": 0, "complete": False})
    day["pulls"] += 1
    day["objects"] += objects
    day["complete"] = bool(day.get("complete") or complete)
    # keep about six weeks of history
    cutoff = (today - timedelta(days=45)).isoformat()
    sgo["days"] = {d: v for d, v in sgo["days"].items() if d > cutoff}
    save_usage(usage)


# =========================================================
# INJURIES
# =========================================================
def injury_pull_allowed(usage: dict, now=None):
    """Returns (allowed, reason)."""
    now = now or datetime.now(EASTERN)
    inj = usage["injury"]

    if not INJURY_PULL_ENABLED:
        return False, "off (set INJURY_PULL_ENABLED=1 in .env to turn on)"
    if inj.get("last_pull_date") == now.date().isoformat():
        return False, "already pulled today"

    reset_at = inj.get("reset_at")
    if reset_at and now >= datetime.fromisoformat(reset_at):
        return True, "ok"
    if inj.get("quota_unknown"):
        return False, "the provider did not report a quota, so it is not safe to continue"
    limit, remaining = inj.get("limit"), inj.get("remaining")
    if limit is not None and remaining is not None:
        reserve = max(INJURY_RESERVE_MIN, int(limit * INJURY_RESERVE_SHARE))
        if remaining <= reserve:
            return False, "monthly request limit reached"
    return True, "ok"


def record_injury_request(usage: dict, headers, now=None) -> None:
    now = now or datetime.now(EASTERN)
    inj = usage["injury"]
    inj["last_pull_date"] = now.date().isoformat()

    limit = _to_int(headers.get("x-ratelimit-requests-limit"))
    remaining = _to_int(headers.get("x-ratelimit-requests-remaining"))
    reset_sec = _to_int(headers.get("x-ratelimit-requests-reset"))
    inj["quota_unknown"] = limit is None or remaining is None
    inj["limit"] = limit
    inj["remaining"] = remaining
    if reset_sec is not None:
        inj["reset_at"] = (now + timedelta(seconds=reset_sec)).isoformat()
    elif inj["quota_unknown"]:
        # no information at all: stay blocked until the start of next month
        first_next = (now.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        inj["reset_at"] = first_next.isoformat()
    save_usage(usage)
