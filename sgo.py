# sgo.py
# DraftKings player prop lines from SportsGameOdds (free "amateur" plan).
# One request returns today's games with points, rebounds and assists lines, and
# each game counts as one object against the plan's 2,500 per 30 days, however
# many lines it carries. Every request is cleared by budget.py first.

from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests

import budget
from common import SGO_API_KEY, clean_player_name

EVENTS_URL = "https://api.sportsgameodds.com/v2/events"
EASTERN = pytz.timezone("US/Eastern")
BOOKMAKER = "draftkings"
# their stat names -> the market names used in ODDS_TODAY.csv
STAT_TO_MARKET = {"points": "player_points", "rebounds": "player_rebounds", "assists": "player_assists"}
MAX_GAMES_PER_REQUEST = 20  # an NBA day has at most 15 games; this caps what one request can cost


def _decimal_price(american) -> float | None:
    """'+105' -> 2.05, '-111' -> 1.9 (the price format ODDS_TODAY.csv already uses)"""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return round(1 + (a / 100 if a > 0 else 100 / -a), 2)


def _name_from_id(player_id: str) -> str:
    """'JAYSON_TATUM_1_NBA' -> 'Jayson Tatum' (only used if the players list lacks a name)"""
    parts = [p for p in str(player_id).split("_") if p and not p.isdigit() and p != "NBA"]
    return " ".join(p.capitalize() for p in parts)


def parse_events(events: list, today, now=None, bookmaker: str = BOOKMAKER) -> tuple[pd.DataFrame, int, int]:
    """Rows shaped like ODDS_TODAY.csv, how many games had at least one line, and
    how many of today's games had not started yet."""
    now = now or datetime.now(pytz.utc)
    rows, games_with_lines, games_open = [], 0, 0
    for e in events:
        try:
            tip_utc = datetime.fromisoformat(e["status"]["startsAt"].replace("Z", "+00:00"))
            tip_eastern = tip_utc.astimezone(EASTERN)
        except Exception:
            continue
        if tip_eastern.date() != today:
            continue
        # a game that has tipped off would carry live in-game odds, not the pregame line
        if e.get("status", {}).get("started") or tip_utc <= now:
            continue
        games_open += 1
        players = e.get("players") or {}
        teams = e.get("teams") or {}
        before = len(rows)
        for odd in (e.get("odds") or {}).values():
            market = STAT_TO_MARKET.get(odd.get("statID"))
            book = (odd.get("byBookmaker") or {}).get(bookmaker)
            if (
                market is None or book is None or not book.get("available")
                or odd.get("betTypeID") != "ou" or odd.get("periodID") != "game"
            ):
                continue
            line = pd.to_numeric(book.get("overUnder"), errors="coerce")
            player_id = odd.get("playerID") or odd.get("statEntityID")
            if pd.isna(line) or not player_id:
                continue
            name = (players.get(player_id) or {}).get("name") or _name_from_id(player_id)
            rows.append({
                "event_id": e.get("eventID"),
                "game_start_utc": tip_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "game_start_eastern": tip_eastern.isoformat(),
                "game_date_eastern": tip_eastern.date().isoformat(),
                "home_team": ((teams.get("home") or {}).get("names") or {}).get("long"),
                "away_team": ((teams.get("away") or {}).get("names") or {}).get("long"),
                "bookmaker": bookmaker,
                "market": market,
                "player_name": name,
                "player_name_clean": clean_player_name(name),
                "line": float(line),
                "price": _decimal_price(book.get("odds")),
            })
        games_with_lines += int(len(rows) > before)
    return pd.DataFrame(rows), games_with_lines, games_open


def fetch_player_props_today() -> tuple[pd.DataFrame, dict]:
    """Today's DraftKings lines. Returns (rows, status); status["ok"] is False when
    the service could not or must not be used, so the caller can fall back."""
    today = budget.today_eastern()
    status = {
        "date": today.isoformat(), "source": "sgo", "ok": False,
        "events": 0, "pulled": 0, "credits": 0, "note": "", "final": False,
    }
    if not SGO_API_KEY:
        status["note"] = "no API key"
        return pd.DataFrame(), status

    usage = budget.load_usage()
    account = budget.sgo_account()
    allowed, reason = budget.plan_sgo_pull(usage, account, MAX_GAMES_PER_REQUEST, today)
    budget.record_sgo_account(usage, account)
    print(f"SportsGameOdds allowance left: {usage['sgo'].get('remaining')} ({reason}).")
    if not allowed:
        status["note"] = reason
        # lines already pulled today is not a failure, and must not trigger the fallback
        status["ok"] = reason == "already pulled today"
        status["final"] = status["ok"]
        return pd.DataFrame(), status

    day_start = EASTERN.localize(datetime(today.year, today.month, today.day))
    now = datetime.now(pytz.utc)
    params = {
        "leagueID": "NBA",
        "oddsAvailable": "true",
        "bookmakerID": BOOKMAKER,
        "oddIDs": ",".join(f"{stat}-PLAYER_ID-game-ou-over" for stat in STAT_TO_MARKET),
        "includeOpposingOdds": "true",
        "startsAfter": max(day_start.astimezone(pytz.utc), now).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "startsBefore": (day_start + timedelta(days=1)).astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": MAX_GAMES_PER_REQUEST,
    }
    try:
        resp = requests.get(EVENTS_URL, headers={"X-Api-Key": SGO_API_KEY}, params=params, timeout=30)
    except Exception as e:
        # unknown whether it was counted: book the most it could have cost
        budget.record_sgo_pull(usage, MAX_GAMES_PER_REQUEST, False, today)
        status["note"] = f"request failed: {e}"
        return pd.DataFrame(), status

    try:
        resp.raise_for_status()
        events = resp.json().get("data") or []
    except Exception as e:
        budget.record_sgo_pull(usage, MAX_GAMES_PER_REQUEST, False, today)
        status["note"] = f"request failed: {e}"
        return pd.DataFrame(), status

    df, games_with_lines, games_open = parse_events(events, today, now)
    cost = max(1, len(events))  # every response counts as at least one object
    # done for the day once every game still to play has lines, or, late in the
    # day, once any do (a later run tries again for the missing ones)
    late = now.astimezone(EASTERN).hour >= budget.LINES_GIVE_UP_HOUR
    complete = games_with_lines > 0 and (games_with_lines >= games_open or late)
    budget.record_sgo_pull(usage, cost, complete, today)
    budget.record_sgo_account(usage, budget.sgo_account())

    status.update({
        "ok": True, "events": games_open, "pulled": games_with_lines, "credits": cost,
        "note": "ok" if complete else ("some games have no lines yet" if games_with_lines else "no lines posted yet"),
        "final": complete,
    })
    print(f"SportsGameOdds: {len(events)} games, {games_with_lines} with DraftKings lines, "
          f"{cost} objects used; {usage['sgo'].get('remaining')} left.")
    return df, status
