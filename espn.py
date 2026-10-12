# espn.py
# Player stats from ESPN's public scoreboard and box score feed, shaped like the
# NBA stats frames the scraper already uses. This is the source that GitHub's
# servers can reach (stats.nba.com times out from there).
#
# Box scores are fetched once per game and kept in CSVs/ESPN_<season>_LOG.csv,
# so a daily run only asks for the games played since the last run.

import time
from datetime import datetime

import pandas as pd
import pytz
import requests

from common import data_path, load_state, save_state

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_SEC = 0.15
EASTERN = pytz.timezone("US/Eastern")

PRESEASON, REGULAR_SEASON = 1, 2
NBA_TEAM_IDS = set(range(1, 31))  # ESPN's ids for the 30 NBA teams (All-Star teams have others)
# ESPN's short names -> the abbreviations used everywhere else in this project
NBA_ABBR = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}

LOG_COLS = [
    "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "MATCHUP", "MIN", "PTS", "REB", "AST", "STL", "BLK",
]
STAT_KEYS = {"MIN": "minutes", "PTS": "points", "REB": "rebounds", "AST": "assists", "STL": "steals", "BLK": "blocks"}

_events_cache: dict[str, pd.DataFrame] = {}
_log_cache: dict[str, pd.DataFrame] = {}


def season_short(season: str) -> str:
    """'2024-25' -> '24-25'"""
    return f"{season[2:4]}-{season[-2:]}"


def _get(path: str, params: dict) -> dict:
    last_error = None
    for attempt in range(3):
        try:
            time.sleep(SLEEP_SEC)
            resp = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def _abbr(team: dict) -> str:
    a = team.get("abbreviation", "")
    return NBA_ABBR.get(a, a)


# =========================================================
# SCHEDULE
# =========================================================
def season_events(season: str) -> pd.DataFrame:
    """Every game between NBA teams on ESPN's calendar for a season ('2025-26'),
    past and future, one row per game."""
    if season in _events_cache:
        return _events_cache[season]

    start = int(season[:4])
    months = [f"{start}{m:02d}" for m in (10, 11, 12)] + [f"{start + 1}{m:02d}" for m in range(1, 7)]
    rows = {}
    for month in months:
        for e in _get("scoreboard", {"dates": month, "limit": 500}).get("events", []):
            comp = e["competitions"][0]
            sides = {c["homeAway"]: c["team"] for c in comp["competitors"]}
            if "home" not in sides or "away" not in sides:
                continue
            home_id, away_id = int(sides["home"]["id"]), int(sides["away"]["id"])
            if home_id not in NBA_TEAM_IDS or away_id not in NBA_TEAM_IDS:
                continue
            tip = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(EASTERN)
            notes = " ".join(n.get("headline", "") for n in comp.get("notes", []))
            rows[str(e["id"])] = {
                "GAME_ID": str(e["id"]),
                "GAME_DATE": tip.date().isoformat(),
                "SEASON_TYPE": int(e["season"]["type"]),
                "FINAL": e["status"]["type"]["name"] == "STATUS_FINAL",
                "HOME_TEAM_ID": home_id,
                "HOME_ABBR": _abbr(sides["home"]),
                "VISITOR_TEAM_ID": away_id,
                "VISITOR_ABBR": _abbr(sides["away"]),
                "NOTES": notes,
            }
    events = pd.DataFrame(list(rows.values()), columns=[
        "GAME_ID", "GAME_DATE", "SEASON_TYPE", "FINAL",
        "HOME_TEAM_ID", "HOME_ABBR", "VISITOR_TEAM_ID", "VISITOR_ABBR", "NOTES",
    ]).sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    _events_cache[season] = events
    return events


def _counted_games(events: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games that count in the standings. The NBA Cup championship
    game is listed with the regular season but does not count toward season stats
    (the Cup's group, quarterfinal and semifinal games do)."""
    is_cup_final = events["NOTES"].str.contains("Cup Championship", case=False)
    return events[(events["SEASON_TYPE"] == REGULAR_SEASON) & ~is_cup_final]


def schedule(season: str) -> pd.DataFrame:
    """Same shape as the scraper's _fetch_schedule."""
    events = season_events(season)
    return pd.DataFrame({
        "GAME_ID": events["GAME_ID"],
        "GAME_DATE": pd.to_datetime(events["GAME_DATE"]).dt.tz_localize("US/Eastern"),
        "HOME_TEAM_ID": events["HOME_TEAM_ID"],
        "VISITOR_TEAM_ID": events["VISITOR_TEAM_ID"],
        "COUNTS_FOR_ODDS": events["SEASON_TYPE"] != PRESEASON,
    }).sort_values("GAME_DATE", kind="stable").reset_index(drop=True)


# =========================================================
# BOX SCORES
# =========================================================
def _box_score_rows(game) -> list[dict]:
    summary = _get("summary", {"event": game["GAME_ID"]})
    rows = []
    for team in summary.get("boxscore", {}).get("players", []):
        team_id = int(team["team"]["id"])
        is_home = team_id == game["HOME_TEAM_ID"]
        abbr = game["HOME_ABBR"] if is_home else game["VISITOR_ABBR"]
        opp = game["VISITOR_ABBR"] if is_home else game["HOME_ABBR"]
        matchup = f"{abbr} vs. {opp}" if is_home else f"{abbr} @ {opp}"
        for block in team.get("statistics", [])[:1]:
            keys = block.get("keys", [])
            if not all(k in keys for k in STAT_KEYS.values()):
                continue
            for a in block.get("athletes", []):
                values = a.get("stats") or []
                athlete = a.get("athlete") or {}
                # skip players who sat out, and the occasional placeholder entry with no id
                if a.get("didNotPlay") or len(values) != len(keys) or "id" not in athlete:
                    continue
                row = {
                    "GAME_ID": game["GAME_ID"],
                    "GAME_DATE": game["GAME_DATE"],
                    "PLAYER_ID": int(athlete["id"]),
                    "PLAYER_NAME": athlete.get("displayName", ""),
                    "TEAM_ID": team_id,
                    "TEAM_ABBREVIATION": abbr,
                    "MATCHUP": matchup,
                }
                for col, key in STAT_KEYS.items():
                    row[col] = pd.to_numeric(values[keys.index(key)], errors="coerce")
                if pd.isna(row["MIN"]):
                    continue
                rows.append(row)
    return rows


def _log_path(season: str):
    return data_path(f"ESPN_{season_short(season)}_LOG.csv")


def update_season_log(season: str) -> pd.DataFrame:
    """The season's stored box scores, after fetching any finished games that are missing."""
    if season in _log_cache:
        return _log_cache[season]

    path = _log_path(season)
    log = pd.read_csv(path, dtype={"GAME_ID": str}) if path.exists() else pd.DataFrame(columns=LOG_COLS)

    state = load_state()
    done_seasons = state.get("espn_complete_seasons", [])
    if season in done_seasons and not log.empty:
        _log_cache[season] = log
        return log

    def save(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"]).reset_index(drop=True)
        frame[LOG_COLS].to_csv(path, index=False)
        return frame

    games = _counted_games(season_events(season))
    finished = games[games["FINAL"]]
    missing = finished[~finished["GAME_ID"].isin(set(log["GAME_ID"].astype(str)))]

    new_rows, failed = [], 0
    for i, (_, game) in enumerate(missing.iterrows(), start=1):
        try:
            rows = _box_score_rows(game)
        except Exception as e:
            failed += 1
            print(f"⚠️  ESPN box score {game['GAME_ID']} failed: {e}")
            continue
        if not rows:
            failed += 1
            continue
        new_rows.extend(rows)
        if i % 100 == 0:  # keep progress during a long first fill
            log = save(pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True))
            new_rows = []
            print(f"  ESPN {season}: {i} of {len(missing)} games fetched")

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    if len(missing):
        log = save(log)
    print(f"ESPN {season}: {len(missing) - failed} new games, {log['GAME_ID'].nunique()} stored"
          + (f", {failed} not available yet" if failed else ""))

    # once a season is over and fully stored, stop asking ESPN about it
    season_over = datetime.now(EASTERN).date().isoformat() > f"{int(season[:4]) + 1}-07-01"
    if season_over and failed == 0 and not log.empty and season not in done_seasons:
        state = load_state()
        state["espn_complete_seasons"] = sorted(set(state.get("espn_complete_seasons", [])) | {season})
        save_state(state)

    _log_cache[season] = log
    return log


# =========================================================
# FRAMES FOR THE SCRAPER
# =========================================================
def game_logs(season: str) -> pd.DataFrame:
    """All player game logs for a season, like fetch_league_player_gamelogs."""
    df = update_season_log(season).copy()
    for c in ["PTS", "REB", "AST", "BLK", "STL", "MIN"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MIN_float"] = df["MIN"].astype(float)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    return df.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=[True, True]).reset_index(drop=True)


def per_game(season: str) -> pd.DataFrame:
    """League-wide per-game stats for a season, like fetch_per_game.
    Averages are rounded to one decimal, the precision the NBA stats site reports."""
    logs = game_logs(season)
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "BLK", "STL"]
    if logs.empty:
        return pd.DataFrame(columns=cols)
    latest = logs.groupby("PLAYER_ID")[["PLAYER_NAME", "TEAM_ABBREVIATION"]].last()
    means = logs.groupby("PLAYER_ID")[["MIN", "PTS", "REB", "AST", "BLK", "STL"]].mean().round(1)
    return latest.join(means).reset_index()[cols]
