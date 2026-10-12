# nba_pipeline.py
# Requirements:
#   pip install -r requirements.txt

import re
import math
import time
import os
import requests
import pandas as pd
from typing import List, Tuple
import json
import warnings
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguegamelog,
    scheduleleaguev2,
)
from nba_api.stats.library.parameters import PerModeDetailed
from datetime import datetime, timedelta
import pytz  # pip install pytz

# pandas 3 warns that .round() leaves date columns alone, which is what we want
warnings.filterwarnings("ignore", message="obj.round has no effect")

import budget
import espn
from common import data_path, clean_player_name, load_state, save_state, ODDS_API_KEY, INJURY_API_KEY, STATS_SOURCE


def season_str(start_year: int) -> str:
    """2025 -> '2025-26'"""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def default_seasons(today=None) -> List[str]:
    """[previous, current] seasons. A new season becomes 'current' in October."""
    today = today or datetime.now(pytz.timezone("US/Eastern")).date()
    start = today.year if today.month >= 10 else today.year - 1
    return [season_str(start - 1), season_str(start)]


# ----- CONFIG -----
SEASONS: List[str] = default_seasons()  # newest last
SEASON_TYPE = "Regular Season"
SLEEP_SEC = 0.6
MIN_THRESHOLD = 15.0
LAST_N = 5
NEXT_DAYS_SCAN = 30  # how far ahead we look for next opponent

# ----- ODDS CONFIG -----
# ODDS_API_KEY is loaded in common.py (environment variable or .env file)

# base sports root
ODDS_SPORT_ROOT = "https://api.the-odds-api.com/v4/sports/basketball_nba"

# only the markets you wanted (P/R/A)
# Each market costs one credit per game. Points and rebounds are pulled for every
# game; assists only when the month's credits allow (budget.py decides).
ODDS_BASE_MARKETS = [
    "player_points",
    "player_rebounds",
]
ODDS_EXTRA_MARKETS = [
    "player_assists",
]
ODDS_MARKETS = ODDS_BASE_MARKETS + ODDS_EXTRA_MARKETS

# only draftkings
ODDS_BOOKS = {"draftkings"}

# ----- INJURY CONFIG (API-BASKETBALL NBA via RapidAPI) -----
# INJURY_API_KEY is loaded in common.py (environment variable or .env file)

INJURY_URL = "https://api-basketball-nba.p.rapidapi.com/nba-injuries"
INJURY_HOST = "api-basketball-nba.p.rapidapi.com"

# =========================================================
# BASIC HELPERS
# =========================================================
def season_short(season: str) -> str:
    """'2024-25' -> '24-25'"""
    return f"{season[2:4]}-{season[-2:]}"


def minutes_from_any(val) -> float:
    """Robustly convert NBA API 'MIN' into float minutes."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return float("nan")
    if ":" in s:
        try:
            m, sec = s.split(":")
            return float(m) + float(sec) / 60.0
        except Exception:
            pass
    if s.startswith("PT"):
        m = re.search(r"(\d+)M", s)
        sec = re.search(r"(\d+)S", s)
        minutes = float(m.group(1)) if m else 0.0
        seconds = float(sec.group(1)) if sec else 0.0
        return minutes + seconds / 60.0
    try:
        return float(s)
    except Exception:
        return float("nan")


# =========================================================
# NBA FETCHERS
# =========================================================
def fetch_per_game(season: str) -> pd.DataFrame:
    """League-wide per-game stats for a season."""
    if STATS_SOURCE == "espn":
        return espn.per_game(season)
    time.sleep(SLEEP_SEC)
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=PerModeDetailed.per_game,
        measure_type_detailed_defense="Base",
    )
    df = resp.get_data_frames()[0]
    min_col = "MIN" if "MIN" in df.columns else ("MINUTES" if "MINUTES" in df.columns else None)
    df["MIN"] = pd.to_numeric(df[min_col], errors="coerce") if min_col else pd.NA
    for c in ["PTS", "REB", "AST", "BLK", "STL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_league_player_gamelogs(season: str, season_type: str) -> pd.DataFrame:
    """All player game logs for a season (one call)."""
    if STATS_SOURCE == "espn":
        return espn.game_logs(season)
    time.sleep(SLEEP_SEC)
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="P",
    )
    df = resp.get_data_frames()[0]
    for c in ["PTS", "REB", "AST", "BLK", "STL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["MIN_float"] = df["MIN"].map(minutes_from_any)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=[True, True]).reset_index(drop=True)
    if "TEAM_ID" in df.columns:
        df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    return df


# =========================================================
# LAST-N AVERAGES
# =========================================================
def last_n_averages(glogs: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = ["PTS", "REB", "AST", "BLK", "STL"]
    glogs = glogs.copy()
    for c in cols:
        glogs[c] = pd.to_numeric(glogs[c], errors="coerce")
    glogs["GAME_DATE"] = pd.to_datetime(glogs["GAME_DATE"], errors="coerce")
    glogs = glogs.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=[True, False])
    gp = glogs.groupby("PLAYER_ID")["GAME_ID"].nunique().rename("GP_so_far")

    def agg_lastN(df):
        window = df.head(n)
        return pd.Series({
            "PLAYER_NAME":        df["PLAYER_NAME"].iloc[0],
            "TEAM_ABBREVIATION":  df["TEAM_ABBREVIATION"].iloc[0],
            "PTS_last5":          window["PTS"].mean(),
            "REB_last5":          window["REB"].mean(),
            "AST_last5":          window["AST"].mean(),
            "BLK_last5":          window["BLK"].mean(),
            "STL_last5":          window["STL"].mean(),
            "MIN_last5":          window["MIN_float"].mean(),
        })

    try:
        last5 = glogs.groupby("PLAYER_ID", group_keys=False).apply(
            agg_lastN, include_groups=False
        ).reset_index()
    except TypeError:
        last5 = glogs.groupby("PLAYER_ID", group_keys=False).apply(agg_lastN).reset_index()

    last5 = last5.merge(gp.reset_index(), on="PLAYER_ID", how="left")
    return last5


# =========================================================
# BASELINE CHOOSER
# =========================================================
def choose_baseline(
    current_pg: pd.DataFrame,
    prev_pg: pd.DataFrame | None,
    last5: pd.DataFrame,
    min_threshold: float,
    n: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keepers = current_pg.loc[
        current_pg["MIN"] >= min_threshold,
        ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","MIN","PTS","REB","AST","BLK","STL"]
    ].copy().rename(columns={
        "MIN":"MIN_season","PTS":"PTS_season","REB":"REB_season","AST":"AST_season",
        "BLK":"BLK_season","STL":"STL_season"
    })

    if prev_pg is not None and not prev_pg.empty:
        prev_base = prev_pg[["PLAYER_ID","MIN","PTS","REB","AST","BLK","STL"]].copy().rename(columns={
            "MIN":"MIN_prev","PTS":"PTS_prev","REB":"REB_prev","AST":"AST_prev",
            "BLK":"BLK_prev","STL":"STL_prev"
        })
    else:
        prev_base = pd.DataFrame(columns=[
            "PLAYER_ID","MIN_prev","PTS_prev","REB_prev","AST_prev","BLK_prev","STL_prev"
        ])

    keepers_for_merge = keepers.drop(columns=["PLAYER_NAME","TEAM_ABBREVIATION"], errors="ignore")
    base = (
        last5
        .merge(keepers_for_merge, on="PLAYER_ID", how="left")
        .merge(prev_base, on="PLAYER_ID", how="left")
    )

    def row_baseline(r):
        gp = r.get("GP_so_far")
        has_prev = pd.notna(r.get("PTS_prev"))
        use_prev = (pd.notna(gp) and gp < n and has_prev)
        src = "previous" if use_prev else "current"
        return pd.Series({
            "baseline_source": src,
            "MIN_baseline": r["MIN_prev"] if src == "previous" else r["MIN_season"],
            "PTS_baseline": r["PTS_prev"] if src == "previous" else r["PTS_season"],
            "REB_baseline": r["REB_prev"] if src == "previous" else r["REB_season"],
            "AST_baseline": r["AST_prev"] if src == "previous" else r["AST_season"],
            "BLK_baseline": r["BLK_prev"] if src == "previous" else r["BLK_season"],
            "STL_baseline": r["STL_prev"] if src == "previous" else r["STL_season"],
        })

    baseline = base.apply(row_baseline, axis=1)
    out = pd.concat([base, baseline], axis=1)
    return keepers, out


# =========================================================
# RECENT VS BASELINE
# =========================================================
def build_recent_vs_baseline(last5: pd.DataFrame, kee_and_base: pd.DataFrame) -> pd.DataFrame:
    merged = kee_and_base.copy()
    for stat in ["PTS", "REB", "AST", "BLK", "STL"]:
        merged[f"{stat}_delta"] = merged[f"{stat}_last5"] - merged[f"{stat}_baseline"]
    merged["Usage_delta_MIN"] = merged["MIN_last5"] - merged["MIN_baseline"]
    keep_cols = [
        "PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","GP_so_far","baseline_source",
        "MIN_last5","PTS_last5","REB_last5","AST_last5","BLK_last5","STL_last5",
        "MIN_baseline","PTS_baseline","REB_baseline","AST_baseline","BLK_baseline","STL_baseline",
        "Usage_delta_MIN","PTS_delta","REB_delta","AST_delta","BLK_delta","STL_delta"
    ]
    merged = merged[[c for c in keep_cols if c in merged.columns]].copy()
    return merged


# =========================================================
# SIMILAR_NEXT (per-stat windows)
# =========================================================
_SIM_THRESH = {"PTS": 3, "REB": 2, "AST": 1, "STL": 1, "BLK": 1}

def build_similar_next_csv(
    logs_two_seasons: pd.DataFrame,
    logs_current_season: pd.DataFrame,
    keepers_current: pd.DataFrame
) -> pd.DataFrame:
    logs_two = logs_two_seasons.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    logs_cur = logs_current_season.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    for stat in ["PTS", "REB", "AST", "STL", "BLK"]:
        logs_two[f"{stat}_next"] = logs_two.groupby("PLAYER_ID")[stat].shift(-1)

    mr_idx = logs_cur.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
    mr = logs_cur.loc[mr_idx, ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION",
                               "GAME_DATE","PTS","REB","AST","STL","BLK"]].copy()
    mr = mr.rename(columns={
        "GAME_DATE":"MR_GAME_DATE",
        "PTS":"PTS_mr","REB":"REB_mr","AST":"AST_mr","STL":"STL_mr","BLK":"BLK_mr"
    })
    players = keepers_current["PLAYER_ID"].unique()
    mr = mr[mr["PLAYER_ID"].isin(players)].copy()

    rows = []
    for _, r in mr.iterrows():
        pid = r["PLAYER_ID"]; mr_date = r["MR_GAME_DATE"]
        history = logs_two[logs_two["PLAYER_ID"] == pid]
        pool = history[history["GAME_DATE"] < mr_date].copy()

        out = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": r["PLAYER_NAME"],
            "TEAM_ABBREVIATION": r["TEAM_ABBREVIATION"],
            "MR_GAME_DATE": mr_date,
            "PTS_mr": r["PTS_mr"],
            "REB_mr": r["REB_mr"],
            "AST_mr": r["AST_mr"],
            "STL_mr": r["STL_mr"],
            "BLK_mr": r["BLK_mr"],
        }

        for stat in ["PTS","REB","AST","STL","BLK"]:
            mr_val = r[f"{stat}_mr"]; thr = _SIM_THRESH[stat]
            sim = pool[pool[stat].between(mr_val - thr, mr_val + thr, inclusive="both")]
            sim = sim[pd.notna(sim[f"{stat}_next"])]
            out[f"{stat}_similar_n"] = int(sim.shape[0])
            out[f"{stat}_next_avg"]  = sim[f"{stat}_next"].mean() if not sim.empty else float("nan")
        rows.append(out)

    result = pd.DataFrame(rows)
    ordered = [
        "PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","MR_GAME_DATE",
        "PTS_mr","PTS_similar_n","PTS_next_avg",
        "REB_mr","REB_similar_n","REB_next_avg",
        "AST_mr","AST_similar_n","AST_next_avg",
        "STL_mr","STL_similar_n","STL_next_avg",
        "BLK_mr","BLK_similar_n","BLK_next_avg",
    ]
    result = result[[c for c in ordered if c in result.columns]]
    return result


# =========================================================
# MOST_SIMILAR_GAME (kept for compatibility, no longer called in main)
# =========================================================
_SIM_WEIGHTS = {"PTS": 3.0, "REB": 2.0, "AST": 2.0, "STL": 1.0, "BLK": 1.0}
_SIM_STATS = ["PTS", "REB", "AST", "STL", "BLK"]

def _weighted_distance(row, ref) -> float:
    d2 = 0.0
    for s in _SIM_STATS:
        rw = row.get(s)
        rv = ref.get(s)
        if pd.isna(rw) or pd.isna(rv):
            return float("inf")
        diff = float(rw) - float(rv)
        w = _SIM_WEIGHTS[s]
        d2 += w * (diff * diff)
    return math.sqrt(d2)

def build_most_similar_game_csv(
    logs_two_seasons: pd.DataFrame,
    logs_current_season: pd.DataFrame,
    keepers_current: pd.DataFrame
) -> pd.DataFrame:
    logs_two = logs_two_seasons.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    logs_cur = logs_current_season.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()

    logs_two["GAME_DATE_next"] = logs_two.groupby("PLAYER_ID")["GAME_DATE"].shift(-1)
    for s in _SIM_STATS:
        logs_two[f"{s}_next"] = logs_two.groupby("PLAYER_ID")[s].shift(-1)

    mr_idx = logs_cur.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
    mr = logs_cur.loc[mr_idx, ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","GAME_DATE"] + _SIM_STATS].copy()
    mr = mr.rename(columns={"GAME_DATE":"MR_GAME_DATE", **{s: f"{s}_mr" for s in _SIM_STATS}})
    players = keepers_current["PLAYER_ID"].unique()
    mr = mr[mr["PLAYER_ID"].isin(players)].copy()

    rows = []
    for _, r in mr.iterrows():
        pid = r["PLAYER_ID"]; mr_date = r["MR_GAME_DATE"]
        ref = {s: r[f"{s}_mr"] for s in _SIM_STATS}
        history = logs_two[logs_two["PLAYER_ID"] == pid]
        pool = history[history["GAME_DATE"] < mr_date].copy()
        if pool.empty:
            rows.append({
                "PLAYER_ID": pid, "PLAYER_NAME": r["PLAYER_NAME"], "TEAM_ABBREVIATION": r["TEAM_ABBREVIATION"],
                "MR_GAME_DATE": mr_date,
                **{f"{s}_mr": ref[s] for s in _SIM_STATS},
                "SIM_GAME_DATE": pd.NaT, "distance": float("nan"),
                "NEXT_GAME_DATE": pd.NaT,
                **{f"{s}_next": float("nan") for s in _SIM_STATS},
                **{f"{s}_sim": float("nan") for s in _SIM_STATS},
            })
            continue

        ref_series = pd.Series(ref)
        pool = pool.copy()
        pool["__dist__"] = pool.apply(lambda row: _weighted_distance(row, ref_series), axis=1)
        pool = pool.sort_values("__dist__")
        best = None
        for _, cand in pool.iterrows():
            if any(pd.isna(cand[f"{s}_next"]) for s in _SIM_STATS):
                continue
            best = cand
            break
        if best is None:
            rows.append({
                "PLAYER_ID": pid, "PLAYER_NAME": r["PLAYER_NAME"], "TEAM_ABBREVIATION": r["TEAM_ABBREVIATION"],
                "MR_GAME_DATE": mr_date,
                **{f"{s}_mr": ref[s] for s in _SIM_STATS},
                "SIM_GAME_DATE": pd.NaT, "distance": pool["__dist__"].min() if "__dist__" in pool else float("nan"),
                "NEXT_GAME_DATE": pd.NaT,
                **{f"{s}_next": float("nan") for s in _SIM_STATS},
                **{f"{s}_sim": float("nan") for s in _SIM_STATS},
            })
            continue

        out = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": r["PLAYER_NAME"],
            "TEAM_ABBREVIATION": r["TEAM_ABBREVIATION"],
            "MR_GAME_DATE": mr_date,
            **{f"{s}_mr": ref[s] for s in _SIM_STATS},
            "SIM_GAME_DATE": best["GAME_DATE"],
            "distance": best["__dist__"],
            "NEXT_GAME_DATE": best["GAME_DATE_next"],
            **{f"{s}_sim": best[s] for s in _SIM_STATS},
            **{f"{s}_next": best[f"{s}_next"] for s in _SIM_STATS},
        }
        rows.append(out)

    result = pd.DataFrame(rows)
    ordered = [
        "PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION",
        "MR_GAME_DATE",
        "PTS_mr","REB_mr","AST_mr","STL_mr","BLK_mr",
        "SIM_GAME_DATE","distance",
        "PTS_sim","REB_sim","AST_sim","STL_sim","BLK_sim",
        "NEXT_GAME_DATE",
        "PTS_next","REB_next","AST_next","STL_next","BLK_next",
    ]
    result = result[[c for c in ordered if c in result.columns]]
    return result


# =========================================================
# OPPONENT BIAS (next opponent only)
# =========================================================
def _parse_opponent(matchup: str) -> str | None:
    if pd.isna(matchup):
        return None
    parts = str(matchup).strip().split()
    if len(parts) >= 3:
        return parts[-1]
    return None


_SCHEDULE_CACHE: dict[str, pd.DataFrame] = {}

def _fetch_schedule(season: str) -> pd.DataFrame:
    """Full league schedule for a season in one call:
    GAME_ID, GAME_DATE (Eastern), HOME_TEAM_ID, VISITOR_TEAM_ID, COUNTS_FOR_ODDS."""
    if season in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[season]
    if STATS_SOURCE == "espn":
        _SCHEDULE_CACHE[season] = espn.schedule(season)
        return _SCHEDULE_CACHE[season]
    time.sleep(SLEEP_SEC)
    raw = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00").get_data_frames()[0]
    sched = pd.DataFrame({
        "GAME_ID": raw["gameId"].astype(str),
        "GAME_DATE": pd.to_datetime(raw["gameDateEst"], errors="coerce")
                       .dt.tz_localize(None).dt.normalize().dt.tz_localize("US/Eastern"),
        "HOME_TEAM_ID": pd.to_numeric(raw["homeTeam_teamId"], errors="coerce"),
        "VISITOR_TEAM_ID": pd.to_numeric(raw["awayTeam_teamId"], errors="coerce"),
    }).dropna().sort_values("GAME_DATE", kind="stable").reset_index(drop=True)
    # the odds service lists everything except preseason and the All-Star game
    sched["COUNTS_FOR_ODDS"] = ~sched["GAME_ID"].str.startswith(("001", "003"))
    _SCHEDULE_CACHE[season] = sched
    return sched


def _games_left_in_month(today) -> int | None:
    """Games from today to the end of the month that the odds service will list
    (everything except preseason and the All-Star game). None if unknown."""
    try:
        sched = _fetch_schedule(default_seasons(today)[-1])
        dates = sched["GAME_DATE"].dt.date
        last_day = today.replace(day=1) + timedelta(days=32)
        last_day = last_day.replace(day=1) - timedelta(days=1)
        return int((sched["COUNTS_FOR_ODDS"] & (dates >= today) & (dates <= last_day)).sum())
    except Exception:
        return None


def _fetch_next_opponents_for_teams(team_ids: List[int], days_ahead: int = NEXT_DAYS_SCAN) -> pd.DataFrame:
    team_ids = [int(t) for t in pd.Series(team_ids).dropna().astype(int).unique().tolist()]
    if not team_ids:
        return pd.DataFrame(columns=["TEAM_ID","NEXT_GAME_DATE","NEXT_OPP_TEAM_ID","NEXT_OPP_ABBR"])

    today = pd.Timestamp.now(tz="US/Eastern").normalize()
    found: dict[int, tuple[pd.Timestamp, int]] = {}

    # each team's first game from today through today + days_ahead
    # (ScoreboardV2, one call per day, no longer returns data reliably)
    try:
        sched = _fetch_schedule(default_seasons()[-1])
        upcoming = sched[
            (sched["GAME_DATE"] >= today) &
            (sched["GAME_DATE"] <= today + pd.Timedelta(days=days_ahead))
        ]
        for _, g in upcoming.iterrows():
            home = int(g["HOME_TEAM_ID"])
            away = int(g["VISITOR_TEAM_ID"])
            for team, opp in [(home, away), (away, home)]:
                if team in team_ids and team not in found:
                    found[team] = (g["GAME_DATE"], opp)
            if len(found) == len(team_ids):
                break
    except Exception as e:
        print(f"⚠️  Failed to fetch the schedule, next opponents will be blank: {e}")

    rows = []
    for tid in team_ids:
        if tid in found:
            dt, opp_id = found[tid]
            rows.append({"TEAM_ID": tid, "NEXT_GAME_DATE": dt, "NEXT_OPP_TEAM_ID": opp_id})
        else:
            rows.append({"TEAM_ID": tid, "NEXT_GAME_DATE": pd.NaT, "NEXT_OPP_TEAM_ID": pd.NA})

    out = pd.DataFrame(rows)
    out["TEAM_ID"] = out["TEAM_ID"].astype("Int64")
    out["NEXT_OPP_TEAM_ID"] = pd.to_numeric(out["NEXT_OPP_TEAM_ID"], errors="coerce").astype("Int64")
    out["NEXT_OPP_ABBR"] = pd.NA
    return out


def build_opponent_bias_next_only(
    logs_two_seasons: pd.DataFrame,
    logs_current_season: pd.DataFrame,
    keepers_current: pd.DataFrame,
) -> pd.DataFrame:
    cur_sorted = logs_current_season.sort_values(["PLAYER_ID","GAME_DATE"])
    mr_idx = cur_sorted.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
    mr = cur_sorted.loc[mr_idx, ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","TEAM_ID","GAME_DATE"]].copy()
    mr = mr.rename(columns={"GAME_DATE":"MR_GAME_DATE"})
    mr["TEAM_ID"] = pd.to_numeric(mr["TEAM_ID"], errors="coerce").astype("Int64")

    team_ids = mr["TEAM_ID"].dropna().astype("Int64").unique().tolist()
    next_opp_df = _fetch_next_opponents_for_teams(team_ids, days_ahead=NEXT_DAYS_SCAN)

    team_abbr_map = logs_current_season.dropna(subset=["TEAM_ID"]) \
                                       .groupby("TEAM_ID")["TEAM_ABBREVIATION"] \
                                       .agg(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else pd.NA)
    team_abbr_map.index = pd.to_numeric(team_abbr_map.index, errors="coerce").astype("Int64")
    team_abbr_map = team_abbr_map.to_dict()
    next_opp_df["NEXT_OPP_ABBR"] = next_opp_df["NEXT_OPP_TEAM_ID"].map(team_abbr_map)

    mr = mr.merge(next_opp_df, on="TEAM_ID", how="left")

    cur = logs_current_season.copy()
    for c in ["PTS","REB","AST","STL","BLK","MIN_float","TEAM_ID"]:
        if c in cur.columns:
            cur[c] = pd.to_numeric(cur[c], errors="coerce")
    season_avgs = cur.groupby("PLAYER_ID").agg(
        SEASON_PTS_avg=("PTS","mean"),
        SEASON_REB_avg=("REB","mean"),
        SEASON_AST_avg=("AST","mean"),
        SEASON_STL_avg=("STL","mean"),
        SEASON_BLK_avg=("BLK","mean"),
        SEASON_MIN_avg=("MIN_float","mean"),
        SEASON_GP=("GAME_ID","nunique"),
    ).reset_index()

    two = logs_two_seasons.copy()
    for c in ["PTS","REB","AST","STL","BLK","MIN_float","TEAM_ID"]:
        if c in two.columns:
            two[c] = pd.to_numeric(two[c], errors="coerce")
    if "MATCHUP" in two.columns:
        two["OPPONENT"] = two["MATCHUP"].map(_parse_opponent)
    else:
        two["OPPONENT"] = pd.NA
    pid_set = set(keepers_current["PLAYER_ID"].unique())
    two = two[two["PLAYER_ID"].isin(pid_set)].copy()

    rows = []
    for _, r in mr.iterrows():
        pid   = r["PLAYER_ID"]
        name  = r.get("PLAYER_NAME")
        team  = r.get("TEAM_ABBREVIATION")
        nopp  = r.get("NEXT_OPP_ABBR")
        ndate = r.get("NEXT_GAME_DATE")

        if pd.isna(nopp):
            rows.append({
                "PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ABBREVIATION": team,
                "NEXT_OPP": pd.NA, "NEXT_GAME_DATE": pd.NaT,
                "VS_OPP_GP": 0,
                "PTS_vs_opp": float("nan"), "REB_vs_opp": float("nan"), "AST_vs_opp": float("nan"),
                "STL_vs_opp": float("nan"), "BLK_vs_opp": float("nan"), "MIN_vs_opp": float("nan"),
                "SEASON_PTS_avg": float("nan"), "SEASON_REB_avg": float("nan"), "SEASON_AST_avg": float("nan"),
                "SEASON_STL_avg": float("nan"), "SEASON_BLK_avg": float("nan"), "SEASON_MIN_avg": float("nan"),
                "DIFF_PTS": float("nan"), "DIFF_REB": float("nan"), "DIFF_AST": float("nan"),
                "DIFF_STL": float("nan"), "DIFF_BLK": float("nan"), "DIFF_MIN": float("nan"),
            })
            continue

        hist = two[(two["PLAYER_ID"] == pid) & (two["OPPONENT"] == nopp)]
        vs_gp = hist["GAME_ID"].nunique() if "GAME_ID" in hist.columns else len(hist)

        vs_means = {
            "PTS_vs_opp": hist["PTS"].mean(),
            "REB_vs_opp": hist["REB"].mean(),
            "AST_vs_opp": hist["AST"].mean(),
            "STL_vs_opp": hist["STL"].mean(),
            "BLK_vs_opp": hist["BLK"].mean(),
            "MIN_vs_opp": hist["MIN_float"].mean(),
        }

        sa = season_avgs[season_avgs["PLAYER_ID"] == pid]
        if sa.empty:
            seas = {"SEASON_PTS_avg": float("nan"), "SEASON_REB_avg": float("nan"), "SEASON_AST_avg": float("nan"),
                    "SEASON_STL_avg": float("nan"), "SEASON_BLK_avg": float("nan"), "SEASON_MIN_avg": float("nan")}
        else:
            seas = sa.iloc[0][[
                "SEASON_PTS_avg","SEASON_REB_avg","SEASON_AST_avg",
                "SEASON_STL_avg","SEASON_BLK_avg","SEASON_MIN_avg"
            ]].to_dict()

        diffs = {
            "DIFF_PTS": vs_means["PTS_vs_opp"] - seas["SEASON_PTS_avg"] if pd.notna(vs_means["PTS_vs_opp"]) and pd.notna(seas["SEASON_PTS_avg"]) else float("nan"),
            "DIFF_REB": vs_means["REB_vs_opp"] - seas["SEASON_REB_avg"] if pd.notna(vs_means["REB_vs_opp"]) and pd.notna(seas["SEASON_REB_avg"]) else float("nan"),
            "DIFF_AST": vs_means["AST_vs_opp"] - seas["SEASON_AST_avg"] if pd.notna(vs_means["AST_vs_opp"]) and pd.notna(seas["SEASON_AST_avg"]) else float("nan"),
            "DIFF_STL": vs_means["STL_vs_opp"] - seas["SEASON_STL_avg"] if pd.notna(vs_means["STL_vs_opp"]) and pd.notna(seas["SEASON_STL_avg"]) else float("nan"),
            "DIFF_BLK": vs_means["BLK_vs_opp"] - seas["SEASON_BLK_avg"] if pd.notna(vs_means["BLK_vs_opp"]) and pd.notna(seas["SEASON_BLK_avg"]) else float("nan"),
            "DIFF_MIN": vs_means["MIN_vs_opp"] - seas["SEASON_MIN_avg"] if pd.notna(vs_means["MIN_vs_opp"]) and pd.notna(seas["SEASON_MIN_avg"]) else float("nan"),
        }

        rows.append({
            "PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ABBREVIATION": team,
            "NEXT_OPP": nopp, "NEXT_GAME_DATE": ndate,
            "VS_OPP_GP": int(vs_gp),
            **vs_means, **seas, **diffs
        })

    out = pd.DataFrame(rows)
    ordered = [
        "PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","NEXT_OPP","NEXT_GAME_DATE","VS_OPP_GP",
        "PTS_vs_opp","REB_vs_opp","AST_vs_opp","STL_vs_opp","BLK_vs_opp","MIN_vs_opp",
        "SEASON_PTS_avg","SEASON_REB_avg","SEASON_AST_avg","SEASON_STL_avg","SEASON_BLK_avg","SEASON_MIN_avg",
        "DIFF_PTS","DIFF_REB","DIFF_AST","DIFF_STL","DIFF_BLK","DIFF_MIN",
    ]
    out = out[[c for c in ordered if c in out.columns]]
    return out


# =========================================================
# REST-DAY NEXT PERFORMANCE (0,1,2,3 days between games)
# =========================================================
def build_rest_day_next_only(
    logs_two_seasons: pd.DataFrame,
    logs_current_season: pd.DataFrame,
    keepers_current: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each current-season keeper:
    - find that player's most recent game in the current season
    - find that player's team next game date (same helper as opponent bias)
    - compute upcoming REST_DAYS = max(0, (next - last).days - 1), clamped to 3
    - from two-season logs, compute that player's historical average PTS/REB/AST/STL/BLK
      for games that were played with that exact rest value
    - output one row per player with the chosen bucket
    """
    # 1) most recent game in current season
    cur_sorted = logs_current_season.sort_values(["PLAYER_ID", "GAME_DATE"])
    mr_idx = cur_sorted.groupby("PLAYER_ID")["GAME_DATE"].idxmax()
    mr = cur_sorted.loc[mr_idx, [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "TEAM_ID",
        "GAME_DATE",
    ]].copy()
    mr = mr.rename(columns={"GAME_DATE": "MR_GAME_DATE"})
    mr["TEAM_ID"] = pd.to_numeric(mr["TEAM_ID"], errors="coerce").astype("Int64")

    pid_keep = set(keepers_current["PLAYER_ID"].unique())
    mr = mr[mr["PLAYER_ID"].isin(pid_keep)].copy()

    # 2) get that team's next game date
    team_ids = mr["TEAM_ID"].dropna().astype("Int64").unique().tolist()
    next_games_df = _fetch_next_opponents_for_teams(team_ids, days_ahead=NEXT_DAYS_SCAN)
    # we only need TEAM_ID -> NEXT_GAME_DATE
    next_games_df = next_games_df[["TEAM_ID", "NEXT_GAME_DATE"]].copy()
    mr = mr.merge(next_games_df, on="TEAM_ID", how="left")

    # 3) build historical rest-day buckets from two-season logs
    logs_two = logs_two_seasons.copy()
    logs_two = logs_two.sort_values(["PLAYER_ID", "GAME_DATE"])
    logs_two["GAME_DATE"] = pd.to_datetime(logs_two["GAME_DATE"], errors="coerce")

    # previous game per player
    logs_two["PREV_GAME_DATE"] = logs_two.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    logs_two["DAYS_BETWEEN"] = (logs_two["GAME_DATE"] - logs_two["PREV_GAME_DATE"]).dt.days
    # rest days = days_between - 1; clamp 0..3
    logs_two["REST_DAYS"] = logs_two["DAYS_BETWEEN"] - 1
    logs_two["REST_DAYS"] = logs_two["REST_DAYS"].clip(lower=0, upper=3)

    for c in ["PTS", "REB", "AST", "STL", "BLK"]:
        if c in logs_two.columns:
            logs_two[c] = pd.to_numeric(logs_two[c], errors="coerce")

    # group (player, rest_days) -> averages
    if "GAME_ID" in logs_two.columns:
        gp_agg = ("GAME_ID", "nunique")
    else:
        gp_agg = ("PTS", "count")

    agg = (
        logs_two
        .dropna(subset=["REST_DAYS"])
        .groupby(["PLAYER_ID", "REST_DAYS"])
        .agg(
            PTS_avg=("PTS", "mean"),
            REB_avg=("REB", "mean"),
            AST_avg=("AST", "mean"),
            STL_avg=("STL", "mean"),
            BLK_avg=("BLK", "mean"),
            GP=gp_agg,
        )
        .reset_index()
    )

    rows = []
    for _, r in mr.iterrows():
        pid = r["PLAYER_ID"]
        name = r["PLAYER_NAME"]
        team = r["TEAM_ABBREVIATION"]
        last_date = r["MR_GAME_DATE"]
        next_date = r.get("NEXT_GAME_DATE")

        rest_next = pd.NA
        pts_next = reb_next = ast_next = stl_next = blk_next = float("nan")
        gp_for_rest = 0

        if pd.notna(next_date) and pd.notna(last_date):
            gap_days = (pd.to_datetime(next_date).date() - pd.to_datetime(last_date).date()).days
            rest_val = max(0, gap_days - 1)
            rest_val = min(rest_val, 3)
            rest_next = rest_val

            sub = agg[(agg["PLAYER_ID"] == pid) & (agg["REST_DAYS"] == rest_val)]
            if not sub.empty:
                row0 = sub.iloc[0]
                pts_next = row0["PTS_avg"]
                reb_next = row0["REB_avg"]
                ast_next = row0["AST_avg"]
                stl_next = row0["STL_avg"]
                blk_next = row0["BLK_avg"]
                gp_for_rest = int(row0["GP"])

        rows.append({
            "PLAYER_ID": pid,
            "PLAYER_NAME": name,
            "TEAM_ABBREVIATION": team,
            "MR_GAME_DATE": last_date,
            "NEXT_GAME_DATE": next_date,
            "REST_NEXT": rest_next,
            "REST_NEXT_GP": gp_for_rest,
            "PTS_next": pts_next,
            "REB_next": reb_next,
            "AST_next": ast_next,
            "STL_next": stl_next,
            "BLK_next": blk_next,
        })

    out = pd.DataFrame(rows)
    ordered = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "MR_GAME_DATE",
        "NEXT_GAME_DATE",
        "REST_NEXT",
        "REST_NEXT_GP",
        "PTS_next",
        "REB_next",
        "AST_next",
        "STL_next",
        "BLK_next",
    ]
    out = out[[c for c in ordered if c in out.columns]]
    return out

# =========================================================
# INJURY REPORT (API-BASKETBALL NBA via RapidAPI)
# =========================================================

def _convert_to_est(utc_str: str) -> str | None:
    """
    Convert an ISO timestamp like '2025-11-13T18:00Z' (UTC)
    to 'YYYY-%m-%d %H:%M:%S EST/EDT' in US/Eastern.
    """
    if not utc_str or not isinstance(utc_str, str):
        return None
    try:
        dt_utc = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        tz_est = pytz.timezone("US/Eastern")
        dt_est = dt_utc.astimezone(tz_est)
        return dt_est.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return utc_str


def _find_injury_lists(obj):
    """
    Walk the JSON and collect any list whose elements look like
    injury objects (have 'athlete' and 'details' keys).
    """
    found = []

    def walk(o):
        if isinstance(o, list):
            if o and all(isinstance(x, dict) and "athlete" in x and "details" in x for x in o):
                found.append(o)
            else:
                for item in o:
                    walk(item)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(obj)
    return found


def fetch_nba_injury_report() -> pd.DataFrame:
    """
    One-call injury fetch from API-BASKETBALL (NBA) via RapidAPI.
    Flattens into one row per injured player and converts dates to EST.
    """
    if not INJURY_API_KEY:
        print("⚠️  No INJURY_API_KEY set, skipping injury pull.")
        return pd.DataFrame()

    usage = budget.load_usage()
    allowed, reason = budget.injury_pull_allowed(usage)
    if not allowed:
        print(f"Injury pull skipped: {reason}.")
        return pd.DataFrame()

    headers = {
        "X-RapidAPI-Key": INJURY_API_KEY,
        "X-RapidAPI-Host": INJURY_HOST,
    }

    # one attempt per day, whether or not it succeeds
    usage["injury"]["last_pull_date"] = budget.today_eastern().isoformat()
    budget.save_usage(usage)
    try:
        resp = requests.get(INJURY_URL, headers=headers, timeout=20)
        budget.record_injury_request(usage, resp.headers)
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️  Failed to fetch injuries: {e}")
        return pd.DataFrame()

    # Show remaining credits
    limit = resp.headers.get("x-ratelimit-requests-limit")
    remaining = resp.headers.get("x-ratelimit-requests-remaining")
    print(f"Injury API credits: {remaining}/{limit} remaining (if provided by API).")

    try:
        data = resp.json()
    except ValueError:
        print("⚠️  Injury response not valid JSON.")
        return pd.DataFrame()

    # Optional: save raw JSON for debugging
    with open(data_path("INJURY_RAW.json"), "w") as f:
        json.dump(data, f, indent=2)

    injury_lists = _find_injury_lists(data)
    if not injury_lists:
        print("⚠️  No injury lists found in JSON. Check INJURY_RAW.json.")
        return pd.DataFrame()

    rows = []
    for lst in injury_lists:
        for inj in lst:
            athlete = inj.get("athlete", {}) or {}
            team = athlete.get("team", {}) or {}
            details = inj.get("details", {}) or {}
            fantasy = details.get("fantasyStatus", {}) or {}

            rows.append({
                "team_name": team.get("displayName") or team.get("name"),
                "team_abbr": team.get("abbreviation"),
                "player_first": athlete.get("firstName"),
                "player_last": athlete.get("lastName"),
                "player_name": athlete.get("displayName"),
                "status": inj.get("status"),  # 'Out', 'Day-To-Day', etc.
                "report_date_utc": inj.get("date"),
                "report_date_est": _convert_to_est(inj.get("date")),
                "fantasy_status": fantasy.get("description"),
                "fantasy_status_abbr": fantasy.get("abbreviation"),
                "injury_type": details.get("type"),
                "injury_location": details.get("location"),
                "injury_detail": details.get("detail"),
                "injury_side": details.get("side"),
                "return_date_utc": details.get("returnDate"),
                "return_date_est": _convert_to_est(details.get("returnDate")),
                "long_comment": inj.get("longComment"),
                "short_comment": inj.get("shortComment"),
            })

    if not rows:
        print("⚠️  Injury lists found but no rows built.")
        return pd.DataFrame()

    return pd.DataFrame(rows)

# =========================================================
# ODDS: per-event props (points, rebounds, assists) for ALL today's games, DK only
# =========================================================
def _clean_player_name_odds(name: str) -> str:
    return clean_player_name(name)


def fetch_nba_player_props_today() -> pd.DataFrame:
    """
    Fetch NBA player props for *today's* NBA events using the per-event odds endpoint.
    We'll interpret "today" as US/Eastern, because NBA games are scheduled by that most often.

    Every request that costs credits is cleared by budget.py first: one pull per day,
    points and rebounds for every game, assists for as many games as the month's
    free credits can spare.
    """
    eastern = pytz.timezone("US/Eastern")
    today_eastern = datetime.now(eastern).date()

    usage = budget.load_usage()
    status = {
        "date": today_eastern.isoformat(),
        "events": 0, "allowed": 0, "pulled": 0, "assists_games": 0, "credits": 0,
        "note": "", "final": False,
    }

    def finish(note: str, final: bool) -> None:
        status["note"], status["final"] = note, final
        usage["odds"]["today"] = status
        budget.save_usage(usage)

    if not ODDS_API_KEY:
        print("⚠️  No ODDS_API_KEY set, skipping odds pull.")
        finish("no API key", True)
        return pd.DataFrame()

    # the events list is free
    events_url = f"{ODDS_SPORT_ROOT}/events"
    try:
        ev_resp = requests.get(
            events_url,
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "dateFormat": "iso",
            },
            timeout=15,
        )
        ev_resp.raise_for_status()
        events = ev_resp.json()
    except Exception as e:
        print(f"⚠️  Failed to fetch NBA events: {e}")
        finish("could not reach the odds service", False)
        return pd.DataFrame()

    todays = []
    for event in events or []:
        commence_iso = event.get("commence_time")  # this is UTC
        try:
            dt_utc = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            dt_eastern = dt_utc.astimezone(eastern)
            event_date_eastern = dt_eastern.date()
        except Exception:
            dt_eastern = None
            event_date_eastern = None

        if event_date_eastern is not None and event_date_eastern != today_eastern:
            continue
        todays.append((event, dt_eastern, event_date_eastern))

    status["events"] = len(todays)
    base_cost, extra_cost = len(ODDS_BASE_MARKETS), len(ODDS_EXTRA_MARKETS)

    # also free: the provider's own count of credits left
    remaining = budget.odds_credits_remaining()
    usage["odds"]["remaining"] = remaining
    usage["odds"]["checked_at"] = datetime.now(eastern).isoformat(timespec="seconds")

    games_left = _games_left_in_month(today_eastern)
    n_games, n_extra, reason = budget.plan_odds_pull(
        usage, len(todays), remaining, base_cost, extra_cost, games_left, today_eastern
    )
    print(f"Odds credits left: {remaining}. Games today: {len(todays)}, left this month: {games_left}. "
          f"Pulling: {n_games}, with assists: {n_extra} ({reason}).")
    if n_games == 0:
        if reason == "already pulled today":
            budget.save_usage(usage)  # keep the earlier pull's record for today
        else:
            finish(reason, reason != "could not confirm credits left")
        return pd.DataFrame()

    chosen_ids = {e.get("id") for e in budget.choose_events([t[0] for t in todays], n_games, today_eastern)}
    chosen = [t for t in todays if t[0].get("id") in chosen_ids]
    extra_ids = {e.get("id") for e in budget.choose_events([t[0] for t in chosen], n_extra, today_eastern)}
    status["allowed"] = len(chosen)

    month = usage["odds"]["months"].setdefault(today_eastern.strftime("%Y-%m"), {"credits": 0, "requests": 0})
    credits_before = month["credits"]
    rows = []

    for event, dt_eastern, event_date_eastern in chosen:
        event_id = event.get("id")
        markets = ODDS_BASE_MARKETS + (ODDS_EXTRA_MARKETS if event_id in extra_ids else [])
        cost_per_game = len(markets)
        if not budget.can_spend_odds(usage, cost_per_game, today_eastern):
            print("Odds pull stopped: credit limit reached.")
            break

        home = event.get("home_team")
        away = event.get("away_team")
        commence_iso = event.get("commence_time")

        odds_url = f"{ODDS_SPORT_ROOT}/events/{event_id}/odds"
        try:
            o_resp = requests.get(
                odds_url,
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us",
                    "markets": ",".join(markets),
                    "bookmakers": "draftkings",
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                timeout=15,
            )
        except Exception as e:
            # unknown whether it was charged: book the full cost and stop
            budget.record_odds_request(usage, {}, cost_per_game, today_eastern)
            print(f"⚠️  Failed to fetch odds for event {event_id}: {e}. Stopping the odds pull.")
            break

        reported = budget.record_odds_request(usage, o_resp.headers, cost_per_game, today_eastern)
        try:
            o_resp.raise_for_status()
            data = o_resp.json()
        except Exception as e:
            print(f"⚠️  Failed to fetch odds for event {event_id}: {e}")
            data = {}

        game_rows = 0
        for bk in data.get("bookmakers", []):
            if bk.get("key") not in ODDS_BOOKS:
                continue
            for market in bk.get("markets", []):
                mkey = market.get("key")
                if mkey not in ODDS_MARKETS:
                    continue
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description") or outcome.get("name")
                    game_rows += 1
                    rows.append({
                        "event_id": event_id,
                        "game_start_utc": commence_iso,
                        "game_start_eastern": dt_eastern.isoformat() if dt_eastern else None,
                        "game_date_eastern": event_date_eastern.isoformat() if event_date_eastern else None,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bk["key"],
                        "market": mkey,
                        "player_name": player_name,
                        "player_name_clean": _clean_player_name_odds(player_name),
                        "line": outcome.get("point"),
                        "price": outcome.get("price"),
                    })
        if game_rows:
            status["pulled"] += 1
            status["assists_games"] += int(event_id in extra_ids)

        if not reported:
            print("Odds pull stopped: the odds service did not report credits left.")
            break

    status["credits"] = month["credits"] - credits_before
    if status["credits"] > 0:
        # credits were spent, so this was today's one pull
        usage["odds"]["last_pull_date"] = today_eastern.isoformat()
        finish("ok", True)
    else:
        # nothing was charged (lines not posted yet): a later run today may try again
        finish("no lines posted yet", False)
    print(f"Odds pull used {status['credits']} credits; {usage['odds'].get('remaining')} left.")

    df = pd.DataFrame(rows)
    return df


# =========================================================
# OU RATE (evaluate yesterday's DK props vs game logs)
# =========================================================
def _safe_to_date(x):
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _result_vs_line(actual: float, line: float) -> str | None:
    """Return 'over', 'under', or None for pushes/invalid."""
    if pd.isna(actual) or pd.isna(line):
        return None
    if actual > line:
        return "over"
    if actual < line:
        return "under"
    return None  # push or equal -> neither

def _load_latest_odds_date(odds_path: str):
    """Load ODDS_TODAY.csv, return (exists, max_game_date_eastern, df)."""
    if not os.path.exists(odds_path):
        return False, None, None
    try:
        df = pd.read_csv(odds_path)
    except Exception:
        return False, None, None
    if df.empty or "game_date_eastern" not in df.columns:
        return True, None, df
    dates = df["game_date_eastern"].map(_safe_to_date)
    dates = [d for d in dates if d is not None]
    if not dates:
        return True, None, df
    return True, max(dates), df

def _clean_player_name_logs(name: str) -> str:
    # align with _clean_player_name_odds rules
    return _clean_player_name_odds(name)

def _evaluate_ou_for_day(gamelogs_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a gamelog dataframe for the season and an odds dataframe from 'yesterday',
    compute per-player counts:
      points_over/under, rebounds_over/under, assists_over/under, total_picks
    """
    # Prepare logs: date & cleaned player name
    logs = gamelogs_df.copy()
    logs["GAME_DATE_only"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce").dt.date
    logs["PLAYER_NAME_CLEAN"] = logs["PLAYER_NAME"].astype(str).map(_clean_player_name_logs)

    # Prepare odds: only DK and the 3 markets, take date (EST) and cleaned player name
    odds = odds_df.copy()
    if "bookmaker" in odds.columns:
        odds = odds[odds["bookmaker"] == "draftkings"].copy()
    if "market" in odds.columns:
        odds = odds[odds["market"].isin(["player_points", "player_rebounds", "player_assists"])].copy()

    odds["ODDS_DATE"] = odds["game_date_eastern"].map(_safe_to_date)
    odds["PLAYER_NAME_CLEAN"] = odds["player_name"].map(_clean_player_name_odds)
    # every line comes back twice (one Over outcome, one Under outcome) -> score it once
    odds = odds.drop_duplicates(subset=["PLAYER_NAME_CLEAN", "ODDS_DATE", "market", "line"])

    from collections import defaultdict
    counts = defaultdict(lambda: {
        "player_name": None,
        "points_over": 0, "points_under": 0,
        "rebounds_over": 0, "rebounds_under": 0,
        "assists_over": 0, "assists_under": 0,
        "total_picks": 0
    })

    # For each odds row, find that player's game on that EST date and compare
    for _, row in odds.iterrows():
        pname_clean = row.get("PLAYER_NAME_CLEAN")
        est_date = row.get("ODDS_DATE")
        line = row.get("line")
        market = row.get("market")

        if not pname_clean or est_date is None or pd.isna(line) or not market:
            continue

        g = logs[(logs["PLAYER_NAME_CLEAN"] == pname_clean) & (logs["GAME_DATE_only"] == est_date)]
        if g.empty:
            continue

        if market == "player_points":
            stat_col = "PTS"
            over_key, under_key = "points_over", "points_under"
        elif market == "player_rebounds":
            stat_col = "REB"
            over_key, under_key = "rebounds_over", "rebounds_under"
        elif market == "player_assists":
            stat_col = "AST"
            over_key, under_key = "assists_over", "assists_under"
        else:
            continue

        actual = pd.to_numeric(g.iloc[0].get(stat_col), errors="coerce")
        res = _result_vs_line(actual, line)
        if res is None:
            continue

        bucket = counts[pname_clean]
        if not bucket["player_name"]:
            bucket["player_name"] = g.iloc[0].get("PLAYER_NAME") or row.get("player_name") or pname_clean

        if res == "over":
            bucket[over_key] += 1
        elif res == "under":
            bucket[under_key] += 1
        bucket["total_picks"] += 1

    if not counts:
        return pd.DataFrame(columns=[
            "player_name", "player_name_clean",
            "points_over", "points_under",
            "rebounds_over", "rebounds_under",
            "assists_over", "assists_under",
            "total_picks"
        ])

    rows = []
    for cname, data in counts.items():
        rows.append({
            "player_name": data["player_name"] or cname,
            "player_name_clean": cname,
            "points_over": int(data["points_over"]),
            "points_under": int(data["points_under"]),
            "rebounds_over": int(data["rebounds_over"]),
            "rebounds_under": int(data["rebounds_under"]),
            "assists_over": int(data["assists_over"]),
            "assists_under": int(data["assists_under"]),
            "total_picks": int(data["total_picks"]),
        })
    return pd.DataFrame(rows)

def _merge_into_existing_ou_rate(new_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    If OU_RATE.csv exists, add counts (cumulative per player_name_clean).
    Otherwise just return new_df. Writes to disk and returns merged.
    """
    cols = [
        "player_name", "player_name_clean",
        "points_over", "points_under",
        "rebounds_over", "rebounds_under",
        "assists_over", "assists_under",
        "total_picks"
    ]
    for c in cols:
        if c not in new_df.columns:
            new_df[c] = 0 if c != "player_name" and c != "player_name_clean" else ""

    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=cols)
        for c in cols:
            if c not in old.columns:
                old[c] = 0 if c not in ("player_name", "player_name_clean") else ""
        # keys written by older versions used a looser cleaning rule
        old["player_name_clean"] = old["player_name_clean"].map(_clean_player_name_odds)
        both = pd.concat([old[cols], new_df[cols]], ignore_index=True)

        def pick_name(s):
            s = s.dropna().astype(str)
            s = s[s.str.strip() != ""]
            return s.iloc[-1] if len(s) else ""
        merged = both.groupby("player_name_clean", as_index=False).agg({
            "player_name": pick_name,
            "points_over": "sum", "points_under": "sum",
            "rebounds_over": "sum", "rebounds_under": "sum",
            "assists_over": "sum", "assists_under": "sum",
            "total_picks": "sum",
        })
    else:
        merged = new_df[cols].copy()

    merged.to_csv(path, index=False)
    return merged

def maybe_update_ou_rate_after_gamelogs(tag_short: str):
    """
    Called right after writing the most recent season's *_GAMELOG.csv but BEFORE re-writing ODDS_TODAY.
    If ODDS_TODAY.csv exists, is from an earlier day (US/Eastern) and has not been scored yet,
    evaluate over/under results against the gamelogs and write/update OU_RATE.csv.
    """
    eastern = pytz.timezone("US/Eastern")
    today_eastern = datetime.now(eastern).date()

    odds_path = data_path("ODDS_TODAY.csv")
    gl_path = data_path(f"{tag_short}_GAMELOG.csv")  # e.g., "25-26_GAMELOG.csv"

    exists, odds_max_date, odds_df = _load_latest_odds_date(odds_path)
    if not exists:
        return
    if odds_max_date is None:
        return

    # Score each saved set of lines exactly once, on any day after its games
    # (not only the very next day, so a missed day doesn't lose the results).
    state = load_state()
    scored_dates = state.get("ou_scored_dates", [])
    if odds_max_date >= today_eastern or odds_max_date.isoformat() in scored_dates:
        return

    if not os.path.exists(gl_path):
        return

    try:
        gamelogs_df = pd.read_csv(gl_path, parse_dates=["GAME_DATE"])
    except Exception:
        return

    new_scores = _evaluate_ou_for_day(gamelogs_df, odds_df)
    if new_scores.empty:
        return
    _merge_into_existing_ou_rate(new_scores, data_path("OU_RATE.csv"))
    state["ou_scored_dates"] = (scored_dates + [odds_max_date.isoformat()])[-60:]
    save_state(state)
    print(f"Scored the {odds_max_date} lines → OU_RATE.csv")


# =========================================================
# PIPELINE
# =========================================================
def main():
    # fetch per-game for all seasons
    seasons = list(SEASONS)
    per_game_by_season = {s: fetch_per_game(s) for s in seasons}

    # October before opening night: the new season has no games yet -> step back one
    if per_game_by_season[seasons[-1]].empty:
        not_started = seasons.pop()
        older = season_str(int(seasons[0][:4]) - 1)
        print(f"{not_started} has no games yet, using {older} and {seasons[0]} instead.")
        per_game_by_season.pop(not_started)
        per_game_by_season[older] = fetch_per_game(older)
        seasons.insert(0, older)

    most_recent = seasons[-1]
    prev_season = seasons[-2] if len(seasons) >= 2 else None

    for idx, season in enumerate(seasons):
        tag = season_short(season)
        print(f"\n=== {season} ({tag}) ===")

        current_pg = per_game_by_season[season]
        prev_pg = per_game_by_season.get(seasons[idx - 1]) if idx - 1 >= 0 else None

        # keepers & AVGS
        keepers = current_pg.loc[
            current_pg["MIN"] >= MIN_THRESHOLD,
            ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","MIN","PTS","REB","AST","BLK","STL"]
        ].copy()
        out_pg = f"{tag}_AVGS.csv"
        keepers.to_csv(data_path(out_pg), index=False, float_format="%.2f")
        print(f"Saved {len(keepers)} players (≥{MIN_THRESHOLD} MIN)  → {out_pg}")

        # GAMELOG
        logs_all = fetch_league_player_gamelogs(season, SEASON_TYPE)
        logs = logs_all[logs_all["PLAYER_ID"].isin(keepers["PLAYER_ID"])].copy()
        out_logs = f"{tag}_GAMELOG.csv"
        logs.to_csv(data_path(out_logs), index=False, float_format="%.2f")
        print(f"Saved {len(logs)} game rows (filtered) → {out_logs}")

        # LAST5 build
        last5 = last_n_averages(logs, n=LAST_N)
        keepers_norm, kee_and_base = choose_baseline(
            current_pg=current_pg,
            prev_pg=prev_pg,
            last5=last5,
            min_threshold=MIN_THRESHOLD,
            n=LAST_N
        )
        recent_vs_baseline = build_recent_vs_baseline(last5, kee_and_base)

        if season == most_recent:
            # write LAST5.csv
            out_cmp = "LAST5.csv"
            recent_vs_baseline = recent_vs_baseline.round(2)
            recent_vs_baseline.to_csv(data_path(out_cmp), index=False, float_format="%.2f")
            print(f"Saved last-{LAST_N} vs baseline comparison → {out_cmp}")

            # make 2-season logs
            if prev_season is not None:
                logs_prev_all = fetch_league_player_gamelogs(prev_season, SEASON_TYPE)
                logs_prev = logs_prev_all[logs_prev_all["PLAYER_ID"].isin(keepers["PLAYER_ID"])].copy()
                logs_two = pd.concat([logs_prev, logs], ignore_index=True)
            else:
                logs_two = logs.copy()

            # SIMILAR_NEXT
            similar_next = build_similar_next_csv(logs_two, logs, keepers).round(2)
            similar_next.to_csv(data_path("SIMILAR_NEXT.csv"), index=False, float_format="%.2f")
            print("Saved similar-game next-performance averages → SIMILAR_NEXT.csv")

            # REST_DAYS (replaces MOST_SIMILAR_GAME in the pipeline)
            rest_days = build_rest_day_next_only(logs_two, logs, keepers).round(2)
            rest_days.to_csv(data_path("REST_DAYS.csv"), index=False, float_format="%.2f")
            print("Saved rest-day-based next performance → REST_DAYS.csv")

            # OPPONENT_BIAS
            opponent_bias = build_opponent_bias_next_only(logs_two, logs, keepers).round(2)
            opponent_bias.to_csv(data_path("OPPONENT_BIAS.csv"), index=False, float_format="%.2f")
            print("Saved next-opponent bias vs season averages → OPPONENT_BIAS.csv")

            # OU_RATE: if yesterday's ODDS_TODAY matches the rule, score and update OU_RATE.csv
            maybe_update_ou_rate_after_gamelogs(season_short(season))

            # INJURIES: league-wide injury report
            injury_df = fetch_nba_injury_report()
            if not injury_df.empty:
                injury_df.to_csv(data_path("INJURY_REPORT.csv"), index=False)
                print("Saved injury report → INJURY_REPORT.csv")
            else:
                print("No injury data saved (injury fetch returned empty).")

            # ODDS: per-event props
            odds_df = fetch_nba_player_props_today()
            if not odds_df.empty:
                odds_df.to_csv(data_path("ODDS_TODAY.csv"), index=False)
                print("Saved odds → ODDS_TODAY.csv")
            else:
                print("No odds saved (no props returned).")


if __name__ == "__main__":
    main()