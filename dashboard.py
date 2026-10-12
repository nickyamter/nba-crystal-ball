# dashboard.py
# Writes dashboard.html: one self-contained page with today's data embedded.
# Open it in any browser; no server needed. The weight sliders re-blend in the page.

import json
from datetime import datetime

import pandas as pd

import budget
import model
from common import BASE_DIR, data_path

TEMPLATE_PATH = BASE_DIR / "dashboard_template.html"
OUTPUT_PATH = BASE_DIR / "dashboard.html"

METRIC_LABELS = {
    "season": "Season avg",
    "last5": "Last 5",
    "similar": "Similar next",
    "rest": "Rest days",
    "opp": "Vs opponent",
}
TOP_BLENDS_SHOWN = 15
TOP_BLENDS_USED = 5  # same as the app's "Suggested Parlays"


def _num(val, digits=2):
    """float -> rounded number, NaN/None -> None (so it becomes JSON null)."""
    if val is None or pd.isna(val):
        return None
    return round(float(val), digits)


def _int(val):
    if val is None or pd.isna(val):
        return None
    return int(val)


def build_data() -> dict:
    now = datetime.now(budget.EASTERN)
    today = now.date().isoformat()
    tag = model.latest_season_tag()
    players = model.build_players()

    odds_df = model.load_csv("ODDS_TODAY.csv")
    lines_date = model.odds_date(odds_df)
    games_with_lines = int(odds_df["event_id"].nunique()) if "event_id" in odds_df.columns else 0

    stats_through = None
    if tag and data_path(f"{tag}_GAMELOG.csv").exists():
        dates = pd.read_csv(data_path(f"{tag}_GAMELOG.csv"), usecols=["GAME_DATE"])["GAME_DATE"]
        if len(dates):
            stats_through = str(dates.max())[:10]

    rows = []
    for _, r in players.iterrows():
        opp = r.get("NEXT_OPP")
        rows.append({
            "name": r["PLAYER_NAME"],
            "team": r["TEAM_ABBREVIATION"],
            "opp": None if pd.isna(opp) else str(opp),
            "sig": {s: [_num(r[c]) for c in model.signal_columns(s)] for s in model.STATS},
            "line": {s: _num(r[f"DK_LINE_{s}"], 1) for s in model.LINE_STATS},
            "ou": {
                s: [_int(r[f"{model.OU_PREFIX[s]}_over"]) or 0, _int(r[f"{model.OU_PREFIX[s]}_under"]) or 0]
                for s in model.LINE_STATS
            },
            "n": {
                "similar": {s: _int(r[f"{s}_similar_n"]) for s in model.STATS},
                "rest_days": _int(r["REST_NEXT"]),
                "rest_gp": _int(r["REST_NEXT_GP"]),
                "opp_gp": _int(r["VS_OPP_GP"]),
            },
        })

    blends = [
        {
            "w": [_num(b[c]) for c in model.WEIGHT_COLS],
            "picks": _int(b["total_picks"]),
            "hits": _int(b["total_hits"]),
            "rate": _num(b["hit_rate"], 3),
        }
        for _, b in model.top_weight_combos(TOP_BLENDS_SHOWN).iterrows()
    ]

    usage = budget.load_usage()
    odds_usage = usage["odds"]
    odds_today = odds_usage.get("today") or {}
    if odds_today.get("date") != today:
        odds_today = {}

    return {
        "generated_at": now.strftime("%a %b %-d, %-I:%M %p ET"),
        "today": today,
        "season": tag,
        "stats_through": stats_through,
        "metrics": [{"key": k, "label": METRIC_LABELS[k]} for k in model.METRICS],
        "stats": model.STATS,
        "line_stats": model.LINE_STATS,
        "players": rows,
        "lines_date": lines_date,
        "lines_today": lines_date == today,
        "games_with_lines": games_with_lines,
        "blends": blends,
        "blends_used": TOP_BLENDS_USED,
        "credits": {
            "remaining": odds_usage.get("remaining"),
            "plan": budget.ODDS_FREE_CREDITS,
            "reserve": budget.ODDS_RESERVE,
            "spent_this_month": odds_usage["months"].get(today[:7], {}).get("credits", 0),
            "today": odds_today,
        },
    }


def build() -> str:
    data = json.dumps(build_data(), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    return str(OUTPUT_PATH)


if __name__ == "__main__":
    print(f"Wrote {build()}")
