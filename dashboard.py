# dashboard.py
# Writes dashboard.html: one self-contained page with today's data embedded.
# Open it in any browser; no server needed. The weight sliders re-blend in the page.

import json
from datetime import datetime

import pandas as pd

import budget
import model
from common import BASE_DIR, data_path, clean_player_name

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


def _sentence(*parts) -> str:
    """('Left', 'Knee', 'Tendinitis') -> 'Left knee tendinitis'"""
    text = " ".join(str(p).strip() for p in parts if p is not None and not pd.isna(p) and str(p).strip())
    return text[:1].upper() + text[1:].lower() if text else ""


def _injury_label(status, abbr) -> str:
    """ESPN's status codes in plain words."""
    abbr = "" if abbr is None or pd.isna(abbr) else str(abbr).upper()
    if abbr == "GTD":
        return "Questionable"
    if abbr == "OFS":
        return "Out for season"
    if abbr == "OUT":
        return "Out"
    return "" if status is None or pd.isna(status) else str(status)


def _injury_entry(status, abbr, side, kind, detail, ret) -> dict:
    return {
        "label": _injury_label(status, abbr),
        "what": _sentence(side, kind, detail),
        "ret": None if ret is None or pd.isna(ret) else str(ret)[:10],
    }


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

    # today's injury report, by team (everyone listed, not only the players projected here)
    report = model.load_csv("INJURY_REPORT.csv")
    injury_today = (
        not report.empty and "snapshot_date" in report.columns
        and (report["snapshot_date"].astype(str) == today).any()
    )
    injured = {}
    if injury_today:
        for _, i in report.iterrows():
            entry = _injury_entry(i["status"], i["fantasy_status_abbr"], i["injury_side"],
                                  i["injury_type"], i["injury_detail"], i["return_date"])
            entry["name"] = i["player_name"]
            entry["key"] = clean_player_name(i["player_name"])
            injured.setdefault(str(i["team_abbr"]), []).append(entry)

    rows = []
    for _, r in players.iterrows():
        opp = r.get("NEXT_OPP")
        inj = None
        if injury_today and str(r["INJ_DATE"]) == today and not pd.isna(r["INJ_STATUS"]):
            inj = _injury_entry(r["INJ_STATUS"], r["INJ_ABBR"], r["INJ_SIDE"], r["INJ_TYPE"], r["INJ_DETAIL"], r["INJ_RETURN"])
        # "first game back": he sat out his team's most recent game(s). Counted when the
        # listed reason is an injury, or when no reason is listed and the games were
        # this season (so last spring's rested starters aren't all flagged on opening night).
        missed = _int(r["GAMES_MISSED"]) or 0
        injury = r["MISSED_FOR_INJURY"] == True
        unexplained = not injury and r["MISSED_FOR_OTHER"] != True and (_int(r["MISSED_THIS_SEASON"]) or 0) > 0
        back = None
        if missed > 0 and (injury or unexplained):
            back = {"games": missed, "reason": _sentence(r["LAST_REASON"]) if injury else ""}
        rows.append({
            "name": r["PLAYER_NAME"],
            "key": r["player_name_clean"],
            "inj": inj,
            "back": back,
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
        "injury_today": bool(injury_today),
        "injured": injured,
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
