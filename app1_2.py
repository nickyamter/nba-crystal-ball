import streamlit as st
import pandas as pd
from pathlib import Path
import random
import datetime
import itertools
import re
import numpy as np

from common import DATA_DIR, data_path, clean_player_name

st.set_page_config(page_title="NBA Prop Blender", layout="wide")
st.title("NBA Metric Blender")

# ============================================================
# CONFIG / PATHS
# ============================================================
def latest_season_tag() -> str | None:
    """Newest season the scraper has written, e.g. '25-26' (from 25-26_AVGS.csv)."""
    tags = sorted(
        m.group(1)
        for m in (re.fullmatch(r"(\d\d-\d\d)_AVGS\.csv", p.name) for p in DATA_DIR.glob("*_AVGS.csv"))
        if m
    )
    return tags[-1] if tags else None

SEASON_TAG = latest_season_tag()
EXPERIMENT_CSV = data_path("weight_experiments.csv")
GAMELOG_CSV = data_path(f"{SEASON_TAG}_GAMELOG.csv")
HITRATE_CSV = data_path("WEIGHT_HITRATE.csv")

# ============================================================
# helper to load csvs safely
# ============================================================
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

# ============================================================
# load base data
# ============================================================
season_df = load_csv(data_path(f"{SEASON_TAG}_AVGS.csv"))
last5_df = load_csv(data_path("LAST5.csv"))
similar_df = load_csv(data_path("SIMILAR_NEXT.csv"))
rest_days_df = load_csv(data_path("REST_DAYS.csv"))  # new metric
opp_bias_df = load_csv(data_path("OPPONENT_BIAS.csv"))
odds_df = load_csv(data_path("ODDS_TODAY.csv"))
ou_df = load_csv(data_path("OU_RATE.csv"))  # NEW

# ============================================================
# sliders
# ============================================================
st.sidebar.header("Metric Weights")
w_season = st.sidebar.slider("Season Avg weight", 0.0, 1.0, 0.2, 0.05)
w_last5 = st.sidebar.slider("Last 5 weight", 0.0, 1.0, 0.2, 0.05)
w_similar = st.sidebar.slider("Similar Next weight", 0.0, 1.0, 0.2, 0.05)
w_rest = st.sidebar.slider("Rest Days weight", 0.0, 1.0, 0.2, 0.05)  # renamed
w_opp = st.sidebar.slider("Opponent Bias weight", 0.0, 1.0, 0.2, 0.05)

if season_df.empty:
    st.error(f"No season averages (*_AVGS.csv) found in {DATA_DIR}. Run your scraping script first.")
    st.stop()

# ============================================================
# build players frame
# ============================================================
players = season_df[[
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
]].copy()

players = players.rename(columns={
    "PTS": "SEASON_PTS",
    "REB": "SEASON_REB",
    "AST": "SEASON_AST",
    "STL": "SEASON_STL",
    "BLK": "SEASON_BLK",
})

# Clean player name once (used for odds + OU_RATE merges)
players["player_name_clean"] = players["PLAYER_NAME"].map(clean_player_name)

def safe_merge(base, df, cols):
    if not df.empty:
        valid = ["PLAYER_ID"] + [c for c in cols if c in df.columns]
        return base.merge(df[valid], on="PLAYER_ID", how="left")
    else:
        for c in cols:
            base[c] = float("nan")
        return base

players = safe_merge(players, last5_df, ["PTS_last5","REB_last5","AST_last5","STL_last5","BLK_last5"])
players = safe_merge(players, similar_df, ["PTS_next_avg","REB_next_avg","AST_next_avg","STL_next_avg","BLK_next_avg"])
players = safe_merge(players, rest_days_df, ["PTS_next","REB_next","AST_next","STL_next","BLK_next"])
players = safe_merge(players, opp_bias_df, ["PTS_vs_opp","REB_vs_opp","AST_vs_opp","STL_vs_opp","BLK_vs_opp"])

# ============================================================
# weighted prediction using CURRENT SLIDERS
# ============================================================
def weighted_pred(row, stat: str, weights=None):
    if weights is None:
        weights = {
            "season": w_season,
            "last5": w_last5,
            "similar": w_similar,
            "rest": w_rest,
            "opp": w_opp,
        }
    metric_values = {
        "season": row.get(f"SEASON_{stat}", float("nan")),
        "last5": row.get(f"{stat}_last5", float("nan")),
        "similar": row.get(f"{stat}_next_avg", float("nan")),
        "rest": row.get(f"{stat}_next", float("nan")),
        "opp": row.get(f"{stat}_vs_opp", float("nan")),
    }
    num, denom = 0.0, 0.0
    for key, val in metric_values.items():
        w = weights.get(key, 0.0)
        if pd.notna(val) and w > 0:
            num += val * w
            denom += w
    if denom == 0:
        return float("nan")
    return round(num / denom, 2)

for stat in ["PTS","REB","AST","STL","BLK"]:
    players[f"PRED_{stat}"] = players.apply(lambda r: weighted_pred(r, stat), axis=1)

# Predicted PRA
players["Predicted PRA"] = (
    players["PRED_PTS"].fillna(0)
    + players["PRED_REB"].fillna(0)
    + players["PRED_AST"].fillna(0)
)

# ============================================================
# integrate DK odds
# ============================================================
if not odds_df.empty:
    odds_df["player_name_clean"] = odds_df["player_name"].map(clean_player_name)

    tmp = []
    for _, r in odds_df.iterrows():
        tmp.append({
            "player_name_clean": r["player_name_clean"],
            r["market"]: r["line"],
        })
    odds_pivot = pd.DataFrame(tmp).groupby("player_name_clean").agg("max").reset_index()

    players = players.merge(odds_pivot, on="player_name_clean", how="left")
    players = players.rename(columns={
        "player_points": "DK_LINE_PTS",
        "player_rebounds": "DK_LINE_REB",
        "player_assists": "DK_LINE_AST",
    })

    players["PTS Difference"] = players["PRED_PTS"] - players["DK_LINE_PTS"]
    players["REB Difference"] = players["PRED_REB"] - players["DK_LINE_REB"]
    players["AST Difference"] = players["PRED_AST"] - players["DK_LINE_AST"]
else:
    players[["PTS Difference","REB Difference","AST Difference"]] = float("nan")

# ============================================================
# MERGE OU_RATE + compute Over/Under Rates
# ============================================================
def _rate(numer, denom):
    try:
        if pd.isna(numer) or pd.isna(denom) or denom <= 0:
            return np.nan
        return round(float(numer) / float(denom), 3)
    except Exception:
        return np.nan

if not ou_df.empty:
    # Make sure clean key is uppercase and normalized
    if "player_name_clean" in ou_df.columns:
        ou_df["player_name_clean"] = ou_df["player_name_clean"].map(clean_player_name)
    else:
        ou_df["player_name_clean"] = ou_df.get("player_name", "").map(clean_player_name)

    # Totals per stat
    ou_df["PTS_total"] = ou_df.get("points_over", 0) + ou_df.get("points_under", 0)
    ou_df["REB_total"] = ou_df.get("rebounds_over", 0) + ou_df.get("rebounds_under", 0)
    ou_df["AST_total"] = ou_df.get("assists_over", 0) + ou_df.get("assists_under", 0)

    rates = pd.DataFrame({
        "player_name_clean": ou_df["player_name_clean"],
        "PTS Over Rate": [_rate(o, t) for o, t in zip(ou_df.get("points_over", 0), ou_df["PTS_total"])],
        "PTS Under Rate": [_rate(u, t) for u, t in zip(ou_df.get("points_under", 0), ou_df["PTS_total"])],
        "REB Over Rate": [_rate(o, t) for o, t in zip(ou_df.get("rebounds_over", 0), ou_df["REB_total"])],
        "REB Under Rate": [_rate(u, t) for u, t in zip(ou_df.get("rebounds_under", 0), ou_df["REB_total"])],
        "AST Over Rate": [_rate(o, t) for o, t in zip(ou_df.get("assists_over", 0), ou_df["AST_total"])],
        "AST Under Rate": [_rate(u, t) for u, t in zip(ou_df.get("assists_under", 0), ou_df["AST_total"])],
    }).drop_duplicates(subset=["player_name_clean"])

    players = players.merge(rates, on="player_name_clean", how="left")
else:
    for c in ["PTS Over Rate","PTS Under Rate","REB Over Rate","REB Under Rate","AST Over Rate","AST Under Rate"]:
        players[c] = np.nan

# ============================================================
# rename for display
# ============================================================
players = players.rename(columns={
    "PLAYER_NAME": "Player",
    "TEAM_ABBREVIATION": "Team",
    "PRED_PTS": "Predicted PTS",
    "PRED_AST": "Predicted AST",
    "PRED_REB": "Predicted REB",
    "PRED_STL": "Predicted STL",
    "PRED_BLK": "Predicted BLK",
})

# ============================================================
# main table
# ============================================================
display_cols = [
    "Player",
    "Team",
    "PTS Difference",
    "AST Difference",
    "REB Difference",
    "Predicted PTS",
    "Predicted AST",
    "Predicted REB",
    "Predicted STL",
    "Predicted BLK",
    "Predicted PRA",
]

# Append the 6 OU-rate columns at the end
ou_display_cols = [
    "PTS Over Rate","PTS Under Rate",
    "REB Over Rate","REB Under Rate",
    "AST Over Rate","AST Under Rate",
]
display_cols = [c for c in display_cols if c in players.columns] + [c for c in ou_display_cols if c in players.columns]

st.subheader("Predictions vs DK Lines")
st.dataframe(players[display_cols].sort_values("Player"), width="stretch")

# ============================================================
# Helper to fetch a player's OU rate for a given stat + pick direction
# ============================================================
def get_ou_rate(player_name: str, stat: str, pick_dir: str) -> float | None:
    """
    stat in {"PTS","REB","AST"}; pick_dir in {"OVER","UNDER"}
    returns float or None
    """
    row = players.loc[players["Player"] == player_name]
    if row.empty:
        return None
    col = f"{stat} {'Over' if pick_dir=='OVER' else 'Under'} Rate"
    if col in players.columns:
        val = row.iloc[0].get(col)
        if pd.notna(val):
            return float(val)
    return None

def fmt_rate(rate: float | None, pick_dir: str) -> str:
    if rate is None or pd.isna(rate):
        return ""
    # Put in parentheses immediately after diff as requested
    return f"; {pick_dir.lower()} rate: {rate:.2f}"

def load_top_weight_combos(k: int = 5):
    path = Path(HITRATE_CSV)
    if not path.exists():
        return []

    hitrate_df = pd.read_csv(path)

    # Handle legacy 'w_most' -> 'w_rest' migration if needed
    if "w_rest" not in hitrate_df.columns and "w_most" in hitrate_df.columns:
        hitrate_df = hitrate_df[hitrate_df["w_most"] == 0].rename(columns={"w_most": "w_rest"})

    needed_cols = ["w_season", "w_last5", "w_similar", "w_rest", "w_opp", "hit_rate"]
    if not all(c in hitrate_df.columns for c in needed_cols):
        return []

    hitrate_df = hitrate_df.sort_values("hit_rate", ascending=False)
    top = hitrate_df.head(k)

    combos = []
    for _, r in top.iterrows():
        combos.append({
            "season": r["w_season"],
            "last5": r["w_last5"],
            "similar": r["w_similar"],
            "rest": r["w_rest"],
            "opp": r["w_opp"],
        })
    return combos

# ============================================================
# FUNCTIONS FOR RANDOM BLENDS (suggested parlays)
# ============================================================
def predict_with_custom_weights(df, weights):
    temp = df.copy()

    def wp(row, stat):
        return weighted_pred(row, stat, weights=weights)

    for stat in ["PTS","REB","AST"]:
        temp[f"PRED_{stat}_TMP"] = temp.apply(lambda r: wp(r, stat), axis=1)

    temp["PTS_Diff_TMP"] = temp["PRED_PTS_TMP"] - temp["DK_LINE_PTS"]
    temp["REB_Diff_TMP"] = temp["PRED_REB_TMP"] - temp["DK_LINE_REB"]
    temp["AST_Diff_TMP"] = temp["PRED_AST_TMP"] - temp["DK_LINE_AST"]

    cols_wanted = [
        "PLAYER_ID",
        "Player",
        "Team",
        "DK_LINE_PTS",
        "DK_LINE_REB",
        "DK_LINE_AST",
        "PTS_Diff_TMP",
        "REB_Diff_TMP",
        "AST_Diff_TMP",
    ]
    cols_present = [c for c in cols_wanted if c in temp.columns]
    return temp[cols_present]

def run_wildcards_for_parlay(players_df, num_trials=5):
    trials = []

    # Use top hit-rate combos first
    weight_combos = load_top_weight_combos(num_trials)
    if weight_combos:
        for w_combo in weight_combos:
            trials.append(predict_with_custom_weights(players_df, w_combo))
    else:
        # Fallback to random if no hit-rate data yet
        for _ in range(num_trials):
            raw = [random.choice([i / 20 for i in range(21)]) for _ in range(5)]
            s = sum(raw) if sum(raw) != 0 else 1.0
            w_combo = {
                "season": raw[0] / s,
                "last5": raw[1] / s,
                "similar": raw[2] / s,
                "rest": raw[3] / s,
                "opp": raw[4] / s,
            }
            trials.append(predict_with_custom_weights(players_df, w_combo))

    base = players_df[[
        "PLAYER_ID",
        "Player",
        "Team",
        "DK_LINE_PTS",
        "DK_LINE_REB",
        "DK_LINE_AST",
    ]].drop_duplicates(subset=["PLAYER_ID"]).copy()

    for stat in ["PTS","REB","AST"]:
        base[f"{stat}_sum"] = 0.0
        base[f"{stat}_cnt"] = 0

    for tdf in trials:
        merged = base[["PLAYER_ID"]].merge(tdf, on="PLAYER_ID", how="left")
        for stat in ["PTS","REB","AST"]:
            diff_col = f"{stat}_Diff_TMP"
            if diff_col not in merged.columns:
                continue
            vals = merged[diff_col]
            mask = vals.notna()
            base.loc[mask, f"{stat}_sum"] += vals[mask]
            base.loc[mask, f"{stat}_cnt"] += 1

    for stat in ["PTS","REB","AST"]:
        base[f"{stat}_avg_diff"] = base.apply(
            lambda r: (r[f"{stat}_sum"] / r[f"{stat}_cnt"]) if r[f"{stat}_cnt"] > 0 else float("nan"),
            axis=1,
        )

    picks_by_stat = {}
    for stat, dk_col in [("PTS","DK_LINE_PTS"),("REB","DK_LINE_REB"),("AST","DK_LINE_AST")]:
        if dk_col not in base.columns:
            continue
        sdf = base.dropna(subset=[dk_col, f"{stat}_avg_diff"]).copy()
        if sdf.empty:
            continue
        sdf["abs_diff"] = sdf[f"{stat}_avg_diff"].abs()
        sdf = sdf.sort_values("abs_diff", ascending=False)
        picks_by_stat[stat] = sdf
    return picks_by_stat

# ============================================================
# Suggested Parlays (random blends)
# ============================================================
st.subheader("Suggested Parlays")

if "DK_LINE_PTS" in players.columns and "DK_LINE_REB" in players.columns and "DK_LINE_AST" in players.columns:
    p1_picks_by_stat = run_wildcards_for_parlay(players)
    parlay1 = []
    for stat, dk_col in [("PTS","DK_LINE_PTS"),("REB","DK_LINE_REB"),("AST","DK_LINE_AST")]:
        sdf = p1_picks_by_stat.get(stat)
        if sdf is None or sdf.empty:
            continue
        row = sdf.iloc[0]
        pick_dir = "OVER" if row[f"{stat}_avg_diff"] > 0 else "UNDER"
        rate = get_ou_rate(row["Player"], stat, pick_dir)
        parlay1.append({
            "stat": stat,
            "player": row["Player"],
            "team": row.get("Team",""),
            "line": row[dk_col],
            "avg_diff": row[f"{stat}_avg_diff"],
            "pick": pick_dir,
            "rate": rate,
        })

    p2_picks_by_stat = run_wildcards_for_parlay(players)
    parlay2 = []
    for stat, dk_col in [("PTS","DK_LINE_PTS"),("REB","DK_LINE_REB"),("AST","DK_LINE_AST")]:
        sdf = p2_picks_by_stat.get(stat)
        if sdf is None or sdf.empty:
            continue
        row = sdf.iloc[0]

        p1_player_for_stat = None
        for p in parlay1:
            if p["stat"] == stat:
                p1_player_for_stat = p["player"]
                break

        if p1_player_for_stat is not None and row["Player"] == p1_player_for_stat and len(sdf) > 1:
            row = sdf.iloc[1]

        pick_dir = "OVER" if row[f"{stat}_avg_diff"] > 0 else "UNDER"
        rate = get_ou_rate(row["Player"], stat, pick_dir)
        parlay2.append({
            "stat": stat,
            "player": row["Player"],
            "team": row.get("Team",""),
            "line": row[dk_col],
            "avg_diff": row[f"{stat}_avg_diff"],
            "pick": pick_dir,
            "rate": rate,
        })

    if parlay1:
        st.markdown("**Parlay 1**")
        for pick in parlay1:
            rate_text = fmt_rate(pick["rate"], pick["pick"])
            st.write(
                f"{pick['player']} ({pick['team']}) — {pick['stat']} {pick['pick']} {pick['line']} "
                f"(avg diff: {pick['avg_diff']:.2f}{rate_text})"
            )
    if parlay2:
        st.markdown("**Parlay 2**")
        for pick in parlay2:
            rate_text = fmt_rate(pick["rate"], pick["pick"])
            st.write(
                f"{pick['player']} ({pick['team']}) — {pick['stat']} {pick['pick']} {pick['line']} "
                f"(avg diff: {pick['avg_diff']:.2f}{rate_text})"
            )
else:
    st.info("Need DK lines for PTS/REB/AST to build parlay suggestions.")

# ============================================================
# Weighted Suggestions (use current slider-weighted predictions)
# ============================================================
st.subheader("Weighted Suggestions")

needed_cols = [
    "Player", "Team",
    "DK_LINE_PTS", "DK_LINE_REB", "DK_LINE_AST",
    "PTS Difference", "REB Difference", "AST Difference",
]
if all(c in players.columns for c in needed_cols):
    wdf = players.copy()

    stat_info = {
        "PTS": ("PTS Difference", "DK_LINE_PTS"),
        "REB": ("REB Difference", "DK_LINE_REB"),
        "AST": ("AST Difference", "DK_LINE_AST"),
    }

    w_parlay1 = []
    for stat, (diff_col, dk_col) in stat_info.items():
        sdf = wdf.dropna(subset=[diff_col, dk_col]).copy()
        if sdf.empty:
            continue
        sdf["abs_diff_tmp"] = sdf[diff_col].abs()
        sdf = sdf.sort_values("abs_diff_tmp", ascending=False)
        top = sdf.iloc[0]
        pick_dir = "OVER" if top[diff_col] > 0 else "UNDER"
        rate = get_ou_rate(top["Player"], stat, pick_dir)
        w_parlay1.append({
            "stat": stat,
            "player": top["Player"],
            "team": top["Team"],
            "line": top[dk_col],
            "diff": top[diff_col],
            "pick": pick_dir,
            "rate": rate,
        })

    w_parlay2 = []
    for stat, (diff_col, dk_col) in stat_info.items():
        sdf = wdf.dropna(subset=[diff_col, dk_col]).copy()
        if sdf.empty:
            continue
        sdf["abs_diff_tmp"] = sdf[diff_col].abs()
        sdf = sdf.sort_values("abs_diff_tmp", ascending=False)

        p1_player = None
        for p in w_parlay1:
            if p["stat"] == stat:
                p1_player = p["player"]
                break

        chosen = None
        for _, row in sdf.iterrows():
            if row["Player"] != p1_player:
                chosen = row
                break

        if chosen is not None:
            pick_dir = "OVER" if chosen[diff_col] > 0 else "UNDER"
            rate = get_ou_rate(chosen["Player"], stat, pick_dir)
            w_parlay2.append({
                "stat": stat,
                "player": chosen["Player"],
                "team": chosen["Team"],
                "line": chosen[dk_col],
                "diff": chosen[diff_col],
                "pick": pick_dir,
                "rate": rate,
            })

    if w_parlay1:
        st.markdown("**Parlay 1 (weighted)**")
        for pick in w_parlay1:
            rate_text = fmt_rate(pick["rate"], pick["pick"])
            st.write(
                f"{pick['player']} ({pick['team']}) — {pick['stat']} {pick['pick']} {pick['line']} "
                f"(diff: {pick['diff']:.2f}{rate_text})"
            )
    if w_parlay2:
        st.markdown("**Parlay 2 (weighted)**")
        for pick in w_parlay2:
            rate_text = fmt_rate(pick["rate"], pick["pick"])
            st.write(
                f"{pick['player']} ({pick['team']}) — {pick['stat']} {pick['pick']} {pick['line']} "
                f"(diff: {pick['diff']:.2f}{rate_text})"
            )
else:
    st.info("Need DK lines and current differences to build weighted suggestions.")

# ============================================================
# DAILY EXPLORATION (FULL COMBINATIONS SUMMING TO 1.0)
# ============================================================
st.subheader("Weight Exploration")

STEP_VALUES = [i / 20 for i in range(21)]
# Eastern date, to line up with game_date_eastern in ODDS_TODAY.csv
today_str = pd.Timestamp.now(tz="US/Eastern").date().isoformat()
exp_path = Path(EXPERIMENT_CSV)
hitrate_path = Path(HITRATE_CSV)

# Load or create experiment CSV
if exp_path.exists():
    exp_df = pd.read_csv(exp_path)
    # migrate old name
    if "w_rest" not in exp_df.columns and "w_most" in exp_df.columns:
        old_rows = len(exp_df)
        exp_df = exp_df.rename(columns={"w_most": "w_rest"})
        exp_df = exp_df[exp_df["w_rest"] == 0].copy()
        st.success(f"✅ Migrated {len(exp_df)} legacy experiment rows from w_most → w_rest (where w_most == 0).")
else:
    exp_df = pd.DataFrame(columns=[
        "date", "player", "team", "stat", "dk_line", "pick", "actual", "hit",
        "w_season", "w_last5", "w_similar", "w_rest", "w_opp"
    ])

# Load or create WEIGHT_HITRATE.csv
if hitrate_path.exists():
    hitrate_df = pd.read_csv(hitrate_path)
    if "w_rest" not in hitrate_df.columns and "w_most" in hitrate_df.columns:
        old_rows = len(hitrate_df)
        migrated = hitrate_df[hitrate_df["w_most"] == 0].copy()
        migrated = migrated.rename(columns={"w_most": "w_rest"})
        for col in ["w_season","w_last5","w_similar","w_opp","total_picks","total_hits","hit_rate"]:
            if col not in migrated.columns:
                migrated[col] = 0 if col not in ["hit_rate"] else 0.0
        hitrate_df = migrated
        st.success(f"✅ Migrated {len(migrated)} of {old_rows} legacy hit-rate rows from w_most → w_rest (where w_most == 0).")
else:
    hitrate_df = pd.DataFrame(columns=[
        "w_season","w_last5","w_similar","w_rest","w_opp","total_picks","total_hits","hit_rate"
    ])

# check odds date
odds_is_for_today = False
if not odds_df.empty:
    if "game_date_eastern" in odds_df.columns:
        odds_dates = pd.to_datetime(
            odds_df["game_date_eastern"], errors="coerce"
        ).dt.date.astype(str)
        odds_is_for_today = (odds_dates == today_str).any()
    else:
        odds_is_for_today = False

# 1) score old picks
gamelog_df = load_csv(GAMELOG_CSV)
if not gamelog_df.empty and not exp_df.empty:
    if "PLAYER" in gamelog_df.columns:
        gamelog_df["player_clean"] = gamelog_df["PLAYER"].str.upper().str.strip()
    elif "PLAYER_NAME" in gamelog_df.columns:
        gamelog_df["player_clean"] = gamelog_df["PLAYER_NAME"].str.upper().str.strip()
    else:
        gamelog_df["player_clean"] = ""

    if "GAME_DATE" in gamelog_df.columns:
        gamelog_df["GAME_DATE"] = pd.to_datetime(
            gamelog_df["GAME_DATE"], errors="coerce"
        ).dt.date.astype(str)

    mask_unscored = exp_df["hit"].isna() | (exp_df["hit"] == "")
    mask_past = exp_df["date"] < today_str
    to_score = exp_df[mask_unscored & mask_past].copy()
    scored_now = []

    for idx, row in to_score.iterrows():
        pl = str(row["player"]).upper().strip()
        dt = str(row["date"])
        stat = row["stat"]

        if "GAME_DATE" in gamelog_df.columns:
            gl_match = gamelog_df[
                (gamelog_df["player_clean"] == pl) &
                (gamelog_df["GAME_DATE"] == dt)
            ]
        else:
            gl_match = pd.DataFrame()

        if gl_match.empty:
            continue

        actual_row = gl_match.iloc[0]

        if stat == "PTS":
            stat_col = "PTS"
        elif stat == "REB":
            stat_col = "TRB" if "TRB" in gl_match.columns else "REB"
        elif stat == "AST":
            stat_col = "AST"
        else:
            stat_col = None

        if stat_col is None or stat_col not in actual_row:
            continue

        actual_val = actual_row[stat_col]
        line_val = row["dk_line"]
        pick_dir = row["pick"]

        if pd.isna(actual_val) or pd.isna(line_val):
            continue

        if pick_dir == "OVER":
            hit = 1 if actual_val > line_val else 0
        else:
            hit = 1 if actual_val < line_val else 0

        exp_df.loc[idx, ["actual", "hit"]] = [actual_val, hit]
        scored_now.append(idx)

    exp_df.to_csv(exp_path, index=False)

    # update WEIGHT_HITRATE with only the picks scored in this run
    # (adding every already-scored pick again on each rerun inflated the totals)
    scored = exp_df.loc[scored_now].copy()
    scored["hit"] = pd.to_numeric(scored["hit"], errors="coerce")
    if not scored.empty:
        grouped = (
            scored.groupby(["w_season","w_last5","w_similar","w_rest","w_opp"])["hit"]
            .agg(["sum","count"])
            .reset_index()
        )
        grouped = grouped.rename(columns={"sum": "hits", "count": "picks"})

        for _, g in grouped.iterrows():
            cond = (
                (hitrate_df["w_season"] == g["w_season"]) &
                (hitrate_df["w_last5"] == g["w_last5"]) &
                (hitrate_df["w_similar"] == g["w_similar"]) &
                (hitrate_df["w_rest"] == g["w_rest"]) &
                (hitrate_df["w_opp"] == g["w_opp"])
            )
            if cond.any():
                i = hitrate_df.index[cond][0]
                hitrate_df.at[i, "total_picks"] += g["picks"]
                hitrate_df.at[i, "total_hits"] += g["hits"]
            else:
                hitrate_df = pd.concat([
                    hitrate_df,
                    pd.DataFrame([{
                        "w_season": g["w_season"],
                        "w_last5": g["w_last5"],
                        "w_similar": g["w_similar"],
                        "w_rest": g["w_rest"],
                        "w_opp": g["w_opp"],
                        "total_picks": g["picks"],
                        "total_hits": g["hits"],
                        "hit_rate": 0.0,
                    }])
                ], ignore_index=True)

        hitrate_df["hit_rate"] = (
            hitrate_df["total_hits"] / hitrate_df["total_picks"]
        ).round(3)
        hitrate_df.to_csv(hitrate_path, index=False)
        st.success("✅ WEIGHT_HITRATE.csv updated with latest results.")

# 2) generate today's experiments
already_today = (not exp_df.empty) and (exp_df["date"] == today_str).any()

def generate_valid_weight_combos():
    for w in itertools.product(STEP_VALUES, repeat=5):
        if abs(sum(w) - 1.0) < 1e-9:
            yield {
                "season": w[0],
                "last5": w[1],
                "similar": w[2],
                "rest": w[3],
                "opp": w[4],
            }

if not odds_is_for_today:
    st.warning("ODDS_TODAY.csv is not for today's Eastern date — skipping weight exploration.")
elif not already_today and "DK_LINE_PTS" in players.columns:
    all_combos = list(generate_valid_weight_combos())
    total = len(all_combos)
    st.write(f"Running full {total} weight combinations × 5 picks each = {total*5} total predictions.")

    progress = st.progress(0)
    progress_text = st.empty()

    new_rows = []

    # multipliers to spread categories:
    stat_multiplier = {
        "PTS": 1.0,
        "REB": 2.0,
        "AST": 2.5,
    }

    for idx, w_combo in enumerate(all_combos):
        tmp = predict_with_custom_weights(players, w_combo)
        long_rows = []
        for _, r in tmp.iterrows():
            for stat, diff_col, dk_col in [
                ("PTS","PTS_Diff_TMP","DK_LINE_PTS"),
                ("REB","REB_Diff_TMP","DK_LINE_REB"),
                ("AST","AST_Diff_TMP","DK_LINE_AST"),
            ]:
                if diff_col not in tmp.columns or dk_col not in tmp.columns:
                    continue
                diff_val = r[diff_col]
                line_val = r[dk_col]
                if pd.isna(diff_val) or pd.isna(line_val):
                    continue
                long_rows.append({
                    "player": r["Player"],
                    "team": r.get("Team",""),
                    "stat": stat,
                    "diff": diff_val,
                    "dk_line": line_val,
                    "pick": "OVER" if diff_val > 0 else "UNDER",
                })

        long_df = pd.DataFrame(long_rows)
        if not long_df.empty:
            long_df["abs_diff"] = long_df["diff"].abs()
            # apply your scaling: AST = 2.5x, REB = 2x, PTS = 1x
            long_df["scaled_abs_diff"] = long_df.apply(
                lambda row: row["abs_diff"] * stat_multiplier.get(row["stat"], 1.0),
                axis=1,
            )
            long_df = long_df.sort_values("scaled_abs_diff", ascending=False).head(5)
            for _, pick in long_df.iterrows():
                new_rows.append({
                    "date": today_str,
                    "player": pick["player"],
                    "team": pick["team"],
                    "stat": pick["stat"],
                    "dk_line": pick["dk_line"],
                    "pick": pick["pick"],
                    "actual": "",
                    "hit": "",
                    "w_season": w_combo["season"],
                    "w_last5": w_combo["last5"],
                    "w_similar": w_combo["similar"],
                    "w_rest": w_combo["rest"],
                    "w_opp": w_combo["opp"],
                })

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            pct = int(((idx + 1) / total) * 100)
            progress.progress(pct)
            progress_text.write(f"Progress: {pct}%")

    if new_rows:
        exp_df = pd.DataFrame(new_rows)
        exp_df.to_csv(EXPERIMENT_CSV, index=False)
        progress.progress(100)
        progress_text.write("Progress: 100%")
        st.success(f"✅ Generated {len(new_rows)} exploration picks for {today_str}.")
else:
    if already_today:
        st.info("Exploration picks for today already exist.")
    else:
        st.info("Exploration skipped (probably missing DK lines).")

# 3) show best combos
st.subheader("Best Weight Distributions (by hit rate)")
if hitrate_path.exists():
    hitrate_df = pd.read_csv(hitrate_path)
    if "w_rest" not in hitrate_df.columns and "w_most" in hitrate_df.columns:
        hitrate_df = hitrate_df[hitrate_df["w_most"] == 0].rename(columns={"w_most": "w_rest"})
    if not hitrate_df.empty:
        hitrate_df = hitrate_df.sort_values("hit_rate", ascending=False)
        st.dataframe(hitrate_df.head(15), width="stretch")
    else:
        st.info("No scored exploration picks yet — come back tomorrow after games are played.")
else:
    st.info("No exploration data yet.")