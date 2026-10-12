# model.py
# The blend, without a UI: turns the scraper's CSVs into predictions, logs each
# weight combination's daily picks, scores them the next day, and keeps the
# weight hit-rate table. Same rules as the original Streamlit app (app1_2.py).

import itertools
import re

import numpy as np
import pandas as pd

from common import DATA_DIR, data_path, clean_player_name

# ----- CONFIG -----
METRICS = ["season", "last5", "similar", "rest", "opp"]
STATS = ["PTS", "REB", "AST", "STL", "BLK"]
LINE_STATS = ["PTS", "REB", "AST"]  # stats with a DraftKings line
MARKET_TO_STAT = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST"}
OU_PREFIX = {"PTS": "points", "REB": "rebounds", "AST": "assists"}

STEP_VALUES = [i / 20 for i in range(21)]
PICKS_PER_COMBO = 5
# multipliers to spread categories: AST = 2.5x, REB = 2x, PTS = 1x
STAT_MULTIPLIER = {"PTS": 1.0, "REB": 2.0, "AST": 2.5}

WEIGHT_COLS = ["w_season", "w_last5", "w_similar", "w_rest", "w_opp"]
EXPERIMENT_COLS = [
    "date", "player", "team", "stat", "dk_line", "pick", "actual", "hit",
] + WEIGHT_COLS
HITRATE_COLS = WEIGHT_COLS + ["total_picks", "total_hits", "hit_rate"]

EXPERIMENT_CSV = data_path("weight_experiments.csv")
HITRATE_CSV = data_path("WEIGHT_HITRATE.csv")


# =========================================================
# LOADING
# =========================================================
def load_csv(name: str) -> pd.DataFrame:
    p = data_path(name)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def latest_season_tag() -> str | None:
    """Newest season the scraper has written, e.g. '25-26' (from 25-26_AVGS.csv)."""
    tags = sorted(
        m.group(1)
        for m in (re.fullmatch(r"(\d\d-\d\d)_AVGS\.csv", p.name) for p in DATA_DIR.glob("*_AVGS.csv"))
        if m
    )
    return tags[-1] if tags else None


def signal_columns(stat: str) -> list[str]:
    """The five signals for a stat, in METRICS order."""
    return [f"SEASON_{stat}", f"{stat}_last5", f"{stat}_next_avg", f"{stat}_next", f"{stat}_vs_opp"]


def odds_date(odds_df: pd.DataFrame) -> str | None:
    """Latest Eastern game date in ODDS_TODAY.csv."""
    if odds_df.empty or "game_date_eastern" not in odds_df.columns:
        return None
    dates = pd.to_datetime(odds_df["game_date_eastern"], errors="coerce").dropna()
    return dates.max().date().isoformat() if len(dates) else None


def build_players() -> pd.DataFrame:
    """One row per player: the five signals per stat, DK lines, and over/under history."""
    tag = latest_season_tag()
    season_df = load_csv(f"{tag}_AVGS.csv") if tag else pd.DataFrame()
    if season_df.empty:
        return pd.DataFrame()

    players = season_df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"] + STATS].copy()
    players = players.rename(columns={s: f"SEASON_{s}" for s in STATS})
    players["player_name_clean"] = players["PLAYER_NAME"].map(clean_player_name)

    def safe_merge(base, df, cols):
        if not df.empty:
            valid = ["PLAYER_ID"] + [c for c in cols if c in df.columns]
            base = base.merge(df[valid], on="PLAYER_ID", how="left")
        for c in cols:
            if c not in base.columns:
                base[c] = float("nan")
        return base

    players = safe_merge(players, load_csv("LAST5.csv"), [f"{s}_last5" for s in STATS])
    players = safe_merge(players, load_csv("SIMILAR_NEXT.csv"),
                         [f"{s}_next_avg" for s in STATS] + [f"{s}_similar_n" for s in STATS])
    players = safe_merge(players, load_csv("REST_DAYS.csv"),
                         [f"{s}_next" for s in STATS] + ["REST_NEXT", "REST_NEXT_GP"])
    players = safe_merge(players, load_csv("OPPONENT_BIAS.csv"),
                         [f"{s}_vs_opp" for s in STATS] + ["NEXT_OPP", "NEXT_GAME_DATE", "VS_OPP_GP"])

    # DK lines
    odds_df = load_csv("ODDS_TODAY.csv")
    if not odds_df.empty:
        odds_df["player_name_clean"] = odds_df["player_name"].map(clean_player_name)
        odds_df = odds_df[odds_df["market"].isin(MARKET_TO_STAT)]
        pivot = (
            odds_df.pivot_table(index="player_name_clean", columns="market", values="line", aggfunc="max")
            .rename(columns={m: f"DK_LINE_{s}" for m, s in MARKET_TO_STAT.items()})
            .reset_index()
        )
        players = players.merge(pivot, on="player_name_clean", how="left")
    for s in LINE_STATS:
        if f"DK_LINE_{s}" not in players.columns:
            players[f"DK_LINE_{s}"] = float("nan")

    # over/under history
    ou_df = load_csv("OU_RATE.csv")
    ou_cols = [f"{OU_PREFIX[s]}_{d}" for s in LINE_STATS for d in ("over", "under")]
    if not ou_df.empty:
        if "player_name_clean" in ou_df.columns:
            ou_df["player_name_clean"] = ou_df["player_name_clean"].map(clean_player_name)
        else:
            ou_df["player_name_clean"] = ou_df["player_name"].map(clean_player_name)
        for c in ou_cols:
            if c not in ou_df.columns:
                ou_df[c] = 0
        ou_df = ou_df[["player_name_clean"] + ou_cols].drop_duplicates(subset=["player_name_clean"])
        players = players.merge(ou_df, on="player_name_clean", how="left")
    for c in ou_cols:
        if c not in players.columns:
            players[c] = float("nan")

    return players


# =========================================================
# PREDICTION
# =========================================================
def weighted_pred(row, stat: str, weights: dict) -> float:
    """Weighted average of the signals that have a value and a weight above zero."""
    num, denom = 0.0, 0.0
    for key, col in zip(METRICS, signal_columns(stat)):
        val = row.get(col, float("nan"))
        w = weights.get(key, 0.0)
        if pd.notna(val) and w > 0:
            num += val * w
            denom += w
    if denom == 0:
        return float("nan")
    return round(float(num / denom), 2)


def predict_matrix(players: pd.DataFrame, stat: str, W: np.ndarray) -> np.ndarray:
    """weighted_pred for every weight combination at once.
    W is (combos, 5) in METRICS order; returns (combos, players).
    Adds the signals in the same order and rounds the same way as weighted_pred,
    so the two agree to the last digit."""
    X = players[signal_columns(stat)].to_numpy(dtype=float)
    num = np.zeros((len(W), len(X)))
    denom = np.zeros((len(W), len(X)))
    for i in range(X.shape[1]):
        has = ~np.isnan(X[:, i])
        w = W[:, i][:, None]
        num += np.where(has, np.nan_to_num(X[:, i]) * w, 0.0)
        denom += np.where(has, w, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.where(denom > 0, num / denom, np.nan)
    rounded = [round(v, 2) for v in pred.ravel().tolist()]
    return np.array(rounded, dtype=float).reshape(pred.shape)


def all_weight_combos() -> np.ndarray:
    """Every split of the five weights in steps of 0.05 that sums to 1.0."""
    combos = [w for w in itertools.product(STEP_VALUES, repeat=5) if abs(sum(w) - 1.0) < 1e-9]
    return np.array(combos, dtype=float)


# =========================================================
# WEIGHT EXPLORATION: score old picks
# =========================================================
def _load_experiments() -> pd.DataFrame:
    if EXPERIMENT_CSV.exists():
        return pd.read_csv(EXPERIMENT_CSV)
    return pd.DataFrame(columns=EXPERIMENT_COLS)


def _load_hitrate() -> pd.DataFrame:
    if HITRATE_CSV.exists():
        return pd.read_csv(HITRATE_CSV)
    return pd.DataFrame(columns=HITRATE_COLS)


def score_experiments(today_str: str) -> int:
    """Score unscored picks from earlier days against the game logs, then add
    those newly scored picks (and only those) to WEIGHT_HITRATE.csv."""
    exp_df = _load_experiments()
    tag = latest_season_tag()
    gamelog_df = load_csv(f"{tag}_GAMELOG.csv") if tag else pd.DataFrame()
    if gamelog_df.empty or exp_df.empty:
        return 0

    gl = pd.DataFrame({
        "_player": gamelog_df["PLAYER_NAME"].map(clean_player_name),
        "_date": pd.to_datetime(gamelog_df["GAME_DATE"], errors="coerce").dt.date.astype(str),
        "PTS": pd.to_numeric(gamelog_df["PTS"], errors="coerce"),
        "REB": pd.to_numeric(gamelog_df["REB"], errors="coerce"),
        "AST": pd.to_numeric(gamelog_df["AST"], errors="coerce"),
    }).drop_duplicates(subset=["_player", "_date"], keep="first")

    hit_col = pd.to_numeric(exp_df["hit"], errors="coerce")
    to_score = exp_df[hit_col.isna() & (exp_df["date"].astype(str) < today_str)].copy()
    if to_score.empty:
        return 0

    to_score["_player"] = to_score["player"].map(clean_player_name)
    to_score["_date"] = to_score["date"].astype(str)
    m = to_score.reset_index().merge(gl, on=["_player", "_date"], how="left").set_index("index")

    actual = pd.Series(np.nan, index=m.index)
    for stat in LINE_STATS:
        is_stat = m["stat"] == stat
        actual[is_stat] = m.loc[is_stat, stat]
    line = pd.to_numeric(m["dk_line"], errors="coerce")
    ok = actual.notna() & line.notna()
    if not ok.any():
        return 0

    over = m["pick"] == "OVER"
    hit = np.where(over, actual > line, actual < line).astype(int)
    scored_idx = m.index[ok]

    exp_df["actual"] = pd.to_numeric(exp_df["actual"], errors="coerce")
    exp_df["hit"] = hit_col
    exp_df.loc[scored_idx, "actual"] = actual[ok]
    exp_df.loc[scored_idx, "hit"] = hit[ok.to_numpy()]
    exp_df.to_csv(EXPERIMENT_CSV, index=False)

    # update WEIGHT_HITRATE with only the picks scored in this run
    scored = exp_df.loc[scored_idx]
    grouped = (
        scored.groupby(WEIGHT_COLS)["hit"].agg(hits="sum", picks="count").reset_index()
    )
    hitrate_df = _load_hitrate()
    for c in WEIGHT_COLS:
        grouped[c] = grouped[c].round(2)
        hitrate_df[c] = pd.to_numeric(hitrate_df[c], errors="coerce").round(2)
    merged = hitrate_df.merge(grouped, on=WEIGHT_COLS, how="outer", sort=False)
    merged["total_picks"] = merged["total_picks"].fillna(0) + merged["picks"].fillna(0)
    merged["total_hits"] = merged["total_hits"].fillna(0) + merged["hits"].fillna(0)
    merged["hit_rate"] = (merged["total_hits"] / merged["total_picks"]).round(3)
    merged[HITRATE_COLS].to_csv(HITRATE_CSV, index=False)

    return int(ok.sum())


# =========================================================
# WEIGHT EXPLORATION: log today's picks for every combination
# =========================================================
def generate_experiments(players: pd.DataFrame, today_str: str) -> int:
    """For each weight combination, log its 5 biggest disagreements with the DK
    lines (PTS x1, REB x2, AST x2.5). Runs once per day, only when the saved
    lines are for today."""
    if players.empty:
        return 0
    if odds_date(load_csv("ODDS_TODAY.csv")) != today_str:
        return 0
    exp_df = _load_experiments()
    if not exp_df.empty and (exp_df["date"].astype(str) == today_str).any():
        return 0

    lines = players[[f"DK_LINE_{s}" for s in LINE_STATS]].to_numpy(dtype=float)
    lined = players[~np.isnan(lines).all(axis=1)].reset_index(drop=True)
    if lined.empty:
        return 0
    lines = lined[[f"DK_LINE_{s}" for s in LINE_STATS]].to_numpy(dtype=float)

    W = all_weight_combos()
    n_players = len(lined)

    # (combos, players, stats): player-major so ties keep the original row order
    diff = np.stack(
        [predict_matrix(lined, s, W) - lines[:, i] for i, s in enumerate(LINE_STATS)],
        axis=2,
    )
    mult = np.array([STAT_MULTIPLIER[s] for s in LINE_STATS])
    scaled = np.abs(diff) * mult
    scaled = np.where(np.isnan(scaled), -np.inf, scaled).reshape(len(W), n_players * len(LINE_STATS))
    top = np.argsort(-scaled, axis=1, kind="stable")[:, :PICKS_PER_COMBO]

    combo_idx = np.repeat(np.arange(len(W)), top.shape[1])
    flat = top.ravel()
    valid = np.isfinite(scaled[combo_idx, flat])
    combo_idx, flat = combo_idx[valid], flat[valid]
    p_idx, s_idx = flat // len(LINE_STATS), flat % len(LINE_STATS)

    new = pd.DataFrame({
        "date": today_str,
        "player": lined["PLAYER_NAME"].to_numpy()[p_idx],
        "team": lined["TEAM_ABBREVIATION"].to_numpy()[p_idx],
        "stat": np.array(LINE_STATS)[s_idx],
        "dk_line": lines[p_idx, s_idx],
        "pick": np.where(diff[combo_idx, p_idx, s_idx] > 0, "OVER", "UNDER"),
        "actual": "",
        "hit": "",
    })
    for i, c in enumerate(WEIGHT_COLS):
        new[c] = W[combo_idx, i]

    if new.empty:
        return 0
    new[EXPERIMENT_COLS].to_csv(EXPERIMENT_CSV, index=False)
    return len(new)


# =========================================================
# BEST WEIGHTS
# =========================================================
def top_weight_combos(k: int = 5) -> pd.DataFrame:
    hitrate_df = _load_hitrate()
    if hitrate_df.empty or not all(c in hitrate_df.columns for c in HITRATE_COLS):
        return pd.DataFrame(columns=HITRATE_COLS)
    return hitrate_df.sort_values("hit_rate", ascending=False, kind="stable").head(k)[HITRATE_COLS]
