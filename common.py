# common.py
# Shared by BRscrape1_2.py, app1_2.py and InjuryFetcherTest1_2.py:
# where the CSVs live, the API keys, and the player-name cleaning rule.

import json
import os
import unicodedata
from pathlib import Path

# ----- PATHS -----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "CSVs"
DATA_DIR.mkdir(exist_ok=True)


def data_path(name: str) -> Path:
    """'LAST5.csv' -> <project>/CSVs/LAST5.csv (works from any working directory)."""
    return DATA_DIR / name


# ----- PIPELINE STATE (CSVs/STATE.json) -----
# Remembers what has already been done, so nothing is counted twice.
def load_state() -> dict:
    try:
        with open(data_path("STATE.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    tmp = str(data_path("STATE.json")) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, data_path("STATE.json"))


# ----- STATS SOURCE -----
# "espn" = ESPN's public box score feed (works from GitHub's servers and from home)
# "nba"  = stats.nba.com through nba_api (works from a home connection only)
# The two use different player ids, so pick one and stay with it. ESPN is the
# default because the daily job runs on GitHub. Set STATS_SOURCE in the
# environment or in .env to override.
def _stats_source() -> str:
    return os.environ.get("STATS_SOURCE", "espn").strip().lower()


# ----- API KEYS -----
# Keys are read from environment variables, or from a .env file next to this
# script with lines like:
#   SGO_API_KEY=your_sportsgameodds_key   (lines: the main source)
#   ODDS_API_KEY=your_odds_api_key        (lines: the fallback)
#   INJURY_API_KEY=your_rapidapi_key      (unused unless INJURY_PULL_ENABLED=1)
def _load_env_file(path: Path = BASE_DIR / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # real environment variables win over the file
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
STATS_SOURCE = _stats_source()
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
INJURY_API_KEY = os.environ.get("INJURY_API_KEY", "")
SGO_API_KEY = os.environ.get("SGO_API_KEY", "")  # SportsGameOdds (free plan)


# ----- PLAYER NAMES -----
# stats.nba.com and DraftKings spell some players differently
# ("Nikola Jokić" / "Nikola Jokic", "Ronald Holland II" / "Ron Holland"),
# so both sides go through the same rule before they are matched.
_NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV"}
_NAME_ALIASES = {
    "HERB JONES": "HERBERT JONES",
    "RON HOLLAND": "RONALD HOLLAND",
}


def clean_player_name(name) -> str:
    if name is None or (isinstance(name, float) and name != name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = (
        s.upper()
        .replace(".", "")
        .replace("'", "")
        .replace("’", "")
        .replace("-", " ")
    )
    parts = s.split()
    while len(parts) > 1 and parts[-1] in _NAME_SUFFIXES:
        parts.pop()
    s = " ".join(parts)
    return _NAME_ALIASES.get(s, s)
