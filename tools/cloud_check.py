# tools/cloud_check.py
# Which data sources can be reached from where this runs? Uses no API keys.
# Run on GitHub (Actions tab > Cloud reachability check) and on the Mac to compare.

import time
import warnings

import requests

warnings.simplefilter("ignore")
UA = {"User-Agent": "Mozilla/5.0"}
results = []


def check(name, fn):
    start = time.time()
    try:
        detail, ok = fn(), True
    except Exception as e:
        detail, ok = f"{type(e).__name__}: {str(e)[:120]}", False
    results.append((ok, name, round(time.time() - start, 1), detail))


def nba_gamelog():
    from nba_api.stats.endpoints import leaguegamelog
    df = leaguegamelog.LeagueGameLog(
        season="2025-26", season_type_all_star="Regular Season",
        player_or_team_abbreviation="P", timeout=30,
    ).get_data_frames()[0]
    return f"{len(df)} player-game rows"


def nba_averages():
    from nba_api.stats.endpoints import leaguedashplayerstats
    df = leaguedashplayerstats.LeagueDashPlayerStats(season="2025-26", timeout=30).get_data_frames()[0]
    return f"{len(df)} players"


def nba_schedule():
    from nba_api.stats.endpoints import scheduleleaguev2
    df = scheduleleaguev2.ScheduleLeagueV2(season="2026-27", league_id="00", timeout=30).get_data_frames()[0]
    return f"{len(df)} games"


def nba_cdn():
    r = requests.get("https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022501194.json", headers=UA, timeout=20)
    r.raise_for_status()
    return f"box score for game {r.json()['game']['gameId']}"


def espn():
    base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
    events = requests.get(f"{base}/scoreboard?dates=20260412&limit=100", headers=UA, timeout=20).json()["events"]
    box = requests.get(f"{base}/summary?event={events[0]['id']}", headers=UA, timeout=20).json()["boxscore"]["players"]
    players = sum(len(t["statistics"][0]["athletes"]) for t in box)
    return f"{len(events)} games on 2026-04-12, {players} player lines in the first box score"


def odds_api():
    r = requests.get("https://api.the-odds-api.com/v4/sports", timeout=20)
    return f"answered with HTTP {r.status_code} (401 is expected without a key)"


check("stats.nba.com game logs", nba_gamelog)
check("stats.nba.com season averages", nba_averages)
check("stats.nba.com schedule", nba_schedule)
check("cdn.nba.com box score", nba_cdn)
check("ESPN scoreboard + box score", espn)
check("The Odds API", odds_api)

print()
for ok, name, secs, detail in results:
    print(f"{'REACHABLE' if ok else 'BLOCKED  '}  {name:32s} {secs:5.1f}s  {detail}")
