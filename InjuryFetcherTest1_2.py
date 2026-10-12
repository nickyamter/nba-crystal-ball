#InjuryFetcherTest.py

import requests
import pandas as pd
import json
import sys
from datetime import datetime
import pytz

from common import data_path, INJURY_API_KEY

url = "https://api-basketball-nba.p.rapidapi.com/nba-injuries"

headers = {
    "X-RapidAPI-Key": INJURY_API_KEY,  # loaded in common.py (environment variable or .env file)
    "X-RapidAPI-Host": "api-basketball-nba.p.rapidapi.com",
}

# ---------- helpers ----------

def convert_to_est(utc_str):
    """
    Convert an ISO timestamp like '2025-11-13T18:00Z' (UTC)
    to 'YYYY-MM-DD HH:MM:SS EST/EDT' in US/Eastern.
    """
    if not utc_str or not isinstance(utc_str, str):
        return None
    try:
        # Make it ISO-8601 friendly: replace Z with +00:00
        dt_utc = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        tz_est = pytz.timezone("US/Eastern")
        dt_est = dt_utc.astimezone(tz_est)
        return dt_est.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        # If anything goes wrong, just return the original string
        return utc_str

def find_injury_lists(obj):
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

# ---------- request ----------

response = requests.get(url, headers=headers)

# --- RATE LIMIT INFO ONLY ---
print("\n--- RapidAPI Rate Limits ---")
print("Request limit:", response.headers.get("x-ratelimit-requests-limit"))
print("Requests remaining:", response.headers.get("x-ratelimit-requests-remaining"))
print("Requests reset at (UTC):", response.headers.get("x-ratelimit-requests-reset"))
print("--------------------------------\n")

# --- PARSE JSON ---
try:
    data = response.json()
except ValueError:
    print("Error: response was not valid JSON.")
    sys.exit(1)

# Optional: save raw JSON for debugging
with open(data_path("INJURY_RAW.json"), "w") as f:
    json.dump(data, f, indent=2)

# --- FIND & FLATTEN INJURY LISTS ---

injury_lists = find_injury_lists(data)

if not injury_lists:
    print("⚠️ No injury lists found in JSON. Check INJURY_RAW.json to inspect the structure.")
    sys.exit(0)

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
            "status": inj.get("status"),  # 'Out', 'Day-To-Day', ...
            "report_date_utc": inj.get("date"),
            "report_date_est": convert_to_est(inj.get("date")),
            "fantasy_status": fantasy.get("description"),
            "fantasy_status_abbr": fantasy.get("abbreviation"),
            "injury_type": details.get("type"),          # e.g. 'Knee'
            "injury_location": details.get("location"),  # e.g. 'Leg'
            "injury_detail": details.get("detail"),      # e.g. 'Sprain'
            "injury_side": details.get("side"),          # e.g. 'Right'
            "return_date_utc": details.get("returnDate"),
            "return_date_est": convert_to_est(details.get("returnDate")),
            "long_comment": inj.get("longComment"),
            "short_comment": inj.get("shortComment"),
        })

if not rows:
    print("⚠️ Found injury lists but no rows were built (unexpected).")
    sys.exit(0)

df = pd.DataFrame(rows)
df.to_csv(data_path("INJURY_REPORT.csv"), index=False)

print(f"Saved INJURY_REPORT.csv with {len(df)} rows and {len(df.columns)} columns.")