from functools import lru_cache
from datetime import date

import pandas as pd
from pybaseball import playerid_lookup, statcast, statcast_pitcher

PITCH_MAP = {
    "FF": "Four-Seam",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "KC": "Knuckle Curve",
    "FS": "Splitter",
    "ST": "Sweeper",
}


@lru_cache(maxsize=512)
def _lookup_pitcher_id(first_name, last_name):
    matches = playerid_lookup(last_name, first_name)
    if matches.empty:
        raise ValueError(f"No pitcher found for: {first_name} {last_name}")

    # Prefer exact first/last-name matches when available.
    exact = matches[
        (matches["name_first"].str.lower() == first_name.lower())
        & (matches["name_last"].str.lower() == last_name.lower())
    ]
    selected = exact.iloc[0] if not exact.empty else matches.iloc[0]
    return int(selected["key_mlbam"])


@lru_cache(maxsize=256)
def _load_statcast_pitcher(start, end, pid):
    df = statcast_pitcher(start, end, pid)
    if df is None:
        return pd.DataFrame()
    return df


@lru_cache(maxsize=128)
def _load_statcast_team(start, end, team):
    df = statcast(start, end, team=team)
    if df is None:
        return pd.DataFrame()
    return df


def get_pitcher_data(first_name, last_name, season):
    pid = _lookup_pitcher_id(first_name, last_name)

    start = f"{season}-03-01"
    end = f"{season}-11-30"

    raw = _load_statcast_pitcher(start, end, pid)
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df = df[df["release_speed"].notna()].copy()

    df["count"] = (
        df["balls"].astype(int).astype(str)
        + "-"
        + df["strikes"].astype(int).astype(str)
    )

    df["pitch_name"] = df["pitch_type"].map(PITCH_MAP).fillna(df["pitch_type"])
    df = df[df["stand"].isin(["R", "L"])]

    # Keep only fields used in app logic/UI while still trimming unused columns.
    keep_cols = [
        "stand",
        "count",
        "release_speed",
        "pitch_type",
        "pitch_name",
        "inning",
        "game_date",
        "home_team",
        "away_team",
        "inning_topbot",
        "p_throws",
    ]
    existing_cols = [col for col in keep_cols if col in df.columns]
    return df[existing_cols].copy()


def get_team_hitting_data(team, season):
    if not team or not isinstance(team, str):
        return pd.DataFrame()

    team = team.strip().upper()
    if len(team) != 3:
        return pd.DataFrame()

    start = f"{season}-03-01"
    current_year = date.today().year
    end = date.today().isoformat() if int(season) == current_year else f"{season}-11-30"

    raw = _load_statcast_team(start, end, team)
    if raw.empty:
        return pd.DataFrame()

    keep_cols = [
        "bat_team",
        "player_name",
        "batter",
        "pitch_type",
        "description",
        "estimated_woba_using_speedangle",
        "woba_value",
        "p_throws",
    ]
    existing_cols = [col for col in keep_cols if col in raw.columns]
    if not existing_cols:
        return pd.DataFrame()

    df = raw[existing_cols].copy()
    if "bat_team" in df.columns:
        df = df[df["bat_team"] == team].copy()

    return df
