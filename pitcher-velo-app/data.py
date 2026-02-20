from functools import lru_cache

import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher

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
