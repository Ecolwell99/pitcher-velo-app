from pybaseball import statcast_pitcher, playerid_lookup
import pandas as pd

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

def get_pitcher_data(first_name, last_name, season):
    matches = playerid_lookup(last_name, first_name)
    if matches.empty:
        raise ValueError(f"No pitcher found for: {first_name} {last_name}")

    pid = int(matches.iloc[0]["key_mlbam"])

    start = f"{season}-03-01"
    end = f"{season}-11-30"

    df = statcast_pitcher(start, end, pid)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["release_speed"].notna()].copy()

    df["count"] = (
        df["balls"].astype(int).astype(str)
        + "-"
        + df["strikes"].astype(int).astype(str)
    )

    df["pitch_name"] = df["pitch_type"].map(PITCH_MAP).fillna(df["pitch_type"])
    df = df[df["stand"].isin(["R", "L"])]

    return df
