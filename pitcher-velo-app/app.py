import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data")

# =============================
# Load & cache pitcher list (Option B heuristic)
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers_heuristic():
    df = chadwick_register().copy()

    df["name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()

    pitcher_positions = {"P", "SP", "RP"}

    if "mlb_pos" in df.columns:
        df = df[df["mlb_pos"].isna() | df["mlb_pos"].isin(pitcher_positions)]
    elif "primary_position" in df.columns:
        df = df[df["primary_position"].isna() | df["primary_position"].isin(pitcher_positions)]
    elif "pos" in df.columns:
        df = df[df["pos"].isna() | df["pos"].isin(pitcher_positions)]

    return sorted(df["name"].dropna().unique().tolist())

PITCHER_LIST = load_pitchers_heuristic()

# =============================
# Styling helper
# =============================
def dark_zebra(df):
    return df.style.apply(
        lambda _: [
            "background-color: rgba(255,255,255,0.045)" if i % 2 else ""
            for i in range(len(df))
        ],
        axis=0
    )

# =============================
# Helpers
# =============================
MIN_PITCHES = 1

def parse_name(full):
    if not full or " " not in full:
        return None, None
    return full.split(" ", 1)

def filter_by_inning(df, inning_group):
    if inning_group == "All":
        return df
    if inning_group == "Early (1–2)":
        return df[df["inning"].isin([1, 2])]
    if inning_group == "Middle (3–4)":
        return df[df["inning"].isin([3, 4])]
    if inning_group == "Late (5+)":
        return df[df["inning"] >= 5]
    return df

def build_pitch_mix(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["pitch_name"] != "PO"]

    mix = (
        df.groupby("pitch_name")
          .agg(
              pitches=("release_speed", "count"),
              avg_mph=("release_speed", "mean"),
          )
          .reset_index()
    )

    total = mix["pitches"].sum()
    if total == 0:
        return pd.DataFrame()

    mix["usage_pct"] = mix["pitches"] / total * 100
    mix = mix.sort_values("usage_pct", ascending=False)

    mix["Usage %"] = mix["usage_pct"].map(lambda x: f"{x:.1f}")
    mix["Avg MPH"] = mix["avg_mph"].map(lambda x: f"{x:.1f}")
    mix = mix.rename(columns={"pitch_name": "Pitch Type"})

    return mix[["Pitch Type", "Usage %", "Avg MPH"]]

def build_count_tables(df):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["stand"].isin(["R", "L"])]
    output = {}

    for stand_value, label in [("L", "LHB"), ("R", "RHB")]:
        df_side = df[df["stand"] == stand_value]
        rows = []

        for count, group in df_side.groupby("count"):
            speeds = group["release_speed"].dropna().to_numpy()
            if len(speeds) < MIN_PITCHES:
                continue

            cutoff = round(float(np.mean(speeds)), 1)
            over = float((speeds >= cutoff).mean())

            bias = (
                f"{round(over*100,1)}% Over {cutoff:.1f} MPH"
                if over >= 0.5
                else f"{round((1-over)*100,1)}% Under {cutoff:.1f} MPH"
            )

            rows.append({"Count": count, "Bias": bias})

        if rows:
            df_out = pd.DataFrame(rows)
            df_out["sort"] = df_out["Count"].apply(
                lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1])
            )
            df_out = df_out.sort_values("sort").drop(columns="sort").reset_index(drop=True)
            output[label] = df_out
        else:
            output[label] = pd.DataFrame()

    return output["LHB"], output["RHB"]

# =============================
# Controls
# =============================
st.markdown("### Matchup")

c1, c2, c3, c4, c5 = st.columns([3, 3, 2, 2, 1])

with c1:
    away_pitcher = st.selectbox("Away Pitcher", PITCHER_LIST)

with c2:
    home_pitcher = st.selectbox("Home Pitcher", PITCHER_LIST)

with c3:
    season = st.selectbox("Season", [2025, 2026])

with c4:
    inning_group = st.selectbox(
        "Inning",
        ["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"],
    )

with c5:
    run = st.button("Run Matchup")

st.divider()

# =============================
# Run matchup
# =============================
if not run:
    st.info("Select pitchers, season, and inning group, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

away_df, home_df = None, None
errors = []

with st.spinner("Pulling Statcast data..."):
    try:
        away_df = get_pitcher_data(away_first, away_last, season)
        away_df = filter_by_inning(away_df, inning_group)
    except Exception:
        errors.append(f"No data for Away Pitcher ({inning_group})")

    try:
        home_df = get_pitcher_data(home_first, home_last, season)
        home_df = filter_by_inning(home_df, inning_group)
    except Exception:
        errors.append(f"No data for Home Pitcher ({inning_group})")

for e in errors:
    st.warning(e)

# =============================
# Display
# =============================
if away_df is not None:
    st.subheader(f"✈️ Away Pitcher — {inning_group}")
    st.markdown(f"**{away_pitcher} — {season}**")

    with st.expander("Show Pitch Mix (Season Overall)"):
        st.dataframe(dark_zebra(build_pitch_mix(away_df)), use_container_width=True, hide_index=True)

    lhb, rhb = build_count_tables(away_df)
    c6, c7 = st.columns(2)

    with c6:
        st.markdown("**[LHB]**")
        st.dataframe(dark_zebra(lhb), use_container_width=True, hide_index=True)

    with c7:
        st.markdown("**[RHB]**")
        st.dataframe(dark_zebra(rhb), use_container_width=True, hide_index=True)

st.divider()

if home_df is not None:
    st.subheader(f"🏠 Home Pitcher — {inning_group}")
    st.markdown(f"**{home_pitcher} — {season}**")

    with st.expander("Show Pitch Mix (Season Overall)"):
        st.dataframe(dark_zebra(build_pitch_mix(home_df)), use_container_width=True, hide_index=True)

    lhb, rhb = build_count_tables(home_df)
    c8, c9 = st.columns(2)

    with c8:
        st.markdown("**[LHB]**")
        st.dataframe(dark_zebra(lhb), use_container_width=True, hide_index=True)

    with c9:
        st.markdown("**[RHB]**")
        st.dataframe(dark_zebra(rhb), use_container_width=True, hide_index=True)

