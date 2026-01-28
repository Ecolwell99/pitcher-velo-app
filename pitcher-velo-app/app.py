import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import statcast, chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data • 2025 season")

# =============================
# Load & cache pitcher list (ONLY real pitchers)
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers():
    """
    Only include players who have thrown at least one pitch in Statcast.
    This prevents non-pitchers from appearing.
    """
    df = statcast(start_dt="2025-03-01", end_dt="2025-11-30")
    pitchers = (
        df[["pitcher", "player_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values("player_name")["player_name"]
        .unique()
        .tolist()
    )
    return pitchers

PITCHER_LIST = load_pitchers()

# =============================
# Matchup header (TOP)
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 1])

with c1:
    away_pitcher = st.selectbox(
        "Away Pitcher",
        options=PITCHER_LIST,
        index=PITCHER_LIST.index("Zac Gallen") if "Zac Gallen" in PITCHER_LIST else 0,
    )

with c2:
    home_pitcher = st.selectbox(
        "Home Pitcher",
        options=PITCHER_LIST,
        index=PITCHER_LIST.index("Gerrit Cole") if "Gerrit Cole" in PITCHER_LIST else 0,
    )

with c3:
    run = st.button("Run Matchup")

st.divider()

# =============================
# Helpers
# =============================
def parse_name(full):
    return full.split(" ", 1)

def style_rows(df):
    """Excel-style alternating row shading"""
    return df.style.apply(
        lambda x: ["background-color: #f4f4f4" if i % 2 else "" for i in range(len(x))],
        axis=0,
    )

def build_pitch_mix(df):
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
    mix["Usage %"] = (mix["pitches"] / total * 100).round(1)
    mix["Avg MPH"] = mix["avg_mph"].round(1)

    mix = mix.sort_values("Usage %", ascending=False)
    mix = mix.rename(columns={"pitch_name": "Pitch Type"})

    return mix[["Pitch Type", "Usage %", "Avg MPH"]]

def build_count_tables(df):
    df = df[df["stand"].isin(["R", "L"])]
    output = {}

    for stand_value, stand_label in [("L", "LHB"), ("R", "RHB")]:
        df_side = df[df["stand"] == stand_value]
        rows = []

        for count_val, group in df_side.groupby("count"):
            speeds = group["release_speed"].dropna().to_numpy()
            if len(speeds) == 0:
                continue

            avg_mph = float(np.mean(speeds))
            cutoff = round(avg_mph, 1)

            over_share = float((speeds >= cutoff).mean())
            under_share = 1 - over_share

            over_pct = round(over_share * 100, 1)
            under_pct = round(under_share * 100, 1)

            if over_share >= 0.5:
                bias = f"{over_pct}% Over {cutoff:.1f} MPH"
            else:
                bias = f"{under_pct}% Under {cutoff:.1f} MPH"

            rows.append({
                "Count": count_val,
                "Bias": bias,
            })

        if not rows:
            output[stand_label] = pd.DataFrame()
            continue

        result = pd.DataFrame(rows)

        def count_sort_key(c):
            balls, strikes = c.split("-")
            return int(balls) * 10 + int(strikes)

        result["sort"] = result["Count"].apply(count_sort_key)
        result = result.sort_values("sort").drop(columns="sort").reset_index(drop=True)

        output[stand_label] = result

    return output.get("LHB"), output.get("RHB")

# =============================
# Run matchup
# =============================
if not run:
    st.info("Select Away and Home pitchers, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

with st.spinner("Pulling Statcast data for both pitchers..."):
    away_df = get_pitcher_data(away_first, away_last, 2025)
    home_df = get_pitcher_data(home_first, home_last, 2025)

# =============================
# Display — Away Pitcher
# =============================
st.subheader("✈️ Away Pitcher")
st.markdown(f"**{away_pitcher}**")

with st.expander("Show Pitch Mix (Season Overall)"):
    away_mix = build_pitch_mix(away_df)
    st.dataframe(style_rows(away_mix), use_container_width=True, hide_index=True)

away_lhb, away_rhb = build_count_tables(away_df)

c4, c5 = st.columns(2)

with c4:
    st.markdown("**[LHB]**")
    st.dataframe(style_rows(away_lhb), use_container_width=True, hide_index=True)

with c5:
    st.markdown("**[RHB]**")
    st.dataframe(style_rows(away_rhb), use_container_width=True, hide_index=True)

st.divider()

# =============================
# Display — Home Pitcher
# =============================
st.subheader("🏠 Home Pitcher")
st.markdown(f"**{home_pitcher}**")

with st.expander("Show Pitch Mix (Season Overall)"):
    home_mix = build_pitch_mix(home_df)
    st.dataframe(style_rows(home_mix), use_container_width=True, hide_index=True)

home_lhb, home_rhb = build_count_tables(home_df)

c6, c7 = st.columns(2)

with c6:
    st.markdown("**[LHB]**")
    st.dataframe(style_rows(home_lhb), use_container_width=True, hide_index=True)

with c7:
    st.markdown("**[RHB]**")
    st.dataframe(style_rows(home_rhb), use_container_width=True, hide_index=True)

