import streamlit as st
import pandas as pd
from data import get_pitcher_data

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Pitcher Matchup Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data • 2025 season • Bias uses TOP pitch-type avg MPH per count")

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Pitchers")
    away_pitcher = st.text_input("Away Pitcher (First Last)", "Gerrit Cole")
    home_pitcher = st.text_input("Home Pitcher (First Last)", "Corbin Burnes")

    st.divider()

    min_pitches = st.number_input(
        "Minimum pitches per count",
        min_value=1,
        max_value=200,
        value=1,
        step=1,
    )
    run = st.button("Run Matchup")

# -----------------------------
# Helper: compute table for one pitcher
# -----------------------------
def build_pitcher_table(first: str, last: str, min_pitches: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_pitcher_data(first, last, 2025)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def top_pitch_avg_bias(group: pd.DataFrame) -> pd.Series:
        avg_velocity = float(group["release_speed"].mean())

        pitch_avgs = group.groupby("pitch_name")["release_speed"].mean()
        top_pitch_name = pitch_avgs.idxmax()
        top_pitch_avg = float(pitch_avgs.loc[top_pitch_name])

        usage = float((group["pitch_name"] == top_pitch_name).mean())

        if usage >= 0.5:
            bias = f"{int(round(usage * 100))}% Over {top_pitch_avg:.1f}"
        else:
            bias = f"{int(round((1 - usage) * 100))}% Under {top_pitch_avg:.1f}"

        return pd.Series({
            "avg_velocity": avg_velocity,
            "bias": bias,
            "pitches": len(group),
        })

    result = (
        df.groupby(["stand", "count"])
          .apply(top_pitch_avg_bias)
          .reset_index()
    )

    result = result[result["pitches"] >= min_pitches]

    if result.empty:
        return pd.DataFrame(), pd.DataFrame()

    result["avg_velocity"] = result["avg_velocity"].round(1)
    result["stand"] = result["stand"].map({"R": "vs RHB", "L": "vs LHB"})

    def count_sort_key(c: str) -> int:
        balls, strikes = c.split("-")
        return int(balls) * 10 + int(strikes)

    result["count_sort"] = result["count"].apply(count_sort_key)
    result = result.sort_values(["stand", "count_sort"])
    result = result.drop(columns=["count_sort", "pitches"])

    display_cols = ["count", "avg_velocity", "bias"]

    vs_lhb = result[result["stand"] == "vs LHB"][display_cols]
    vs_rhb = result[result["stand"] == "vs RHB"][display_cols]

    return vs_lhb, vs_rhb

# -----------------------------
# Run matchup
# -----------------------------
if not run:
    st.info("Enter Away and Home pitchers, then click **Run Matchup**.")
    st.stop()

def parse_name(name: str):
    if " " not in name.strip():
        return None, None
    return name.strip().split(" ", 1)

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

if not away_first or not home_first:
    st.error("Please enter both pitcher names as: First Last")
    st.stop()

with st.spinner("Pulling Statcast data for both pitchers..."):
    away_lhb, away_rhb = build_pitcher_table(away_first, away_last, min_pitches)
    home_lhb, home_rhb = build_pitcher_table(home_first, home_last, min_pitches)

# -----------------------------
# Display
# -----------------------------
st.subheader("Away Pitcher")
st.markdown(f"**{away_first} {away_last}**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟥 vs Left-Handed Batters")
    if away_lhb.empty:
        st.info("No data vs LHB.")
    else:
        st.dataframe(away_lhb, use_container_width=True)

with col2:
    st.markdown("### 🟦 vs Right-Handed Batters")
    if away_rhb.empty:
        st.info("No data vs RHB.")
    else:
        st.dataframe(away_rhb, use_container_width=True)

st.divider()

st.subheader("Home Pitcher")
st.markdown(f"**{home_first} {home_last}**")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🟥 vs Left-Handed Batters")
    if home_lhb.empty:
        st.info("No data vs LHB.")
    else:
        st.dataframe(home_lhb, use_container_width=True)

with col4:
    st.markdown("### 🟦 vs Right-Handed Batters")
    if home_rhb.empty:
        st.info("No data vs RHB.")
    else:
        st.dataframe(home_rhb, use_container_width=True)
