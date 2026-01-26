import streamlit as st
import pandas as pd
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data • 2025 season • Bias uses TOP pitch-type AVG MPH per count")

# =============================
# Sidebar inputs
# =============================
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

# =============================
# Helpers
# =============================
def parse_name(name: str):
    if " " not in name.strip():
        return None, None
    return name.strip().split(" ", 1)

def top_pitch_avg_bias(group: pd.DataFrame) -> pd.Series:
    """
    For a (stand, count) group:
    - Compute avg MPH per pitch type IN THIS COUNT
    - Anchor MPH = highest pitch-type avg MPH
    - Usage = % of pitches that are that pitch type (tie-safe)
    - If usage >= 50% -> "X% OVER anchor"
      else            -> "(100-X)% UNDER anchor"
    """
    avg_velocity = float(group["release_speed"].mean())

    pitch_avg = group.groupby("pitch_name")["release_speed"].mean()
    top_mph = float(pitch_avg.max())

    top_pitch_types = set(pitch_avg[pitch_avg == top_mph].index.tolist())
    top_usage = float(group["pitch_name"].isin(top_pitch_types).mean())

    top_usage_pct = round(top_usage * 100, 1)
    under_pct = round((1 - top_usage) * 100, 1)

    if top_usage >= 0.5:
        bias = f"{top_usage_pct}% OVER {top_mph:.1f}"
    else:
        bias = f"{under_pct}% UNDER {top_mph:.1f}"

    return pd.Series({
        "avg_velocity": avg_velocity,
        "bias": bias,
        "pitches": len(group),
    })

def build_pitcher_tables(first: str, last: str, min_pitches: int):
    try:
        df = get_pitcher_data(first, last, 2025)
    except Exception as e:
        return None, None, str(e)

    if df.empty:
        return None, None, "No Statcast data found."

    result = (
        df.groupby(["stand", "count"])
          .apply(top_pitch_avg_bias)
          .reset_index()
    )

    result = result[result["pitches"] >= min_pitches]
    if result.empty:
        return None, None, "No rows meet pitch threshold."

    result["avg_velocity"] = result["avg_velocity"].round(1)
    result["stand"] = result["stand"].map({"R": "vs RHB", "L": "vs LHB"})

    def count_sort_key(c):
        balls, strikes = c.split("-")
        return int(balls) * 10 + int(strikes)

    result["count_sort"] = result["count"].apply(count_sort_key)
    result = result.sort_values(["stand", "count_sort"])
    result = result.drop(columns=["count_sort", "pitches"])

    display_cols = ["count", "avg_velocity", "bias"]

    vs_lhb = result[result["stand"] == "vs LHB"][display_cols]
    vs_rhb = result[result["stand"] == "vs RHB"][display_cols]

    return vs_lhb, vs_rhb, None

# =============================
# Run matchup
# =============================
if not run:
    st.info("Enter Away and Home pitchers, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

if not away_first or not home_first:
    st.error("Please enter both pitcher names as: First Last")
    st.stop()

with st.spinner("Pulling Statcast data for both pitchers..."):
    away_lhb, away_rhb, away_error = build_pitcher_tables(away_first, away_last, min_pitches)
    home_lhb, home_rhb, home_error = build_pitcher_tables(home_first, home_last, min_pitches)

# =============================
# Display — Away Pitcher
# =============================
st.subheader("Away Pitcher")
st.markdown(f"**{away_first} {away_last}**")

if away_error:
    st.error(f"Away pitcher error: {away_error}")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟥 vs Left-Handed Batters")
        st.dataframe(away_lhb, use_container_width=True)

    with col2:
        st.markdown("### 🟦 vs Right-Handed Batters")
        st.dataframe(away_rhb, use_container_width=True)

st.divider()

# =============================
# Display — Home Pitcher
# =============================
st.subheader("Home Pitcher")
st.markdown(f"**{home_first} {home_last}**")

if home_error:
    st.error(f"Home pitcher error: {home_error}")
else:
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🟥 vs Left-Handed Batters")
        st.dataframe(home_lhb, use_container_width=True)

    with col4:
        st.markdown("### 🟦 vs Right-Handed Batters")
        st.dataframe(home_rhb, use_container_width=True)

