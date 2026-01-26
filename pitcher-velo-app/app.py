import streamlit as st
import pandas as pd
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Avg Velocity by Count", layout="wide")
st.title("⚾ Pitcher Matchup — Average Velocity by Count")
st.caption("Public Statcast data • 2025 season")

# =============================
# Load & cache pitcher list
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers():
    """
    Load all MLB pitchers from Chadwick register.
    Returns sorted list of 'First Last'.
    """
    df = chadwick_register()
    pitchers = (
        df[df["mlb_played_last"] >= 2015]   # modern era filter
        .assign(name=lambda x: x["name_first"] + " " + x["name_last"])
        .sort_values("name")["name"]
        .unique()
        .tolist()
    )
    return pitchers

PITCHER_LIST = load_pitchers()

# =============================
# Matchup header (TOP OF PAGE)
# =============================
st.markdown("### Matchup")

col_input_1, col_input_2, col_input_3 = st.columns([3, 3, 1])

with col_input_1:
    away_pitcher = st.selectbox(
        "Away Pitcher",
        options=PITCHER_LIST,
        index=PITCHER_LIST.index("Gerrit Cole") if "Gerrit Cole" in PITCHER_LIST else 0,
    )

with col_input_2:
    home_pitcher = st.selectbox(
        "Home Pitcher",
        options=PITCHER_LIST,
        index=PITCHER_LIST.index("Corbin Burnes") if "Corbin Burnes" in PITCHER_LIST else 0,
    )

with col_input_3:
    run = st.button("Run Matchup")

st.divider()

# =============================
# Constants
# =============================
MIN_PITCHES = 1  # locked

# =============================
# Helpers
# =============================
def parse_name(name: str):
    return name.split(" ", 1)

def build_pitcher_tables(first: str, last: str):
    try:
        df = get_pitcher_data(first, last, 2025)
    except Exception as e:
        return None, None, str(e)

    if df is None or df.empty:
        return None, None, "No Statcast data found."

    df = df[df["stand"].isin(["R", "L"])]

    result = (
        df.groupby(["stand", "count"])
          .agg(
              avg_velocity=("release_speed", "mean"),
              pitches=("release_speed", "count"),
          )
          .reset_index()
    )

    result = result[result["pitches"] >= MIN_PITCHES]

    if result.empty:
        return None, None, "No rows meet pitch threshold."

    result["avg_velocity"] = result["avg_velocity"].round(1)
    result["stand"] = result["stand"].map({"L": "vs LHB", "R": "vs RHB"})

    def count_sort_key(c):
        balls, strikes = c.split("-")
        return int(balls) * 10 + int(strikes)

    result["count_sort"] = result["count"].apply(count_sort_key)
    result = result.sort_values(["stand", "count_sort"])
    result = result.drop(columns=["count_sort", "pitches"])

    display_cols = ["count", "avg_velocity"]

    vs_lhb = result[result["stand"] == "vs LHB"][display_cols]
    vs_rhb = result[result["stand"] == "vs RHB"][display_cols]

    return vs_lhb, vs_rhb, None

# =============================
# Run matchup
# =============================
if not run:
    st.info("Select Away and Home pitchers, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

with st.spinner("Pulling Statcast data for both pitchers..."):
    away_lhb, away_rhb, away_error = build_pitcher_tables(away_first, away_last)
    home_lhb, home_rhb, home_error = build_pitcher_tables(home_first, home_last)

# =============================
# Display — Away Pitcher
# =============================
st.subheader("✈️ Away Pitcher")
st.markdown(f"**{away_pitcher}**")

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
st.subheader("🏠 Home Pitcher")
st.markdown(f"**{home_pitcher}**")

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

