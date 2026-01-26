import streamlit as st
import pandas as pd
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data • 2025 season • Season-level pitch-type velocity anchors")

# =============================
# Sidebar inputs
# =============================
with st.sidebar:
    st.header("Pitchers")
    away_pitcher = st.text_input("Away Pitcher (First Last)", "Jack Flaherty")
    home_pitcher = st.text_input("Home Pitcher (First Last)", "Slade Cecconi")

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

def build_pitcher_tables(first: str, last: str, min_pitches: int):
    try:
        df = get_pitcher_data(first, last, 2025)
    except Exception as e:
        return None, None, str(e)

    if df.empty:
        return None, None, "No Statcast data found."

    # Ensure handedness is clean
    df = df[df["stand"].isin(["R", "L"])]

    output_tables = {}

    for stand_value, stand_label in [("L", "vs LHB"), ("R", "vs RHB")]:
        df_side = df[df["stand"] == stand_value]

        if df_side.empty:
            output_tables[stand_label] = pd.DataFrame()
            continue

        # --------------------------------
        # SEASON-LEVEL anchor (per handedness)
        # --------------------------------
        pitch_type_avg = (
            df_side.groupby("pitch_name")["release_speed"]
            .mean()
        )

        anchor_pitch = pitch_type_avg.idxmax()
        anchor_mph = float(pitch_type_avg.loc[anchor_pitch])

        # --------------------------------
        # Per-count aggregation
        # --------------------------------
        def bias_by_count(group: pd.DataFrame) -> pd.Series:
            avg_velocity = float(group["release_speed"].mean())

            usage = float((group["pitch_name"] == anchor_pitch).mean())

            usage_pct = round(usage * 100, 1)
            under_pct = round((1 - usage) * 100, 1)

            if usage >= 0.5:
                bias = f"{usage_pct}% over {anchor_mph:.1f} MPH"
            else:
                bias = f"{under_pct}% under {anchor_mph:.1f} MPH"

            return pd.Series({
                "avg_velocity": avg_velocity,
                "bias": bias,
                "pitches": len(group),
            })

        result = (
            df_side.groupby("count")
            .apply(bias_by_count)
            .reset_index()
        )

        result = result[result["pitches"] >= min_pitches]

        if result.empty:
            output_tables[stand_label] = pd.DataFrame()
            continue

        # Formatting
        result["avg_velocity"] = result["avg_velocity"].round(1)

        # Logical count order
        def count_sort_key(c):
            balls, strikes = c.split("-")
            return int(balls) * 10 + int(strikes)

        result["count_sort"] = result["count"].apply(count_sort_key)
        result = result.sort_values("count_sort")
        result = result.drop(columns=["count_sort", "pitches"])

        result = result.rename(columns={
            "count": "Count",
            "avg_velocity": "Avg MPH",
            "bias": "% Over / Under MPH"
        })

        output_tables[stand_label] = result

    return output_tables.get("vs LHB"), output_tables.get("vs RHB"), None

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


