import streamlit as st
import pandas as pd
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data • 2025 season • Bias = % of pitches over/under anchor MPH")

# =============================
# Sidebar inputs
# =============================
with st.sidebar:
    st.header("Pitchers")
    away_pitcher = st.text_input("Away Pitcher (First Last)", "Yoshinobu Yamamoto")
    home_pitcher = st.text_input("Home Pitcher (First Last)", "Gerrit Cole")

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

    df = df[df["stand"].isin(["R", "L"])]

    output = {}

    for stand_value, stand_label in [("L", "vs LHB"), ("R", "vs RHB")]:
        df_side = df[df["stand"] == stand_value]

        if df_side.empty:
            output[stand_label] = pd.DataFrame()
            continue

        # --------------------------------
        # Anchor MPH (season-level, handedness-specific)
        # --------------------------------
        pitch_type_avg = (
            df_side.groupby("pitch_name")["release_speed"]
            .mean()
        )

        anchor_mph = float(pitch_type_avg.max())

        # --------------------------------
        # Per-count aggregation
        # --------------------------------
        def bias_by_count(group: pd.DataFrame) -> pd.Series:
            avg_velocity = float(group["release_speed"].mean())

            over_share = float((group["release_speed"] >= anchor_mph).mean())
            under_share = 1 - over_share

            over_pct = round(over_share * 100, 1)
            under_pct = round(under_share * 100, 1)

            if over_share >= 0.5:
                bias = f"{over_pct}% Over {anchor_mph:.1f}"
            else:
                bias = f"{under_pct}% Under {anchor_mph:.1f}"

            return pd.Series({
                "Avg MPH": avg_velocity,
                "% Over / Under MPH": bias,
                "pitches": len(group),
            })

        result = (
            df_side.groupby("count")
            .apply(bias_by_count)
            .reset_index()
        )

        result = result[result["pitches"] >= min_pitches]
        if result.empty:
            output[stand_label] = pd.DataFrame()
            continue

        result["Avg MPH"] = result["Avg MPH"].round(1)

        def count_sort_key(c):
            balls, strikes = c.split("-")
            return int(balls) * 10 + int(strikes)

        result["count_sort"] = result["count"].apply(count_sort_key)
        result = result.sort_values("count_sort")
        result = result.drop(columns=["count_sort", "pitches"])

        result = result.rename(columns={"count": "Count"})
        output[stand_label] = result

    return output.get("vs LHB"), output.get("vs RHB"), None

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

