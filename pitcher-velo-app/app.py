import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Count Avg Over/Under", layout="wide")
st.title("⚾ Pitcher Matchup — Over / Under vs Count Average Velocity")
st.caption("Public Statcast data • 2025 season • Threshold = avg velocity in that count")

# =============================
# Load & cache pitcher list
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers():
    df = chadwick_register()
    df = df.assign(
        name=df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    )
    names = sorted(df["name"].dropna().unique().tolist())
    return names

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
# Constants
# =============================
MIN_PITCHES = 1

# =============================
# Helpers
# =============================
def parse_name(full):
    return full.split(" ", 1)

def build_pitcher_tables(first, last):
    try:
        df = get_pitcher_data(first, last, 2025)
    except Exception as e:
        return None, None, str(e)

    if df is None or df.empty:
        return None, None, "No Statcast data found."

    df = df[df["stand"].isin(["R", "L"])]

    output = {}

    for stand_value, stand_label in [("L", "vs LHB"), ("R", "vs RHB")]:
        df_side = df[df["stand"] == stand_value]
        if df_side.empty:
            output[stand_label] = pd.DataFrame()
            continue

        rows = []
        for count_val, group in df_side.groupby("count"):
            speeds = group["release_speed"].dropna().to_numpy()
            if len(speeds) < MIN_PITCHES:
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
                "Avg MPH": round(avg_mph, 1),
                "% Over": over_pct,
                "% Under": under_pct,
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

    return output.get("vs LHB"), output.get("vs RHB"), None

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
        if away_lhb is None or away_lhb.empty:
            st.info("No data vs LHB.")
        else:
            st.dataframe(away_lhb, use_container_width=True)

    with col2:
        st.markdown("### 🟦 vs Right-Handed Batters")
        if away_rhb is None or away_rhb.empty:
            st.info("No data vs RHB.")
        else:
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
        if home_lhb is None or home_lhb.empty:
            st.info("No data vs LHB.")
        else:
            st.dataframe(home_lhb, use_container_width=True)

    with col4:
        st.markdown("### 🟦 vs Right-Handed Batters")
        if home_rhb is None or home_rhb.empty:
            st.info("No data vs RHB.")
        else:
            st.dataframe(home_rhb, use_container_width=True)

