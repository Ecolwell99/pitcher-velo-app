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
st.caption("Public Statcast data • 2025 season • Threshold = avg velocity in that count (pitcher + handedness + count-specific)")

# =============================
# Load & cache pitcher list
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers():
    df = chadwick_register()
    if "name_first" in df.columns and "name_last" in df.columns:
        df = df.assign(name=lambda x: x["name_first"].fillna("") + " " + x["name_last"].fillna(""))
        names = df["name"].dropna().unique().tolist()
        names = sorted([n for n in names if n.strip() != ""])
        return names
    return []

PITCHER_LIST = load_pitchers()

# =============================
# Matchup header (TOP OF PAGE)
# =============================
st.markdown("### Matchup")

col_input_1, col_input_2, col_input_3 = st.columns([3, 3, 1])

with col_input_1:
    away_pitcher = st.selectbox(
        "Away Pitcher (type to search)",
        options=PITCHER_LIST,
        index=PITCHER_LIST.index("Zac Gallen") if "Zac Gallen" in PITCHER_LIST else 0,
    )

with col_input_2:
    home_pitcher = st.selectbox(
        "Home Pitcher (type to search)",
        options=PITCHER_LIST,
        index=PITCHER_LIST.index("Gerrit Cole") if "Gerrit Cole" in PITCHER_LIST else 0,
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
def parse_name(full: str):
    if not full or " " not in full:
        return None, None
    return full.split(" ", 1)

def build_pitcher_tables(first: str, last: str):
    """
    Returns vs_lhb_df, vs_rhb_df, error (or None)
    Threshold for Over/Under is the AVG MPH for that count (pitcher+handedness+count-specific).
    """
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
            pitches = len(speeds)
            if pitches < MIN_PITCHES:
                continue

            avg_mph = float(np.mean(speeds))
            # IMPORTANT: use the displayed (rounded) avg as the cutoff so output matches trader intuition
            cutoff = round(avg_mph, 1)

            over_share = float((speeds >= cutoff).mean()) if pitches > 0 else 0.0
            under_share = 1.0 - over_share

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

        # logical count ordering
        def count_sort_key(c):
            try:
                balls, strikes = c.split("-")
                return int(balls) * 10 + int(strikes)
            except Exception:
                return 999

        result["sort"] = result["Count"].apply(count_sort_key)
        result = result.sort_values("sort").drop(columns=["sort"]).reset_index(drop=True)

        output[stand_label] = result[["Count", "Avg MPH", "% Over", "% Under", "Bias"]]

    return output.get("vs LHB"), output.get("vs RHB"), None

# =============================
# Run matchup
# =============================
if not run:
    st.info("Select Away and Home pitchers, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

if not away_first or not home_first:
    st.error("Please select both pitcher names.")
    st.stop()

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
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🟥 vs Left-Handed Batters")
        st.dataframe(away_lhb, use_container_width=True) if away_lhb is not None else st.info("No data vs LHB.")
    with c2:
        st.markdown("### 🟦 vs Right-Handed Batters")
        st.dataframe(away_rhb, use_container_width=True) if away_rhb is not None else st.info("No data vs RHB.")

st.divider()

# =============================
# Display — Home Pitcher
# =============================
st.subheader("🏠 Home Pitcher")
st.markdown(f"**{home_pitcher}**")

if home_error:
    st.error(f"Home pitcher error: {home_error}")
else:
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 🟥 vs Left-Handed Batters")
        st.dataframe(home_lhb, use_container_width=True) if home_lhb is not None else st.info("No data vs LHB.")
    with c4:
        st.markdown("### 🟦 vs Right-Handed Batters")
        st.dataframe(home_rhb, use_container_width=True) if home_rhb is not None else st.info("No data vs RHB.")

