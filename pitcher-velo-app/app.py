
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Count-Specific Over/Under", layout="wide")
st.title("⚾ Pitcher Matchup — Count-Specific Over / Under by Velocity")
st.caption("Public Statcast data • 2025 season • Cutoff = natural break in release_speed distribution per count")

# =============================
# Load & cache pitcher list
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers():
    df = chadwick_register()
    # keep modern-era players and build "First Last" list
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
        index=PITCHER_LIST.index("Yoshinobu Yamamoto") if "Yoshinobu Yamamoto" in PITCHER_LIST else 0,
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

def natural_break_threshold(speeds: np.ndarray) -> float:
    """
    Given a numpy array of speeds (floats), return a threshold value that is
    the midpoint of the largest gap between sorted unique speeds.
    Fallback: if only one unique speed, return the mean.
    """
    if len(speeds) == 0:
        return float("nan")
    uniq = np.unique(speeds.round(2))  # round small noise
    if uniq.size <= 1:
        return float(np.mean(speeds))
    # find largest gap between consecutive unique speeds
    diffs = np.diff(uniq)
    max_idx = int(np.argmax(diffs))
    # threshold is midpoint between uniq[max_idx] and uniq[max_idx+1]
    thresh = float((uniq[max_idx] + uniq[max_idx + 1]) / 2.0)
    return thresh

def build_pitcher_tables(first: str, last: str):
    """
    Returns vs_lhb_df, vs_rhb_df, error (or None)
    vs_*_df columns: Count, Avg MPH, % Over, % Under, Bias
    """
    try:
        df = get_pitcher_data(first, last, 2025)
    except Exception as e:
        return None, None, str(e)

    if df is None or df.empty:
        return None, None, "No Statcast data found."

    # Filter to valid stands
    df = df[df["stand"].isin(["R", "L"])]

    output = {}

    for stand_value, stand_label in [("L", "vs LHB"), ("R", "vs RHB")]:
        df_side = df[df["stand"] == stand_value]

        if df_side.empty:
            output[stand_label] = pd.DataFrame()
            continue

        rows = []
        # group by count
        for count_val, group in df_side.groupby("count"):
            speeds = group["release_speed"].dropna().to_numpy()
            pitches = len(speeds)
            if pitches < MIN_PITCHES:
                continue

            avg_mph = float(np.mean(speeds)) if pitches > 0 else np.nan

            # determine threshold using natural break
            if pitches >= 2:
                thresh = natural_break_threshold(speeds)
            else:
                thresh = avg_mph

            # compute shares
            over_share = float((speeds >= thresh).mean()) if pitches > 0 else 0.0
            under_share = 1.0 - over_share
            over_pct = round(over_share * 100, 1)
            under_pct = round(under_share * 100, 1)

            # bias string uses majority side, but we also expose both % columns
            if over_share >= 0.5:
                bias = f"{over_pct}% Over {thresh:.1f} MPH"
            else:
                bias = f"{under_pct}% Under {thresh:.1f} MPH"

            rows.append({
                "Count": count_val,
                "Avg MPH": round(avg_mph, 1),
                "% Over": over_pct,
                "% Under": under_pct,
                "Bias": bias,
                # keep raw threshold so trader can inspect if needed (not displayed by default)
                "_threshold": round(thresh, 1),
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
        result = result.sort_values("sort").drop(columns=["sort"])

        # final displayed columns
        display = result[["Count", "Avg MPH", "% Over", "% Under", "Bias"]].reset_index(drop=True)
        output[stand_label] = display

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
    st.error("Please select both pitcher names")
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
