import streamlit as st
import pandas as pd
from data import get_pitcher_data

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Pitcher Velocity Bias by Count", layout="wide")
st.title("⚾ Pitcher Velocity by Count (vs LHB / vs RHB)")
st.caption("Public Statcast data • 2025 season • Bias uses TOP pitch-type AVG MPH per count")

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    pitcher = st.text_input("Pitcher name (First Last)", "Gerrit Cole")
    min_pitches = st.number_input(
        "Minimum pitches per count",
        min_value=1,
        max_value=200,
        value=1,
        step=1,
    )
    run = st.button("Run")

# -----------------------------
# Main logic
# -----------------------------
if not run:
    st.info("Enter a pitcher name and click **Run**.")
    st.stop()

if " " not in pitcher.strip():
    st.error("Please enter pitcher name as: First Last")
    st.stop()

first, last = pitcher.strip().split(" ", 1)

with st.spinner("Pulling Statcast data (may take ~30 seconds on first run)..."):
    try:
        df = get_pitcher_data(first, last, 2025)
    except Exception as e:
        st.error(str(e))
        st.stop()

if df.empty:
    st.warning("No Statcast data returned for this pitcher.")
    st.stop()

# -----------------------------
# Helper: top pitch-type AVG mph + usage (per count/handedness)
# -----------------------------
def top_pitch_avg_bias(group: pd.DataFrame) -> pd.Series:
    """
    For a (stand, count) group:
    1) Compute avg mph for each pitch_name in this group
    2) Find pitch_name with the highest avg mph (top pitch-type avg)
    3) Compute usage % of that pitch type in this group
    4) Display:
       - If usage >= 50%:  "{usage}% Over {top_avg}"
       - Else:            "{100-usage}% Under {top_avg}"
    """
    speeds = group["release_speed"]
    avg_velocity = float(speeds.mean())

    # Avg mph by pitch type within this count
    pitch_avgs = group.groupby("pitch_name")["release_speed"].mean()

    top_pitch_name = pitch_avgs.idxmax()
    top_pitch_avg = float(pitch_avgs.loc[top_pitch_name])

    # Usage of that top pitch type in this count
    usage = float((group["pitch_name"] == top_pitch_name).mean())  # 0..1

    if usage >= 0.5:
        label = f"{int(round(usage * 100))}% Over {top_pitch_avg:.1f}"
    else:
        label = f"{int(round((1 - usage) * 100))}% Under {top_pitch_avg:.1f}"

    return pd.Series({
        "avg_velocity": avg_velocity,
        "top_pitch_type": top_pitch_name,
        "top_pitch_avg": top_pitch_avg,
        "bias": label,
        "pitches": len(group),
    })

# -----------------------------
# Aggregate by count & handedness
# -----------------------------
result = (
    df.groupby(["stand", "count"])
      .apply(top_pitch_avg_bias)
      .reset_index()
)

# Minimum pitch filter
result = result[result["pitches"] >= min_pitches]
if result.empty:
    st.warning("No rows meet the minimum pitch threshold.")
    st.stop()

# Formatting
result["avg_velocity"] = result["avg_velocity"].round(1)
result["top_pitch_avg"] = result["top_pitch_avg"].round(1)
result["stand"] = result["stand"].map({"R": "vs RHB", "L": "vs LHB"})

# Sort counts logically (0-0 → 3-2)
def count_sort_key(c: str) -> int:
    balls, strikes = c.split("-")
    return int(balls) * 10 + int(strikes)

result["count_sort"] = result["count"].apply(count_sort_key)
result = result.sort_values(["stand", "count_sort"]).drop(columns=["count_sort", "pitches"])

# -----------------------------
# Columns to display (NO pitch-count column)
# -----------------------------
display_cols = ["count", "avg_velocity", "bias", "top_pitch_type", "top_pitch_avg"]

# Split into two tables (LHB LEFT, RHB RIGHT)
vs_lhb = result[result["stand"] == "vs LHB"][display_cols]
vs_rhb = result[result["stand"] == "vs RHB"][display_cols]

st.subheader(f"{first} {last} — 2025")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟥 vs Left-Handed Batters")
    if vs_lhb.empty:
        st.info("No data vs LHB.")
    else:
        st.dataframe(vs_lhb, use_container_width=True)

with col2:
    st.markdown("### 🟦 vs Right-Handed Batters")
    if vs_rhb.empty:
        st.info("No data vs RHB.")
    else:
        st.dataframe(vs_rhb, use_container_width=True)
