import streamlit as st
import pandas as pd
from data import get_pitcher_data

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Pitcher Velocity Bias by Count", layout="wide")
st.title("⚾ Pitcher Velocity Bias by Count (vs LHB / vs RHB)")
st.caption("Public Statcast data • 2025 season • Exact MPH • Over / Under framing")

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

# Validate name
if " " not in pitcher.strip():
    st.error("Please enter pitcher name as: First Last")
    st.stop()

first, last = pitcher.strip().split(" ", 1)

# Pull Statcast data
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
# Helper: velocity bias per group
# -----------------------------
def velocity_bias(group: pd.DataFrame):
    """
    For a (count, handedness) group:
    - Find modal velocity (exact decimal)
    - Find max velocity
    - If modal == max:
        % Over modal (>=)
    - Else:
        % Under max (<)
    """
    speeds = group["release_speed"]

    # Modal velocity (most frequent exact value)
    mode_val = speeds.value_counts().idxmax()
    mode_freq = (speeds >= mode_val).mean()  # used only if mode == max

    max_val = speeds.max()
    max_freq = (speeds >= max_val).mean()

    if mode_val == max_val:
        pct = (speeds >= mode_val).mean()
        label = f"{int(round(pct * 100))}% Over {mode_val:.1f}"
    else:
        pct = 1 - max_freq
        label = f"{int(round(pct * 100))}% Under {max_val:.1f}"

    return pd.Series({
        "avg_velocity": speeds.mean(),
        "velocity_bias": label,
        "pitches": len(group),
    })

# -----------------------------
# Aggregate by count & handedness
# -----------------------------
result = (
    df.groupby(["stand", "count"])
      .apply(velocity_bias)
      .reset_index()
)

# Apply minimum pitch filter
result = result[result["pitches"] >= min_pitches]

if result.empty:
    st.warning("No rows meet the minimum pitch threshold.")
    st.stop()

# Formatting
result["avg_velocity"] = result["avg_velocity"].round(1)
result["stand"] = result["stand"].map({"R": "vs RHB", "L": "vs LHB"})

# Sort counts logically (0-0 → 3-2)
def count_sort_key(c):
    balls, strikes = c.split("-")
    return int(balls) * 10 + int(strikes)

result["count_sort"] = result["count"].apply(count_sort_key)
result = result.sort_values(["stand", "count_sort"])
result = result.drop(columns=["count_sort", "pitches"])

# -----------------------------
# Split into two tables (LHB LEFT, RHB RIGHT)
# -----------------------------
vs_lhb = result[result["stand"] == "vs LHB"].drop(columns=["stand"])
vs_rhb = result[result["stand"] == "vs RHB"].drop(columns=["stand"])

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
