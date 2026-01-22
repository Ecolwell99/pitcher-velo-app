import streamlit as st
from data import get_pitcher_data

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Pitcher Velocity by Count", layout="wide")
st.title("⚾ Pitcher Avg Velocity by Count (vs RHB / LHB)")
st.caption("Public Statcast data • 2025 season")

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
        value=1,   # 👈 DEFAULT IS NOW 1
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
# Aggregate: ONE avg velo per count per handedness
# -----------------------------
result = (
    df.groupby(["stand", "count"])
      .agg(
          avg_velocity=("release_speed", "mean"),
          pitches=("release_speed", "count"),
      )
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
result = result.drop(columns=["count_sort"])

# -----------------------------
# Split into two tables
# -----------------------------
vs_rhb = result[result["stand"] == "vs RHB"].drop(columns=["stand"])
vs_lhb = result[result["stand"] == "vs LHB"].drop(columns=["stand"])

st.subheader(f"{first} {last} — 2025")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟦 vs Right-Handed Batters")
    if vs_rhb.empty:
        st.info("No data vs RHB.")
    else:
        st.dataframe(vs_rhb, use_container_width=True)

with col2:
    st.markdown("### 🟥 vs Left-Handed Batters")
    if vs_lhb.empty:
        st.info("No data vs LHB.")
    else:
        st.dataframe(vs_lhb, use_container_width=True)


