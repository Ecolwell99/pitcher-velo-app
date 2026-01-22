import streamlit as st
from data import get_pitcher_data

st.set_page_config(page_title="Pitcher Velocity", layout="wide")
st.title("⚾ Pitcher Avg Velocity by Count & Batter Handedness (2025)")

with st.sidebar:
    st.header("Inputs")
    pitcher = st.text_input("Pitcher name (First Last)", "Gerrit Cole")
    season = 2025
    min_pitches = st.number_input(
        "Minimum pitches per row",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )
    run = st.button("Run")

if run:
    if " " not in pitcher:
        st.error("Enter pitcher name like: First Last")
        st.stop()

    first, last = pitcher.strip().split(" ", 1)

    with st.spinner("Pulling Statcast data (first run may take ~30 seconds)..."):
        try:
            df = get_pitcher_data(first, last, season)
        except Exception as e:
            st.error(str(e))
            st.stop()

    if df.empty:
        st.warning("No data returned for this pitcher/season.")
        st.stop()

    result = (
        df.groupby(["stand", "count", "pitch_name"])
          .agg(
              avg_velocity=("release_speed", "mean"),
              pitches=("release_speed", "count"),
          )
          .reset_index()
    )

    result = result[result["pitches"] >= min_pitches]
    result["avg_velocity"] = result["avg_velocity"].round(1)

    result = result.sort_values(
        ["stand", "count", "avg_velocity"],
        ascending=[True, True, False],
    )

    result["stand"] = result["stand"].map({"R": "vs RHB", "L": "vs LHB"})

    st.subheader("Results")
    st.dataframe(result, use_container_width=True)
