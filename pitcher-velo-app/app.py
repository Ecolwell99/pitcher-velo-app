import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup")
st.caption("Public Statcast data • Safe mode (timeouts enabled)")

# =============================
# Matchup inputs (TOP)
# =============================
c1, c2, c3 = st.columns([3, 3, 1])

with c1:
    away_pitcher = st.text_input("Away Pitcher (First Last)", "Gerrit Cole")

with c2:
    home_pitcher = st.text_input("Home Pitcher (First Last)", "Corbin Burnes")

with c3:
    run = st.button("Run Matchup")

st.divider()

# =============================
# Helpers
# =============================
def parse_name(name: str):
    if not name or " " not in name:
        return None, None
    return name.split(" ", 1)

def fetch_with_timeout(first, last, season, timeout=10):
    """
    Run get_pitcher_data with a hard timeout.
    Prevents Streamlit Cloud from hanging forever.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_pitcher_data, first, last, season)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None
        except Exception:
            return None

def build_tables(df):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["stand"].isin(["R", "L"])]

    output = {}
    for stand_value, label in [("L", "LHB"), ("R", "RHB")]:
        df_side = df[df["stand"] == stand_value]
        rows = []

        for count, group in df_side.groupby("count"):
            speeds = group["release_speed"].dropna().to_numpy()
            if len(speeds) == 0:
                continue

            avg_mph = np.mean(speeds)
            cutoff = round(avg_mph, 1)

            over_share = (speeds >= cutoff).mean()
            under_share = 1 - over_share

            if over_share >= 0.5:
                bias = f"{round(over_share*100,1)}% Over {cutoff} MPH"
            else:
                bias = f"{round(under_share*100,1)}% Under {cutoff} MPH"

            rows.append({"Count": count, "Bias": bias})

        table = pd.DataFrame(rows)
        if not table.empty:
            table["sort"] = table["Count"].apply(lambda x: int(x.split("-")[0])*10 + int(x.split("-")[1]))
            table = table.sort_values("sort").drop(columns="sort").reset_index(drop=True)

        output[label] = table

    return output["LHB"], output["RHB"]

# =============================
# Run logic
# =============================
if not run:
    st.info("Enter pitchers and click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

if not away_first or not home_first:
    st.error("Please enter names as: First Last")
    st.stop()

with st.spinner("Fetching data (safe mode, max 10s per pitcher)…"):
    away_df = fetch_with_timeout(away_first, away_last, 2025)
    home_df = fetch_with_timeout(home_first, home_last, 2025)

# =============================
# Display — Away
# =============================
st.subheader("✈️ Away Pitcher")
st.markdown(f"**{away_pitcher}**")

if away_df is None:
    st.error("Away pitcher data unavailable (Statcast timeout).")
else:
    lhb, rhb = build_tables(away_df)
    c4, c5 = st.columns(2)

    with c4:
        st.markdown("**[LHB]**")
        st.dataframe(lhb, hide_index=True, use_container_width=True)

    with c5:
        st.markdown("**[RHB]**")
        st.dataframe(rhb, hide_index=True, use_container_width=True)

st.divider()

# =============================
# Display — Home
# =============================
st.subheader("🏠 Home Pitcher")
st.markdown(f"**{home_pitcher}**")

if home_df is None:
    st.error("Home pitcher data unavailable (Statcast timeout).")
else:
    lhb, rhb = build_tables(home_df)
    c6, c7 = st.columns(2)

    with c6:
        st.markdown("**[LHB]**")
        st.dataframe(lhb, hide_index=True, use_container_width=True)

    with c7:
        st.markdown("**[RHB]**")
        st.dataframe(rhb, hide_index=True, use_container_width=True)

