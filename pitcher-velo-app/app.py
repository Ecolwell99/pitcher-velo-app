import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup — Velocity Bias", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count")
st.caption("Public Statcast data")

# =============================
# Load & cache pitcher list (heuristic)
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers():
    df = chadwick_register().copy()
    df["name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()

    pitcher_positions = {"P", "SP", "RP"}
    if "mlb_pos" in df.columns:
        df = df[df["mlb_pos"].isna() | df["mlb_pos"].isin(pitcher_positions)]
    elif "primary_position" in df.columns:
        df = df[df["primary_position"].isna() | df["primary_position"].isin(pitcher_positions)]
    elif "pos" in df.columns:
        df = df[df["pos"].isna() | df["pos"].isin(pitcher_positions)]

    return sorted(df["name"].dropna().unique().tolist())

PITCHER_LIST = load_pitchers()

# =============================
# Styling
# =============================
def dark_zebra(df):
    return df.style.apply(
        lambda _: [
            "background-color: rgba(255,255,255,0.045)" if i % 2 else ""
            for i in range(len(df))
        ],
        axis=0
    )

# =============================
# Helpers
# =============================
MIN_PITCHES = 1

def parse_name(full):
    if not full or " " not in full:
        return None, None
    return full.split(" ", 1)

def split_by_inning(df):
    return {
        "All": df,
        "Early": df[df["inning"].isin([1, 2])],
        "Middle": df[df["inning"].isin([3, 4])],
        "Late": df[df["inning"] >= 5],
    }

def build_pitch_mix(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["pitch_name"] != "PO"]

    mix = (
        df.groupby("pitch_name")
          .agg(
              pitches=("release_speed", "count"),
              avg_mph=("release_speed", "mean"),
          )
          .reset_index()
    )

    total = mix["pitches"].sum()
    if total == 0:
        return pd.DataFrame()

    mix["usage_pct"] = mix["pitches"] / total * 100
    mix = mix.sort_values("usage_pct", ascending=False)

    mix["Usage %"] = mix["usage_pct"].map(lambda x: f"{x:.1f}")
    mix["Avg MPH"] = mix["avg_mph"].map(lambda x: f"{x:.1f}")
    mix = mix.rename(columns={"pitch_name": "Pitch Type"})

    return mix[["Pitch Type", "Usage %", "Avg MPH"]]

def build_count_tables(df):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["stand"].isin(["R", "L"])]
    out = {}

    for stand, label in [("L", "LHB"), ("R", "RHB")]:
        rows = []
        for count, grp in df[df["stand"] == stand].groupby("count"):
            speeds = grp["release_speed"].dropna().to_numpy()
            if len(speeds) < MIN_PITCHES:
                continue

            cutoff = round(float(np.mean(speeds)), 1)
            over = float((speeds >= cutoff).mean())

            bias = (
                f"{round(over*100,1)}% Over {cutoff:.1f} MPH"
                if over >= 0.5
                else f"{round((1-over)*100,1)}% Under {cutoff:.1f} MPH"
            )
            rows.append({"Count": count, "Bias": bias})

        if rows:
            df_out = pd.DataFrame(rows)
            df_out["sort"] = df_out["Count"].apply(
                lambda x: int(x.split("-")[0])*10 + int(x.split("-")[1])
            )
            out[label] = df_out.sort_values("sort").drop(columns="sort").reset_index(drop=True)
        else:
            out[label] = pd.DataFrame()

    return out["LHB"], out["RHB"]

# =============================
# Controls
# =============================
st.markdown("### Matchup")

c1, c2, c3, c4 = st.columns([3, 3, 2, 1])

with c1:
    away_pitcher = st.selectbox("Away Pitcher", PITCHER_LIST)

with c2:
    home_pitcher = st.selectbox("Home Pitcher", PITCHER_LIST)

with c3:
    season = st.selectbox("Season", [2025, 2026])

with c4:
    run = st.button("Run Matchup")

st.divider()

# =============================
# Run once
# =============================
if not run:
    st.info("Select pitchers and season, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

with st.spinner("Pulling Statcast data..."):
    away_raw = get_pitcher_data(away_first, away_last, season)
    home_raw = get_pitcher_data(home_first, home_last, season)

away_groups = split_by_inning(away_raw)
home_groups = split_by_inning(home_raw)

# =============================
# Tabs
# =============================
tabs = st.tabs(["All", "Early", "Middle", "Late"])

for tab, key in zip(tabs, ["All", "Early", "Middle", "Late"]):
    with tab:
        st.subheader(f"✈️ Away Pitcher — {key}")
        st.markdown(f"**{away_pitcher} — {season}**")

        with st.expander("Show Pitch Mix (Season Overall)"):
            st.dataframe(
                dark_zebra(build_pitch_mix(away_groups[key])),
                use_container_width=True,
                hide_index=True
            )

        lhb, rhb = build_count_tables(away_groups[key])
        c5, c6 = st.columns(2)

        with c5:
            st.markdown("**[LHB]**")
            st.dataframe(dark_zebra(lhb), use_container_width=True, hide_index=True)

        with c6:
            st.markdown("**[RHB]**")
            st.dataframe(dark_zebra(rhb), use_container_width=True, hide_index=True)

        st.divider()

        st.subheader(f"🏠 Home Pitcher — {key}")
        st.markdown(f"**{home_pitcher} — {season}**")

        with st.expander("Show Pitch Mix (Season Overall)"):
            st.dataframe(
                dark_zebra(build_pitch_mix(home_groups[key])),
                use_container_width=True,
                hide_index=True
            )

        lhb, rhb = build_count_tables(home_groups[key])
        c7, c8 = st.columns(2)

        with c7:
            st.markdown("**[LHB]**")
            st.dataframe(dark_zebra(lhb), use_container_width=True, hide_index=True)

        with c8:
            st.markdown("**[RHB]**")
            st.dataframe(dark_zebra(rhb), use_container_width=True, hide_index=True)

