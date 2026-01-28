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
# Load & cache player list (SAFE)
# =============================
@st.cache_data(show_spinner=False)
def load_players():
    df = chadwick_register()
    df = df.assign(
        name=df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    )
    names = df["name"].dropna().unique().tolist()
    return sorted(names)

PITCHER_LIST = load_players()

# =============================
# Styling helper (dark-mode zebra rows)
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

def build_pitch_mix(df):
    if df is None or df.empty:
        return pd.DataFrame()

    # Exclude pitch outs
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
    output = {}

    for stand_value, label in [("L", "LHB"), ("R", "RHB")]:
        df_side = df[df["stand"] == stand_value]
        rows = []

        for count, group in df_side.groupby("count"):
            speeds = group["release_speed"].dropna().to_numpy()
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
                lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1])
            )
            df_out = df_out.sort_values("sort").drop(columns="sort").reset_index(drop=True)
            output[label] = df_out
        else:
            output[label] = pd.DataFrame()

    return output["LHB"], output["RHB"]

# =============================
# Matchup header
# =============================
st.markdown("### Matchup")

c1, c2, c3, c4 = st.columns([3, 3, 2, 1])

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
    season = st.selectbox(
        "Season",
        options=[2025, 2026],
        index=0,  # default = 2025
    )

with c4:
    run = st.button("Run Matchup")

st.divider()

# =============================
# Run matchup
# =============================
if not run:
    st.info("Select pitchers and season, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

away_df, home_df = None, None
errors = []

with st.spinner(f"Pulling Statcast data for {season}..."):
    try:
        away_df = get_pitcher_data(away_first, away_last, season)
    except Exception:
        errors.append(f"No Statcast data found for Away Pitcher: {away_pitcher} ({season})")

    try:
        home_df = get_pitcher_data(home_first, home_last, season)
    except Exception:
        errors.append(f"No Statcast data found for Home Pitcher: {home_pitcher} ({season})")

for e in errors:
    st.warning(e)

# =============================
# Display — Away Pitcher
# =============================
if away_df is not None:
    st.subheader("✈️ Away Pitcher")
    st.markdown(f"**{away_pitcher} — {season}**")

    with st.expander("Show Pitch Mix (Season Overall)"):
        st.dataframe(
            dark_zebra(build_pitch_mix(away_df)),
            use_container_width=True,
            hide_index=True
        )

    away_lhb, away_rhb = build_count_tables(away_df)
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("**[LHB]**")
        st.dataframe(dark_zebra(away_lhb), use_container_width=True, hide_index=True)

    with c6:
        st.markdown("**[RHB]**")
        st.dataframe(dark_zebra(away_rhb), use_container_width=True, hide_index=True)

st.divider()

# =============================
# Display — Home Pitcher
# =============================
if home_df is not None:
    st.subheader("🏠 Home Pitcher")
    st.markdown(f"**{home_pitcher} — {season}**")

    with st.expander("Show Pitch Mix (Season Overall)"):
        st.dataframe(
            dark_zebra(build_pitch_mix(home_df)),
            use_container_width=True,
            hide_index=True
        )

    home_lhb, home_rhb = build_count_tables(home_df)
    c7, c8 = st.columns(2)

    with c7:
        st.markdown("**[LHB]**")
        st.dataframe(dark_zebra(home_lhb), use_container_width=True, hide_index=True)

    with c8:
        st.markdown("**[RHB]**")
        st.dataframe(dark_zebra(home_rhb), use_container_width=True, hide_index=True)

