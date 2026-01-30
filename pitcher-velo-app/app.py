import streamlit as st
import pandas as pd
import numpy as np
import re
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Velocity Profiles", layout="wide")

st.markdown(
    """
    # Pitcher Velocity Profiles
    *Velocity behavior by count, inning, and handedness (Statcast)*
    """,
    unsafe_allow_html=True,
)

# =============================
# Load & cache registry (names + MLBAM IDs)
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()

    for col in ["key_mlbam", "mlbam_id", "key_mlb"]:
        if col in df.columns:
            df["mlbam_id"] = df[col]
            break
    else:
        df["mlbam_id"] = np.nan

    return df[["name", "mlbam_id"]]

REGISTRY = load_registry()

@st.cache_data(show_spinner=False)
def load_pitchers():
    return sorted(REGISTRY["name"].dropna().unique().tolist())

PITCHER_LIST = load_pitchers()

# =============================
# Helpers
# =============================
MIN_PITCHES = 1

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

def savant_url(name: str):
    row = REGISTRY[REGISTRY["name"] == name]
    if row.empty or pd.isna(row.iloc[0]["mlbam_id"]):
        return None
    return f"https://baseballsavant.mlb.com/savant-player/{slugify(name)}-{int(row.iloc[0]['mlbam_id'])}"

def render_pitcher_header(name: str, context: str):
    url = savant_url(name)
    if url:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="margin:0;">{name}</h2>
                <a href="{url}" target="_blank"
                   style="text-decoration:none; font-size:18px; opacity:0.8;">
                    🔗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {name}")

    st.markdown(f"*{context}*")

def dark_zebra(df):
    return df.style.apply(
        lambda _: [
            "background-color: rgba(255,255,255,0.045)" if i % 2 else ""
            for i in range(len(df))
        ],
        axis=0
    )

def parse_name(full):
    return full.split(" ", 1)

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_pitch_mix(df):
    if df.empty:
        return pd.DataFrame()

    df = df[df["pitch_name"] != "PO"]

    mix = (
        df.groupby("pitch_name")
          .agg(pitches=("release_speed", "count"),
               avg_mph=("release_speed", "mean"))
          .reset_index()
    )

    total = mix["pitches"].sum()
    mix["usage_pct"] = mix["pitches"] / total * 100
    mix = mix.sort_values("usage_pct", ascending=False)

    mix["Usage %"] = mix["usage_pct"].map(lambda x: f"{x:.1f}")
    mix["Avg MPH"] = mix["avg_mph"].map(lambda x: f"{x:.1f}")

    return mix.rename(columns={"pitch_name": "Pitch Type"})[
        ["Pitch Type", "Usage %", "Avg MPH"]
    ]

def build_count_tables(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["stand"].isin(["R", "L"])]
    out = {}

    for stand, label in [("L", "LHB"), ("R", "RHB")]:
        rows = []
        for count, grp in df[df["stand"] == stand].groupby("count"):
            speeds = grp["release_speed"].dropna().to_numpy()
            if len(speeds) < MIN_PITCHES:
                continue

            cutoff = round(np.mean(speeds), 1)
            over = (speeds >= cutoff).mean()

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
# Matchup
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 2])

with c1:
    away_pitcher = st.selectbox("Away Pitcher", PITCHER_LIST)

with c2:
    home_pitcher = st.selectbox("Home Pitcher", PITCHER_LIST)

with c3:
    season = st.selectbox("Season", [2025, 2026])

c_spacer, c_button = st.columns([8, 1])
with c_button:
    run = st.button("Run Matchup", use_container_width=True)

st.divider()

if not run:
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

away_raw = get_pitcher_data(away_first, away_last, season)
home_raw = get_pitcher_data(home_first, home_last, season)

away_groups = split_by_inning(away_raw)
home_groups = split_by_inning(home_raw)

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, key in zip(tabs, away_groups.keys()):
    with tab:
        render_pitcher_header(away_pitcher, f"Away Pitcher • {key} • {season}")

        lhb, rhb = build_count_tables(away_groups[key])
        c4, c5 = st.columns(2)
        c4.dataframe(dark_zebra(lhb), use_container_width=True, hide_index=True)
        c5.dataframe(dark_zebra(rhb), use_container_width=True, hide_index=True)

        st.divider()

        render_pitcher_header(home_pitcher, f"Home Pitcher • {key} • {season}")

        lhb, rhb = build_count_tables(home_groups[key])
        c6, c7 = st.columns(2)
        c6.dataframe(dark_zebra(lhb), use_container_width=True, hide_index=True)
        c7.dataframe(dark_zebra(rhb), use_container_width=True, hide_index=True)

