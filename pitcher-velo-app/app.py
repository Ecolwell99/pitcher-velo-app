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
# Load & cache registry
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
                   style="text-decoration:none; font-size:16px; opacity:0.75;">
                    🔗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {name}")

    st.markdown(f"*{context}*")

def zebra(df):
    return df.style.apply(
        lambda _: [
            "background-color: rgba(255,255,255,0.045)" if i % 2 else ""
            for i in range(len(df))
        ],
        axis=0
    )

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_bias_tables(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df["stand"].isin(["R", "L"])]

    def make_side(side):
        rows = []
        for count, g in df[df["stand"] == side].groupby("count"):
            speeds = g["release_speed"].dropna().to_numpy()
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

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            return df_out

        df_out["sort"] = df_out["Count"].apply(
            lambda x: int(x.split("-")[0])*10 + int(x.split("-")[1])
        )

        return (
            df_out
            .sort_values("sort")
            .drop(columns="sort")
            .reset_index(drop=True)  # 🔑 REMOVE INDEX
        )

    return make_side("L"), make_side("R")

# =============================
# Matchup controls
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 2])
away = st.selectbox("Away Pitcher", PITCHER_LIST)
home = st.selectbox("Home Pitcher", PITCHER_LIST)
season = st.selectbox("Season", [2025, 2026])

c_spacer, c_btn = st.columns([8, 1])
run = c_btn.button("Run Matchup", use_container_width=True)

st.divider()

if not run:
    st.stop()

away_df = get_pitcher_data(*away.split(" ", 1), season)
home_df = get_pitcher_data(*home.split(" ", 1), season)

away_groups = split_by_inning(away_df)
home_groups = split_by_inning(home_df)

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, key in zip(tabs, away_groups.keys()):
    with tab:
        render_pitcher_header(away, f"Away Pitcher • {key} • {season}")
        lhb, rhb = build_bias_tables(away_groups[key])
        st.columns(2)[0].table(zebra(lhb))
        st.columns(2)[1].table(zebra(rhb))

        st.divider()

        render_pitcher_header(home, f"Home Pitcher • {key} • {season}")
        lhb, rhb = build_bias_tables(home_groups[key])
        st.columns(2)[0].table(zebra(lhb))
        st.columns(2)[1].table(zebra(rhb))

