import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import date

from pybaseball import chadwick_register, statcast
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
# Chadwick registry (Savant link fallback only)
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

# =============================
# Helpers
# =============================
MIN_PITCHES = 1

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

def savant_url(name: str, mlbam_id: int | None = None):
    pid = mlbam_id
    if pid is None:
        row = REGISTRY[REGISTRY["name"] == name]
        if not row.empty and not pd.isna(row.iloc[0]["mlbam_id"]):
            pid = int(row.iloc[0]["mlbam_id"])

    if pid is None:
        return None

    return f"https://baseballsavant.mlb.com/savant-player/{slugify(name)}-{int(pid)}"

def split_first_last(name: str):
    parts = name.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])

def get_pitcher_throws(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "p_throws" not in df.columns:
        return None
    v = df["p_throws"].dropna()
    if v.empty:
        return None
    return "RHP" if v.iloc[0] == "R" else "LHP"

def render_pitcher_header(name: str, context: str, mlbam_id: int):
    url = savant_url(name, mlbam_id)
    if url:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="margin:0;">{name}</h2>
                <a href="{url}" target="_blank"
                   style="text-decoration:none; font-size:16px; opacity:0.75;">🔗</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {name}")

    st.markdown(f"*{context}*")

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_bias_tables(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Count", "Bias"]), pd.DataFrame(columns=["Count", "Bias"])

    df = df[df["stand"].isin(["R", "L"])]

    def make_side(side):
        rows = []
        for count, g in df[df["stand"] == side].groupby("count"):
            speeds = g["release_speed"].dropna().to_numpy()
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

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["sort"] = out["Count"].apply(lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1]))
        return out.sort_values("sort").drop(columns="sort").reset_index(drop=True)

    return make_side("L"), make_side("R")

# =============================
# Pitcher pool from REAL Statcast pitch logs (ROBUST OPTION A)
# =============================
@st.cache_data(show_spinner=False)
def load_pitcher_pool(season: int):
    start = f"{season}-03-01"
    end = date.today().strftime("%Y-%m-%d")

    df = statcast(start, end)
    df = df.dropna(subset=["pitcher", "player_name"])

    out = (
        df[["pitcher", "player_name"]]
        .drop_duplicates()
        .rename(columns={"pitcher": "mlbam_id", "player_name": "name"})
    )

    out["name"] = out["name"].astype(str).str.strip()
    out = out.sort_values("name").reset_index(drop=True)

    fl = out["name"].apply(split_first_last)
    out["first"] = fl.apply(lambda x: x[0])
    out["last"] = fl.apply(lambda x: x[1])

    return out

# =============================
# Matchup controls
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 2])

with c3:
    season = st.selectbox("Season", [2025, 2026])

pitcher_pool = load_pitcher_pool(season)

PITCHER_OPTIONS = ["— Select Pitcher —"] + pitcher_pool["name"].tolist()

PITCHER_MAP = {
    row["name"]: {
        "first": row["first"],
        "last": row["last"],
        "mlbam_id": int(row["mlbam_id"]),
    }
    for _, row in pitcher_pool.iterrows()
}

with c1:
    away = st.selectbox("Away Pitcher", PITCHER_OPTIONS)
with c2:
    home = st.selectbox("Home Pitcher", PITCHER_OPTIONS)

c_spacer, c_btn = st.columns([8, 1])
with c_btn:
    run = st.button("Run Matchup", use_container_width=True)

st.divider()

if not run or away.startswith("—") or home.startswith("—"):
    st.stop()

away_meta = PITCHER_MAP[away]
home_meta = PITCHER_MAP[home]

away_df = get_pitcher_data(away_meta["first"], away_meta["last"], season)
home_df = get_pitcher_data(home_meta["first"], home_meta["last"], season)

away_throw = get_pitcher_throws(away_df)
home_throw = get_pitcher_throws(home_df)

away_groups = split_by_inning(away_df)
home_groups = split_by_inning(home_df)

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, key in zip(tabs, ["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"]):
    with tab:
        render_pitcher_header(
            away,
            f"{away_throw} | Away Pitcher • {key} • {season}",
            away_meta["mlbam_id"],
        )

        lhb, rhb = build_bias_tables(away_groups[key])
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**vs LHB**")
            st.markdown(lhb.to_html(index=False), unsafe_allow_html=True)
        with col_r:
            st.markdown("**vs RHB**")
            st.markdown(rhb.to_html(index=False), unsafe_allow_html=True)

        st.divider()

        render_pitcher_header(
            home,
            f"{home_throw} | Home Pitcher • {key} • {season}",
            home_meta["mlbam_id"],
        )

        lhb, rhb = build_bias_tables(home_groups[key])
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown("**vs LHB**")
            st.markdown(lhb.to_html(index=False), unsafe_allow_html=True)
        with col_r2:
            st.markdown("**vs RHB**")
            st.markdown(rhb.to_html(index=False), unsafe_allow_html=True)

