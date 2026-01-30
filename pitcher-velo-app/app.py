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
# Load curated pitcher list
# =============================
@st.cache_data(show_spinner=False)
def load_pitchers_from_csv():
    df = pd.read_csv("pitchers.csv")
    df["name"] = df["name"].astype(str).str.strip()
    return sorted(df["name"].unique().tolist())

PITCHER_LIST = load_pitchers_from_csv()

# =============================
# Load registry ONLY for Savant links
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

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

# ---------- Bias table logic ----------
def build_bias_tables(df):
    if df.empty:
        empty = pd.DataFrame(columns=["Count", "Bias"])
        return empty, empty

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

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["sort"] = out["Count"].apply(lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1]))
        return out.sort_values("sort").drop(columns="sort").reset_index(drop=True)

    return make_side("L"), make_side("R")

# ---------- HTML table renderer (no index, zebra, no scroll) ----------
TABLE_CSS = """
<style>
.dk-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.dk-table th, .dk-table td {
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,0.08);
}
.dk-table th {
  text-align: left;
  background: rgba(255,255,255,0.06);
}
.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.04);
}
</style>
"""

def render_bias_table(df):
    df = df.copy().reset_index(drop=True)
    html = df.to_html(index=False, classes="dk-table", escape=False)
    st.markdown(TABLE_CSS + html, unsafe_allow_html=True)

# =============================
# Matchup controls
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 2])

with c1:
    away_pitcher = st.selectbox("Away Pitcher", PITCHER_LIST)

with c2:
    home_pitcher = st.selectbox("Home Pitcher", PITCHER_LIST)

with c3:
    season = st.selectbox("Season", [2024, 2025])

c_spacer, c_btn = st.columns([8, 1])
with c_btn:
    run = st.button("Run Matchup", use_container_width=True)

st.divider()

if not run:
    st.stop()

# =============================
# Load data
# =============================
away_df = get_pitcher_data(*away_pitcher.split(" ", 1), season)
home_df = get_pitcher_data(*home_pitcher.split(" ", 1), season)

away_groups = split_by_inning(away_df)
home_groups = split_by_inning(home_df)

# =============================
# Tabs
# =============================
tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, key in zip(tabs, away_groups.keys()):
    with tab:
        # Away
        render_pitcher_header(away_pitcher, f"Away Pitcher • {key} • {season}")
        lhb, rhb = build_bias_tables(away_groups[key])

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**vs LHB**")
            render_bias_table(lhb)
        with col_r:
            st.markdown("**vs RHB**")
            render_bias_table(rhb)

        st.divider()

        # Home
        render_pitcher_header(home_pitcher, f"Home Pitcher • {key} • {season}")
        lhb, rhb = build_bias_tables(home_groups[key])

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown("**vs LHB**")
            render_bias_table(lhb)
        with col_r2:
            st.markdown("**vs RHB**")
            render_bias_table(rhb)

