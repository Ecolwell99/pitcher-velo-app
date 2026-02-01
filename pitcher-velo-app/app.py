import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
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
# Global CSS
# =============================
TABLE_CSS = """
<style>
.dk-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  table-layout: fixed;
}
.dk-table th, .dk-table td {
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,0.08);
}
.dk-table th {
  text-align: left;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.85);
}
.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.045);
}
.dk-bias th:nth-child(1), .dk-bias td:nth-child(1) { width: 110px; }
.dk-bias th:nth-child(2), .dk-bias td:nth-child(2) { width: auto; }
.dk-mix th:nth-child(1), .dk-mix td:nth-child(1) { width: 140px; }
.dk-mix th:nth-child(2), .dk-mix td:nth-child(2) { width: 100px; text-align:right; }
.dk-mix th:nth-child(3), .dk-mix td:nth-child(3) { width: 100px; text-align:right; }
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# =============================
# Load pitcher list (CSV)
# =============================
BASE_DIR = Path(__file__).resolve().parent
PITCHER_CSV_PATH = BASE_DIR / "assets" / "pitchers.csv"

PITCHERS_DF = pd.read_csv(PITCHER_CSV_PATH)
PITCHER_OPTIONS = ["— Select Pitcher —"] + sorted(PITCHERS_DF["name"].astype(str).tolist())
PITCHER_MAP = {
    r["name"]: {"first": r["first"], "last": r["last"]}
    for _, r in PITCHERS_DF.iterrows()
}

# =============================
# Registry (Savant links)
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["name"] = (df.get("name_first", "") + " " + df.get("name_last", "")).str.strip()
    df["mlbam_id"] = df.filter(regex="mlbam").bfill(axis=1).iloc[:, 0]
    return df[["name", "mlbam_id"]]

REGISTRY = load_registry()

# =============================
# Helpers
# =============================
def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

def savant_url(name):
    r = REGISTRY[REGISTRY["name"] == name]
    return None if r.empty else f"https://baseballsavant.mlb.com/savant-player/{slugify(name)}-{int(r.iloc[0]['mlbam_id'])}"

def get_pitcher_throws(df):
    return None if df.empty else ("RHP" if df["p_throws"].iloc[0] == "R" else "LHP")

def render_pitcher_header(name, context):
    url = savant_url(name)
    st.markdown(
        f"""
        <h2 style="margin-bottom:4px;">
          {name}
          <a href="{url}" target="_blank"
             style="font-size:16px; opacity:.7; text-decoration:none; border-bottom:none;">
            🔗
          </a>
        </h2>
        <i>{context}</i>
        """,
        unsafe_allow_html=True,
    )

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_bias_tables(df):
    def make(side):
        rows = []
        for c, g in df[df["stand"] == side].groupby("count"):
            v = g["release_speed"].dropna()
            if v.empty:
                continue
            m = v.mean()
            p = (v >= m).mean()
            rows.append({
                "Count": c,
                "Bias": f"{round(max(p,1-p)*100,1)}% {'Over' if p>=.5 else 'Under'} {m:.1f}"
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["s"] = out["Count"].apply(lambda x: int(x.split("-")[0])*10+int(x.split("-")[1]))
        return out.sort_values("s").drop(columns="s")
    return make("L"), make("R")

def build_pitch_mix_overall(df):
    if df.empty:
        return pd.DataFrame(columns=["Pitch Type","Usage %","Avg MPH"])
    g = df[df["pitch_type"] != "PO"].dropna(subset=["pitch_type"])
    if g.empty:
        return pd.DataFrame(columns=["Pitch Type","Usage %","Avg MPH"])
    mix = g.groupby("pitch_type").agg(
        P=("pitch_type","size"),
        V=("release_speed","mean")
    ).reset_index().rename(columns={"pitch_type":"Pitch Type"})
    mix["Usage %"] = (mix["P"]/mix["P"].sum()*100).round(1).astype(str)+"%"
    mix["Avg MPH"] = mix["V"].round(1).astype(str)
    return mix.sort_values("Usage %", ascending=False)[["Pitch Type","Usage %","Avg MPH"]]

def render_table(df, cls):
    st.markdown(df.to_html(index=False, classes=f"dk-table {cls}", escape=False), unsafe_allow_html=True)

# =============================
# Controls
# =============================
c1, c2, c3 = st.columns([3,3,2])
with c1: away = st.selectbox("Away Pitcher", PITCHER_OPTIONS)
with c2: home = st.selectbox("Home Pitcher", PITCHER_OPTIONS)
with c3: season = st.selectbox("Season", [2025, 2026])

run = st.button("Run Matchup", use_container_width=True)
if not run or away.startswith("—") or home.startswith("—"):
    st.stop()

try:
    away_df = get_pitcher_data(PITCHER_MAP[away]["first"], PITCHER_MAP[away]["last"], season)
    home_df = get_pitcher_data(PITCHER_MAP[home]["first"], PITCHER_MAP[home]["last"], season)
except ValueError as e:
    st.error(str(e))
    st.stop()

away_mix = build_pitch_mix_overall(away_df)
home_mix = build_pitch_mix_overall(home_df)

tabs = st.tabs(["All","Early (1–2)","Middle (3–4)","Late (5+)"])

for t, key in zip(tabs, ["All","Early (1–2)","Middle (3–4)","Late (5+)"]):
    with t:
        # Away
        render_pitcher_header(
            away,
            f"{get_pitcher_throws(away_df)} | Away Pitcher • {key} • {season}"
        )

        # spacing before expander (this is the requested fix)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        with st.expander("Show Pitch Mix (Season Overall)"):
            render_table(away_mix, "dk-mix")

        lhb, rhb = build_bias_tables(split_by_inning(away_df)[key])
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**vs LHB**")
            render_table(lhb,"dk-bias")
        with cR:
            st.markdown("**vs RHB**")
            render_table(rhb,"dk-bias")

        st.divider()

        # Home
        render_pitcher_header(
            home,
            f"{get_pitcher_throws(home_df)} | Home Pitcher • {key} • {season}"
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        with st.expander("Show Pitch Mix (Season Overall)"):
            render_table(home_mix, "dk-mix")

        lhb, rhb = build_bias_tables(split_by_inning(home_df)[key])
        cL2, cR2 = st.columns(2)
        with cL2:
            st.markdown("**vs LHB**")
            render_table(lhb,"dk-bias")
        with cR2:
            st.markdown("**vs RHB**")
            render_table(rhb,"dk-bias")

        st.divider()

