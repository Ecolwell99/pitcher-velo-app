import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
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
# Name normalization
# =============================
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"\s+", " ", name.lower()).strip()
    return name

# =============================
# Load Chadwick registry (cached)
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["display_name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()
    df["norm_name"] = df["display_name"].apply(normalize_name)
    return df

REGISTRY = load_registry()

# =============================
# Pitcher resolution with disambiguation
# =============================
def resolve_pitcher(input_name: str, role: str):
    if not input_name or len(input_name.strip().split()) < 2:
        raise ValueError("Please enter full first and last name.")

    norm = normalize_name(input_name)
    matches = REGISTRY[REGISTRY["norm_name"] == norm]

    if matches.empty:
        raise ValueError(f"No pitcher found for '{input_name}'.")

    if len(matches) == 1:
        row = matches.iloc[0]
        return row["name_first"], row["name_last"], row["display_name"]

    # Multiple matches → disambiguation
    st.warning(f'Multiple pitchers named "{input_name}" found. Please select the correct pitcher:')

    options = []
    for _, r in matches.iterrows():
        label = r["display_name"]
        options.append(label)

    choice = st.radio(
        label=f"Select {role} Pitcher",
        options=options,
        key=f"disambiguate_{role}",
    )

    row = matches[matches["display_name"] == choice].iloc[0]
    return row["name_first"], row["name_last"], row["display_name"]

# =============================
# Analytics helpers
# =============================
def get_pitcher_throws(df):
    return None if df.empty else ("RHP" if df["p_throws"].iloc[0] == "R" else "LHP")

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
with c1:
    away_input = st.text_input("Away Pitcher (First Last)")
with c2:
    home_input = st.text_input("Home Pitcher (First Last)")
with c3:
    season = st.selectbox("Season", [2025, 2026])

run = st.button("Run Matchup", use_container_width=True)
if not run:
    st.stop()

# =============================
# Resolve pitchers (with radio disambiguation)
# =============================
try:
    away_first, away_last, away_name = resolve_pitcher(away_input, "Away")
    home_first, home_last, home_name = resolve_pitcher(home_input, "Home")
except ValueError as e:
    st.error(str(e))
    st.stop()

# =============================
# Pull Statcast data
# =============================
try:
    away_df = get_pitcher_data(away_first, away_last, season)
    home_df = get_pitcher_data(home_first, home_last, season)
except ValueError as e:
    st.error(str(e))
    st.stop()

away_mix = build_pitch_mix_overall(away_df)
home_mix = build_pitch_mix_overall(home_df)

tabs = st.tabs(["All","Early (1–2)","Middle (3–4)","Late (5+)"])

for t, key in zip(tabs, ["All","Early (1–2)","Middle (3–4)","Late (5+)"]):
    with t:
        # Away
        st.markdown(f"## {away_name}")
        st.markdown(f"*{get_pitcher_throws(away_df)} | Away Pitcher • {key} • {season}*")
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
        st.markdown(f"## {home_name}")
        st.markdown(f"*{get_pitcher_throws(home_df)} | Home Pitcher • {key} • {season}*")
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

