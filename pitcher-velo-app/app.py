import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pybaseball import chadwick_register, statcast_pitcher

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
}
.dk-table th, .dk-table td {
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,0.08);
}
.dk-table th {
  background: rgba(255,255,255,0.06);
}
.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.045);
}
.dk-bias th:nth-child(1), .dk-bias td:nth-child(1) { width: 110px; }
.dk-mix th:nth-child(1), .dk-mix td:nth-child(1) { width: 140px; }
.dk-mix th:nth-child(2), .dk-mix td:nth-child(2),
.dk-mix th:nth-child(3), .dk-mix td:nth-child(3) {
  text-align: right;
  width: 100px;
}
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# =============================
# Name normalization
# =============================
def normalize_name(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name.lower()).strip()

# =============================
# Load Chadwick registry (cached)
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["display_name"] = (
        df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    ).str.strip()
    df["norm_name"] = df["display_name"].apply(normalize_name)

    # pick MLBAM id
    for col in ["key_mlbam", "mlbam_id", "key_mlb"]:
        if col in df.columns:
            df["mlbam"] = df[col]
            break

    return df[["name_first", "name_last", "display_name", "norm_name", "mlbam"]]

REGISTRY = load_registry()

# =============================
# Resolve pitcher (ID-based)
# =============================
def resolve_pitcher(input_name, season, role):
    if not input_name or len(input_name.split()) < 2:
        raise ValueError("Please enter full first and last name.")

    norm = normalize_name(input_name)
    matches = REGISTRY[REGISTRY["norm_name"] == norm]

    if matches.empty:
        raise ValueError(f"No pitcher found for '{input_name}'.")

    enriched = []

    for _, r in matches.iterrows():
        if pd.isna(r["mlbam"]):
            continue

        df = statcast_pitcher(
            start_dt=f"{season}-03-01",
            end_dt=f"{season}-11-30",
            player_id=int(r["mlbam"]),
        )

        if df.empty:
            continue

        throws = "LHP" if df["p_throws"].iloc[0] == "L" else "RHP"
        team = df["home_team"].mode().iloc[0]

        enriched.append({
            "first": r["name_first"],
            "last": r["name_last"],
            "name": r["display_name"],
            "mlbam": int(r["mlbam"]),
            "throws": throws,
            "team": team,
            "df": df,
        })

    if not enriched:
        raise ValueError(f"No Statcast data found for '{input_name}' in {season}.")

    if len(enriched) == 1:
        e = enriched[0]
        return e

    st.warning(f'Multiple pitchers named "{input_name}" found in {season}. Please select:')

    options = {
        f'{e["name"]} — {e["throws"]} — {e["team"]}': e
        for e in enriched
    }

    choice = st.radio(f"Select {role} Pitcher", list(options.keys()))
    return options[choice]

# =============================
# Analytics helpers
# =============================
def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1,2])],
        "Middle (3–4)": df[df["inning"].isin([3,4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_bias_tables(df):
    def make(side):
        rows = []
        for c, g in df[df["stand"] == side].groupby("count"):
            v = g["release_speed"].dropna()
            if v.empty: continue
            m = v.mean()
            p = (v >= m).mean()
            rows.append({
                "Count": c,
                "Bias": f"{round(max(p,1-p)*100,1)}% {'Over' if p>=.5 else 'Under'} {m:.1f}"
            })
        out = pd.DataFrame(rows)
        if out.empty: return out
        out["s"] = out["Count"].apply(lambda x: int(x.split("-")[0])*10+int(x.split("-")[1]))
        return out.sort_values("s").drop(columns="s")
    return make("L"), make("R")

def build_pitch_mix(df):
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
with c1: away_input = st.text_input("Away Pitcher (First Last)")
with c2: home_input = st.text_input("Home Pitcher (First Last)")
with c3: season = st.selectbox("Season", [2025, 2026])

if not st.button("Run Matchup", use_container_width=True):
    st.stop()

# =============================
# Resolve pitchers
# =============================
try:
    away = resolve_pitcher(away_input, season, "Away")
    home = resolve_pitcher(home_input, season, "Home")
except ValueError as e:
    st.error(str(e))
    st.stop()

tabs = st.tabs(["All","Early (1–2)","Middle (3–4)","Late (5+)"])

for t, key in zip(tabs, ["All","Early (1–2)","Middle (3–4)","Late (5+)"]):
    with t:
        for label, p in [("Away", away), ("Home", home)]:
            df = p["df"]
            st.markdown(f"## {p['name']}")
            st.markdown(f"*{p['throws']} | {label} Pitcher • {key} • {season}*")
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            with st.expander("Show Pitch Mix (Season Overall)"):
                render_table(build_pitch_mix(df), "dk-mix")

            lhb, rhb = build_bias_tables(split_by_inning(df)[key])
            cL, cR = st.columns(2)
            with cL:
                st.markdown("**vs LHB**")
                render_table(lhb, "dk-bias")
            with cR:
                st.markdown("**vs RHB**")
                render_table(rhb, "dk-bias")

            st.divider()

