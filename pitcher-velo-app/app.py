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
  background: rgba(255,255,255,0.06);
}
.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.045);
}
.dk-info {
  opacity: 0.6;
  margin-left: 6px;
  cursor: help;
}
.dk-low {
  opacity: 0.45;
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
# Load Chadwick registry
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
# Resolve pitcher
# =============================
def resolve_pitcher(input_name, season, role):
    norm = normalize_name(input_name)
    matches = REGISTRY[REGISTRY["norm_name"] == norm]

    enriched = []
    for _, r in matches.iterrows():
        try:
            df = get_pitcher_data(r["name_first"], r["name_last"], season)
            if not df.empty:
                enriched.append((r["name_first"], r["name_last"], r["display_name"]))
        except:
            pass

    if not enriched:
        raise ValueError(f"No Statcast data found for '{input_name}' in {season}.")

    if len(enriched) == 1:
        return enriched[0]

    choice = st.radio(
        f"Select {role} Pitcher",
        [e[2] for e in enriched],
        key=f"pick_{role}",
    )
    return next(e for e in enriched if e[2] == choice)

# =============================
# Bias logic (Rule 1)
# =============================
FAST_PITCHES = {"FF", "SI", "FC", "CT"}
SOFT_PITCHES = {"SL", "CU", "KC", "CH"}

def build_bias_tables(df):
    def make(side):
        rows = []

        for count, g in df[df["stand"] == side].groupby("count"):
            speeds = g["release_speed"].dropna()
            if speeds.empty:
                continue

            n = len(speeds)

            fast = g[g["pitch_type"].isin(FAST_PITCHES)]
            soft = g[g["pitch_type"].isin(SOFT_PITCHES)]

            fast_pct = len(fast) / n if n else 0
            soft_pct = len(soft) / n if n else 0

            # --- Boundary selection ---
            if fast_pct >= 0.55 and not fast.empty:
                boundary = fast["release_speed"].mean()
                favored = "Over"
            elif soft_pct >= 0.55 and not soft.empty:
                boundary = soft["release_speed"].mean()
                favored = "Under"
            else:
                mid = g[g["pitch_type"].isin({"SL", "CH"})]
                if not mid.empty:
                    boundary = mid["release_speed"].mean()
                elif not soft.empty:
                    boundary = soft["release_speed"].mean()
                else:
                    boundary = speeds.mean()
                favored = "Over" if (speeds >= boundary).mean() >= 0.5 else "Under"

            if favored == "Over":
                pct = (speeds >= boundary).mean()
            else:
                pct = (speeds < boundary).mean()

            bias = f"{round(pct*100,1)}% {favored} {boundary:.1f}"

            cls = ""
            tip = None
            if n < 10:
                cls = "dk-low"
                tip = "Very small sample (<10 pitches)."
            elif n < 20:
                tip = "Low sample size (10–19 pitches). Interpret directionally."

            if tip:
                bias += f' <span class="dk-info" title="{tip}">ⓘ</span>'

            if cls:
                bias = f'<span class="{cls}">{bias}</span>'

            rows.append({"Count": count, "Bias": bias})

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["s"] = out["Count"].apply(lambda x: int(x.split("-")[0])*10 + int(x.split("-")[1]))
        return out.sort_values("s").drop(columns="s").reset_index(drop=True)

    return make("L"), make("R")

# =============================
# Pitch mix
# =============================
def build_pitch_mix(df):
    g = df[df["pitch_type"] != "PO"].dropna(subset=["pitch_type"])
    mix = g.groupby("pitch_type").agg(
        Count=("pitch_type","size"),
        MPH=("release_speed","mean")
    ).reset_index()
    mix["Usage %"] = (mix["Count"]/mix["Count"].sum()*100).round(1)
    return mix.sort_values("Usage %", ascending=False)[["pitch_type","Usage %","MPH"]]

def render_table(df):
    st.markdown(df.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)

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

if not st.button("Run Matchup", use_container_width=True):
    st.stop()

away_first, away_last, away_name = resolve_pitcher(away_input, season, "Away")
home_first, home_last, home_name = resolve_pitcher(home_input, season, "Home")

away_df = get_pitcher_data(away_first, away_last, season)
home_df = get_pitcher_data(home_first, home_last, season)

tabs = st.tabs(["All","Early (1–2)","Middle (3–4)","Late (5+)"])

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1,2])],
        "Middle (3–4)": df[df["inning"].isin([3,4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

for tab, key in zip(tabs, ["All","Early (1–2)","Middle (3–4)","Late (5+)"]):
    with tab:
        st.markdown(f"## {away_name}")
        lhb, rhb = build_bias_tables(split_by_inning(away_df)[key])
        st.markdown("**vs LHB**")
        render_table(lhb)
        st.markdown("**vs RHB**")
        render_table(rhb)

        st.divider()

        st.markdown(f"## {home_name}")
        lhb, rhb = build_bias_tables(split_by_inning(home_df)[key])
        st.markdown("**vs LHB**")
        render_table(lhb)
        st.markdown("**vs RHB**")
        render_table(rhb)

