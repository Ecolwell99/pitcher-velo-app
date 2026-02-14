import streamlit as st
import pandas as pd
import re
import unicodedata
import numpy as np
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitch Tendencies", layout="wide")

st.markdown(
    """
    <div style="font-size:30px; font-weight:700;">
        Pitch Tendencies
    </div>
    <div style="opacity:0.6; margin-top:4px; margin-bottom:18px;">
        By count & split
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# CSS
# =============================
TABLE_CSS = """
<style>
.dk-table {
    width: 600px;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 13px;
}
.dk-table th, .dk-table td {
    padding: 5px 6px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}
.dk-table td {
    color: #ffffff;
}
.dk-table th:first-child,
.dk-table td:first-child {
    text-align: left;
    width: 60px;
    font-weight: 600;
}
.dk-table th {
    background: rgba(255,255,255,0.08);
    font-weight: 600;
}
.dk-table tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.04);
}
.dk-fav {
    font-weight: 600;
    background-color: rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 2px 8px;
}
.dk-subtitle {
    opacity: 0.6;
    margin-bottom: 12px;
}
.dk-mix {
    font-size: 12px;
    margin-bottom: 8px;
    opacity: 0.75;
}
.dk-flags {
    font-size: 12px;
    margin-bottom: 8px;
    line-height: 1.4;
}
.dk-link {
    color: #ffffff;
    text-decoration: none;
}
.dk-link:hover {
    text-decoration: none;
    opacity: 0.85;
}
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def normalize_name(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name.lower()).strip()

@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["display_name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()
    df["norm"] = df["display_name"].apply(normalize_name)
    return df

REGISTRY = load_registry()

def resolve_pitcher(name, season, role):
    norm = normalize_name(name)
    rows = REGISTRY[REGISTRY["norm"] == norm]

    valid = []
    for _, r in rows.iterrows():
        try:
            df = get_pitcher_data(r["name_first"], r["name_last"], season)
            if not df.empty:
                valid.append((r["name_first"], r["name_last"], r["display_name"]))
        except:
            pass

    if not valid:
        raise ValueError

    if len(valid) == 1:
        return valid[0]

    choice = st.radio(f"Select {role} Pitcher", [v[2] for v in valid])
    return next(v for v in valid if v[2] == choice)

def get_mlbam_id(first, last):
    rows = REGISTRY[
        (REGISTRY["name_first"].str.lower() == first.lower()) &
        (REGISTRY["name_last"].str.lower() == last.lower())
    ]
    if not rows.empty and "key_mlbam" in rows.columns:
        return rows.iloc[0]["key_mlbam"]
    return None

def get_current_team(df):
    if df.empty or "game_date" not in df.columns:
        return None
    latest_row = df.sort_values("game_date", ascending=False).iloc[0]
    if "home_team" in df.columns and "away_team" in df.columns:
        if "inning_topbot" in df.columns:
            return latest_row["home_team"] if latest_row["inning_topbot"] == "Top" else latest_row["away_team"]
        return latest_row["home_team"]
    return None

FASTBALLS = {"FF", "SI", "FC"}
BREAKING = {"SL", "CU", "KC", "SV", "ST"}
OFFSPEED = {"CH", "FS", "FO"}

# =============================
# Inline Mix
# =============================
def build_inline_mix(df, side):
    g = df[df["stand"] == side].dropna(subset=["pitch_type"])
    if g.empty:
        return None

    mix = g.groupby("pitch_type").size().reset_index(name="n")
    total = mix["n"].sum()
    mix["pct"] = (mix["n"] / total * 100).round(0)
    mix = mix[mix["pct"] >= 2]
    mix = mix.sort_values("pct", ascending=False)

    return " | ".join(f"{r['pitch_type']} {int(r['pct'])}%" for _, r in mix.iterrows())

# =============================
# Structural Flags
# =============================
def build_structure_flags(df, side):
    dominance = {}
    for count, g in df[df["stand"] == side].groupby("count"):
        g = g.dropna(subset=["pitch_type"])
        if g.empty:
            continue
        counts = g["pitch_type"].value_counts(normalize=True) * 100
        if counts.empty:
            continue
        top_pitch = counts.idxmax()
        if top_pitch in FASTBALLS:
            group = "Fastball"
        elif top_pitch in BREAKING:
            group = "Breaking"
        elif top_pitch in OFFSPEED:
            group = "Offspeed"
        else:
            continue
        dominance[count] = group

    early = {"0-0", "1-0", "0-1"}
    two_strike = {"0-2", "1-2", "2-2"}
    full = {"3-2"}

    def most_common(counts):
        vals = [dominance[c] for c in counts if c in dominance]
        return max(set(vals), key=vals.count) if vals else None

    flags = []
    if e := most_common(early):
        flags.append(f"• Early Counts: {e}")
    if t := most_common(two_strike):
        flags.append(f"• 2-Strike: {t}")
    if f := most_common(full):
        flags.append(f"• Full Count: {f}")
    return flags

# =============================
# Pitch Table
# =============================
def build_pitch_table(df, side):
    rows = []
    for count, g in d
