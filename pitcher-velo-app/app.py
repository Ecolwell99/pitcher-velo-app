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
a.dk-link {
    color: #ffffff !important;
    text-decoration: none !important;
}
a.dk-link:hover {
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
                mlbam = r["key_mlbam"] if "key_mlbam" in r.index else None
                valid.append((
                    r["name_first"],
                    r["name_last"],
                    r["display_name"],
                    mlbam
                ))
        except:
            pass

    if not valid:
        raise ValueError

    if len(valid) == 1:
        return valid[0]

    choice = st.radio(f"Select {role} Pitcher", [v[2] for v in valid])
    return next(v for v in valid if v[2] == choice)

def get_hand_from_statcast(df):
    if df is None or df.empty:
        return None
    if "p_throws" not in df.columns:
        return None
    s = df["p_throws"].dropna().astype(str).str.upper()
    if s.empty:
        return None
    v = s.mode().iloc[0]
    if v in ["R", "L"]:
        return f"{v}HP"
    return None

def get_current_team(df):
    if df.empty or "game_date" not in df.columns:
        return None
    latest = df.sort_values("game_date", ascending=False).iloc[0]
    if "home_team" in df.columns and "away_team" in df.columns:
        if "inning_topbot" in df.columns:
            return latest["home_team"] if latest["inning_topbot"] == "Top" else latest["away_team"]
        return latest["home_team"]
    return None

FASTBALLS = {"FF", "SI", "FC"}
BREAKING = {"SL", "CU", "KC", "SV", "ST"}
OFFSPEED = {"CH", "FS", "FO"}

def build_pitch_table(df, side):
    rows = []
    for count, g in df[df["stand"] == side].groupby("count"):
        g = g.dropna(subset=["release_speed", "pitch_type"])
        if g.empty:
            continue
        total = len(g)
        if total < 5:
            continue

        summary = (
            g.groupby("pitch_type")
            .agg(n=("pitch_type", "size"),
                 mph_list=("release_speed", list))
            .reset_index()
        )
        summary["pct"] = (summary["n"] / total * 100).round(0)

        group_totals = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}
        dominant_velos = {"Fastball": None, "Breaking": None, "Offspeed": None}
        dominant_pct = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}

        for _, r in summary.iterrows():
            pt = r["pitch_type"]
            pct = int(r["pct"])
            velocities = np.array(r["mph_list"])

            if pt in FASTBALLS:
                group = "Fastball"
            elif pt in BREAKING:
                group = "Breaking"
            elif pt in OFFSPEED:
                group = "Offspeed"
            else:
                continue

            group_totals[group] += pct
            if pct > dominant_pct[group]:
                dominant_pct[group] = pct
                dominant_velos[group] = velocities

        row_data = {"Count": count}

        for group in ["Fastball", "Breaking", "Offspeed"]:
            if group_totals[group] > 0 and dominant_velos[group] is not None:
                velocities = dominant_velos[group]
                pct = group_totals[group]
                if len(velocities) >= 15:
                    low = int(round(np.percentile(velocities, 10)))
                    high = int(round(np.percentile(velocities, 90)))
                else:
                    mean = velocities.mean()
                    low = int(round(mean - 1))
                    high = int(round(mean + 1))
                row_data[group] = f"{pct}% ({low}-{high})"
            else:
                row_data[group] = "—"

        sorted_groups = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_groups) > 1:
            top, second = sorted_groups[0], sorted_groups[1]
            if top[1] >= second[1] + 10:
                row_data[top[0]] = f"<span class='dk-fav'>{row_data[top[0]]}</span>"

        rows.append(row_data)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # 🔁 Sort by strikes first, then balls
    out["balls"] = out["Count"].apply(lambda x: int(x.split("-")[0]))
    out["strikes"] = out["Count"].apply(lambda x: int(x.split("-")[1]))

    return (
        out.sort_values(["strikes", "balls"])
           .drop(columns=["balls", "strikes"])
           .reset_index(drop=True)
    )

