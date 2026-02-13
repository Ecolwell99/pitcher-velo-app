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
    # Pitch Tendencies
    *By count & split*
    """,
    unsafe_allow_html=True,
)

# =============================
# Global CSS
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
    margin-top: -6px;
    margin-bottom: 8px;
}

.dk-mix {
    font-size: 12px;
    margin-bottom: 8px;
    opacity: 0.75;
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

# =============================
# Build pitch table (dominant pitch band logic)
# =============================
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

        data = {"Fastball": "—",
                "Breaking": "—",
                "Offspeed": "—"}

        group_totals = {"Fastball": 0, "Breaking": 0, "Offspeed": 0}
        dominant_pitch = {"Fastball": None, "Breaking": None, "Offspeed": None}
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
                dominant_pitch[group] = velocities

        for group in data.keys():
            if group_totals[group] > 0:
                velocities = dominant_pitch[group]
                pct = group_totals[group]

                if len(velocities) >= 15:
                    lower = int(round(np.percentile(velocities, 10)))
                    upper = int(round(np.percentile(velocities, 90)))
                else:
                    mean = velocities.mean()
                    lower = int(round(mean - 1))
                    upper = int(round(mean + 1))

                cluster = f"{lower}-{upper}"
                data[group] = f"{pct}% ({cluster})"

        # Determine dominant group
        sorted_groups = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_groups) > 1:
            top, second = sorted_groups[0], sorted_groups[1]
            if top[1] >= second[1] + 10:
                data[top[0]] = f"<span class='dk-fav'>{data[top[0]]}</span>"

        row = {"Count": count}
        row.update(data)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["s"] = out["Count"].apply(
        lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1])
    )
    return out.sort_values("s").drop(columns="s").reset_index(drop=True)

# =============================
# Controls
# =============================
c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    away = st.text_input("Away Pitcher (First Last)")
with c2:
    home = st.text_input("Home Pitcher (First Last)")
with c3:
    season = st.selectbox("Season", [2025, 2026])

if not st.button("Run Matchup", use_container_width=True):
    st.stop()

away_f, away_l, away_name = resolve_pitcher(away, season, "Away")
home_f, home_l, home_name = resolve_pitcher(home, season, "Home")

away_df = get_pitcher_data(away_f, away_l, season)
home_df = get_pitcher_data(home_f, home_l, season)

for name, df, role in [
    (away_name, away_df, "Away"),
    (home_name, home_df, "Home"),
]:
    st.markdown(f"## {name}")
    st.markdown(
        f'<div class="dk-subtitle">{role} Pitcher • All • {season}</div>',
        unsafe_allow_html=True,
    )

    for side in ["L", "R"]:
        label = "vs LHB" if side == "L" else "vs RHB"
        st.markdown(f"### {label}")

        table = build_pitch_table(df, side)
        st.markdown(
            table.to_html(index=False, classes="dk-table", escape=False),
            unsafe_allow_html=True,
        )

    st.divider()

