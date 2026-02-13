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
# Global CSS (balanced hierarchy)
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

/* Subtle neutral dominant highlight */
.dk-fav {
    font-weight: 600;
    background-color: rgba(255,255,255,0.06);
    border-radius: 3px;
    padding: 2px 4px;
}

.dk-subtitle {
    opacity: 0.6;
    margin-top: -6px;
    margin-bottom: 8px;
}

.dk-flags {
    margin-bottom: 6px;
    line-height: 1.4;
    font-size: 12px;
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
# Name normalization
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

FASTBALLS = {"FF", "SI", "FC"}
BREAKING = {"SL", "CU", "KC", "SV", "ST"}
OFFSPEED = {"CH", "FS", "FO"}

def classify_pitch(pt):
    if pt in FASTBALLS:
        return "Fastball"
    if pt in BREAKING:
        return "Breaking"
    if pt in OFFSPEED:
        return "Offspeed"
    return None

def build_inline_mix(df, side):
    g = df[df["stand"] == side].dropna(subset=["pitch_type"])
    if g.empty:
        return None

    mix = (
        g.groupby("pitch_type")
        .agg(n=("pitch_type", "size"))
        .reset_index()
    )

    total = mix["n"].sum()
    mix["pct"] = (mix["n"] / total * 100).round(0)
    mix = mix.sort_values("pct", ascending=False)

    parts = []
    for _, r in mix.iterrows():
        parts.append(f"{r['pitch_type']} {int(r['pct'])}%")

    return " | ".join(parts)

def build_pitch_table(df, side):

    rows = []
    dominance_tracker = {}

    for count, g in df[df["stand"] == side].groupby("count"):
        g = g.dropna(subset=["release_speed", "pitch_type"])
        if g.empty:
            continue

        g = g.copy()
        g["group"] = g["pitch_type"].apply(classify_pitch)
        g = g.dropna(subset=["group"])

        total = len(g)
        if total < 5:
            continue

        summary = (
            g.groupby("group")
            .agg(n=("group", "size"),
                 mph_list=("release_speed", list))
            .reset_index()
        )

        summary["pct"] = (summary["n"] / total * 100).round(1)

        data = {"Fastball": "—",
                "Breaking": "—",
                "Offspeed": "—"}

        pct_dict = {}

        for _, r in summary.iterrows():
            pct = r["pct"]
            velocities = np.array(r["mph_list"])
            grp = r["group"]

            if len(velocities) >= 15:
                lower = int(round(np.percentile(velocities, 10)))
                upper = int(round(np.percentile(velocities, 90)))
            else:
                mean = velocities.mean()
                lower = int(round(mean - 1))
                upper = int(round(mean + 1))

            cluster = f"{lower}-{upper}"
            label = f"{pct}% ({cluster})"

            data[grp] = label
            pct_dict[grp] = pct

        if pct_dict:
            sorted_groups = sorted(pct_dict.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_groups) > 1:
                top, second = sorted_groups[0], sorted_groups[1]
                if top[1] >= second[1] + 10:
                    fav = top[0]
                    data[fav] = f"<span class='dk-fav'>{data[fav]}</span>"
                    dominance_tracker[count] = fav

        row = {"Count": count}
        row.update(data)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}

    out["s"] = out["Count"].apply(
        lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1])
    )
    out = out.sort_values("s").drop(columns="s").reset_index(drop=True)

    return out, dominance_tracker

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

tabs = st.tabs(["All"])

for tab in tabs:
    with tab:
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

                mix_line = build_inline_mix(df, side)
                if mix_line:
                    st.markdown(
                        f"<div class='dk-mix'>Mix: {mix_line}</div>",
                        unsafe_allow_html=True,
                    )

                table, _ = build_pitch_table(df, side)
                st.markdown(
                    table.to_html(index=False, classes="dk-table", escape=False),
                    unsafe_allow_html=True,
                )

            st.divider()

