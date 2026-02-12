import streamlit as st
import pandas as pd
import re
import unicodedata
from pybaseball import chadwick_register
from data import get_pitcher_data

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Pitch Profiles", layout="wide")

st.markdown(
    """
    # Pitcher Pitch Profiles
    *Pitch selection and velocity by count & handedness*
    """,
    unsafe_allow_html=True,
)

# =============================
# Global CSS
# =============================
TABLE_CSS = """
<style>
.dk-table {
    width: 620px;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 14px;
}
.dk-table th, .dk-table td {
    padding: 8px 10px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

/* Slightly mute all cell text */
.dk-table td {
    color: rgba(255,255,255,0.75);
}

.dk-table th:first-child,
.dk-table td:first-child {
    text-align: left;
    width: 80px;
}
.dk-table th {
    background: rgba(255,255,255,0.08);
    font-weight: 600;
    color: rgba(255,255,255,0.85);
}
.dk-table tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.04);
}

/* Dominant pitch styling */
.dk-fav {
    color: #ffffff;
    font-weight: 600;
}

.dk-subtitle {
    opacity: 0.6;
    margin-top: -6px;
    margin-bottom: 12px;
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
# Registry
# =============================
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

# =============================
# Resolve pitcher
# =============================
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
# Pitch group mapping
# =============================
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

# =============================
# Build pitch table
# =============================
def build_pitch_table(df, side):

    rows = []

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
            .agg(
                n=("group", "size"),
                mph=("release_speed", "mean")
            )
            .reset_index()
        )

        summary["pct"] = (summary["n"] / total * 100).round(1)
        summary["mph"] = summary["mph"].round(1)

        data = {"Fastball": "—", "Breaking": "—", "Offspeed": "—"}
        pct_dict = {}

        for _, r in summary.iterrows():
            pct = r["pct"]
            mph = r["mph"]
            grp = r["group"]
            data[grp] = f"{pct}% ({mph})"
            pct_dict[grp] = pct

        # Determine dominant pitch (must be 10% higher than second)
        if pct_dict:
            sorted_groups = sorted(pct_dict.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_groups) > 1:
                top, second = sorted_groups[0], sorted_groups[1]
                if top[1] >= second[1] + 10:
                    fav = top[0]
                    data[fav] = f"<span class='dk-fav'>{data[fav]}</span>"

        rows.append({
            "Count": count,
            "Fastball": data["Fastball"],
            "Breaking": data["Breaking"],
            "Offspeed": data["Offspeed"],
        })

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

try:
    away_f, away_l, away_name = resolve_pitcher(away, season, "Away")
except ValueError:
    st.error("Away pitcher not found — check spelling or season availability.")
    st.stop()

try:
    home_f, home_l, home_name = resolve_pitcher(home, season, "Home")
except ValueError:
    st.error("Home pitcher not found — check spelling or season availability.")
    st.stop()

away_df = get_pitcher_data(away_f, away_l, season)
home_df = get_pitcher_data(home_f, home_l, season)

def split(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, segment in zip(tabs, split(away_df).keys()):
    with tab:
        for name, df, role in [
            (away_name, split(away_df)[segment], "Away"),
            (home_name, split(home_df)[segment], "Home"),
        ]:
            st.markdown(f"## {name}")
            st.markdown(
                f'<div class="dk-subtitle">{role} Pitcher • {segment} • {season}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**vs LHB**")
            lhb = build_pitch_table(df, "L")
            st.markdown(
                lhb.to_html(index=False, classes="dk-table", escape=False),
                unsafe_allow_html=True,
            )

            st.markdown("**vs RHB**")
            rhb = build_pitch_table(df, "R")
            st.markdown(
                rhb.to_html(index=False, classes="dk-table", escape=False),
                unsafe_allow_html=True,
            )

            st.divider()

