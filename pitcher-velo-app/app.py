import streamlit as st
import pandas as pd
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
    *Velocity behavior by count, inning, and handedness*
    """,
    unsafe_allow_html=True,
)

# =============================
# Global CSS — MATCH MAIN VERSION
# =============================
TABLE_CSS = """
<style>
.dk-table {
    width: 520px;
    max-width: 520px;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 14px;
    margin-left: 0 !important;
    margin-right: auto !important;
}

.dk-table th, .dk-table td {
    padding: 8px 10px;
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.dk-table th {
    background: rgba(255,255,255,0.08);
    font-weight: 600;
}

/* Zebra striping */
.dk-table tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.05);
}

/* Count column */
.dk-table th:first-child,
.dk-table td:first-child {
    width: 110px;
    text-align: left;
}

/* Bias column — NARROW (this was the bug) */
.dk-table th:last-child,
.dk-table td:last-child {
    width: 160px;
    text-align: right;
    font-weight: 600;
}

/* Subtitle under pitcher */
.dk-subtitle {
    opacity: 0.6;
    margin-top: -6px;
    margin-bottom: 12px;
}

/* Pitch Mix expander width */
.dk-expander {
    width: 520px;
    max-width: 520px;
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
        raise ValueError(f"No Statcast data for '{name}'")

    if len(valid) == 1:
        return valid[0]

    choice = st.radio(f"Select {role} Pitcher", [v[2] for v in valid])
    return next(v for v in valid if v[2] == choice)

# =============================
# Pitch Mix (overall)
# =============================
def build_pitch_mix(df):
    g = df.dropna(subset=["release_speed", "pitch_type"])
    if g.empty:
        return pd.DataFrame(columns=["Pitch", "%", "MPH"])

    total_n = len(g)
    pt = (
        g.groupby("pitch_type")
        .agg(n=("pitch_type", "size"), mph=("release_speed", "mean"))
        .reset_index()
    )
    pt["usage"] = pt["n"] / total_n
    pt = pt.sort_values("usage", ascending=False)

    return pt.assign(
        usage_pct=(pt["usage"] * 100).round(1),
        mph=pt["mph"].round(1)
    )[["pitch_type", "usage_pct", "mph"]].rename(
        columns={"pitch_type": "Pitch", "usage_pct": "%", "mph": "MPH"}
    )

# =============================
# Bias logic — FINAL RULE
# =============================
def build_bias_table(df, side):
    HARD_FLOOR = 89.0
    rows = []

    for count, g in df[df["stand"] == side].groupby("count"):
        g = g.dropna(subset=["release_speed", "pitch_type"])
        if g.empty:
            continue

        total_n = len(g)

        pt = (
            g.groupby("pitch_type")
            .agg(n=("pitch_type", "size"), mph=("release_speed", "mean"))
            .reset_index()
        )
        pt["usage"] = pt["n"] / total_n
        pt = pt.sort_values("mph", ascending=False).reset_index(drop=True)

        boundary_idx = 0
        for i in range(len(pt)):
            if pt.loc[i, "mph"] >= HARD_FLOOR:
                boundary_idx = i
            else:
                break

        boundary = pt.loc[boundary_idx, "mph"]

        over_pct = pt.loc[pt["mph"] >= boundary, "usage"].sum()
        under_pct = 1 - over_pct

        if over_pct >= under_pct:
            pct = over_pct
            label = "Over"
        else:
            pct = under_pct
            label = "Under"

        bias = f"{round(pct*100,1)}% {label} {boundary:.1f}"

        if total_n < 10:
            bias = f'<span class="dk-low">{bias} <span class="dk-info" title="Very small sample">ⓘ</span></span>'
        elif total_n < 20:
            bias += ' <span class="dk-info" title="Low sample size">ⓘ</span>'

        rows.append({"Count": count, "Bias": bias})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["s"] = out["Count"].apply(lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1]))
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
                f'<div class="dk-subtitle">RHP | {role} Pitcher • {segment} • {season}</div>',
                unsafe_allow_html=True,
            )

            mix_df = build_pitch_mix(df)
            st.markdown('<div class="dk-expander">', unsafe_allow_html=True)
            with st.expander("Pitch Mix", expanded=False):
                st.markdown(
                    mix_df.to_html(index=False, classes="dk-table", escape=False),
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("**vs LHB**")
            lhb = build_bias_table(df, "L")
            st.markdown(lhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)

            st.markdown("**vs RHB**")
            rhb = build_bias_table(df, "R")
            st.markdown(rhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)

            st.divider()
