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
# Global CSS
# =============================
TABLE_CSS = """
<style>
.dk-table {
    width: 520px;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 14px;
    margin-left: 0;
    margin-right: auto;
}
.dk-table th, .dk-table td {
    padding: 8px 10px;
    border: 1px solid rgba(255,255,255,0.08);
    white-space: nowrap;
}
.dk-table th {
    background: rgba(255,255,255,0.08);
    font-weight: 600;
}
.dk-table tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.05);
}
.dk-table th:first-child,
.dk-table td:first-child {
    width: 110px;
    text-align: left;
}
.dk-table th:last-child,
.dk-table td:last-child {
    width: 200px;
    text-align: left;      /* Bias text left-aligned */
    font-weight: 600;
}
.dk-subtitle {
    opacity: 0.6;
    margin-top: -6px;
    margin-bottom: 12px;
}
.dk-expander {
    width: 520px;
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
def normalize_name(name: str) -> str:
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
# Build the DISPLAY pitch-type table for a given slice
# (This table is the single source of truth for Bias.)
# =============================
def pitch_type_table_display(g: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pitch-type table with DISPLAY values:
    - MPH is rounded to 1 decimal
    - % is rounded to 1 decimal (the thing traders add)
    """
    g = g.dropna(subset=["release_speed", "pitch_type"])
    if g.empty:
        return pd.DataFrame(columns=["Pitch Type", "#", "%", "MPH"])

    total_n = len(g)

    pt = (
        g.groupby("pitch_type")
        .agg(
            n=("pitch_type", "size"),
            mph=("release_speed", "mean"),
        )
        .reset_index()
    )
    pt["usage"] = pt["n"] / total_n

    # DISPLAY TRUTH
    pt["MPH"] = pt["mph"].round(1)
    pt["%"] = (pt["usage"] * 100).round(1)

    # Build display frame (sorted by MPH desc like your screenshots)
    out = pt.rename(columns={"pitch_type": "Pitch Type", "n": "#"}).copy()
    out = out[["Pitch Type", "#", "%", "MPH"]].sort_values("MPH", ascending=False).reset_index(drop=True)
    return out

# =============================
# Pitch Mix (overall expander)
# =============================
def build_pitch_mix_overall(df: pd.DataFrame) -> pd.DataFrame:
    return pitch_type_table_display(df)

# =============================
# Bias logic — computed directly from DISPLAY pitch-type table
# =============================
def bias_from_display_table(pt_disp: pd.DataFrame, min_line_usage_pct: float) -> tuple[float, str, float]:
    """
    pt_disp columns: Pitch Type, #, %, MPH  (all DISPLAY values)

    Rule:
    1) Boundary = highest MPH row whose % >= min_line_usage_pct
       (if none, boundary = top MPH)
    2) Over% = sum of DISPLAY % where MPH >= boundary
       Under% = sum of DISPLAY % where MPH <  boundary
    3) Show favored side; ties go to Over (keeps it non-trivial).
    """
    if pt_disp.empty:
        return (0.0, "Under", 0.0)

    # Boundary selection on DISPLAY values
    boundary = None
    for _, r in pt_disp.iterrows():
        if float(r["%"]) >= min_line_usage_pct:
            boundary = float(r["MPH"])
            break
    if boundary is None:
        boundary = float(pt_disp.iloc[0]["MPH"])

    over_sum = float(pt_disp.loc[pt_disp["MPH"] >= boundary, "%"].sum())
    under_sum = float(pt_disp.loc[pt_disp["MPH"] < boundary, "%"].sum())

    # Favor ties to Over
    if over_sum >= under_sum:
        return (over_sum, "Over", boundary)
    return (under_sum, "Under", boundary)

def build_bias_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    MIN_LINE_USAGE_PCT = 30.0  # 30%+ can stand alone as the O/U line (DISPLAY percent)

    rows = []
    df_side = df[df["stand"] == side].copy()

    for count, g in df_side.groupby("count"):
        g = g.dropna(subset=["release_speed", "pitch_type"])
        if g.empty:
            continue

        total_n = len(g)

        # SINGLE SOURCE OF TRUTH: display pitch-type table for THIS exact slice
        pt_disp = pitch_type_table_display(g)

        pct, label, boundary = bias_from_display_table(pt_disp, MIN_LINE_USAGE_PCT)

        bias = f"{pct:.1f}% {label} {boundary:.1f}"

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

def split(df: pd.DataFrame):
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

            # One overall pitch mix expander (season/segment overall)
            mix_df = build_pitch_mix_overall(df)
            st.markdown('<div class="dk-expander">', unsafe_allow_html=True)
            with st.expander("Pitch Mix", expanded=False):
                st.markdown(
                    mix_df.to_html(index=False, classes="dk-table", escape=False),
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # Bias tables
            st.markdown("**vs LHB**")
            lhb = build_bias_table(df, "L")
            st.markdown(lhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)

            st.markdown("**vs RHB**")
            rhb = build_bias_table(df, "R")
            st.markdown(rhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)

            st.divider()

