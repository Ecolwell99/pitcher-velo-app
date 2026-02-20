import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from pybaseball import chadwick_register

from data import get_pitcher_data

# =============================
# Constants
# =============================
COUNT_ORDER = [
    "0-0",
    "0-1",
    "0-2",
    "1-0",
    "1-1",
    "1-2",
    "2-0",
    "2-1",
    "2-2",
    "3-0",
    "3-1",
    "3-2",
]
COUNT_ORDER_MAP = {c: i for i, c in enumerate(COUNT_ORDER)}
SEGMENTS = ["All", "Early (1-2)", "Middle (3-4)", "Late (5+)"]
DEFAULT_COLOR_COLUMNS = False  # Default for user toggle
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
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap");
:root {
    --dk-font: "IBM Plex Sans", "Segoe UI", sans-serif;
    --dk-border: rgba(128,128,128,0.36);
    --dk-border-soft: rgba(128,128,128,0.24);
    --dk-header-bg: rgba(128,128,128,0.18);
    --dk-row-alt: rgba(128,128,128,0.10);
    --dk-row-hover: rgba(128,128,128,0.15);
    --dk-pill-border: rgba(128,128,128,0.68);
    --dk-radius-sm: 8px;
    --dk-radius-md: 12px;
    --dk-fastball: #F06A46;
    --dk-breaking: #5A92FF;
    --dk-offspeed: #977AFF;
}

html, body, [class*="css"], .stApp {
    font-family: var(--dk-font);
}

.dk-table, .dk-mix, .dk-flags, .dk-subtitle {
    font-family: var(--dk-font);
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--dk-border-soft);
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border-radius: var(--dk-radius-sm);
}

[data-testid="stButton"] button {
    border-radius: var(--dk-radius-sm);
    border: 1px solid var(--dk-border);
    font-weight: 600;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
}

[data-testid="stTabs"] button[data-baseweb="tab"] {
    border-radius: var(--dk-radius-sm) var(--dk-radius-sm) 0 0;
}

.dk-table {
    width: min(760px, 100%);
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 13px;
    border-radius: var(--dk-radius-md);
    overflow: hidden;
}

.dk-table th,
.dk-table td {
    padding: 6px 7px;
    border: 1px solid var(--dk-border);
    text-align: center;
}

.dk-table td {
    color: inherit;
}

.dk-table th:first-child,
.dk-table td:first-child {
    text-align: center;
    width: 66px;
    font-weight: 700;
    color: inherit;
}

.dk-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--dk-header-bg);
    font-weight: 600;
}

.dk-table tbody tr:nth-child(even) td {
    background: var(--dk-row-alt);
}

.dk-table tbody tr:hover td {
    background: var(--dk-row-hover);
}

.dk-fav {
    font-weight: 600;
    background-color: transparent;
    border: 1.5px solid var(--dk-pill-border);
    border-radius: var(--dk-radius-sm);
    padding: 2px 8px;
}

.dk-subtitle {
    opacity: 0.75;
    margin-bottom: 12px;
}

.dk-mix {
    font-size: 12px;
    margin-bottom: 8px;
    opacity: 0.8;
}

.dk-flags {
    font-size: 12px;
    margin-bottom: 8px;
    line-height: 1.4;
}
a.dk-link {
    color: inherit !important;
    text-decoration: none !important;
}

a.dk-link:hover {
    opacity: 0.88;
}
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)
with st.sidebar:
    st.header("Display")
    color_columns = st.checkbox("Color columns", value=DEFAULT_COLOR_COLUMNS)
# =============================
# Helpers
# =============================
def normalize_name(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name.lower()).strip()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["display_name"] = (
        df.get("name_first", "").fillna("") + " " + df.get("name_last", "").fillna("")
    ).str.strip()
    df["norm"] = df["display_name"].apply(normalize_name)
    return df


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_pitcher_data(first_name, last_name, season):
    return get_pitcher_data(first_name, last_name, season)


REGISTRY = load_registry()


def resolve_pitcher(name, season, role):
    norm = normalize_name(name)
    rows = REGISTRY[REGISTRY["norm"] == norm]

    valid = []
    for _, r in rows.iterrows():
        try:
            df = load_pitcher_data(r["name_first"], r["name_last"], season)
            if not df.empty:
                mlbam = r["key_mlbam"] if "key_mlbam" in r.index else None
                valid.append((r["name_first"], r["name_last"], r["display_name"], mlbam))
        except Exception:
            pass

    if not valid:
        raise ValueError(f"No Statcast results found for {role.lower()} input: {name}")

    if len(valid) == 1:
        return valid[0]

    choice = st.radio(f"Select {role} Pitcher", [v[2] for v in valid], horizontal=True)
    return next(v for v in valid if v[2] == choice)


# =============================
# Pull handedness from Statcast
# =============================
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


# =============================
# Most recent team
# =============================
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
        top = counts.idxmax()
        if top in FASTBALLS:
            group = "Fastball"
        elif top in BREAKING:
            group = "Breaking"
        elif top in OFFSPEED:
            group = "Offspeed"
        else:
            continue
        dominance[count] = group

    early = {"0-0", "1-0", "0-1"}
    two = {"0-2", "1-2", "2-2"}
    full = {"3-2"}

    def most_common(keys):
        vals = [dominance[k] for k in keys if k in dominance]
        return max(set(vals), key=vals.count) if vals else None

    flags = []
    if e := most_common(early):
        flags.append(f"- Early Counts: {e}")
    if t := most_common(two):
        flags.append(f"- 2-Strike: {t}")
    if f := most_common(full):
        flags.append(f"- Full Count: {f}")
    return flags


# =============================
# Pitch Table
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
            .agg(n=("pitch_type", "size"), mph_list=("release_speed", list))
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
                row_data[group] = "-"

        sorted_groups = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_groups) > 1:
            top, second = sorted_groups[0], sorted_groups[1]
            if top[1] >= second[1] + 10:
                row_data[top[0]] = f"<span class='dk-fav'>{row_data[top[0]]}</span>"

        rows.append(row_data)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["count_rank"] = out["Count"].map(COUNT_ORDER_MAP).fillna(999)
    return out.sort_values("count_rank").drop(columns=["count_rank"]).reset_index(drop=True)


# =============================
# Segment split
# =============================
def split_segments(df):
    return {
        "All": df,
        "Early (1-2)": df[df["inning"].isin([1, 2])],
        "Middle (3-4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }


# =============================
# Controls
# =============================
if "run_matchup" not in st.session_state:
    st.session_state.run_matchup = False

def trigger_run():
    away_val = st.session_state.get("away_input", "").strip()
    home_val = st.session_state.get("home_input", "").strip()
    st.session_state.run_matchup = bool(away_val and home_val)

c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    away = st.text_input("Away Pitcher (First Last)", key="away_input")
with c2:
    home = st.text_input("Home Pitcher (First Last)", key="home_input", on_change=trigger_run)
with c3:
    season = st.selectbox("Season", [2025, 2026], index=0)

if color_columns:
    st.markdown(
        """
        <style>
        .dk-table th:nth-child(2), .dk-table td:nth-child(2) { color: var(--dk-fastball); font-weight: 600; }
        .dk-table th:nth-child(3), .dk-table td:nth-child(3) { color: var(--dk-breaking); font-weight: 600; }
        .dk-table th:nth-child(4), .dk-table td:nth-child(4) { color: var(--dk-offspeed); font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

if st.button("Run Matchup", use_container_width=True):
    st.session_state.run_matchup = True

if not st.session_state.run_matchup:
    st.stop()

away = away.strip()
home = home.strip()
if not away or not home:
    st.info("Enter both away and home pitchers, then press Enter or Run Matchup.")
    st.stop()

try:
    away_f, away_l, away_name, away_mlbam = resolve_pitcher(away, season, "Away")
    home_f, home_l, home_name, home_mlbam = resolve_pitcher(home, season, "Home")
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.spinner("Loading pitcher data..."):
    away_df_full = load_pitcher_data(away_f, away_l, season)
    home_df_full = load_pitcher_data(home_f, home_l, season)

away_team = get_current_team(away_df_full)
home_team = get_current_team(home_df_full)

away_hand = get_hand_from_statcast(away_df_full)
home_hand = get_hand_from_statcast(home_df_full)

away_splits = split_segments(away_df_full)
home_splits = split_segments(home_df_full)


tabs = st.tabs(SEGMENTS)

for tab, segment in zip(tabs, SEGMENTS):
    with tab:
        for name, team, mlbam_id, hand, segment_df in [
            (away_name, away_team, away_mlbam, away_hand, away_splits[segment]),
            (home_name, home_team, home_mlbam, home_hand, home_splits[segment]),
        ]:

            if mlbam_id:
                url = f"https://baseballsavant.mlb.com/savant-player/{int(mlbam_id)}"
                st.markdown(
                    f"""
                    <a href="{url}" target="_blank" class="dk-link">
                        <div style='font-size:24px; font-weight:700; margin-top:10px;'>{name}</div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size:24px; font-weight:700; margin-top:10px;'>{name}</div>",
                    unsafe_allow_html=True,
                )

            hand_display = f"{hand} | " if hand else ""
            team_display = team if team else "-"

            st.markdown(
                f"<div class='dk-subtitle'>{team_display} | {hand_display}{segment} | {season}</div>",
                unsafe_allow_html=True,
            )

            for side in ["L", "R"]:
                label = "vs LHB" if side == "L" else "vs RHB"

                st.markdown(
                    f"<div style='font-weight:600; font-size:18px; margin-top:10px;'>{label}</div>",
                    unsafe_allow_html=True,
                )

                mix_line = build_inline_mix(segment_df, side)
                if mix_line:
                    st.markdown(
                        f"<div class='dk-mix'>Mix: {mix_line}</div>",
                        unsafe_allow_html=True,
                    )

                flags = build_structure_flags(segment_df, side)
                if flags:
                    st.markdown(
                        "<div class='dk-flags'>" + "<br>".join(flags) + "</div>",
                        unsafe_allow_html=True,
                    )

                table = build_pitch_table(segment_df, side)

                if table.empty:
                    st.markdown("<div class='dk-mix'>No rows for current filters.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        table.to_html(index=False, classes="dk-table", escape=False),
                        unsafe_allow_html=True,
                    )







































































