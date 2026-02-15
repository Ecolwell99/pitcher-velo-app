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
# CSS (Polished)
# =============================
TABLE_CSS = """
<style>
.dk-table {
    width: 560px;
    table-layout: fixed;
    border-collapse: collapse;
    font-size: 13px;
}

.dk-table th, .dk-table td {
    padding: 5px 6px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    border-right: none;
    border-left: none;
    border-top: none;
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
    background-color: rgba(255,255,255,0.10);
    border-radius: 6px;
    padding: 2px 6px;
}

.dk-subtitle {
    opacity: 0.55;
    font-size: 13px;
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
        flags.append(f"• Early Counts: {e}")
    if t := most_common(two):
        flags.append(f"• 2-Strike: {t}")
    if f := most_common(full):
        flags.append(f"• Full Count: {f}")
    return flags

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

away_f, away_l, away_name, away_mlbam = resolve_pitcher(away, season, "Away")
home_f, home_l, home_name, home_mlbam = resolve_pitcher(home, season, "Home")

away_df_full = get_pitcher_data(away_f, away_l, season)
home_df_full = get_pitcher_data(home_f, home_l, season)

away_team = get_current_team(away_df_full)
home_team = get_current_team(home_df_full)

away_hand = get_hand_from_statcast(away_df_full)
home_hand = get_hand_from_statcast(home_df_full)

def split(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, segment in zip(tabs, split(away_df_full).keys()):
    with tab:
        for name, df_full, team, mlbam_id, hand in [
            (away_name, away_df_full, away_team, away_mlbam, away_hand),
            (home_name, home_df_full, home_team, home_mlbam, home_hand),
        ]:

            df_segment = split(df_full)[segment]

            if mlbam_id:
                url = f"https://baseballsavant.mlb.com/savant-player/{int(mlbam_id)}"
                st.markdown(
                    f"""
                    <a href="{url}" target="_blank" class="dk-link">
                        <div style='font-size:26px; font-weight:700; letter-spacing:0.2px; margin-top:10px;'>{name}</div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='font-size:26px; font-weight:700; letter-spacing:0.2px; margin-top:10px;'>{name}</div>",
                    unsafe_allow_html=True
                )

            hand_display = f"{hand} • " if hand else ""

            st.markdown(
                f"<div class='dk-subtitle'>{team} • {hand_display}{segment} • {season}</div>",
                unsafe_allow_html=True
            )

            for side in ["L", "R"]:
                label = "vs LHB" if side == "L" else "vs RHB"

                st.markdown(
                    f"<div style='font-weight:600; font-size:18px; margin-top:10px;'>{label}</div>",
                    unsafe_allow_html=True
                )

                mix_line = build_inline_mix(df_segment, side)
                if mix_line:
                    st.markdown(
                        f"<div class='dk-mix'>Mix: {mix_line}</div>",
                        unsafe_allow_html=True,
                    )

                flags = build_structure_flags(df_segment, side)
                if flags:
                    st.markdown(
                        "<div class='dk-flags'>" + "<br>".join(flags) + "</div>",
                        unsafe_allow_html=True,
                    )

                table = build_pitch_table(df_segment, side)
                st.markdown(
                    table.to_html(index=False, classes="dk-table", escape=False),
                    unsafe_allow_html=True,
                )

            st.markdown("<hr style='opacity:0.2;'>", unsafe_allow_html=True)
