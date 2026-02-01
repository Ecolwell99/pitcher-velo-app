import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
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
# Global CSS (dark, zebra, trader-scan)
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
  vertical-align: top;
}

.dk-table th {
  text-align: left;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.85);
  font-weight: 600;
}

.dk-table td {
  color: rgba(255,255,255,0.88);
}

.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.045);
}
.dk-table tr:nth-child(odd) td {
  background: rgba(0,0,0,0);
}

/* Bias table: keep Count narrow, Bias wide */
.dk-bias th:nth-child(1), .dk-bias td:nth-child(1) {
  width: 110px;
  white-space: nowrap;
}
.dk-bias th:nth-child(2), .dk-bias td:nth-child(2) {
  width: calc(100% - 110px);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Pitch mix */
.dk-mix th:nth-child(1), .dk-mix td:nth-child(1) { width: 110px; white-space: nowrap; }
.dk-mix th:nth-child(2), .dk-mix td:nth-child(2) { width: 110px; white-space: nowrap; }
.dk-mix th:nth-child(3), .dk-mix td:nth-child(3) { width: 130px; white-space: nowrap; }
.dk-mix th:nth-child(4), .dk-mix td:nth-child(4) { width: auto;  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# =============================
# Load pitcher list (STATIC CSV — ROBUST PATH)
# =============================
BASE_DIR = Path(__file__).resolve().parent
PITCHER_CSV_PATH = BASE_DIR / "assets" / "pitchers.csv"

try:
    PITCHERS_DF = pd.read_csv(PITCHER_CSV_PATH)
except Exception as e:
    st.error(f"Failed to load pitchers.csv at {PITCHER_CSV_PATH}: {e}")
    st.stop()

required_cols = {"name", "first", "last"}
if not required_cols.issubset(PITCHERS_DF.columns):
    st.error("pitchers.csv must contain columns: name, first, last")
    st.stop()

PITCHERS_DF = PITCHERS_DF.dropna(subset=["name", "first", "last"])
PITCHERS_DF["name"] = PITCHERS_DF["name"].astype(str).str.strip()

PITCHER_OPTIONS = ["— Select Pitcher —"] + sorted(PITCHERS_DF["name"].tolist())

PITCHER_MAP = {
    row["name"]: {"first": row["first"], "last": row["last"]}
    for _, row in PITCHERS_DF.iterrows()
}

# =============================
# Load & cache registry (Savant links only)
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()

    for col in ["key_mlbam", "mlbam_id", "key_mlb"]:
        if col in df.columns:
            df["mlbam_id"] = df[col]
            break
    else:
        df["mlbam_id"] = np.nan

    return df[["name", "mlbam_id"]]

REGISTRY = load_registry()

# =============================
# Helpers
# =============================
MIN_PITCHES = 1

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

def savant_url(name: str):
    row = REGISTRY[REGISTRY["name"] == name]
    if row.empty or pd.isna(row.iloc[0]["mlbam_id"]):
        return None
    return f"https://baseballsavant.mlb.com/savant-player/{slugify(name)}-{int(row.iloc[0]['mlbam_id'])}"

def get_pitcher_throws(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "p_throws" not in df.columns:
        return None
    v = df["p_throws"].dropna()
    if v.empty:
        return None
    return "RHP" if v.iloc[0] == "R" else "LHP"

def render_pitcher_header(name: str, context: str):
    url = savant_url(name)
    if url:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="margin:0;">{name}</h2>
                <a href="{url}" target="_blank"
                   style="text-decoration:none; font-size:16px; opacity:0.75;">🔗</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {name}")

    st.markdown(f"*{context}*")

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_bias_tables(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Count", "Bias"]), pd.DataFrame(columns=["Count", "Bias"])

    df = df[df["stand"].isin(["R", "L"])]

    def make_side(side):
        rows = []
        for count, g in df[df["stand"] == side].groupby("count"):
            speeds = g["release_speed"].dropna().to_numpy()
            if len(speeds) < MIN_PITCHES:
                continue

            cutoff = round(float(np.mean(speeds)), 1)
            over = float((speeds >= cutoff).mean())

            if over >= 0.5:
                bias = f"{round(over*100,1)}% Over {cutoff:.1f}"
            else:
                bias = f"{round((1-over)*100,1)}% Under {cutoff:.1f}"

            rows.append({"Count": count, "Bias": bias})

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["Count", "Bias"])

        out["sort"] = out["Count"].apply(lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1]))
        out = out.sort_values("sort").drop(columns="sort").reset_index(drop=True)
        return out

    return make_side("L"), make_side("R")

def build_pitch_mix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pitch mix table (overall for the inning group):
    - Pitch
    - Usage %
    - Avg Velo
    """
    if df is None or df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame(columns=["Pitch", "Usage", "Avg Velo", "Notes"])

    g = df.dropna(subset=["pitch_type"]).copy()
    if g.empty:
        return pd.DataFrame(columns=["Pitch", "Usage", "Avg Velo", "Notes"])

    total = len(g)
    mix = (
        g.groupby("pitch_type")
         .agg(Pitches=("pitch_type", "size"), AvgVelo=("release_speed", "mean"))
         .reset_index()
         .rename(columns={"pitch_type": "Pitch"})
    )

    mix["Usage"] = (mix["Pitches"] / total) * 100.0
    mix["Usage"] = mix["Usage"].round(1).astype(str) + "%"
    mix["Avg Velo"] = mix["AvgVelo"].round(1)

    mix["Notes"] = ""
    mix = mix.drop(columns=["Pitches", "AvgVelo"])

    # sort by numeric usage desc
    mix = mix.sort_values(
        by="Usage",
        ascending=False,
        key=lambda s: s.str.replace("%", "", regex=False).astype(float)
    ).reset_index(drop=True)

    return mix[["Pitch", "Usage", "Avg Velo", "Notes"]]

def render_table(df: pd.DataFrame, table_class: str):
    df = df.copy().reset_index(drop=True)
    html = df.to_html(index=False, classes=f"dk-table {table_class}", escape=False)
    st.markdown(html, unsafe_allow_html=True)

# =============================
# Matchup controls
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 2])
with c1:
    away = st.selectbox("Away Pitcher", PITCHER_OPTIONS)
with c2:
    home = st.selectbox("Home Pitcher", PITCHER_OPTIONS)
with c3:
    season = st.selectbox("Season", [2025, 2026])

c_spacer, c_btn = st.columns([8, 1])
with c_btn:
    run = st.button("Run Matchup", use_container_width=True)

st.divider()

if not run or away.startswith("—") or home.startswith("—"):
    st.stop()

away_meta = PITCHER_MAP[away]
home_meta = PITCHER_MAP[home]

# =============================
# SAFE Statcast fetch with clear messaging
# =============================
try:
    away_df = get_pitcher_data(away_meta["first"], away_meta["last"], season)
except ValueError:
    st.error(f"No Statcast data available for **{away} ({season})**.")
    st.stop()

try:
    home_df = get_pitcher_data(home_meta["first"], home_meta["last"], season)
except ValueError:
    st.error(f"No Statcast data available for **{home} ({season})**.")
    st.stop()

away_throw = get_pitcher_throws(away_df)
home_throw = get_pitcher_throws(home_df)

away_groups = split_by_inning(away_df)
home_groups = split_by_inning(home_df)

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, key in zip(tabs, ["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"]):
    with tab:
        # -------------------------
        # Away Pitcher block
        # -------------------------
        render_pitcher_header(away, f"{away_throw} | Away Pitcher • {key} • {season}")

        # Pitch Mix expander (the table itself lives inside this dropdown)
        # collapsed by default to keep the view clean; expand to show the table
        with st.expander("Pitch Mix ▾", expanded=False, key=f"mix_expander_away_{key}"):
            away_mix = build_pitch_mix(away_groups[key])
            render_table(away_mix, "dk-mix")

        # Bias tables below
        away_lhb, away_rhb = build_bias_tables(away_groups[key])
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**vs LHB**")
            render_table(away_lhb, "dk-bias")
        with col_r:
            st.markdown("**vs RHB**")
            render_table(away_rhb, "dk-bias")

        st.divider()

        # -------------------------
        # Home Pitcher block
        # -------------------------
        render_pitcher_header(home, f"{home_throw} | Home Pitcher • {key} • {season}")

        with st.expander("Pitch Mix ▾", expanded=False, key=f"mix_expander_home_{key}"):
            home_mix = build_pitch_mix(home_groups[key])
            render_table(home_mix, "dk-mix")

        home_lhb, home_rhb = build_bias_tables(home_groups[key])
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown("**vs LHB**")
            render_table(home_lhb, "dk-bias")
        with col_r2:
            st.markdown("**vs RHB**")
            render_table(home_rhb, "dk-bias")

