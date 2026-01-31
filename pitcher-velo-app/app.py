import streamlit as st
import pandas as pd
import numpy as np
import re

from pybaseball import chadwick_register
from data import get_pitcher_data

# Try to import Statcast pitching leaderboard (used for dependable pitcher list)
try:
    from pybaseball import statcast_pitching_stats
except Exception:
    statcast_pitching_stats = None


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
# Chadwick registry (fallback only)
# - We are no longer using this for the dropdown.
# - We keep it for Savant link fallback when needed.
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()
    df["name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()

    # best-effort MLBAM id
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

def normalize_name_last_first(name: str) -> str:
    """
    Convert 'Last, First' -> 'First Last' when needed.
    If already 'First Last', returns as-is.
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    n = name.strip()
    if "," in n:
        parts = [p.strip() for p in n.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[1]} {parts[0]}".strip()
    return n

def split_first_last(display_name: str) -> tuple[str, str]:
    """
    Best-effort split for get_pitcher_data(first, last, season).
    Uses first token as first name; remaining tokens as last.
    """
    if not isinstance(display_name, str) or not display_name.strip():
        return "", ""
    parts = display_name.strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])

def savant_url(name: str, mlbam_id: int | None = None):
    """
    Prefer the provided mlbam_id (from Statcast pitcher list).
    Fallback to Chadwick registry if needed.
    """
    pid = None

    if mlbam_id is not None and not pd.isna(mlbam_id):
        try:
            pid = int(mlbam_id)
        except Exception:
            pid = None

    if pid is None:
        row = REGISTRY[REGISTRY["name"] == name]
        if not row.empty and not pd.isna(row.iloc[0]["mlbam_id"]):
            try:
                pid = int(row.iloc[0]["mlbam_id"])
            except Exception:
                pid = None

    if pid is None:
        return None

    return f"https://baseballsavant.mlb.com/savant-player/{slugify(name)}-{pid}"

def get_pitcher_throws(df: pd.DataFrame) -> str | None:
    """
    Safely extract pitcher throwing hand from Statcast pitch-level data.
    Returns 'RHP', 'LHP', or None.
    """
    if df is None or df.empty or "p_throws" not in df.columns:
        return None
    val = df["p_throws"].dropna()
    if val.empty:
        return None
    if val.iloc[0] == "R":
        return "RHP"
    if val.iloc[0] == "L":
        return "LHP"
    return None

def render_pitcher_header(name: str, context: str, mlbam_id: int | None = None):
    url = savant_url(name, mlbam_id=mlbam_id)
    if url:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="margin:0;">{name}</h2>
                <a href="{url}" target="_blank"
                   style="text-decoration:none; font-size:16px; opacity:0.75;">
                    🔗
                </a>
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
                bias = f"{round(over*100,1)}% Over {cutoff:.1f} MPH"
            else:
                bias = f"{round((1-over)*100,1)}% Under {cutoff:.1f} MPH"

            rows.append({"Count": count, "Bias": bias})

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["Count", "Bias"])

        out["sort"] = out["Count"].apply(lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1]))
        out = out.sort_values("sort").drop(columns="sort").reset_index(drop=True)
        return out

    return make_side("L"), make_side("R")

# ---- HTML table renderer (no index + zebra + dark) ----
TABLE_CSS = """
<style>
.dk-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.dk-table th, .dk-table td {
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,0.08);
}
.dk-table th {
  text-align: left;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.85);
}
.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.04);
}
.dk-table tr:nth-child(odd) td {
  background: rgba(0,0,0,0);
}
</style>
"""

def render_bias_table(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    html = df.to_html(index=False, classes="dk-table", escape=False)
    st.markdown(TABLE_CSS + html, unsafe_allow_html=True)

# =============================
# Season-backed pitcher pool (Option A)
# =============================
@st.cache_data(show_spinner=False)
def load_pitcher_pool(season: int) -> pd.DataFrame:
    """
    Returns a dependable pitcher list for the selected season based on Statcast pitching stats.
    Output columns: name_display, first, last, mlbam_id
    """
    if statcast_pitching_stats is None:
        # Hard fail is better than silent junk list — but keep app stable:
        # Return empty pool; UI will stop before running.
        return pd.DataFrame(columns=["name_display", "first", "last", "mlbam_id"])

    # Pull leaderboard for the season.
    # Use a few signature fallbacks to stay robust across pybaseball versions.
    df = None
    errors = []

    # Common patterns in the wild:
    # statcast_pitching_stats(year)
    # statcast_pitching_stats(year, min_pitches=1)
    # statcast_pitching_stats(year, qual=1)
    # We'll attempt a couple safely.
    for kwargs in [{"min_pitches": 1}, {"qual": 1}, {}]:
        try:
            df = statcast_pitching_stats(season, **kwargs)
            if df is not None and not df.empty:
                break
        except TypeError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(str(e))

    if df is None or df.empty:
        return pd.DataFrame(columns=["name_display", "first", "last", "mlbam_id"])

    # Identify name column
    name_col = None
    for c in ["player_name", "Name", "name"]:
        if c in df.columns:
            name_col = c
            break

    # Identify id column
    id_col = None
    for c in ["player_id", "mlbam_id", "id", "pitcher"]:
        if c in df.columns:
            id_col = c
            break

    if name_col is None or id_col is None:
        return pd.DataFrame(columns=["name_display", "first", "last", "mlbam_id"])

    out = df[[name_col, id_col]].copy()
    out = out.dropna(subset=[name_col, id_col])

    out["name_display"] = out[name_col].apply(normalize_name_last_first).astype(str).str.strip()
    out["mlbam_id"] = pd.to_numeric(out[id_col], errors="coerce")

    out = out.dropna(subset=["name_display", "mlbam_id"])
    out = out[out["name_display"] != ""]

    # Deduplicate by MLBAM id (best identity); keep first name_display
    out = out.drop_duplicates(subset=["mlbam_id"]).copy()

    # Build first/last fields for get_pitcher_data
    fl = out["name_display"].apply(split_first_last)
    out["first"] = fl.apply(lambda x: x[0])
    out["last"] = fl.apply(lambda x: x[1])

    out = out.sort_values("name_display").reset_index(drop=True)
    return out[["name_display", "first", "last", "mlbam_id"]]

# =============================
# Matchup controls
# =============================
st.markdown("### Matchup")

c1, c2, c3 = st.columns([3, 3, 2])

# Season first, because pitcher list depends on season (Option A)
with c3:
    season = st.selectbox("Season", [2025, 2026])

pitcher_pool = load_pitcher_pool(int(season))

PITCHER_OPTIONS = ["— Select Pitcher —"] + pitcher_pool["name_display"].tolist()

# Build mapping for safe retrieval (no fragile string-splitting reliance)
PITCHER_MAP = {
    row["name_display"]: {
        "first": row["first"],
        "last": row["last"],
        "mlbam_id": int(row["mlbam_id"]),
    }
    for _, row in pitcher_pool.iterrows()
}

with c1:
    away = st.selectbox("Away Pitcher", PITCHER_OPTIONS)
with c2:
    home = st.selectbox("Home Pitcher", PITCHER_OPTIONS)

c_spacer, c_btn = st.columns([8, 1])
with c_btn:
    run = st.button("Run Matchup", use_container_width=True)

st.divider()

# Require explicit selection + click
if not run or away.startswith("—") or home.startswith("—"):
    st.stop()

# Safety: ensure selected names exist in mapping (should always be true)
if away not in PITCHER_MAP or home not in PITCHER_MAP:
    st.stop()

away_meta = PITCHER_MAP[away]
home_meta = PITCHER_MAP[home]

away_df = get_pitcher_data(away_meta["first"], away_meta["last"], int(season))
home_df = get_pitcher_data(home_meta["first"], home_meta["last"], int(season))

away_throw = get_pitcher_throws(away_df)
home_throw = get_pitcher_throws(home_df)

away_groups = split_by_inning(away_df)
home_groups = split_by_inning(home_df)

tabs = st.tabs(["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"])

for tab, key in zip(tabs, ["All", "Early (1–2)", "Middle (3–4)", "Late (5+)"]):
    with tab:
        # Away
        away_context = (
            f"{away_throw} | Away Pitcher • {key} • {season}"
            if away_throw
            else f"Away Pitcher • {key} • {season}"
        )
        render_pitcher_header(away, away_context, mlbam_id=away_meta["mlbam_id"])

        away_lhb, away_rhb = build_bias_tables(away_groups[key])

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**vs LHB**")
            render_bias_table(away_lhb)
        with col_r:
            st.markdown("**vs RHB**")
            render_bias_table(away_rhb)

        st.divider()

        # Home
        home_context = (
            f"{home_throw} | Home Pitcher • {key} • {season}"
            if home_throw
            else f"Home Pitcher • {key} • {season}"
        )
        render_pitcher_header(home, home_context, mlbam_id=home_meta["mlbam_id"])

        home_lhb, home_rhb = build_bias_tables(home_groups[key])

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown("**vs LHB**")
            render_bias_table(home_lhb)
        with col_r2:
            st.markdown("**vs RHB**")
            render_bias_table(home_rhb)

