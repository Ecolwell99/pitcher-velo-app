import streamlit as st
import pandas as pd
import numpy as np
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
    *Velocity behavior by count, inning, and handedness (Statcast)*
    """,
    unsafe_allow_html=True,
)

# =============================
# Global CSS
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
}
.dk-table th {
  text-align: left;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.85);
}
.dk-table tr:nth-child(even) td {
  background: rgba(255,255,255,0.045);
}
.dk-bias th:nth-child(1), .dk-bias td:nth-child(1) { width: 110px; }
.dk-bias th:nth-child(2), .dk-bias td:nth-child(2) { width: auto; }
.dk-mix th:nth-child(1), .dk-mix td:nth-child(1) { width: 140px; }
.dk-mix th:nth-child(2), .dk-mix td:nth-child(2) { width: 100px; text-align:right; }
.dk-mix th:nth-child(3), .dk-mix td:nth-child(3) { width: 100px; text-align:right; }
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# =============================
# Name normalization
# =============================
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"\s+", " ", name.lower()).strip()
    return name

def parse_full_name(input_name: str):
    parts = re.sub(r"\s+", " ", (input_name or "").strip()).split(" ")
    if len(parts) < 2:
        return None, None
    first = " ".join(parts[:-1])
    last = parts[-1]
    return first, last

# =============================
# First-name aliasing (SAFE, controlled)
# =============================
# Bidirectional alias sets. Keep this small + obvious.
_ALIAS_GROUPS = [
    {"matt", "matthew"},
    {"mike", "michael"},
    {"nick", "nicholas"},
    {"chris", "christopher"},
    {"alex", "alexander"},
    {"ben", "benjamin"},
    {"dan", "daniel"},
    {"dave", "david"},
    {"jim", "james"},
    {"joe", "joseph"},
    {"jon", "john", "jonathan"},
    {"tom", "thomas"},
    {"will", "william"},
    {"rob", "robert"},
]
_ALIAS_MAP = {}
for grp in _ALIAS_GROUPS:
    for a in grp:
        _ALIAS_MAP[a] = grp

def first_name_variants(first: str) -> set[str]:
    f = normalize_name(first)
    if not f:
        return set()
    variants = {f}
    if f in _ALIAS_MAP:
        variants |= set(_ALIAS_MAP[f])
    return variants

# =============================
# Load Chadwick registry (cached)
# =============================
@st.cache_data(show_spinner=False)
def load_registry():
    df = chadwick_register().copy()

    df["display_name"] = (
        df.get("name_first", "").fillna("") + " " +
        df.get("name_last", "").fillna("")
    ).str.strip()

    df["norm_name"] = df["display_name"].apply(normalize_name)
    df["norm_first"] = df.get("name_first", "").fillna("").apply(normalize_name)
    df["norm_last"] = df.get("name_last", "").fillna("").apply(normalize_name)

    return df

REGISTRY = load_registry()

def savant_url(display_name: str) -> str | None:
    row = REGISTRY[REGISTRY["display_name"] == display_name]
    if row.empty:
        return None
    for col in ["key_mlbam", "mlbam_id", "key_mlb"]:
        if col in row.columns and not pd.isna(row.iloc[0][col]):
            try:
                return f"https://baseballsavant.mlb.com/savant-player/{int(row.iloc[0][col])}"
            except Exception:
                pass
    return None

# =============================
# Resolve pitcher with Statcast-backed disambiguation
# =============================
def resolve_pitcher(input_name: str, season: int, role: str):
    if not input_name or len(input_name.strip().split()) < 2:
        raise ValueError("Please enter full first and last name.")

    norm_full = normalize_name(input_name)
    matches = REGISTRY[REGISTRY["norm_name"] == norm_full]

    # Fallback 1: last-name match + SAFE first-name alias match
    if matches.empty:
        first, last = parse_full_name(input_name)
        if first and last:
            nf = normalize_name(first)
            nl = normalize_name(last)

            last_matches = REGISTRY[REGISTRY["norm_last"] == nl].copy()

            if not last_matches.empty:
                input_first_variants = first_name_variants(first)

                # Candidate matches if candidate first is in our variants OR input first in candidate variants
                def alias_ok(row):
                    cand_first = row["norm_first"]
                    if not cand_first:
                        return False
                    return (cand_first in input_first_variants) or (nf in first_name_variants(cand_first))

                alias_matches = last_matches[last_matches.apply(alias_ok, axis=1)]

                if not alias_matches.empty:
                    matches = alias_matches

                # Fallback 2: last name exists but alias didn't resolve → offer refine-by-radio
                elif len(last_matches) > 0:
                    st.warning(
                        f'Could not find an exact match for "{input_name}". '
                        f'Pitchers found with last name "{last}": please select the correct one.'
                    )
                    # show up to 25 to avoid overwhelming UI
                    opts = last_matches["display_name"].dropna().unique().tolist()[:25]
                    choice = st.radio(
                        f"Select {role} Pitcher",
                        opts,
                        key=f"refine_lastname_{role}",
                    )
                    matches = REGISTRY[REGISTRY["display_name"] == choice]

    if matches.empty:
        raise ValueError(f"No pitcher found for '{input_name}'.")

    enriched = []

    # Enrich candidates by verifying Statcast data exists (season)
    for _, r in matches.iterrows():
        try:
            df = get_pitcher_data(r["name_first"], r["name_last"], season)
        except ValueError:
            continue

        if df.empty:
            continue

        throws = "LHP" if df["p_throws"].iloc[0] == "L" else "RHP"
        team = df["home_team"].mode().iloc[0] if "home_team" in df else "UNK"

        enriched.append({
            "first": r["name_first"],
            "last": r["name_last"],
            "display": r["display_name"],
            "throws": throws,
            "team": team,
        })

    if not enriched:
        raise ValueError(f"No Statcast data found for '{input_name}' in {season}.")

    if len(enriched) == 1:
        e = enriched[0]
        return e["first"], e["last"], e["display"]

    st.warning(f'Multiple pitchers match "{input_name}" in {season}. Please select:')

    options = {
        f'{e["display"]} — {e["throws"]} — {e["team"]}': e
        for e in enriched
    }

    choice = st.radio(
        f"Select {role} Pitcher",
        list(options.keys()),
        key=f"disambiguate_{role}",
    )

    e = options[choice]
    return e["first"], e["last"], e["display"]

# =============================
# Analytics helpers
# =============================
def get_pitcher_throws(df):
    return None if df.empty else ("RHP" if df["p_throws"].iloc[0] == "R" else "LHP")

def split_by_inning(df):
    return {
        "All": df,
        "Early (1–2)": df[df["inning"].isin([1, 2])],
        "Middle (3–4)": df[df["inning"].isin([3, 4])],
        "Late (5+)": df[df["inning"] >= 5],
    }

def build_bias_tables(df):
    def make(side):
        rows = []
        for c, g in df[df["stand"] == side].groupby("count"):
            v = g["release_speed"].dropna()
            if v.empty:
                continue
            m = v.mean()
            p = (v >= m).mean()
            rows.append({
                "Count": c,
                "Bias": f"{round(max(p,1-p)*100,1)}% {'Over' if p>=.5 else 'Under'} {m:.1f}"
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["s"] = out["Count"].apply(lambda x: int(x.split("-")[0])*10+int(x.split("-")[1]))
        return out.sort_values("s").drop(columns="s")
    return make("L"), make("R")

def build_pitch_mix_overall(df):
    if df.empty:
        return pd.DataFrame(columns=["Pitch Type","Usage %","Avg MPH"])
    g = df[df["pitch_type"] != "PO"].dropna(subset=["pitch_type"])
    if g.empty:
        return pd.DataFrame(columns=["Pitch Type","Usage %","Avg MPH"])
    mix = g.groupby("pitch_type").agg(
        P=("pitch_type","size"),
        V=("release_speed","mean")
    ).reset_index().rename(columns={"pitch_type":"Pitch Type"})
    mix["Usage %"] = (mix["P"]/mix["P"].sum()*100).round(1).astype(str)+"%"
    mix["Avg MPH"] = mix["V"].round(1).astype(str)
    return mix.sort_values("Usage %", ascending=False)[["Pitch Type","Usage %","Avg MPH"]]

def render_table(df, cls):
    st.markdown(df.to_html(index=False, classes=f"dk-table {cls}", escape=False), unsafe_allow_html=True)

def render_pitcher_header(name: str, context: str):
    url = savant_url(name)
    if url:
        st.markdown(
            f"""
            <h2 style="margin-bottom:4px;">
              {name}
              <a href="{url}" target="_blank"
                 style="font-size:16px; opacity:.7; text-decoration:none; border-bottom:none;">
                🔗
              </a>
            </h2>
            <i>{context}</i>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## {name}\n*{context}*")

# =============================
# Controls
# =============================
c1, c2, c3 = st.columns([3,3,2])
with c1:
    away_input = st.text_input("Away Pitcher (First Last)")
with c2:
    home_input = st.text_input("Home Pitcher (First Last)")
with c3:
    season = st.selectbox("Season", [2025, 2026])

run = st.button("Run Matchup", use_container_width=True)
if not run:
    st.stop()

# =============================
# Resolve pitchers
# =============================
try:
    away_first, away_last, away_name = resolve_pitcher(away_input, season, "Away")
    home_first, home_last, home_name = resolve_pitcher(home_input, season, "Home")
except ValueError as e:
    st.error(str(e))
    st.stop()

# =============================
# Pull Statcast data
# =============================
try:
    away_df = get_pitcher_data(away_first, away_last, season)
    home_df = get_pitcher_data(home_first, home_last, season)
except ValueError as e:
    st.error(str(e))
    st.stop()

away_mix = build_pitch_mix_overall(away_df)
home_mix = build_pitch_mix_overall(home_df)

tabs = st.tabs(["All","Early (1–2)","Middle (3–4)","Late (5+)"])

for t, key in zip(tabs, ["All","Early (1–2)","Middle (3–4)","Late (5+)"]):
    with t:
        render_pitcher_header(
            away_name,
            f"{get_pitcher_throws(away_df)} | Away Pitcher • {key} • {season}"
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Show Pitch Mix (Season Overall)"):
            render_table(away_mix, "dk-mix")

        lhb, rhb = build_bias_tables(split_by_inning(away_df)[key])
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**vs LHB**")
            render_table(lhb, "dk-bias")
        with cR:
            st.markdown("**vs RHB**")
            render_table(rhb, "dk-bias")

        st.divider()

        render_pitcher_header(
            home_name,
            f"{get_pitcher_throws(home_df)} | Home Pitcher • {key} • {season}"
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("Show Pitch Mix (Season Overall)"):
            render_table(home_mix, "dk-mix")

        lhb, rhb = build_bias_tables(split_by_inning(home_df)[key])
        cL2, cR2 = st.columns(2)
        with cL2:
            st.markdown("**vs LHB**")
            render_table(lhb, "dk-bias")
        with cR2:
            st.markdown("**vs RHB**")
            render_table(rhb, "dk-bias")

        st.divider()

