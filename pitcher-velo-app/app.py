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
    *Velocity behavior by count, inning, and handedness (Statcast)*
    """,
    unsafe_allow_html=True,
)

# =============================
# Global CSS
# =============================
TABLE_CSS = """
<style>
.dk-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.dk-table th, .dk-table td { padding: 10px 12px; border: 1px solid rgba(255,255,255,0.08); }
.dk-table th { background: rgba(255,255,255,0.06); }
.dk-table tr:nth-child(even) td { background: rgba(255,255,255,0.045); }
.dk-info { opacity: 0.6; margin-left: 6px; cursor: help; }
.dk-low { opacity: 0.45; }
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
        raise ValueError(f"No Statcast data for '{name}' in {season}")

    if len(valid) == 1:
        return valid[0]

    choice = st.radio(f"Select {role} Pitcher", [v[2] for v in valid])
    return next(v for v in valid if v[2] == choice)

# =============================
# Bias logic — Regime-based (FINAL)
# =============================
def build_bias_tables(df):

    MIN_SIDE_USAGE = 0.15     # both sides must matter
    MIN_BOUNDARY_USAGE = 0.08 # boundary pitch must not be fringe

    def make(side):
        rows = []

        for count, g in df[df["stand"] == side].groupby("count"):
            g = g.dropna(subset=["release_speed", "pitch_type"])
            if g.empty:
                continue

            total_n = len(g)

            # ---- pitch-type table ----
            pt = (
                g.groupby("pitch_type")
                .agg(
                    n=("pitch_type", "size"),
                    mph=("release_speed", "mean")
                )
                .reset_index()
            )
            pt["usage"] = pt["n"] / total_n

            # ---- velocity ladder (hard → soft) ----
            pt = pt.sort_values("mph", ascending=False).reset_index(drop=True)

            splits = []

            # ---- evaluate adjacent regime splits ----
            for i in range(len(pt) - 1):
                hard = pt.iloc[: i + 1]
                soft = pt.iloc[i + 1 :]

                hard_usage = hard["usage"].sum()
                soft_usage = soft["usage"].sum()

                boundary_pitch = pt.iloc[i]
                gap = pt.iloc[i]["mph"] - pt.iloc[i + 1]["mph"]

                if (
                    hard_usage >= MIN_SIDE_USAGE
                    and soft_usage >= MIN_SIDE_USAGE
                    and boundary_pitch["usage"] >= MIN_BOUNDARY_USAGE
                ):
                    score = gap * min(hard_usage, soft_usage)
                    splits.append({
                        "boundary_mph": boundary_pitch["mph"],
                        "score": score
                    })

            # ---- choose boundary ----
            if splits:
                boundary = max(splits, key=lambda x: x["score"])["boundary_mph"]
            else:
                # ---- fallback: usage-weighted median pitch ----
                pt["cum_usage"] = pt["usage"].cumsum()
                boundary = pt.loc[pt["cum_usage"] >= 0.5].iloc[0]["mph"]

            # ---- compute bias using ALL pitches (pitch-type regime) ----
            pt_mph_map = dict(zip(pt["pitch_type"], pt["mph"]))
            bucket_mph = g["pitch_type"].map(pt_mph_map)

            over_pct = (bucket_mph >= boundary).mean()
            under_pct = 1 - over_pct

            if over_pct >= under_pct:
                pct = over_pct
                label = "Over"
            else:
                pct = under_pct
                label = "Under"

            bias = f"{round(pct*100,1)}% {label} {boundary:.1f}"

            # ---- sample size context ----
            cls = ""
            tip = None
            if total_n < 10:
                cls = "dk-low"
                tip = "Very small sample (<10 pitches)"
            elif total_n < 20:
                tip = "Low sample size (10–19 pitches). Interpret directionally."

            if tip:
                bias += f' <span class="dk-info" title="{tip}">ⓘ</span>'
            if cls:
                bias = f'<span class="{cls}">{bias}</span>'

            rows.append({"Count": count, "Bias": bias})

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["s"] = out["Count"].apply(
            lambda x: int(x.split("-")[0]) * 10 + int(x.split("-")[1])
        )
        return out.sort_values("s").drop(columns="s").reset_index(drop=True)

    return make("L"), make("R")

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

for tab, key in zip(tabs, split(away_df).keys()):
    with tab:
        st.markdown(f"## {away_name}")
        lhb, rhb = build_bias_tables(split(away_df)[key])
        st.markdown("**vs LHB**")
        st.markdown(lhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)
        st.markdown("**vs RHB**")
        st.markdown(rhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)

        st.divider()

        st.markdown(f"## {home_name}")
        lhb, rhb = build_bias_tables(split(home_df)[key])
        st.markdown("**vs LHB**")
        st.markdown(lhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)
        st.markdown("**vs RHB**")
        st.markdown(rhb.to_html(index=False, classes="dk-table", escape=False), unsafe_allow_html=True)
