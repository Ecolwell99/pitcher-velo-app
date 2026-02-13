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
    # Pitch Tendencies
    *By count & split*
    """,
    unsafe_allow_html=True,
)


# =============================
# Global CSS (TIGHT + COUNT FIX)
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

/* Slightly muted default cells */
.dk-table td {
    color: rgba(255,255,255,0.75);
}

/* Count column emphasized */
.dk-table th:first-child,
.dk-table td:first-child {
    text-align: left;
    width: 60px;
    color: #ffffff;
    font-weight: 600;
}

.dk-table th {
    background: rgba(255,255,255,0.08);
    font-weight: 600;
    color: rgba(255,255,255,0.85);
}

.dk-table tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.04);
}

/* Dominant pitch */
.dk-fav {
    color: #ffffff;
    font-weight: 600;
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
    color: rgba(255,255,255,0.85);
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
                 mph=("release_speed", "mean"))
            .reset_index()
        )

        summary["pct"] = (summary["n"] / total * 100).round(1)
        summary["mph"] = summary["mph"].round(1)

        data = {
            "Fastball (% | MPH)": "—",
            "Breaking (% | MPH)": "—",
            "Offspeed (% | MPH)": "—"
        }

        pct_dict = {}

        for _, r in summary.iterrows():
            pct = r["pct"]
            mean_mph = r["mph"]
            grp = r["group"]

            lower = int(round(mean_mph - 1))
            upper = int(round(mean_mph + 1))
            cluster = f"{lower}-{upper}"

            label = f"{pct}% ({cluster})"

            column_name = f"{grp} (% | MPH)"
            data[column_name] = label
            pct_dict[grp] = pct

        if pct_dict:
            sorted_groups = sorted(pct_dict.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_groups) > 1:
                top, second = sorted_groups[0], sorted_groups[1]
                if top[1] >= second[1] + 10:
                    fav = top[0]
                    fav_col = f"{fav} (% | MPH)"
                    data[fav_col] = f"<span class='dk-fav'>{data[fav_col]}</span>"
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
# Structural Flags
# =============================
def build_structure_flags(dominance_tracker):

    early = {"0-0", "1-0", "0-1"}
    two_strike = {"0-2", "1-2", "2-2"}
    full = {"3-2"}

    def most_common(counts):
        vals = [dominance_tracker[c] for c in counts if c in dominance_tracker]
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    flags = []
    early_flag = most_common(early)
    two_flag = most_common(two_strike)
    full_flag = most_common(full)

    if early_flag:
        flags.append(f"• Early Counts: {early_flag}")
    if two_flag:
        flags.append(f"• 2-Strike: {two_flag}")
    if full_flag:
        flags.append(f"• Full Count: {full_flag}")

    return flags

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

            for side in ["L", "R"]:
                label = "vs LHB" if side == "L" else "vs RHB"
                st.markdown(f"### {label}")

                table, dominance = build_pitch_table(df, side)
                flags = build_structure_flags(dominance)

                if flags:
                    st.markdown(
                        "<div class='dk-flags'>" + "<br>".join(flags) + "</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    table.to_html(index=False, classes="dk-table", escape=False),
                    unsafe_allow_html=True,
                )

            st.divider()


