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
.dk-table th:first-child,
.dk-table td:first-child {
    text-align: left;
    width: 80px;
}
.dk-table th {
    background: rgba(255,255,255,0.08);
    font-weight: 600;
}
.dk-table tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.04);
}

/* Softer institutional green highlight */
.dk-fav {
    background-color: rgba(40, 167, 69, 0.12);
    border-radius: 4px;
    padding: 2px 6px;
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

        # Determine dominant pitch (must be 10% higher than second highest)
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
        lambda x: int(x

