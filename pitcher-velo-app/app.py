import streamlit as st
import pandas as pd
from data import get_pitcher_data
from typing import Tuple

# =============================
# Page setup
# =============================
st.set_page_config(page_title="Pitcher Matchup Velocity Debug", layout="wide")
st.title("⚾ Pitcher Matchup — Velocity Bias by Count (Debuggable)")
st.caption("Public Statcast data • 2025 season • Use Debug Mode to inspect anchors & samples")

# =============================
# Sidebar inputs
# =============================
with st.sidebar:
    st.header("Pitchers / Controls")
    away_pitcher = st.text_input("Away Pitcher (First Last)", "Yoshinobu Yamamoto")
    home_pitcher = st.text_input("Home Pitcher (First Last)", "Gerrit Cole")

    st.divider()

    min_pitches = st.number_input(
        "Minimum pitches per count",
        min_value=1,
        max_value=200,
        value=1,
        step=1,
    )

    debug_mode = st.checkbox("Debug mode — show anchors & samples", value=True)

    st.divider()
    run = st.button("Run Matchup")

# =============================
# Helpers
# =============================
def parse_name(name: str) -> Tuple[str, str]:
    if " " not in name.strip():
        return None, None
    return name.strip().split(" ", 1)

def compute_anchor_and_counts(df_side: pd.DataFrame):
    """
    Returns:
     - anchor_pitch (pitch_name with highest season avg on this side)
     - anchor_mph (float)
     - pitch_type_avgs (Series)
    """
    pitch_type_avg = df_side.groupby("pitch_name")["release_speed"].mean()
    if pitch_type_avg.empty:
        return None, None, pitch_type_avg
    anchor_pitch = pitch_type_avg.idxmax()
    anchor_mph = float(pitch_type_avg.loc[anchor_pitch])
    return anchor_pitch, anchor_mph, pitch_type_avg

def build_pitcher_tables_debug(first: str, last: str, min_pitches: int):
    """
    Returns:
      vs_lhb_df, vs_rhb_df, debug_info (dict or error string)
    debug_info contains:
      - anchor_pitch per side
      - anchor_mph per side
      - pitch_type_avgs per side
      - per-count rows with: count, pitches, over_share (decimal), over_pct, sample_speeds(list)
    """
    try:
        df = get_pitcher_data(first, last, 2025).copy()
    except Exception as e:
        return None, None, f"get_pitcher_data error: {e}"

    if df is None or df.empty:
        return None, None, "No Statcast data found."

    # Ensure expected columns exist
    required_cols = {"release_speed", "pitch_name", "stand", "count"}
    if not required_cols.issubset(set(df.columns)):
        return None, None, f"Missing columns in data: required {required_cols}, got {set(df.columns)}"

    # filter valid stands
    df = df[df["stand"].isin(["R", "L"])]

    debug = {"pitcher": f"{first} {last}", "sides": {}}
    output_tables = {}

    for stand_value, stand_label in [("L", "vs LHB"), ("R", "vs RHB")]:
        df_side = df[df["stand"] == stand_value]

        if df_side.empty:
            output_tables[stand_label] = pd.DataFrame()
            debug["sides"][stand_label] = {
                "anchor_pitch": None,
                "anchor_mph": None,
                "pitch_type_avgs": pd.Series(dtype=float),
                "counts": []
            }
            continue

        # compute anchor using season-level avg by pitch_name (on this side only)
        anchor_pitch, anchor_mph, pitch_type_avgs = compute_anchor_and_counts(df_side)

        # gather per-count rows
        rows = []
        for count_val, group in df_side.groupby("count"):
            sample_speeds = group["release_speed"].tolist()
            pitches = len(group)
            over_share = float((group["release_speed"] >= anchor_mph).mean()) if pitches > 0 else 0.0
            under_share = 1 - over_share
            over_pct = round(over_share * 100, 1)
            under_pct = round(under_share * 100, 1)

            # label chosen so majority side is shown (like your sheet)
            if over_share >= 0.5:
                bias_label = f"{over_pct}% Over {anchor_mph:.1f}"
            else:
                bias_label = f"{under_pct}% Under {anchor_mph:.1f}"

            rows.append({
                "count": count_val,
                "pitches": pitches,
                "over_share": over_share,
                "over_pct": over_pct,
                "under_pct": under_pct,
                "bias_label": bias_label,
                "sample_speeds": sample_speeds[:50],  # cap to 50 to avoid massive tables
            })

        # build result table (final view)
        result = pd.DataFrame(rows)
        if result.empty:
            output_tables[stand_label] = pd.DataFrame()
        else:
            # format table
            result_display = result[["count", "pitches", "over_pct", "bias_label"]].rename(columns={
                "count": "Count",
                "pitches": "Pitches",
                "over_pct": "% >= anchor",
                "bias_label": "% Over / Under Anchor"
            })

            # sort logically
            def count_sort_key(c):
                try:
                    balls, strikes = c.split("-")
                    return int(balls) * 10 + int(strikes)
                except Exception:
                    return 999
            result_display["sort"] = result_display["Count"].apply(count_sort_key)
            result_display = result_display.sort_values("sort").drop(columns=["sort"])
            output_tables[stand_label] = result_display

        debug["sides"][stand_label] = {
            "anchor_pitch": anchor_pitch,
            "anchor_mph": anchor_mph,
            "pitch_type_avgs": pitch_type_avgs.sort_values(ascending=False),
            "counts": rows
        }

    return output_tables.get("vs LHB"), output_tables.get("vs RHB"), debug

# =============================
# Run matchup
# =============================
if not run:
    st.info("Enter Away and Home pitchers, then click **Run Matchup**.")
    st.stop()

away_first, away_last = parse_name(away_pitcher)
home_first, home_last = parse_name(home_pitcher)

if not away_first or not home_first:
    st.error("Please enter both pitcher names as: First Last")
    st.stop()

with st.spinner("Pulling Statcast data for both pitchers..."):
    away_lhb, away_rhb, away_debug = build_pitcher_tables_debug(away_first, away_last, min_pitches)
    home_lhb, home_rhb, home_debug = build_pitcher_tables_debug(home_first, home_last, min_pitches)

# =============================
# Display — Away Pitcher
# =============================
st.subheader("Away Pitcher")
st.markdown(f"**{away_first} {away_last}**")

if isinstance(away_debug, str):
    st.error(f"Away pitcher error: {away_debug}")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟥 vs Left-Handed Batters")
        if away_lhb is None or away_lhb.empty:
            st.info("No data vs LHB.")
        else:
            st.dataframe(away_lhb, use_container_width=True)

        if debug_mode and away_debug and away_debug.get("sides"):
            st.markdown("**DEBUG: Anchor & per-count samples (vs LHB)**")
            side = away_debug["sides"].get("vs LHB")
            st.write(f"Anchor pitch: **{side['anchor_pitch']}**")
            st.write(f"Anchor MPH: **{side['anchor_mph']}**")
            st.write("Pitch-type avgs (season vs LHB):")
            st.dataframe(side["pitch_type_avgs"].reset_index().rename(columns={0: "Avg MPH", "pitch_name": "Pitch"}), use_container_width=True)
            st.markdown("Per-count raw rows (sample_speeds truncated to 50 entries):")
            if side["counts"]:
                counts_df = pd.DataFrame(side["counts"])
                st.dataframe(counts_df[["count", "pitches", "over_pct", "bias_label"]].rename(columns={
                    "count": "Count", "pitches": "Pitches", "over_pct": "% >= anchor", "bias_label": "% Over/Under"
                }), use_container_width=True)
                # allow expanding a particular count to see sample speeds
                st.markdown("Expand one count to see its sample speeds:")
                for r in side["counts"]:
                    with st.expander(f"{r['count']} — {r['pitches']} pitches — {r['bias_label']}"):
                        st.write(r["sample_speeds"])
            else:
                st.info("No counts to show.")

    with col2:
        st.markdown("### 🟦 vs Right-Handed Batters")
        if away_rhb is None or away_rhb.empty:
            st.info("No data vs RHB.")
        else:
            st.dataframe(away_rhb, use_container_width=True)

        if debug_mode and away_debug and away_debug.get("sides"):
            st.markdown("**DEBUG: Anchor & per-count samples (vs RHB)**")
            side = away_debug["sides"].get("vs RHB")
            st.write(f"Anchor pitch: **{side['anchor_pitch']}**")
            st.write(f"Anchor MPH: **{side['anchor_mph']}**")
            st.write("Pitch-type avgs (season vs RHB):")
            st.dataframe(side["pitch_type_avgs"].reset_index().rename(columns={0: "Avg MPH", "pitch_name": "Pitch"}), use_container_width=True)
            st.markdown("Per-count raw rows (sample_speeds truncated to 50 entries):")
            if side["counts"]:
                counts_df = pd.DataFrame(side["counts"])
                st.dataframe(counts_df[["count", "pitches", "over_pct", "bias_label"]].rename(columns={
                    "count": "Count", "pitches": "Pitches", "over_pct": "% >= anchor", "bias_label": "% Over/Under"
                }), use_container_width=True)
                for r in side["counts"]:
                    with st.expander(f"{r['count']} — {r['pitches']} pitches — {r['bias_label']}"):
                        st.write(r["sample_speeds"])
            else:
                st.info("No counts to show.")

st.divider()

# =============================
# Display — Home Pitcher
# =============================
st.subheader("Home Pitcher")
st.markdown(f"**{home_first} {home_last}**")

if isinstance(home_debug, str):
    st.error(f"Home pitcher error: {home_debug}")
else:
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🟥 vs Left-Handed Batters")
        if home_lhb is None or home_lhb.empty:
            st.info("No data vs LHB.")
        else:
            st.dataframe(home_lhb, use_container_width=True)

        if debug_mode and home_debug and home_debug.get("sides"):
            st.markdown("**DEBUG: Anchor & per-count samples (vs LHB)**")
            side = home_debug["sides"].get("vs LHB")
            st.write(f"Anchor pitch: **{side['anchor_pitch']}**")
            st.write(f"Anchor MPH: **{side['anchor_mph']}**")
            st.write("Pitch-type avgs (season vs LHB):")
            st.dataframe(side["pitch_type_avgs"].reset_index().rename(columns={0: "Avg MPH", "pitch_name": "Pitch"}), use_container_width=True)
            st.markdown("Per-count raw rows (sample_speeds truncated to 50 entries):")
            if side["counts"]:
                counts_df = pd.DataFrame(side["counts"])
                st.dataframe(counts_df[["count", "pitches", "over_pct", "bias_label"]].rename(columns={
                    "count": "Count", "pitches": "Pitches", "over_pct": "% >= anchor", "bias_label": "% Over/Under"
                }), use_container_width=True)
                for r in side["counts"]:
                    with st.expander(f"{r['count']} — {r['pitches']} pitches — {r['bias_label']}"):
                        st.write(r["sample_speeds"])
            else:
                st.info("No counts to show.")

    with col4:
        st.markdown("### 🟦 vs Right-Handed Batters")
        if home_rhb is None or home_rhb.empty:
            st.info("No data vs RHB.")
        else:
            st.dataframe(home_rhb, use_container_width=True)

        if debug_mode and home_debug and home_debug.get("sides"):
            st.markdown("**DEBUG: Anchor & per-count samples (vs RHB)**")
            side = home_debug["sides"].get("vs RHB")
            st.write(f"Anchor pitch: **{side['anchor_pitch']}**")
            st.write(f"Anchor MPH: **{side['anchor_mph']}**")
            st.write("Pitch-type avgs (season vs RHB):")
            st.dataframe(side["pitch_type_avgs"].reset_index().rename(columns={0: "Avg MPH", "pitch_name": "Pitch"}), use_container_width=True)
            st.markdown("Per-count raw rows (sample_speeds truncated to 50 entries):")
            if side["counts"]:
                counts_df = pd.DataFrame(side["counts"])
                st.dataframe(counts_df[["count", "pitches", "over_pct", "bias_label"]].rename(columns={
                    "count": "Count", "pitches": "Pitches", "over_pct": "% >= anchor", "bias_label": "% Over/Under"
                }), use_container_width=True)
                for r in side["counts"]:
                    with st.expander(f"{r['count']} — {r['pitches']} pitches — {r['bias_label']}"):
                        st.write(r["sample_speeds"])
            else:
                st.info("No counts to show.")

