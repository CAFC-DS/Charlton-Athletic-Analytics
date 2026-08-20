# =============================================================================
# TEAM RADAR - Interactive Tactical Comparison
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from utils import team_analysis as ta
from utils import ui
from utils import charting

ta.page_header(
    "Team Radar",
    "Interactive tactical profile radar for comparing team performance across the league.",
    ta.TEAM_STYLE_SOURCE,
)

# --- CATEGORIES ---
CATEGORY_ORDER = ["Attacking", "Passing", "Progression", "Defensive"]

# --- PRESETS ---
RADAR_PRESETS = {
    "Standard Tactical": [
        "xG /90", "Pass %", "Passes to Final 3rd /90", 
        "Bypassed Opponents /90", "Ball Wins /90"
    ],
    "Attacking Focus": [
        "Goals /90", "xG /90", "Shots /90", 
        "Dribble Progression /90", "Assists /90"
    ],
    "Progression & Control": [
        "Pass %", "Passes to Final 3rd /90", 
        "Bypassed Opponents /90", "Dribble Progression /90"
    ],
    "Defensive Profile": [
        "Ball Wins /90", "Ball Win Value /90", "Bypassed Opponents /90"
    ],
}

# --- DATA LOAD ---
season = ta.select_season("players", key="team_radar_season")
teams = ta.load_team_style_data(season)
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

# --- CONTROLS ---
with st.expander("Radar Controls & Tactical Filters", expanded=True):
    # Team Selector (Multi-select)
    team_names = sorted(teams["Team"].dropna().unique().tolist())

    if "tr_teams" in st.session_state:
        stale = [t for t in st.session_state["tr_teams"] if t not in team_names]
        if stale:
            del st.session_state["tr_teams"]

    # Default selection: Charlton if available
    charlton_matches = [t for t in team_names if "charlton" in t.lower()]
    default_teams = [charlton_matches[0]] if charlton_matches else [team_names[0]] if team_names else []
    
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_teams = st.multiselect(
            "Select Teams to Compare",
            team_names,
            default=default_teams,
            max_selections=3,
            help="Compare up to 3 teams side-by-side.",
            key="tr_teams"
        )
    with c2:
        compare_league = st.toggle("Show League Median (50th)", value=False, key="tr_league_avg")
    
    st.divider()
    
    # Presets and Metrics in columns
    p1, p2 = st.columns([1, 2])
    
    with p1:
        st.subheader("Radar Presets")
        preset_choice = st.selectbox(
            "Select a template",
            ["Custom Builder"] + list(RADAR_PRESETS.keys()),
            index=0,
            key="tr_preset_select"
        )
    
    with p2:
        st.subheader("Tactical Metrics")
        st.caption("Select at least 3 metrics to build the radar.")
        
        # Determine initial defaults based on preset
        default_selected = []
        if preset_choice != "Custom Builder":
            default_selected = RADAR_PRESETS[preset_choice]
        
        selected_metrics = []
        
        # Group metrics by category from ta.TEAM_METRIC_META
        metrics_by_cat = {cat: [] for cat in CATEGORY_ORDER}
        for m, meta in ta.TEAM_METRIC_META.items():
            if m in teams.columns:
                metrics_by_cat[meta[0]].append(m)
                
        # Use columns for categories to save space
        cat_cols = st.columns(2)
        for i, cat in enumerate(CATEGORY_ORDER):
            cat_metrics = metrics_by_cat[cat]
            if not cat_metrics:
                continue
                
            cat_defaults = [m for m in default_selected if m in cat_metrics]
            
            with cat_cols[i % 2]:
                chosen = st.multiselect(
                    cat,
                    cat_metrics,
                    default=cat_defaults if preset_choice != "Custom Builder" else cat_metrics,
                    format_func=lambda x: ta.TEAM_METRIC_META[x][2],
                    key=f"tr_metrics_{cat}_{preset_choice}"
                )
                selected_metrics.extend(chosen)

# --- PAGE TABS ---
tab_comp, tab_builder = st.tabs(["📊 Performance Comparison", "🛠️ Custom Radar Builder"])

with tab_comp:
    if not selected_teams:
        st.info("Please select at least one team above to begin.")
    elif len(selected_metrics) < 3:
        st.warning("Please select at least 3 metrics above to generate a radar chart.")
    else:
        # --- DATA PROCESSING ---
        radar_data = []
        all_radar_labels = [ta.TEAM_METRIC_META[m][2] for m in selected_metrics]

        for team in selected_teams:
            team_mask = teams["Team"] == team
            team_vals = []
            team_percentiles = []
            
            for metric in selected_metrics:
                meta = ta.TEAM_METRIC_META[metric]
                higher_is_better = meta[1]
                
                values = pd.to_numeric(teams[metric], errors="coerce")
                pct = ta.percentile(values, higher_is_better=higher_is_better)
                
                val = values[team_mask].iloc[0] if any(team_mask) else np.nan
                p = pct[team_mask].iloc[0] if any(team_mask) else 0
                
                team_vals.append(val)
                team_percentiles.append(p)
                
            radar_data.append({
                "name": team,
                "values": team_percentiles,
                "raw_values": team_vals,
                "color": ui.get_team_color(team)
            })

        if compare_league:
            league_vals = [50] * len(selected_metrics)
            radar_data.append({
                "name": "League Median",
                "values": league_vals,
                "raw_values": [np.nan] * len(selected_metrics),
                "color": "#98a2b3"
            })

        # --- VISUALIZATION ---
        ta.section_heading("Tactical Performance Radar")

        c1, c2 = st.columns([1.3, 1], gap="large")

        with c1:
            fig = ta.team_radar(
                all_radar_labels,
                [d["values"] for d in radar_data],
                [d["name"] for d in radar_data],
                colors=[d["color"] for d in radar_data],
                height=650
            )
            st.plotly_chart(fig, width="stretch", key="team_radar_plot")

        with c2:
            st.markdown("##### Performance Breakdown")
            
            if len(selected_teams) == 1:
                team = selected_teams[0]
                data_rows = []
                for i, m in enumerate(selected_metrics):
                    data_rows.append({
                        "Metric": ta.TEAM_METRIC_META[m][2],
                        "Value": radar_data[0]["raw_values"][i],
                        "Percentile": radar_data[0]["values"][i]
                    })
                df_summary = pd.DataFrame(data_rows).sort_values("Percentile", ascending=False)
                
                st.dataframe(
                    df_summary,
                    column_config={
                        "Percentile": st.column_config.ProgressColumn(
                            "League Percentile",
                            help="Relative rank (100 is best)",
                            format="%.0f",
                            min_value=0,
                            max_value=100
                        ),
                        "Value": st.column_config.NumberColumn("Value", format="%.2f")
                    },
                    hide_index=True,
                    width="stretch"
                )

                # Summary Cards
                avg_pct = df_summary["Percentile"].mean()
                st.metric("Overall Style Score", f"{avg_pct:.1f}/100")
                
                s1, s2 = st.columns(2)
                top_metric = df_summary.iloc[0]
                low_metric = df_summary.iloc[-1]
                s1.metric("Strongest Aspect", top_metric["Metric"], f"{top_metric['Percentile']:.0f}th")
                s2.metric("Lowest Aspect", low_metric["Metric"], f"{low_metric['Percentile']:.0f}th")
            else:
                # Comparison table for multiple teams
                comp_rows = []
                for i, m in enumerate(selected_metrics):
                    row = {"Metric": ta.TEAM_METRIC_META[m][2]}
                    for d in radar_data:
                        if d["name"] != "League Median":
                            row[d["name"]] = f"{d['raw_values'][i]:.2f} ({d['values'][i]:.0f}th)"
                    comp_rows.append(row)
                
                st.dataframe(
                    pd.DataFrame(comp_rows),
                    hide_index=True,
                    width="stretch"
                )
                
                st.info("Percentiles in brackets show relative standing in the league (higher is better).")

        with st.expander("Show Detailed Data Table"):
            # Wide table with all selected teams and metrics
            export_rows = []
            for team_data in radar_data:
                if team_data["name"] == "League Median": continue
                for i, m in enumerate(selected_metrics):
                    export_rows.append({
                        "Team": team_data["name"],
                        "Metric": m,
                        "Label": ta.TEAM_METRIC_META[m][2],
                        "Category": ta.TEAM_METRIC_META[m][0],
                        "Value": team_data["raw_values"][i],
                        "Percentile": team_data["values"][i]
                    })
            st.dataframe(pd.DataFrame(export_rows), width="stretch", hide_index=True)

with tab_builder:
    st.header("Custom Radar Builder")
    st.markdown("""
        Create a bespoke radar chart by selecting specific metrics from the database. 
        Use the controls above to pick your team and metrics.
    """)
    
    if not selected_teams:
        st.info("Select a team above to see your custom radar.")
    elif len(selected_metrics) < 3:
        st.warning("Please select at least 3 metrics above.")
    else:
        # Build a single-team focused view but with extra details
        if len(selected_teams) > 1:
            builder_team = st.selectbox("Focus Team for Detailed Breakdown", selected_teams, key="tr_builder_focus")
        else:
            builder_team = selected_teams[0]
            
        st.subheader(f"Custom Profile: {builder_team}")
        
        # Find the data for the selected builder team
        team_data = next((d for d in radar_data if d["name"] == builder_team), radar_data[0])
        
        b1, b2 = st.columns([1, 1])
        
        with b1:
            fig_custom = ta.team_radar(
                all_radar_labels,
                [team_data["values"]],
                [team_data["name"]],
                colors=[team_data["color"]],
                height=600
            )
            st.plotly_chart(fig_custom, width="stretch", key="custom_radar_builder_plot")
            
        with b2:
            st.markdown("### Metric Definitions & Values")
            for i, m in enumerate(selected_metrics):
                meta = ta.TEAM_METRIC_META[m]
                raw_value = team_data["raw_values"][i]
                percentile_value = team_data["values"][i]
                has_value = pd.notna(raw_value) and pd.notna(percentile_value)
                raw_text = f"{raw_value:.2f}" if pd.notna(raw_value) else "N/A"
                percentile_text = f"{percentile_value:.0f}th" if pd.notna(percentile_value) else "N/A"
                with st.expander(f"**{meta[2]}**: {raw_text} ({percentile_text})"):
                    st.write(f"**Category:** {meta[0]}")
                    st.write(f"**Raw Metric:** `{m}`")
                    if has_value:
                        st.progress(percentile_value / 100)
                    else:
                        st.caption("No value available for this team/metric combination.")
                    st.caption(f"Ranked against {len(teams)} teams in the {season} season.")

st.caption(
    "Percentiles are calculated against all teams in the selected season. "
    "100 represents the top performer for that specific metric."
)
