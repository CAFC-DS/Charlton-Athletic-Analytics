# =============================================================================
# PASSING NETWORK - real Impect passer-to-receiver map
# =============================================================================
import pandas as pd
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch
from utils import ui


def _passing_network_css() -> None:
    st.markdown(
        """
        <style>
        .pn-summary-card {
            background: var(--ss-panel);
            border: 1px solid var(--ss-border);
            border-radius: 8px;
            border-top: 3px solid var(--ss-accent);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 102px;
            padding: 14px 16px;
        }

        .pn-summary-label {
            color: var(--ss-muted);
            font-size: 0.875rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 14px;
        }

        .pn-summary-value {
            color: var(--ss-ink);
            display: block;
            font-size: clamp(1.5rem, 1.9vw, 1.9rem);
            font-weight: 400;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .pn-summary-value-text {
            font-size: clamp(0.78rem, 0.92vw, 0.98rem);
            letter-spacing: -0.01em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: object, *, text_value: bool = False) -> None:
    value_class = "pn-summary-value pn-summary-value-text" if text_value else "pn-summary-value"
    st.markdown(
        f"""
        <div class="pn-summary-card">
            <div class="pn-summary-label">{ui.esc(label)}</div>
            <div class="{value_class}">{ui.esc(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ma.page_header(
    "Passing Network",
    "Map passer-to-receiver links from completed CAFC_DB Impect pass events for a selected fixture and team.",
    "The app derives link counts and average passer/receiver coordinates from the underlying provider event rows.",
)
_passing_network_css()

season = ma.select_match_season(key="passing_network_match_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="passing_network_match")
team_name = ma.team_selector_for_match(match_row, key="passing_network_team")
network = data.load_pass_network(match_id=match_row.get("MatchId"), team=team_name)

ma.section_heading("Selected fixture summary")
metric_cols = st.columns(4)
with metric_cols[0]:
    _summary_card("Fixture", str(match_row.get("Match", "Unknown")), text_value=True)
with metric_cols[1]:
    _summary_card("Team", team_name, text_value=True)
with metric_cols[2]:
    _summary_card("Network links", len(network))
with metric_cols[3]:
    _summary_card("Total passes", ma.metric_value(network["Pass Count"].sum() if not network.empty else 0, "Actions"))

ma.section_heading("Network controls")
control_cols = st.columns(2)
max_count = int(pd.to_numeric(network["Pass Count"], errors="coerce").max()) if not network.empty else 2
min_passes = control_cols[0].slider("Minimum link passes", 1, max(max_count, 2), min(3, max(max_count, 1)))
label = f"{team_name} pass network"

ma.section_heading("Passer-to-receiver network")
if network.empty:
    st.info("No pass-network rows are available for this selected match and team. This table currently covers the Impect event seasons only.")
else:
    st.plotly_chart(pitch.passing_network(network, team_name, label, min_passes=min_passes), width="stretch")

ma.section_heading("Pass network table")
if network.empty:
    st.caption("No pass-network links are available for the current selection.")
else:
    display_cols = ["Player", "Receiver", "Pass Count", "Passer X", "Passer Y", "Receiver X", "Receiver Y"]
    st.dataframe(network[display_cols].sort_values("Pass Count", ascending=False), width="stretch", hide_index=True)
