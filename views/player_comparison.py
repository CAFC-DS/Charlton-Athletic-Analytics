# =============================================================================
# PLAYER COMPARISON - side-by-side positional percentile comparison
# =============================================================================
import pandas as pd
import streamlit as st

from utils import player_analysis as pa


def _sort_key(row: pd.Series) -> tuple[str, str, str]:
    last_name = row.get("Last Name")
    if pd.notna(last_name) and str(last_name).strip():
        sort_last = str(last_name).strip().casefold()
    else:
        player = str(row.get("Player", "")).strip()
        sort_last = player.split()[-1].casefold() if player else ""
    return (sort_last, str(row.get("Player", "")).casefold(), str(row.get("Team", "")).casefold())


def _player_options(players: pd.DataFrame) -> list[int]:
    sortable = players.copy()
    sortable["_sort"] = sortable.apply(_sort_key, axis=1)
    return sortable.sort_values("_sort", kind="mergesort").index.tolist()


def _player_label(players: pd.DataFrame, idx: int) -> str:
    row = players.loc[idx]
    return f"{row.get('Player', 'Unknown')} | {row.get('Team', 'Unknown')} | {row.get('_Position Display', 'Unknown')}"


def _comparison_key() -> None:
    st.markdown(
        """
        <style>
            .pc-key {
                display: flex;
                flex-wrap: wrap;
                gap: 10px 18px;
                align-items: center;
                margin: -4px 0 12px;
                color: #344054;
                font-size: 0.88rem;
            }
            .pc-key-item {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                line-height: 1.2;
            }
            .pc-swatch {
                width: 12px;
                height: 12px;
                border-radius: 3px;
                display: inline-block;
            }
            .pc-good { background: #15803d; }
            .pc-bad { background: #dc2626; }
            .pc-note {
                flex-basis: 100%;
                color: #667085;
            }
        </style>
        <div class="pc-key">
            <span class="pc-key-item"><span class="pc-swatch pc-good"></span>Better than positional peer median</span>
            <span class="pc-key-item"><span class="pc-swatch pc-bad"></span>Worse than positional peer median</span>
            <span class="pc-note">Higher percentile always means stronger performance. Only raw metrics where lower is better are inverted before scoring.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


pa.page_header(
    "Player Percentile Comparison",
    "Compare two to four players using role-aware percentiles, so each player is benchmarked against their closest positional peer group.",
)

season = pa.select_season(key="player_comparison_season")
players = pa.add_position_groups(pa.load_player_data(season))
if players.empty:
    st.warning("No players are available for the selected season.")
    st.stop()

pa.section_heading("Comparison controls")
st.caption("Choose two to four players. Percentiles are calculated within each selected player's positional peer group.")
player_options = _player_options(players)
default = player_options[:2] if len(player_options) >= 2 else player_options
selected_indices = st.multiselect(
    "Players",
    player_options,
    default=default,
    max_selections=4,
    format_func=lambda idx: _player_label(players, idx),
)

if len(selected_indices) < 2:
    st.info("Select at least two players to build a comparison.")
    st.stop()

comparison_rows = pa.comparison_percentile_rows(players, selected_indices)
if comparison_rows.empty:
    st.warning("No comparable percentile metrics are available for these players.")
    st.stop()

pa.section_heading("Percentile comparison")
_comparison_key()
st.plotly_chart(
    pa.comparison_chart(players, selected_indices),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key=f"player_percentile_comparison_{'_'.join(str(idx) for idx in selected_indices)}",
)
st.caption(
    "Bars are grouped in the same order as the selected players. The chart uses role-specific metric sets and each player is scored against their positional peer group."
)

pa.section_heading("Comparison detail")
table = comparison_rows.copy()
table["Percentile"] = pd.to_numeric(table["Percentile"], errors="coerce").round(0)
table["Rank"] = table["Rank"].apply(lambda value: "N/A" if pd.isna(value) else f"{int(value)}")
st.dataframe(
    table[
        [
            "Player",
            "Team",
            "Role Group",
            "Category",
            "Metric",
            "Direction",
            "Display Value",
            "Percentile",
            "Performance",
            "Rank",
            "Peer Group",
        ]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "Direction": st.column_config.TextColumn("Raw metric direction"),
        "Display Value": st.column_config.TextColumn("Value"),
        "Percentile": st.column_config.ProgressColumn("Pct", min_value=0, max_value=100, format="%.0f"),
    },
)
