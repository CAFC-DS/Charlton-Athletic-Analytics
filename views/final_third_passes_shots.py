# =============================================================================
# FINAL THIRD & PENALTY BOX ENTRIES - real Impect entry map
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data
from utils import match_analysis as ma
from utils import pitch


ZONE_OPTIONS = ["Final Third", "Penalty Box"]
UNSUCCESSFUL_RESULTS = {"FAIL", "FAILED", "UNSUCCESSFUL"}


def _zone_title(zone: str) -> str:
    return "Penalty Box" if "penalty" in str(zone).lower() or "box" in str(zone).lower() else "Final Third"


def _entry_mask(events: pd.DataFrame, zone: str) -> pd.Series:
    if events.empty:
        return pd.Series(False, index=events.index)

    start_x = pd.to_numeric(events["Start X"], errors="coerce")
    start_y = pd.to_numeric(events["Start Y"], errors="coerce")
    end_x = pd.to_numeric(events["End X"], errors="coerce")
    end_y = pd.to_numeric(events["End Y"], errors="coerce")
    action_type = events["Action Type"].astype(str).str.upper() if "Action Type" in events else pd.Series("", index=events.index)

    if _zone_title(zone) == "Penalty Box":
        end_zone = end_x.ge(pitch.PENALTY_BOX_X) & end_y.between(-pitch.PENALTY_BOX_Y, pitch.PENALTY_BOX_Y)
        start_zone = start_x.ge(pitch.PENALTY_BOX_X) & start_y.between(-pitch.PENALTY_BOX_Y, pitch.PENALTY_BOX_Y)
    else:
        end_zone = end_x.ge(pitch.FINAL_THIRD_X)
        start_zone = start_x.ge(pitch.FINAL_THIRD_X)

    return end_zone & ~start_zone & action_type.ne("SHOT")


def _entry_value(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(dtype="float64")
    values = events.copy()
    for col in ["PXT Pass", "Team xT"]:
        if col not in values:
            values[col] = 0
        values[col] = pd.to_numeric(values[col], errors="coerce")
    return values[["PXT Pass", "Team xT"]].clip(lower=0).max(axis=1).fillna(0)


def _mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[~clean.str.lower().isin(["", "nan", "none", "null"])]
    if clean.empty:
        return ""
    mode = clean.mode()
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0])


def _result_status(values: pd.Series) -> pd.Series:
    result = values.astype(str).str.strip().str.upper()
    return pd.Series(
        pd.Categorical(
            result.map(lambda value: "Successful" if value == "SUCCESS" else "Unsuccessful" if value in UNSUCCESSFUL_RESULTS else "Other"),
            categories=["Successful", "Unsuccessful", "Other"],
            ordered=True,
        ),
        index=values.index,
    )


def _player_entry_summary(entries: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Player",
        "Entries",
        "Successful",
        "Unsuccessful",
        "Other",
        "Success %",
        "Entry Value",
        "Avg Entry Value",
        "PXT Pass",
        "Team xT",
        "Primary Action",
        "Main Receiver",
    ]
    if entries.empty or "Player" not in entries:
        return pd.DataFrame(columns=columns)

    values = entries.copy()
    values["Player"] = values["Player"].fillna("Unknown")
    values["_Outcome"] = _result_status(values["Result"]) if "Result" in values else "Other"
    values["_Successful"] = values["_Outcome"].astype(str).eq("Successful")
    values["_Unsuccessful"] = values["_Outcome"].astype(str).eq("Unsuccessful")
    values["_Other"] = values["_Outcome"].astype(str).eq("Other")
    values["_Entry Value"] = _entry_value(values) if "_Entry Value" not in values else pd.to_numeric(values["_Entry Value"], errors="coerce").fillna(0)
    for col in ["PXT Pass", "Team xT"]:
        values[col] = pd.to_numeric(values[col], errors="coerce") if col in values else 0
    values["_Action"] = values["Action"].fillna(values["Action Type"]) if "Action" in values else values.get("Action Type", "")
    values["_Receiver"] = values["Receiver"] if "Receiver" in values else ""

    summary = values.groupby("Player", as_index=False).agg(
        Entries=("Player", "size"),
        Successful=("_Successful", "sum"),
        Unsuccessful=("_Unsuccessful", "sum"),
        Other=("_Other", "sum"),
        **{
            "Entry Value": ("_Entry Value", "sum"),
            "Avg Entry Value": ("_Entry Value", "mean"),
            "PXT Pass": ("PXT Pass", "sum"),
            "Team xT": ("Team xT", "sum"),
            "Primary Action": ("_Action", _mode_text),
            "Main Receiver": ("_Receiver", _mode_text),
        },
    )
    summary["Successful"] = pd.to_numeric(summary["Successful"], errors="coerce").fillna(0).astype(int)
    summary["Unsuccessful"] = pd.to_numeric(summary["Unsuccessful"], errors="coerce").fillna(0).astype(int)
    summary["Other"] = pd.to_numeric(summary["Other"], errors="coerce").fillna(0).astype(int)
    summary["Success %"] = (summary["Successful"] / summary["Entries"].replace(0, pd.NA) * 100).fillna(0)
    for col in ["Success %", "Entry Value", "Avg Entry Value", "PXT Pass", "Team xT"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").round(2)
    return summary.sort_values(
        ["Entry Value", "Successful", "Entries", "Success %"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)[columns]


def _player_effectiveness_chart(summary: pd.DataFrame, zone_label: str, min_entries: int = 1) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        fig.add_annotation(
            text="No player entry data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    plot_df = summary[pd.to_numeric(summary["Entries"], errors="coerce").fillna(0) >= max(int(min_entries), 1)].copy()
    plot_df = plot_df.head(12).sort_values(["Entry Value", "Successful", "Entries"], ascending=True)
    if plot_df.empty:
        fig.add_annotation(
            text="No players meet the entry threshold",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    customdata = plot_df[["Entries", "Success %", "Entry Value", "Avg Entry Value", "Primary Action", "Main Receiver", "Unsuccessful", "Other"]].to_numpy()
    fig.add_trace(
        go.Bar(
            y=plot_df["Player"],
            x=plot_df["Successful"],
            orientation="h",
            name="Successful",
            marker=dict(color=pitch.GREEN),
            customdata=customdata,
            hovertemplate=(
                "%{y}"
                "<br>Successful entries: %{x:.0f}"
                "<br>Total entries: %{customdata[0]:.0f}"
                "<br>Success rate: %{customdata[1]:.1f}%"
                "<br>Entry value: %{customdata[2]:.2f}"
                "<br>Avg entry value: %{customdata[3]:.2f}"
                "<br>Primary action: %{customdata[4]}"
                "<br>Main receiver: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            y=plot_df["Player"],
            x=plot_df["Unsuccessful"],
            orientation="h",
            name="Unsuccessful",
            marker=dict(color=pitch.RED),
            customdata=customdata,
            hovertemplate=(
                "%{y}"
                "<br>Unsuccessful entries: %{x:.0f}"
                "<br>Total entries: %{customdata[0]:.0f}"
                "<br>Success rate: %{customdata[1]:.1f}%"
                "<br>Entry value: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            y=plot_df["Player"],
            x=plot_df["Other"],
            orientation="h",
            name="Other",
            marker=dict(color="#98a2b3"),
            customdata=customdata,
            hovertemplate=(
                "%{y}"
                "<br>Other entries: %{x:.0f}"
                "<br>Total entries: %{customdata[0]:.0f}"
                "<br>Unsuccessful entries: %{customdata[6]:.0f}"
                "<br>Success rate: %{customdata[1]:.1f}%"
                "<br>Entry value: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{zone_label} Entry Effectiveness by Player",
            font=dict(size=20, color=pitch.DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        barmode="stack",
        height=max(390, 42 * len(plot_df) + 150),
        margin=dict(l=28, r=28, t=104, b=58),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=pitch.DARK, size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            title=dict(text=""),
            bgcolor="rgba(255,255,255,0)",
        ),
        xaxis_title="Entries",
        yaxis_title="",
    )
    fig.update_xaxes(rangemode="tozero", gridcolor="#e8edf3")
    fig.update_yaxes(automargin=True)
    return fig


ma.page_header(
    "Final Third & Penalty Box Entries",
    "Switch between final-third entries and penalty-box entries for the selected match and team.",
    "CAFC_DB Impect provider events supply adjusted start/end coordinates, outcomes, receivers, pass PXT and team xT through the app's event adapter.",
)

season = ma.select_match_season(key="final_third_match_season")
matches = ma.load_matches(season)
if matches.empty:
    st.warning("No match data is available for this season.")
    st.stop()

match_row = ma.match_selector(matches, key="final_third_match")
team_name = ma.team_selector_for_match(match_row, key="final_third_team")
events = data.load_match_events(
    season=season,
    match_id=match_row.get("MatchId"),
    team=team_name,
    limit=20000,
)
spatial = events.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()
if spatial.empty:
    st.info("No mapped action start/end locations are available for this selected match and team.")
    st.stop()

ma.section_heading("Entry controls")
control_cols = st.columns(4)
zone = control_cols[0].selectbox("Entry zone", ZONE_OPTIONS, key="final_third_entry_zone")
zone_label = _zone_title(zone)
entries = spatial[_entry_mask(spatial, zone_label)].copy()

entry_results = sorted(entries["Result"].dropna().astype(str).unique().tolist()) if not entries.empty else []
entry_types = sorted(entries["Action Type"].dropna().astype(str).unique().tolist()) if not entries.empty else []
selected_entry_results = control_cols[1].multiselect("Entry outcomes", entry_results, default=entry_results)
selected_entry_types = control_cols[2].multiselect("Action types", entry_types, default=entry_types)
min_value = control_cols[3].number_input("Minimum entry value", min_value=0.0, value=0.0, step=0.01)
control_cols[3].caption("Entry value uses the larger positive value from PXT Pass and team xT where available.")

summary_source = entries.copy()
if selected_entry_types:
    summary_source = summary_source[summary_source["Action Type"].astype(str).isin(selected_entry_types)]
summary_source["_Entry Value"] = _entry_value(summary_source)
summary_source = summary_source[summary_source["_Entry Value"] >= min_value].copy()

filtered = summary_source.copy()
if selected_entry_results:
    filtered = filtered[filtered["Result"].astype(str).isin(selected_entry_results)]
player_summary = _player_entry_summary(summary_source)

ma.section_heading("Selected fixture summary")
score = ma.team_match_summary(match_row, team_name)
successful = int(filtered["Result"].astype(str).str.upper().eq("SUCCESS").sum()) if not filtered.empty else 0
metric_cols = st.columns(5)
metric_cols[0].metric("Result", score["Result"])
metric_cols[1].metric("Entry zone", zone_label)
metric_cols[2].metric("Entries", len(filtered))
metric_cols[3].metric("Successful", successful)
metric_cols[4].metric("Entry value", ma.metric_value(filtered["_Entry Value"].sum() if not filtered.empty else 0, "PXT Pass"))

ma.section_heading(f"{team_name}: {zone_label} Entries")
if filtered.empty:
    st.info(f"No {zone_label.lower()} entries match the current filters.")
else:
    st.plotly_chart(
        pitch.entry_zone_map(filtered, team_name, f"{team_name}: {zone_label} Entries", zone=zone_label),
        width="stretch",
    )
st.caption(
    "An entry is counted when an action starts outside the selected zone and ends inside it. "
    "Shots are excluded here because they are already covered by the Shot Map and Shooting Dashboard."
)

ma.section_heading("Player entry effectiveness")
if player_summary.empty:
    st.caption("No player entry summary is available for the current zone and action filters.")
else:
    detail_cols = st.columns([1, 3])
    min_player_entries = detail_cols[0].slider(
        "Minimum player entries",
        1,
        max(1, int(player_summary["Entries"].max())),
        1,
        key="final_third_min_player_entries",
    )
    detail_cols[1].caption(
        "This section uses the selected zone, action-type and minimum-value filters before the outcome filter, "
        "so success rate still compares successful entries against all attempted entries."
    )
    eligible_summary = player_summary[
        pd.to_numeric(player_summary["Entries"], errors="coerce").fillna(0) >= min_player_entries
    ].copy()
    st.plotly_chart(_player_effectiveness_chart(player_summary, zone_label, min_entries=min_player_entries), width="stretch")

    table_summary = eligible_summary.copy()
    for col in ["Success %", "Entry Value", "Avg Entry Value", "PXT Pass", "Team xT"]:
        if col in table_summary:
            table_summary[col] = pd.to_numeric(table_summary[col], errors="coerce").round(2)
    with st.expander("Show Player Entry Summary Table"):
        st.dataframe(table_summary, width="stretch", hide_index=True)

ma.section_heading(f"{zone_label} entry table")
display_cols = ma.available_columns(
    filtered,
    [
        "Minute",
        "Action Type",
        "Player",
        "Receiver",
        "Action",
        "Result",
        "PXT Pass",
        "Team xT",
        "_Entry Value",
        "Pass Distance",
        "Start Lane",
        "End Lane",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    ],
)
if filtered.empty:
    st.caption("No entry rows are available for the current selection.")
else:
    table = filtered[display_cols].sort_values(["Minute", "Action Type"]).copy()
    if "_Entry Value" in table:
        table = table.rename(columns={"_Entry Value": "Entry Value"})
    st.dataframe(table, width="stretch", hide_index=True)
