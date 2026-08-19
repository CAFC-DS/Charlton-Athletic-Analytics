"""Inspect the real Opta feeds ingested into CAFC_DB."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data, ui


st.title("Opta Data")
st.caption(
    "Real Opta fixture, F24 event and F7 lineup data from CAFC_DB. The parsed staging views read the "
    "DVMS_RAW payloads directly; the demo scouting snapshot is not used on this page."
)

ui.data_refresh_control()
st.divider()

st.info(
    "Opta and Impect remain separate provider feeds: CAFC_DB does not yet contain populated cross-provider "
    "player/team/fixture identity maps, so the app will not join them by name and risk mismatching records."
)

if data.USE_MOCK_DATA:
    st.warning("Opta feeds are disabled in demo mode. Set CHARLTON_DATA_MODE=production.")
    st.stop()

fixtures = data.load_opta_fixtures()
if fixtures.empty:
    st.warning("No Opta fixtures are available to the current Snowflake role.")
    st.stop()

fixtures = fixtures.sort_values(["Date", "FixtureId"], na_position="last").reset_index(drop=True)
season_options = sorted(fixtures["Season"].dropna().astype(str).unique().tolist())
team_options = sorted(
    pd.concat([fixtures["Home"], fixtures["Away"]], ignore_index=True)
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

controls = st.columns([0.8, 1.2])
with controls[0]:
    season = st.selectbox("Season", season_options, index=len(season_options) - 1)
with controls[1]:
    charlton_index = next(
        (index + 1 for index, team in enumerate(team_options) if "charlton" in team.casefold()),
        0,
    )
    team = st.selectbox("Team", ["All teams", *team_options], index=charlton_index)

filtered = fixtures[fixtures["Season"].astype(str).eq(str(season))].copy()
if team != "All teams":
    filtered = filtered[
        filtered["Home"].astype(str).eq(team) | filtered["Away"].astype(str).eq(team)
    ].copy()
if filtered.empty:
    st.info("No Opta fixtures match those filters.")
    st.stop()

fixture_options = filtered.index.tolist()


def _fixture_label(index: int) -> str:
    row = filtered.loc[index]
    date = pd.to_datetime(row.get("Date"), errors="coerce")
    date_text = date.strftime("%d %b %Y") if pd.notna(date) else "Undated"
    score = ""
    if pd.notna(row.get("Home Goals")) and pd.notna(row.get("Away Goals")):
        score = f" · {row['Home Goals']:.0f}-{row['Away Goals']:.0f}"
    return f"{date_text} · {row.get('Home')} vs {row.get('Away')}{score}"


selected_index = st.selectbox(
    "Fixture",
    fixture_options,
    index=len(fixture_options) - 1,
    format_func=_fixture_label,
)
selected = filtered.loc[selected_index]
fixture_id = selected["FixtureId"]

summary = st.columns(4)
summary[0].metric("Filtered fixtures", len(filtered))
summary[1].metric("Opta match ID", selected.get("Opta Match Id", "—"))
summary[2].metric("Round", selected.get("Round", "—"))
summary[3].metric("Venue", selected.get("Venue", "—"))

fixture_tab, source_tab = st.tabs(["Fixture inventory", "Selected fixture feed"])
with fixture_tab:
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered Opta fixtures",
        filtered.to_csv(index=False),
        file_name="opta_fixtures.csv",
        mime="text/csv",
    )

with source_tab:
    st.caption(
        "Fixture detail is loaded on demand because the F24/F7 XML is parsed at query time. "
        "Provider event and qualifier dictionaries are currently empty, so IDs are shown without invented labels."
    )
    if st.button("Load real Opta fixture detail", type="primary"):
        with st.spinner("Parsing the selected F24 and F7 feeds..."):
            events = data.load_opta_events(fixture_id)
            lineups = data.load_opta_lineups(fixture_id)
            qualifiers = data.load_opta_event_qualifiers(fixture_id)

        detail_metrics = st.columns(4)
        detail_metrics[0].metric("F24 events", f"{len(events):,}")
        detail_metrics[1].metric("F7 lineup rows", f"{len(lineups):,}")
        detail_metrics[2].metric("Qualifier rows", f"{len(qualifiers):,}")
        detail_metrics[3].metric(
            "Qualifier IDs",
            f"{qualifiers['QualifierId'].nunique():,}" if not qualifiers.empty else "0",
        )

        event_tab, lineup_tab, qualifier_tab = st.tabs(["F24 events", "F7 lineups", "F24 qualifiers"])
        with event_tab:
            st.dataframe(events, width="stretch", hide_index=True)
            st.download_button(
                "Download F24 events",
                events.to_csv(index=False),
                file_name=f"opta_f24_{fixture_id}.csv",
                mime="text/csv",
            )
        with lineup_tab:
            st.dataframe(lineups, width="stretch", hide_index=True)
        with qualifier_tab:
            st.dataframe(qualifiers, width="stretch", hide_index=True)
