# =============================================================================
# DATA HUB · EXPORT DATA — pick a dataset, pick columns, download a CSV
# =============================================================================
# Distinct from Raw Data's per-tab download: this lets staff pull a trimmed,
# filtered CSV instead of the full table.
# =============================================================================
import pandas as pd
import streamlit as st
from utils import data, ui

st.title("⬇️ Export Data")
st.caption("Pick a season and dataset, narrow it down to the columns and rows you need, then download a trimmed CSV.")

ui.data_refresh_control()
st.divider()

# Every dataset takes an optional season; without one each loader silently
# resolves to whichever season sorts last (the newest, often still a
# handful of matches old), with no way to export a completed prior season.
season_options = sorted(data.list_seasons().get("players", []))
if not season_options:
    st.warning("No seasons are available to export.")
    st.stop()
preferred_season = data.preferred_season(season_options)
season = st.selectbox("Season", season_options, index=season_options.index(preferred_season))

datasets = {
    "Players": data.load_players(season=season),
    "Teams": data.load_teams(season=season),
    "Matches": data.load_matches(season=season),
    "Opta Fixtures": data.load_opta_fixtures(season=season),
}

dataset_name = st.selectbox("Dataset", list(datasets.keys()))
df = datasets[dataset_name]
if df.empty:
    st.info(f"No {dataset_name.lower()} rows are available for {season}.")
    st.stop()

columns = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist())

# Let the user filter to specific rows via any text/category column. Checking
# dtype == object misses pandas' modern string dtype (Arrow-backed "string"),
# which is what most of this app's text columns (Player, Team, Season, ...)
# actually use -- pd.api.types.is_string_dtype catches both.
text_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
filter_col = st.selectbox("Filter by (optional)", ["None"] + text_columns)

filtered = df
if filter_col != "None":
    values = st.multiselect(f"{filter_col} values", sorted(df[filter_col].dropna().unique()))
    if values:
        filtered = df[df[filter_col].isin(values)]

export_df = filtered[columns] if columns else filtered

st.dataframe(export_df, width="stretch", hide_index=True)
st.caption(f"{len(export_df)} of {len(df)} rows selected · {season}.")

st.download_button(
    f"Download {dataset_name.lower()}.csv",
    export_df.to_csv(index=False),
    file_name=f"{dataset_name.lower()}_{season.replace('/', '-')}.csv",
    mime="text/csv",
)
