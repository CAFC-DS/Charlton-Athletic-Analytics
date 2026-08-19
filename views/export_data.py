# =============================================================================
# DATA HUB · EXPORT DATA — pick a dataset, pick columns, download a CSV
# =============================================================================
# Distinct from Raw Data's per-tab download: this lets staff pull a trimmed,
# filtered CSV instead of the full table.
# =============================================================================
import streamlit as st
from utils import data, ui

st.title("⬇️ Export Data")
st.caption("Pick a dataset, narrow it down to the columns and rows you need, then download a trimmed CSV.")

ui.data_refresh_control()
st.divider()

datasets = data.all_datasets()

dataset_name = st.selectbox("Dataset", list(datasets.keys()))
df = datasets[dataset_name]

columns = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist())

# Let the user filter to specific rows via any text/category column.
text_columns = [c for c in df.columns if df[c].dtype == object]
filter_col = st.selectbox("Filter by (optional)", ["None"] + text_columns)

filtered = df
if filter_col != "None":
    values = st.multiselect(f"{filter_col} values", sorted(df[filter_col].unique()))
    if values:
        filtered = df[df[filter_col].isin(values)]

export_df = filtered[columns] if columns else filtered

st.dataframe(export_df, use_container_width=True, hide_index=True)
st.caption(f"{len(export_df)} of {len(df)} rows selected.")

st.download_button(
    f"Download {dataset_name.lower()}.csv",
    export_df.to_csv(index=False),
    file_name=f"{dataset_name.lower()}.csv",
    mime="text/csv",
)
