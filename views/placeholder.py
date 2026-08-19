# =============================================================================
# PLACEHOLDER - stands in for pages listed in the plan but not built yet
# =============================================================================
import streamlit as st


def render(title: str, section: str) -> None:
    st.title(title)
    st.info(f"Not built yet - part of **{section}**.", icon=":material/construction:")
    st.caption("This page is a placeholder in the navigation skeleton.")
