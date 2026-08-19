# =============================================================================
# TEAM STYLE PROFILE - derived style percentiles from team metrics
# =============================================================================
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import team_analysis as ta


ta.page_header(
    "Team Style Profile",
    "Summarise a team's directly supported style indicators as league-percentile scores.",
    ta.TEAM_STYLE_SOURCE,
    "This page uses only the team metrics currently loaded in the app. It is not a tracking-data tactical model.",
)

season = ta.select_season("players", key="team_style_season")
teams = ta.load_team_style_data(season)
if teams.empty:
    st.warning("No team data is available for this season.")
    st.stop()

st.caption("Use the team dropdown to choose which profile is drawn in the radar chart.")
team_name = ta.team_selector(teams, key="team_style_team")
scores = ta.style_scores(teams)
selected = scores[scores["Team"] == team_name].iloc[0]

labels = [col for col in scores.columns if col != "Team"]
values = [float(selected[col]) for col in labels]

ta.section_heading("Selected team profile")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.plotly_chart(ta.team_radar(labels, values, team_name), width="stretch")
with c2:
    profile = pd.DataFrame({"Style Area": labels, "Percentile": values}).sort_values("Percentile", ascending=False)
    st.dataframe(profile, width="stretch", hide_index=True)
    strongest = profile.iloc[0]
    weakest = profile.iloc[-1]
    st.metric("Strongest area", strongest["Style Area"], f"{strongest['Percentile']:.0f}th percentile")
    st.metric("Lowest area", weakest["Style Area"], f"{weakest['Percentile']:.0f}th percentile")

ta.section_heading("All team style scores")

style_cols = [col for col in scores.columns if col != "Team"]
plot_scores = scores.sort_values("Metric Balance", ascending=False)

heatmap_text = (
    plot_scores[style_cols]
    .round(0)
    .where(plot_scores[style_cols].notna(), "")
    .astype(str)
)

fig = go.Figure(
    data=go.Heatmap(
        z=plot_scores[style_cols],
        x=style_cols,
        y=plot_scores["Team"],
        text=heatmap_text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#dc2626"],
            [0.5, "#f59e0b"],
            [1.0, "#16a34a"],
    ],
    zmin=0,
    zmax=100,
    colorbar=dict(title="Percentile"),
    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}th percentile<extra></extra>",
    )
)



fig.update_layout(
    height=max(520, len(plot_scores) * 24),
    margin=dict(l=130, r=40, t=30, b=90),
    xaxis_title="Style area",
    yaxis_title="Team",
    yaxis=dict(autorange="reversed"),
    template="plotly_white",
)

st.plotly_chart(fig, width="stretch")

ta.section_heading("Terminology key")
st.caption("All style scores are league percentiles for the selected season: 100 = strongest in the league, 0 = weakest.")

style_terms = [
    {
        "label": "Scoring",
        "made_from": "Goals /90",
        "meaning": "How strongly the team ranks for goal output.",
    },
    {
        "label": "Creation",
        "made_from": "Assists /90",
        "meaning": "How strongly the team ranks for assisted chance-ending actions.",
    },
    {
        "label": "Progression",
        "made_from": "Bypassed Opponents /90 + Passes to Final 3rd /90",
        "meaning": "How well the team moves play forward and removes opponents from the game.",
    },
    {
        "label": "Ball Security",
        "made_from": "Pass %",
        "meaning": "How securely the team keeps the ball when passing.",
    },
    {
        "label": "Final Third",
        "made_from": "Passes to Final 3rd /90",
        "meaning": "How often the team moves the ball into advanced attacking areas.",
    },
    {
        "label": "Territory",
        "made_from": "Passes to Final 3rd /90 + Bypassed Opponents /90",
        "meaning": "A current proxy for territory gained through forward access and progression.",
    },
    {
        "label": "Control Proxy",
        "made_from": "Ball Security + Progression",
        "meaning": "A combined proxy for teams that keep the ball and progress it effectively.",
    },
    {
        "label": "Metric Balance",
        "made_from": "Pass % + Bypassed Opponents /90 + Passes to Final 3rd /90",
        "meaning": "A broad balance score used to sort the league heatmap.",
    },
]

term_cards = "".join(
    f"""
    <div class="tsp-term-card">
        <div class="tsp-term-title">{term["label"]}</div>
        <div class="tsp-term-made">Made from: {term["made_from"]}</div>
        <div class="tsp-term-meaning">{term["meaning"]}</div>
    </div>
    """
    for term in style_terms
)

st.markdown(
    f"""
    <style>
    .tsp-term-grid {{
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        margin-top: 12px;
    }}

    .tsp-term-card {{
        background: #ffffff;
        border: 1px solid #e6edf5;
        border-left: 4px solid #c30017;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        padding: 14px 16px;
    }}

    .tsp-term-title {{
        color: #172033;
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 6px;
    }}

    .tsp-term-made {{
        color: #c30017;
        font-size: 0.82rem;
        font-weight: 750;
        margin-bottom: 8px;
    }}

    .tsp-term-meaning {{
        color: #667085;
        font-size: 0.88rem;
        line-height: 1.42;
    }}
    </style>

    <div class="tsp-term-grid">
        {term_cards}
    </div>
    """,
    unsafe_allow_html=True,
)
