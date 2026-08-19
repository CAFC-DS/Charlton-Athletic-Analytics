# =============================================================================
# VISUALISATION HELPERS - reusable chart functions
# =============================================================================
# Kept for compatibility with older pages/scripts. Active analysis pages use
# team_analysis, player_analysis and match_analysis, which share these styling
# conventions through utils.charting.
# =============================================================================

import plotly.graph_objects as go

from utils import charting, ui


def scatter(df, x, y, label_col, highlight=None):
    """Scatter plot with one highlighted label and uncluttered hover details."""
    plot_df = df.copy()
    plot_df["_colour"] = plot_df[label_col].apply(lambda value: "Highlight" if value == highlight else "Other")
    plot_df["_text"] = charting.selected_text(plot_df[label_col], highlight)

    fig = go.Figure(
        go.Scatter(
            x=plot_df[x],
            y=plot_df[y],
            mode="markers+text",
            text=plot_df["_text"],
            textposition="top center",
            marker=dict(
                color=[ui.CHARLTON_RED if colour == "Highlight" else "#7a7f87" for colour in plot_df["_colour"]],
                size=[16 if colour == "Highlight" else 11 for colour in plot_df["_colour"]],
                line=dict(width=1.2, color="#ffffff"),
                opacity=0.9,
            ),
            customdata=plot_df[label_col],
            hovertemplate="%{customdata}<br>" + x + ": %{x}<br>" + y + ": %{y}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, height=550, xaxis_title=x, yaxis_title=y)
    return charting.polish_figure(fig, f"{x} vs {y}")


def radar_chart(labels, values, name):
    """Radar chart for values on a 0-100 percentile scale."""
    label_list = list(labels)
    value_list = list(values)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=value_list + [value_list[0]],
            theta=[charting.wrap_label(label, width=15, max_lines=2) for label in label_list + [label_list[0]]],
            fill="toself",
            name=name,
            line=dict(color=ui.CHARLTON_RED, width=3),
            fillcolor="rgba(195, 0, 23, 0.22)",
            hovertemplate="%{theta}<br>Percentile: %{r:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickformat=".0f")),
        showlegend=False,
        height=550,
    )
    return charting.polish_figure(fig)
