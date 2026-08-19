import math
import textwrap

import pandas as pd
import plotly.graph_objects as go

from utils import ui


RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
GREY = "#7a7f87"
LIGHT_GREY = ui.CHARLTON_BORDER
AXIS_TEXT = "#111111"
AXIS_LINE = "#98a2b3"
AXIS_FONT_FAMILY = "Inter SemiBold, Arial, sans-serif"


def _bold_axis_title_text(value: object) -> str:
    text = "" if value is None else str(value)
    if not text:
        return text
    stripped = text.strip()
    if stripped.lower().startswith("<b>") and stripped.lower().endswith("</b>"):
        return text
    return f"<b>{text}</b>"


def _bold_axis_titles(fig: go.Figure) -> go.Figure:
    for axis_name in fig.layout:
        if not axis_name.startswith(("xaxis", "yaxis")):
            continue
        axis = getattr(fig.layout, axis_name, None)
        title = getattr(axis, "title", None)
        title_text = getattr(title, "text", None)
        if title_text:
            title.text = _bold_axis_title_text(title_text)
    return fig


def wrap_label(value: object, width: int = 18, max_lines: int = 3) -> str:
    text = "" if value is None else str(value)
    if not text or text.lower() == "nan":
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = f"{lines[-1].rstrip()}..."
    return "<br>".join(lines) if lines else text


def value_text(value: object, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    if abs(number) >= 1000:
        return f"{number:,.0f}{suffix}"
    if number.is_integer():
        return f"{number:,.0f}{suffix}"
    return f"{number:,.{digits}f}{suffix}"


def metric_digits(metric: str | None) -> int:
    if metric is None:
        return 2
    text = metric.lower()
    if "percentile" in text:
        return 0
    if "%" in metric or "rating" in text:
        return 1
    if "goals" in text and "/90" not in metric:
        return 0
    if "points" in text and "rolling" not in text:
        return 0
    if "actions" in text or "count" in text:
        return 0
    return 2


def metric_tickformat(metric: str | None) -> str:
    digits = metric_digits(metric)
    return ",.0f" if digits == 0 else f",.{digits}f"


def metric_suffix(metric: str | None) -> str:
    return "%" if metric and "%" in metric else ""


def metric_hover_format(metric: str | None) -> str:
    return metric_tickformat(metric)


def hover_value(ref: str, metric: str | None) -> str:
    return f"%{{{ref}:{metric_tickformat(metric)}}}{metric_suffix(metric)}"


def metric_text(value: object, metric: str | None = None) -> str:
    return value_text(value, metric_digits(metric), metric_suffix(metric))


def format_xaxis(fig: go.Figure, metric: str | None) -> go.Figure:
    fig.update_xaxes(tickformat=metric_tickformat(metric), ticksuffix=metric_suffix(metric))
    return fig


def format_yaxis(fig: go.Figure, metric: str | None) -> go.Figure:
    fig.update_yaxes(tickformat=metric_tickformat(metric), ticksuffix=metric_suffix(metric))
    return fig


def polish_figure(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=19, color=DARK), x=0.01, xanchor="left"))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color=DARK, size=13),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LIGHT_GREY, font_size=13, font_color=DARK),
        margin=dict(l=28, r=34, t=68 if title else 34, b=54),
        uniformtext=dict(minsize=10, mode="hide"),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    if height:
        fig.update_layout(height=height)

    fig.update_xaxes(
        automargin=True,
        gridcolor="#eef2f6",
        linecolor=AXIS_LINE,
        tickcolor=AXIS_LINE,
        tickfont=dict(size=12, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
        title_font=dict(size=14, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
        title_standoff=14,
        zerolinecolor="#e6edf5",
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="#eef2f6",
        linecolor=AXIS_LINE,
        tickcolor=AXIS_LINE,
        tickfont=dict(size=12, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
        title_font=dict(size=14, color=AXIS_TEXT, family=AXIS_FONT_FAMILY),
        title_standoff=14,
        zerolinecolor="#e6edf5",
    )
    _bold_axis_titles(fig)
    return fig


def horizontal_bar_height(rows: int, min_height: int = 430, row_height: int = 32, max_height: int = 720) -> int:
    return min(max(min_height, row_height * max(rows, 1) + 150), max_height)


def outside_bar_text(values: pd.Series, metric: str | None = None) -> list[str]:
    return [metric_text(value, metric) for value in values]


def selected_text(labels: pd.Series, selected: str | None, width: int = 16) -> list[str]:
    if selected is None:
        return ["" for _ in labels]
    return [wrap_label(label, width=width, max_lines=2) if str(label) == str(selected) else "" for label in labels]
