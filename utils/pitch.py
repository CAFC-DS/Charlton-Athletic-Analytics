import base64
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import charting, ui


PITCH_X_MIN = -52.5
PITCH_X_MAX = 52.5
PITCH_Y_MIN = -34.0
PITCH_Y_MAX = 34.0
FINAL_THIRD_X = 17.5
PENALTY_BOX_X = 36.0
PENALTY_BOX_Y = 20.16

RED = ui.CHARLTON_RED
DARK = ui.CHARLTON_BLACK
DEEP_RED = ui.CHARLTON_DEEP_RED
GREY = "#7a7f87"
LIGHT_GREY = ui.CHARLTON_BORDER
GOLD = "#c69214"
BLUE = "#344054"
GREEN = "#16803c"
PITCH_GREEN = "#f7fbf8"
LINE = "#b7c2d0"
PASS_OUTCOME_COLORS = {
    "Complete": GREEN,
    "Incomplete": RED,
    "Neutral": GREY,
}
YELLOW_CARD_ICON = "\U0001f7e8"
RED_CARD_ICON = "\U0001f7e5"
PITCH_IMAGE_CANDIDATES = [
    {
        "path": ui.ASSETS_DIR / "football_pitch_template_landscape_white.png",
        "mime": "image/png",
        "size": (3072, 2127),
        "pitch_bbox": (154, 130, 2916, 1997),
    },
    {
        "path": ui.ASSETS_DIR / "football_pitch_template_landscape.png",
        "mime": "image/png",
        "size": (3072, 2127),
        "pitch_bbox": (154, 130, 2916, 1997),
    },
    {
        "path": ui.ASSETS_DIR / "football_pitch_no_white.jpg",
        "mime": "image/jpeg",
        "size": None,
        "pitch_bbox": None,
    },
    {
        "path": ui.APP_ROOT / "Football Pitch (No White).jpg",
        "mime": "image/jpeg",
        "size": None,
        "pitch_bbox": None,
    },
]


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value)
    return "" if text.lower() == "nan" else text


def _entry_outcome_label(result: object) -> str:
    text = _safe_text(result).strip().upper()
    if text == "SUCCESS":
        return "Successful entries"
    if text in {"FAIL", "FAILED", "UNSUCCESSFUL"}:
        return "Unsuccessful entries"
    return "Other entries"


@lru_cache(maxsize=1)
def _pitch_image_asset() -> tuple[str, tuple[int, int] | None, tuple[int, int, int, int] | None]:
    for candidate in PITCH_IMAGE_CANDIDATES:
        path = candidate["path"]
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{candidate['mime']};base64,{encoded}", candidate["size"], candidate["pitch_bbox"]
    return "", None, None


def _pitch_image_placement(
    image_size: tuple[int, int] | None,
    pitch_bbox: tuple[int, int, int, int] | None,
) -> tuple[float, float, float, float]:
    """Return x, y, sizex, sizey for a pitch image.

    If the image has a margin around the pitch, pitch_bbox defines the pixel
    rectangle that should align to the Impect pitch coordinates.
    """
    if not image_size or not pitch_bbox:
        return PITCH_X_MIN, PITCH_Y_MAX, PITCH_X_MAX - PITCH_X_MIN, PITCH_Y_MAX - PITCH_Y_MIN

    width, height = image_size
    left, top, right, bottom = pitch_bbox
    pitch_width_px = max(right - left, 1)
    pitch_height_px = max(bottom - top, 1)
    pitch_width_m = PITCH_X_MAX - PITCH_X_MIN
    pitch_height_m = PITCH_Y_MAX - PITCH_Y_MIN
    sizex = pitch_width_m * width / pitch_width_px
    sizey = pitch_height_m * height / pitch_height_px
    x = PITCH_X_MIN - pitch_width_m * left / pitch_width_px
    y = PITCH_Y_MAX + pitch_height_m * top / pitch_height_px
    return x, y, sizex, sizey


def _spatial_events(events: pd.DataFrame, require_end: bool = False) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    required = ["Start X", "Start Y"]
    if require_end:
        required += ["End X", "End Y"]
    available = [col for col in required if col in events]
    if len(available) != len(required):
        return pd.DataFrame(columns=events.columns)
    out = events.dropna(subset=required).copy()
    for col in ["Start X", "Start Y", "End X", "End Y", "Shot xG", "Post-Shot xG", "PXT Pass", "PXT Shot"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def pitch_figure(title: str | None = None, height: int = 650, legend: bool = True) -> go.Figure:
    fig = go.Figure()
    _add_pitch_shapes(fig)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=19, color=DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ) if title else None,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PITCH_GREEN,
        font=dict(family="Inter, Arial, sans-serif", color=DARK, size=12),
        height=height,
        margin=dict(l=28, r=28, t=104 if title else 28, b=24),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LIGHT_GREY, font_size=13, font_color=DARK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
            title=dict(text=""),
        ),
        showlegend=bool(legend),
    )
    fig.update_xaxes(
        range=[PITCH_X_MIN - 2.0, PITCH_X_MAX + 2.0],
        visible=False,
        fixedrange=True,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[PITCH_Y_MIN - 2.0, PITCH_Y_MAX + 2.0],
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def pitch_image_figure(title: str | None = None, height: int = 650, legend: bool = True) -> go.Figure:
    """Pitch figure using the supplied black-line pitch image as the plotting surface.

    The app's event coordinates are still the same centred 105 x 68 Impect coordinates
    used by pitch_figure: x -52.5..52.5 and y -34..34.
    """
    fig = go.Figure()
    image_uri, image_size, pitch_bbox = _pitch_image_asset()
    if image_uri:
        x, y, sizex, sizey = _pitch_image_placement(image_size, pitch_bbox)
        fig.add_layout_image(
            dict(
                source=image_uri,
                xref="x",
                yref="y",
                x=x,
                y=y,
                sizex=sizex,
                sizey=sizey,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )
    else:
        _add_pitch_shapes(fig)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=19, color=DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ) if title else None,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff" if image_uri else PITCH_GREEN,
        font=dict(family="Inter, Arial, sans-serif", color=DARK, size=12),
        height=height,
        margin=dict(l=18, r=18, t=74 if title else 22, b=96 if bool(legend) else 18),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LIGHT_GREY, font_size=13, font_color=DARK),
        legend=dict(
            orientation="h",
            title_text="Key",
            yanchor="top",
            y=-0.05,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        ),
        showlegend=bool(legend),
    )
    fig.update_xaxes(
        range=[PITCH_X_MIN - 2.0, PITCH_X_MAX + 2.0],
        visible=False,
        fixedrange=True,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[PITCH_Y_MIN - 2.0, PITCH_Y_MAX + 2.0],
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def half_pitch_vertical_figure(title: str | None = None, height: int = 650, legend: bool = True) -> go.Figure:
    """Attacking half-pitch with the goal at the top of the chart."""
    fig = go.Figure()
    _add_vertical_half_pitch_shapes(fig)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=19, color=DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ) if title else None,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PITCH_GREEN,
        font=dict(family="Inter, Arial, sans-serif", color=DARK, size=12),
        height=height,
        margin=dict(l=28, r=28, t=104 if title else 28, b=24),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=LIGHT_GREY, font_size=13, font_color=DARK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
            title=dict(text=""),
        ),
        showlegend=bool(legend),
    )
    fig.update_xaxes(
        range=[PITCH_Y_MIN - 2.0, PITCH_Y_MAX + 2.0],
        visible=False,
        fixedrange=True,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[-1.5, PITCH_X_MAX + 3.2],
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def _add_pitch_shapes(fig: go.Figure) -> None:
    line = dict(color=LINE, width=1.5)
    box_line = dict(color="#aeb9c7", width=1.3)
    fig.add_shape(
        type="rect",
        x0=PITCH_X_MIN,
        y0=PITCH_Y_MIN,
        x1=PITCH_X_MAX,
        y1=PITCH_Y_MAX,
        line=line,
        fillcolor=PITCH_GREEN,
        layer="below",
    )
    fig.add_shape(type="line", x0=0, y0=PITCH_Y_MIN, x1=0, y1=PITCH_Y_MAX, line=line, layer="below")
    fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line, layer="below")
    fig.add_shape(type="circle", x0=-0.45, y0=-0.45, x1=0.45, y1=0.45, line=dict(color=LINE, width=1), layer="below")

    for side, box_start, box_end, six_start, six_end, spot_x, goal_outer in [
        ("left", PITCH_X_MIN, -36.0, PITCH_X_MIN, -47.0, -41.5, PITCH_X_MIN - 1.4),
        ("right", 36.0, PITCH_X_MAX, 47.0, PITCH_X_MAX, 41.5, PITCH_X_MAX + 1.4),
    ]:
        _ = side
        fig.add_shape(type="rect", x0=box_start, y0=-20.16, x1=box_end, y1=20.16, line=box_line, layer="below")
        fig.add_shape(type="rect", x0=six_start, y0=-9.16, x1=six_end, y1=9.16, line=box_line, layer="below")
        fig.add_shape(
            type="circle",
            x0=spot_x - 0.38,
            y0=-0.38,
            x1=spot_x + 0.38,
            y1=0.38,
            line=dict(color=LINE, width=1),
            fillcolor=LINE,
            layer="below",
        )
        goal_x0, goal_x1 = (goal_outer, PITCH_X_MIN) if spot_x < 0 else (PITCH_X_MAX, goal_outer)
        fig.add_shape(
            type="rect",
            x0=goal_x0,
            y0=-3.66,
            x1=goal_x1,
            y1=3.66,
            line=dict(color="#9aa6b2", width=1.2),
            fillcolor="#ffffff",
            layer="below",
        )

    theta_left = np.linspace(-0.92, 0.92, 48)
    left_x = -41.5 + np.cos(theta_left) * 9.15
    left_y = np.sin(theta_left) * 9.15
    theta_right = np.linspace(np.pi - 0.92, np.pi + 0.92, 48)
    right_x = 41.5 + np.cos(theta_right) * 9.15
    right_y = np.sin(theta_right) * 9.15
    fig.add_trace(go.Scatter(x=left_x, y=left_y, mode="lines", line=line, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=right_x, y=right_y, mode="lines", line=line, hoverinfo="skip", showlegend=False))


def _add_vertical_half_pitch_shapes(fig: go.Figure) -> None:
    line = dict(color=LINE, width=1.5)
    box_line = dict(color="#aeb9c7", width=1.3)
    fig.add_shape(
        type="rect",
        x0=PITCH_Y_MIN,
        y0=0,
        x1=PITCH_Y_MAX,
        y1=PITCH_X_MAX,
        line=line,
        fillcolor=PITCH_GREEN,
        layer="below",
    )
    fig.add_shape(type="line", x0=PITCH_Y_MIN, y0=0, x1=PITCH_Y_MAX, y1=0, line=line, layer="below")
    fig.add_shape(type="rect", x0=-20.16, y0=36.0, x1=20.16, y1=PITCH_X_MAX, line=box_line, layer="below")
    fig.add_shape(type="rect", x0=-9.16, y0=47.0, x1=9.16, y1=PITCH_X_MAX, line=box_line, layer="below")
    fig.add_shape(
        type="rect",
        x0=-3.66,
        y0=PITCH_X_MAX,
        x1=3.66,
        y1=PITCH_X_MAX + 1.55,
        line=dict(color="#9aa6b2", width=1.2),
        fillcolor="#ffffff",
        layer="below",
    )
    fig.add_shape(
        type="circle",
        x0=-0.38,
        y0=41.5 - 0.38,
        x1=0.38,
        y1=41.5 + 0.38,
        line=dict(color=LINE, width=1),
        fillcolor=LINE,
        layer="below",
    )
    theta = np.linspace(np.pi - 0.92, np.pi + 0.92, 48)
    arc_x = np.sin(theta) * 9.15
    arc_y = 41.5 + np.cos(theta) * 9.15
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=line, hoverinfo="skip", showlegend=False))


def _normalise_shots_to_top_goal(shots: pd.DataFrame) -> pd.DataFrame:
    """Rotate attacking-right coordinates into a goal-at-top view.

    Impect's adjusted coordinates already rotate each team and period so the
    attacking goal is at +X. When that frame is turned 90 degrees for a
    vertical half-pitch, attacking-left (+Y) must become screen-left (-X).
    ``direction`` remains as a fallback for any unadjusted rows.
    """
    out = shots.copy()
    start_x = pd.to_numeric(out["Start X"], errors="coerce")
    start_y = pd.to_numeric(out["Start Y"], errors="coerce")
    end_x = pd.to_numeric(out["End X"], errors="coerce") if "End X" in out else pd.Series(np.nan, index=out.index)
    end_y = pd.to_numeric(out["End Y"], errors="coerce") if "End Y" in out else pd.Series(np.nan, index=out.index)
    target_x = end_x.where(end_x.notna(), start_x)
    direction = pd.Series(np.where(target_x < 0, -1.0, 1.0), index=out.index)
    out["_Half X"] = (-start_y * direction).clip(PITCH_Y_MIN, PITCH_Y_MAX)
    out["_Half Y"] = (start_x * direction).clip(0, PITCH_X_MAX)
    out["_Half End X"] = (-end_y * direction).clip(PITCH_Y_MIN, PITCH_Y_MAX)
    out["_Half End Y"] = (end_x * direction).clip(0, PITCH_X_MAX)
    return out


def _shot_goal_label_positions(group: pd.DataFrame) -> list[str]:
    """Place goal labels beside their marker instead of underneath the goal."""
    positions: list[str] = []
    for _, row in group.iterrows():
        half_x = _finite(row.get("_Half X"), 0.0)
        if half_x > 1.0:
            positions.append("middle left")
        elif half_x < -1.0:
            positions.append("middle right")
        else:
            positions.append("top center")
    return positions


def _shot_type_label(body_part: object, action: object = None) -> str:
    text = f"{_safe_text(body_part)} {_safe_text(action)}".upper()
    if "HEAD" in text:
        return "Header"
    if "LEFT" in text:
        return "Left Foot"
    if "RIGHT" in text:
        return "Right Foot"
    if "FOOT" in text:
        return "Foot"
    return "Other"


SHOT_TYPE_SYMBOLS = {
    "Right Foot": "circle",
    "Left Foot": "diamond",
    "Header": "triangle-up",
    "Foot": "circle-open",
    "Other": "square",
}


def pass_map(events: pd.DataFrame, team: str | None, title: str, max_passes: int = 450) -> go.Figure:
    passes = _spatial_events(events, require_end=True)
    if team:
        passes = passes[passes["Team"].astype(str) == str(team)]
    passes = passes[passes["Action Type"].astype(str).str.upper() == "PASS"].copy()
    if len(passes) > max_passes:
        sort_col = "PXT Pass" if "PXT Pass" in passes else "Pass Distance"
        passes = passes.sort_values(sort_col, ascending=False).head(max_passes)

    fig = pitch_figure(title)
    if passes.empty:
        _empty_pitch_message(fig, "No pass locations")
        return fig

    passes["Outcome"] = np.select(
        [
            passes["Result"].astype(str).str.upper().eq("SUCCESS"),
            passes["Result"].astype(str).str.upper().eq("FAIL"),
        ],
        ["Complete", "Incomplete"],
        default="Neutral",
    )
    for outcome, group in passes.groupby("Outcome", sort=False):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for _, row in group.iterrows():
            x_values += [row["Start X"], row["End X"], None]
            y_values += [row["Start Y"], row["End Y"], None]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=outcome,
                line=dict(color=PASS_OUTCOME_COLORS.get(outcome, GREY), width=3.2),
                opacity=0.82,
                hoverinfo="skip",
            )
        )
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Receiver"].fillna("Unknown"),
                group["Pass Distance"].fillna(0),
                group["PXT Pass"].fillna(0),
                group["Minute"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["End X"],
                y=group["End Y"],
                mode="markers",
                name=f"{outcome} end",
                marker=dict(
                    size=6.5,
                    color=PASS_OUTCOME_COLORS.get(outcome, GREY),
                    opacity=0.86,
                    line=dict(color="#ffffff", width=0.8),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} to %{customdata[1]}"
                    "<br>Minute: %{customdata[4]:.0f}"
                    "<br>Distance: %{customdata[2]:.1f}m"
                    "<br>PXT pass: %{customdata[3]:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    return fig


def passing_network(
    network: pd.DataFrame,
    team: str | None,
    title: str,
    min_passes: int = 2,
    use_pitch_image: bool = False,
) -> go.Figure:
    """Passer-to-receiver network map with player average positions and link strengths."""
    edges = network.copy()
    if team:
        edges = edges[edges["Team"].astype(str) == str(team)]

    for col in ["Passer X", "Passer Y", "Receiver X", "Receiver Y", "Pass Count"]:
        if col in edges:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")
    edges = edges.dropna(subset=["Passer X", "Passer Y", "Receiver X", "Receiver Y", "Pass Count"])

    # Calculate nodes from the full team network to ensure stable tactical positions
    nodes = _network_nodes(edges)
    fig = pitch_image_figure(title) if use_pitch_image else pitch_figure(title)
    if nodes.empty:
        _empty_pitch_message(fig, "No pass-network links")
        return fig

    # To make the network actionable, we connect player dots and combine bidirectional links.
    node_map = nodes.set_index("NodeId").to_dict("index")

    # Filter and aggregate edges for display
    plot_links = edges[edges["Pass Count"] >= min_passes].copy()
    if not plot_links.empty:
        plot_links["_P1"] = plot_links["PlayerId"].astype(str)
        plot_links["_P2"] = plot_links["ReceiverId"].astype(str)
        plot_links["_Pair"] = plot_links.apply(lambda r: tuple(sorted([r["_P1"], r["_P2"]])), axis=1)

        combined = plot_links.groupby("_Pair").agg(
            TotalPasses=("Pass Count", "sum")
        ).reset_index()

        max_total = max(_finite(combined["TotalPasses"].max(), 1.0), 1.0)
        for index, row in combined.iterrows():
            id1, id2 = row["_Pair"]
            # print(f"DEBUG: checking {id1} ({type(id1)}) and {id2} ({type(id2)}) in node_map keys {list(node_map.keys())}")
            if id1 not in node_map or id2 not in node_map:
                continue

            n1, n2 = node_map[id1], node_map[id2]
            pair_links = plot_links[plot_links["_Pair"] == row["_Pair"]]
            tooltip_parts = [f"{l['Player']} to {l['Receiver']}: {l['Pass Count']:.0f}" for _, l in pair_links.iterrows()]
            hover_text = f"<b>Total: {row['TotalPasses']:.0f} passes</b><br>" + "<br>".join(tooltip_parts)

            strength = row["TotalPasses"] / max_total
            width = 3.6 + strength * 8.4

            fig.add_trace(
                go.Scatter(
                    x=[n1["X"], n2["X"]],
                    y=[n1["Y"], n2["Y"]],
                    mode="lines",
                    showlegend=False,
                    legendgroup="links",
                    line=dict(color="rgba(255, 255, 255, 0.86)", width=width + 3.0),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[n1["X"], n2["X"]],
                    y=[n1["Y"], n2["Y"]],
                    mode="lines",
                    name="Pass links",
                    legendgroup="links",
                    showlegend=bool(index == combined.index[0]),
                    line=dict(color=f"rgba(195, 0, 23, {0.45 + 0.45 * strength:.2f})", width=width),
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            )

    max_involvement = max(_finite(nodes["Involvement"].max(), 1.0), 1.0)
    nodes["Label"] = ""
    label_limit = 24
    if len(nodes) <= label_limit:
        label_mask = pd.Series(True, index=nodes.index)
    else:
        label_mask = nodes["Involvement"].rank(method="first", ascending=False) <= label_limit
    nodes.loc[label_mask, "Label"] = nodes.loc[label_mask, "Player"].apply(lambda value: charting.wrap_label(value, 13, 2))
    nodes["Label Position"] = _network_label_positions(nodes)
    customdata = np.stack(
        [
            nodes["Player"],
            nodes["Passes Out"].fillna(0),
            nodes["Passes In"].fillna(0),
            nodes["Involvement"].fillna(0),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=nodes["X"],
            y=nodes["Y"],
            mode="markers+text",
            name="Players",
            text=nodes["Label"],
            textposition=nodes["Label Position"],
            textfont=dict(size=10, color=DARK),
            marker=dict(
                size=12 + (nodes["Involvement"] / max_involvement) * 24,
                color=DEEP_RED,
                opacity=0.92,
                line=dict(color="#ffffff", width=1.4),
            ),
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]}"
                "<br>Passes out: %{customdata[1]:.0f}"
                "<br>Passes in: %{customdata[2]:.0f}"
                "<br>Involvement: %{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    return fig


def _network_label_positions(nodes: pd.DataFrame) -> list[str]:
    positions = []
    center_positions = ["top center", "bottom center", "middle right", "middle left"]
    for order, (_, row) in enumerate(nodes.iterrows()):
        x = _finite(row.get("X"))
        y = _finite(row.get("Y"))
        if y >= PITCH_Y_MAX - 8:
            positions.append("bottom center")
        elif y <= PITCH_Y_MIN + 8:
            positions.append("top center")
        elif x >= PITCH_X_MAX - 14:
            positions.append("middle left")
        elif x <= PITCH_X_MIN + 14:
            positions.append("middle right")
        else:
            positions.append(center_positions[order % len(center_positions)])
    return positions


def _network_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    passer = edges[["PlayerId", "Player", "Passer X", "Passer Y", "Pass Count"]].rename(
        columns={"PlayerId": "NodeId", "Passer X": "X", "Passer Y": "Y"}
    )
    passer = passer.groupby(["NodeId", "Player"], as_index=False).agg(
        X=("X", "mean"),
        Y=("Y", "mean"),
        **{"Passes Out": ("Pass Count", "sum")},
    )
    receiver = edges[["ReceiverId", "Receiver", "Receiver X", "Receiver Y", "Pass Count"]].rename(
        columns={"ReceiverId": "NodeId", "Receiver": "Player", "Receiver X": "X", "Receiver Y": "Y"}
    )
    receiver = receiver.groupby(["NodeId", "Player"], as_index=False).agg(
        X=("X", "mean"),
        Y=("Y", "mean"),
        **{"Passes In": ("Pass Count", "sum")},
    )
    nodes = passer.merge(receiver, on=["NodeId", "Player"], how="outer", suffixes=("_out", "_in"))
    nodes["X"] = nodes[["X_out", "X_in"]].mean(axis=1)
    nodes["Y"] = nodes[["Y_out", "Y_in"]].mean(axis=1)
    nodes["Passes Out"] = nodes["Passes Out"].fillna(0)
    nodes["Passes In"] = nodes["Passes In"].fillna(0)
    nodes["Involvement"] = nodes["Passes Out"] + nodes["Passes In"]
    return nodes.dropna(subset=["X", "Y"]).reset_index(drop=True)


def formation_overlay_trace(
    lineup: pd.DataFrame,
    team: str | None,
    formation: str | None,
    marker_color: str = BLUE,
) -> go.Scatter | None:
    """Return a scatter trace overlaying F7 formation places on an existing pitch figure.

    Unlike ``formation_map`` which creates a standalone figure, this returns a
    trace that can be added to an existing ``average_position_map`` figure.
    """
    if lineup.empty:
        return None
    team_lineup = lineup[lineup["Team"].astype(str).eq(str(team))].copy() if team else lineup.copy()
    if team_lineup.empty:
        return None

    f_str = "".join(filter(str.isdigit, str(formation))) if formation else ""
    coords = _formation_coordinates(f_str)
    if not coords:
        return None

    # Map formation place -> player
    team_lineup["Formation Place"] = pd.to_numeric(team_lineup.get("Formation Place"), errors="coerce")
    points = []
    labels = []
    for place, (x, y) in coords.items():
        player_row = team_lineup[team_lineup["Formation Place"] == place]
        if player_row.empty:
            points.append((x, y))
            labels.append("")
        else:
            player_label = str(player_row.iloc[0].get("Player", ""))
            shirt = player_row.iloc[0].get("Shirt Number")
            if pd.notna(shirt):
                player_label = f"{int(shirt)}. {player_label}" if player_label else str(int(shirt))
            points.append((x, y))
            labels.append(player_label)

    if not points:
        return None

    xs, ys = zip(*points)
    return go.Scatter(
        x=list(xs),
        y=list(ys),
        mode="markers+text",
        name=f"Formation ({formation or '?'})",
        text=list(labels),
        textposition="top center",
        textfont=dict(size=9, color="#ffffff"),
        marker=dict(
            size=20,
            color=marker_color,
            opacity=0.75,
            symbol="circle",
            line=dict(color="#ffffff", width=1.5),
        ),
        hoverinfo="skip",
        showlegend=True,
    )


def _formation_coordinates(f_str: str) -> dict[int, tuple[float, float]] | None:
    """Return a dict of formation place -> (x, y) for the given formation string."""
    coords_map = {
        "4231": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-35, 12),
            6: (-35, -12), 7: (22, 30), 8: (-10, -8), 9: (42, 0), 10: (22, 0), 11: (22, -30),
        },
        "3421": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -12), 4: (-30, -30), 5: (-10, 8),
            6: (-10, -8), 7: (22, 30), 8: (22, 0), 9: (42, 0), 10: (22, -30), 11: (-10, 0),
        },
        "433": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-10, -8),
            6: (-35, 0), 7: (22, 30), 8: (22, 0), 9: (42, 0), 10: (22, -30), 11: (-35, 0),
        },
        "4411": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-10, -8),
            6: (-35, 12), 7: (-35, -12), 8: (22, 30), 9: (22, 0), 10: (22, -30), 11: (-10, 0),
        },
        "352": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, 0), 4: (-30, -30), 5: (-10, 8),
            6: (-10, -8), 7: (22, 30), 8: (22, 0), 9: (42, 0), 10: (42, 12), 11: (42, -12),
        },
        "3511": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, 0), 4: (-30, -30), 5: (-10, 8),
            6: (-10, -8), 7: (22, 30), 8: (22, 0), 9: (42, 0), 10: (22, -30), 11: (-10, 0),
        },
        "442": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-10, -8),
            6: (-35, 12), 7: (-35, -12), 8: (22, 30), 9: (42, 0), 10: (22, -30), 11: (-35, 12),
        },
        "532": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, 12), 4: (-30, -12), 5: (-30, -30),
            6: (-10, 8), 7: (-10, -8), 8: (22, 30), 9: (42, 0), 10: (42, 12), 11: (42, -12),
        },
        "541": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, 12), 4: (-30, -12), 5: (-30, -30),
            6: (-10, 8), 7: (-10, -8), 8: (22, 30), 9: (42, 0), 10: (22, -30), 11: (-10, 0),
        },
        "5212": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, 12), 4: (-30, -12), 5: (-30, -30),
            6: (-10, 8), 7: (-10, -8), 8: (22, 30), 9: (42, 0), 10: (22, -30), 11: (-10, 0),
        },
        "4141": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-10, -8),
            6: (-35, 0), 7: (22, 30), 8: (22, 0), 9: (42, 0), 10: (22, -30), 11: (-35, 0),
        },
        "4132": {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-10, -8),
            6: (-35, 0), 7: (22, 30), 8: (22, 0), 9: (42, 0), 10: (42, 12), 11: (42, -12),
        },
    }
    return coords_map.get(f_str)


def average_position_map(events: pd.DataFrame, team: str | None, title: str, min_actions: int = 3) -> go.Figure:
    """Event-derived average player locations for the supplied event frame."""
    spatial = _spatial_events(events)
    if team:
        spatial = spatial[spatial["Team"].astype(str) == str(team)]
    if "Player" in spatial:
        spatial = spatial.dropna(subset=["Player"])

    fig = pitch_figure(title, height=650, legend=False)
    if spatial.empty or "Player" not in spatial:
        _empty_pitch_message(fig, "No player-location data")
        return fig

    spatial = spatial.copy()
    spatial["_Start X"] = pd.to_numeric(spatial["Start X"], errors="coerce")
    spatial["_Start Y"] = pd.to_numeric(spatial["Start Y"], errors="coerce")
    spatial["_Action"] = (
        spatial["Action"].fillna(spatial["Action Type"]) if "Action Type" in spatial
        else spatial.get("Action", pd.Series("", index=spatial.index))
    )
    spatial["_Position"] = spatial["Position"] if "Position" in spatial else ""
    spatial["_Match Minutes"] = (
        pd.to_numeric(spatial["Match Minutes"], errors="coerce")
        if "Match Minutes" in spatial
        else pd.Series(np.nan, index=spatial.index)
    )
    grouped = spatial.groupby("Player", as_index=False).agg(
        Actions=("Player", "size"),
        X=("_Start X", "mean"),
        Y=("_Start Y", "mean"),
        MatchMinutes=("_Match Minutes", "max"),
        Position=("_Position", lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else ""),
        PrimaryAction=("_Action", lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else ""),
    )
    grouped = grouped.dropna(subset=["X", "Y"])
    grouped = grouped[pd.to_numeric(grouped["Actions"], errors="coerce").fillna(0) >= max(int(min_actions), 1)].copy()
    if grouped.empty:
        _empty_pitch_message(fig, "No players meet the action threshold")
        return fig

    grouped["X"] = pd.to_numeric(grouped["X"], errors="coerce").clip(PITCH_X_MIN + 2, PITCH_X_MAX - 2)
    grouped["Y"] = pd.to_numeric(grouped["Y"], errors="coerce").clip(PITCH_Y_MIN + 2, PITCH_Y_MAX - 2)
    if grouped["MatchMinutes"].notna().any():
        grouped = grouped.sort_values(["MatchMinutes", "Actions"], ascending=[False, False]).reset_index(drop=True)
    else:
        grouped = grouped.sort_values("Actions", ascending=False).reset_index(drop=True)
    max_actions = max(_finite(grouped["Actions"].max(), 1.0), 1.0)
    label_limit = 18
    grouped["Label"] = ""
    label_mask = (
        pd.Series(True, index=grouped.index)
        if len(grouped) <= label_limit
        else grouped["Actions"].rank(method="first", ascending=False) <= label_limit
    )
    grouped.loc[label_mask, "Label"] = grouped.loc[label_mask, "Player"].apply(lambda value: charting.wrap_label(value, 12, 2))
    grouped["Label Position"] = _network_label_positions(grouped[["X", "Y"]])
    customdata = np.stack(
        [
            grouped["Player"].fillna("Unknown"),
            grouped["Position"].fillna(""),
            grouped["Actions"].fillna(0),
            grouped["PrimaryAction"].fillna(""),
            grouped["X"].fillna(0),
            grouped["Y"].fillna(0),
            grouped["MatchMinutes"].fillna(0),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["X"],
            y=grouped["Y"],
            mode="markers+text",
            name="Average positions",
            text=grouped["Label"],
            textposition=grouped["Label Position"],
            textfont=dict(size=10, color=DARK),
            marker=dict(
                size=14 + (grouped["Actions"] / max_actions) * 28,
                color=RED,
                opacity=0.88,
                line=dict(color="#ffffff", width=1.8),
            ),
            customdata=customdata,
            hovertemplate=(
                "%{customdata[0]}"
                "<br>Position: %{customdata[1]}"
                "<br>Match minutes: %{customdata[6]:.1f}"
                "<br>Actions: %{customdata[2]:.0f}"
                "<br>Primary action: %{customdata[3]}"
                "<br>Average X: %{customdata[4]:.1f}"
                "<br>Average Y: %{customdata[5]:.1f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    return fig


def formation_map(
    lineup: pd.DataFrame, 
    team: str | None, 
    title: str, 
    formation: str | None = None, 
    mirror: bool = False,
    marker_color: str | None = None
) -> go.Figure:
    """Tactical pitch mapping of a starting XI using Opta F7 formation places."""
    display_title = title
    if formation:
        display_title = f"{title} ({formation})"
        
    fig = pitch_image_figure(display_title, height=600, legend=False)
    if lineup.empty:
        _empty_pitch_message(fig, "No lineup data available")
        return fig

    # Default color
    dot_color = marker_color if marker_color else RED
    border_color = "#ffffff"
    # If the marker is white or very light, use a dark border for visibility
    if str(dot_color).lower() in {"#ffffff", "white", "#fffffe", "#f4f5f7"}:
        border_color = DARK

    # Standard Opta F7 Formation Place to Coordinate mapping (Back-to-front, Right-to-left)
    # Values are normalized for a team attacking Left-to-Right (X: -52.5 to 52.5)
    
    f_str = "".join(filter(str.isdigit, str(formation))) if formation else ""
    
    if f_str == "4231":
        coords = {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-10, 8), 5: (-35, 12), 6: (-35, -12),
            7: (22, 30), 8: (-10, -8), 9: (42, 0), 10: (22, 0), 11: (22, -30)
        }
    elif f_str == "3421":
        coords = {
            1: (-48, 0), 2: (-15, 32), 3: (-15, -32), 4: (-35, -18), 5: (-35, 0), 6: (-35, 18),
            7: (-8, 10), 8: (-8, -10), 9: (42, 0), 10: (22, 12), 11: (22, -12)
        }
    elif f_str == "433":
        coords = {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-18, 0), 5: (-35, 12), 6: (-35, -12),
            7: (25, 32), 8: (0, 15), 9: (42, 0), 10: (0, -15), 11: (25, -32)
        }
    elif f_str == "442":
        coords = {
            1: (-48, 0), 2: (-30, 30), 3: (-30, -30), 4: (-5, 10), 5: (-35, 12), 6: (-35, -12),
            7: (10, 32), 8: (-5, -10), 9: (35, 10), 10: (35, -10), 11: (10, -32)
        }
    elif f_str.startswith("3") or f_str.startswith("5"):
        # Generic 3/5-back
        coords = {
            1: (-48, 0), 2: (-25, 30), 3: (-32, 18), 4: (-32, 0), 5: (-32, -18), 6: (-25, -30),
            7: (-5, 12), 8: (-5, -12), 9: (15, 20), 10: (35, 0), 11: (15, -20)
        }
    else:
        # Default 4-back
        coords = {
            1: (-48, 0), 2: (-30, 25), 3: (-32, 8), 4: (-32, -8), 5: (-30, -25),
            6: (-5, 25), 7: (-8, 8), 8: (-8, -8), 9: (-5, -25), 10: (25, 12), 11: (25, -12)
        }

    starters = lineup[lineup["Lineup Status"].astype(str).str.casefold() == "start"].copy()
    if starters.empty:
        _empty_pitch_message(fig, "No starting XI found")
        return fig

    starters["_Place"] = pd.to_numeric(starters["Formation Place"], errors="coerce")
    starters = starters.dropna(subset=["_Place"])

    plot_data = []
    for _, row in starters.iterrows():
        place = int(row["_Place"])
        xy = coords.get(place)
        if xy:
            x, y = xy
            if mirror:
                x = -x
                y = -y
            plot_data.append({
                "Player": row.get("Player", "Unknown"),
                "Number": row.get("Shirt Number", ""),
                "Position": row.get("Position Group") or row.get("Registered Position") or "",
                "X": x,
                "Y": y,
            })

    if not plot_data:
        _empty_pitch_message(fig, "No players with valid formation places")
        return fig

    df = pd.DataFrame(plot_data)
    df["Label"] = df["Player"].apply(lambda p: charting.wrap_label(p, 10, 2))

    fig.add_trace(
        go.Scatter(
            x=df["X"],
            y=df["Y"],
            mode="markers+text",
            text=df["Label"],
            textposition="bottom center",
            textfont=dict(size=11, color=DARK, family="Inter SemiBold, Arial"),
            marker=dict(
                size=32,
                color=dot_color,
                line=dict(color=border_color, width=2),
                opacity=0.95
            ),
            customdata=np.stack([df["Player"], df["Number"], df["Position"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Number: %{customdata[1]}<br>"
                "Position: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False
        )
    )

    # Add shirt numbers inside markers
    fig.add_trace(
        go.Scatter(
            x=df["X"],
            y=df["Y"],
            mode="text",
            text=df["Number"],
            textfont=dict(size=12, color="#ffffff", family="Inter Bold, Arial"),
            hoverinfo="skip",
            showlegend=False
        )
    )

    return fig


def line_breaking_actions_map(
    events: pd.DataFrame,
    team: str | None,
    title: str,
    min_bypassed_opponents: float = 1,
    min_bypassed_defenders: float = 0,
    max_actions: int = 250,
) -> go.Figure:
    """Map event-level Impect packing actions using bypassed-player values."""
    spatial = _spatial_events(events)
    if team:
        spatial = spatial[spatial["Team"].astype(str) == str(team)]

    fig = pitch_figure(title, height=650, legend=True)
    if spatial.empty:
        _empty_pitch_message(fig, "No line-breaking locations")
        return fig

    actions = spatial.copy()
    for col in ["Bypassed Opponents", "Bypassed Defenders", "Packing xG", "PXT Pass", "PXT Shot", "Minute"]:
        if col not in actions:
            actions[col] = np.nan
        actions[col] = pd.to_numeric(actions[col], errors="coerce")

    actions["Bypassed Opponents"] = actions["Bypassed Opponents"].fillna(0)
    actions["Bypassed Defenders"] = actions["Bypassed Defenders"].fillna(0)
    actions = actions[
        actions["Bypassed Opponents"].ge(float(min_bypassed_opponents))
        & actions["Bypassed Defenders"].ge(float(min_bypassed_defenders))
    ].copy()
    if actions.empty:
        _empty_pitch_message(fig, "No actions meet the bypass filter")
        return fig

    actions["_Threat Sort"] = actions[["Packing xG", "PXT Pass", "PXT Shot"]].clip(lower=0).max(axis=1).fillna(0)
    actions = actions.sort_values(
        ["Bypassed Defenders", "Bypassed Opponents", "_Threat Sort", "Minute"],
        ascending=[False, False, False, True],
    )
    if len(actions) > max_actions:
        actions = actions.head(max(int(max_actions), 1)).copy()

    actions["Action Type"] = actions["Action Type"].fillna("UNKNOWN").astype(str)
    max_bypassed = max(_finite(actions["Bypassed Opponents"].max(), 1.0), 1.0)
    palette = [RED, DARK, GOLD, BLUE, GREEN, GREY]
    action_order = (
        actions.groupby("Action Type")["Bypassed Opponents"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for index, action_type in enumerate(action_order):
        group = actions[actions["Action Type"].astype(str) == str(action_type)].copy()
        color = palette[index % len(palette)]
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for _, row in group.dropna(subset=["End X", "End Y"]).iterrows():
            line_x += [row["Start X"], row["End X"], None]
            line_y += [row["Start Y"], row["End Y"], None]
        if line_x:
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name=f"{_safe_text(action_type).replace('_', ' ').title()} path",
                    line=dict(color=color, width=2.8),
                    opacity=0.58,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        marker_x = group["End X"].where(group["End X"].notna(), group["Start X"])
        marker_y = group["End Y"].where(group["End Y"].notna(), group["Start Y"])
        marker_size = 8 + (group["Bypassed Opponents"] / max_bypassed) * 18
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Action"].fillna(action_type),
                group["Result"].fillna("Unknown"),
                group["Minute"].fillna(0),
                group["Bypassed Opponents"].fillna(0),
                group["Bypassed Defenders"].fillna(0),
                group["Packing xG"].fillna(0),
                group["PXT Pass"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                name=_safe_text(action_type).replace("_", " ").title(),
                marker=dict(
                    size=marker_size,
                    color=color,
                    opacity=0.86,
                    line=dict(color="#ffffff", width=1.2),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[1]}"
                    "<br>Minute: %{customdata[3]:.0f}"
                    "<br>Result: %{customdata[2]}"
                    "<br>Bypassed opponents: %{customdata[4]:.1f}"
                    "<br>Bypassed defenders: %{customdata[5]:.1f}"
                    "<br>Packing xG: %{customdata[6]:.3f}"
                    "<br>PXT pass: %{customdata[7]:.3f}<extra></extra>"
                ),
            )
        )

    return fig


def shot_map(events: pd.DataFrame, team: str | None, title: str) -> go.Figure:
    shots = _spatial_events(events)
    if team:
        shots = shots[shots["Team"].astype(str) == str(team)]
    shots = shots[shots["Action Type"].astype(str).str.upper() == "SHOT"].copy()
    fig = pitch_figure(title)
    if shots.empty:
        _empty_pitch_message(fig, "No shot locations")
        return fig

    shots["Shot xG"] = pd.to_numeric(shots["Shot xG"], errors="coerce").fillna(0)
    shots["Outcome"] = np.where(
        shots["Result"].astype(str).str.upper().eq("SUCCESS") | shots["Action"].astype(str).str.upper().eq("GOAL"),
        "Goal",
        "Shot",
    )
    for _, row in shots.dropna(subset=["End X", "End Y"]).iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Start X"], row["End X"]],
                y=[row["Start Y"], row["End Y"]],
                mode="lines",
                line=dict(color="rgba(17,17,17,0.20)", width=1.1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    max_xg = max(_finite(shots["Shot xG"].max(), 0.1), 0.1)
    colors = {"Goal": RED, "Shot": DARK}
    for outcome, group in shots.groupby("Outcome", sort=False):
        group = group.copy()
        group["Label"] = np.where(
            outcome == "Goal",
            group["Player"].apply(lambda value: charting.wrap_label(value, 13, 2)),
            "",
        )
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Action"].fillna("Shot"),
                group["Minute"].fillna(0),
                group["Shot xG"].fillna(0),
                group["Post-Shot xG"].fillna(0),
                group["Shot Distance"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["Start X"],
                y=group["Start Y"],
                mode="markers+text",
                name=outcome,
                text=group["Label"],
                textposition="top center",
                textfont=dict(size=11, color=DARK),
                marker=dict(
                    size=10 + (group["Shot xG"] / max_xg) * 24,
                    color=colors.get(outcome, GREY),
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1.4),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[1]}"
                    "<br>Minute: %{customdata[2]:.0f}"
                    "<br>xG: %{customdata[3]:.3f}"
                    "<br>Post-shot xG: %{customdata[4]:.3f}"
                    "<br>Distance: %{customdata[5]:.1f}m<extra></extra>"
                ),
            )
        )
    return fig


def shot_map_half_pitch(events: pd.DataFrame, team: str | None, title: str) -> go.Figure:
    shots = _spatial_events(events)
    if team:
        shots = shots[shots["Team"].astype(str) == str(team)]
    shots = shots[shots["Action Type"].astype(str).str.upper() == "SHOT"].copy()
    fig = half_pitch_vertical_figure(title)
    if shots.empty:
        _empty_pitch_message(fig, "No shot locations")
        return fig

    shots = _normalise_shots_to_top_goal(shots)
    shots["Shot xG"] = pd.to_numeric(shots["Shot xG"], errors="coerce").fillna(0)
    shots["Outcome"] = np.where(
        shots["Result"].astype(str).str.upper().eq("SUCCESS") | shots["Action"].astype(str).str.upper().eq("GOAL"),
        "Goal",
        "Shot",
    )
    body_part = shots["Body Part"] if "Body Part" in shots else pd.Series("", index=shots.index)
    action = shots["Action"] if "Action" in shots else pd.Series("", index=shots.index)
    shots["Shot Type"] = [_shot_type_label(body, act) for body, act in zip(body_part, action, strict=False)]
    shots["Shot Symbol"] = shots["Shot Type"].map(SHOT_TYPE_SYMBOLS).fillna(SHOT_TYPE_SYMBOLS["Other"])
    for _, row in shots.dropna(subset=["_Half End X", "_Half End Y"]).iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["_Half X"], row["_Half End X"]],
                y=[row["_Half Y"], row["_Half End Y"]],
                mode="lines",
                line=dict(color="rgba(17,17,17,0.22)", width=1.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    max_xg = max(_finite(shots["Shot xG"].max(), 0.1), 0.1)
    colors = {"Goal": RED, "Shot": DARK}
    for outcome, group in shots.groupby("Outcome", sort=False):
        group = group.copy()
        group["Label"] = np.where(
            outcome == "Goal",
            group["Player"].apply(lambda value: charting.wrap_label(value, 13, 2)),
            "",
        )
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Team"].fillna("Unknown team"),
                group["Action"].fillna("Shot"),
                group["Shot Type"].fillna("Other"),
                group["Minute"].fillna(0),
                group["Shot xG"].fillna(0),
                group["Post-Shot xG"].fillna(0),
                group["Shot Distance"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["_Half X"],
                y=group["_Half Y"],
                mode="markers+text",
                name=outcome,
                text=group["Label"],
                textposition=(
                    _shot_goal_label_positions(group)
                    if outcome == "Goal"
                    else ["top center"] * len(group)
                ),
                textfont=dict(size=10, color=DARK),
                marker=dict(
                    size=10 + (group["Shot xG"] / max_xg) * 24,
                    symbol=group["Shot Symbol"],
                    color=colors.get(outcome, GREY),
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1.4),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[2]}"
                    "<br>Team: %{customdata[1]}"
                    "<br>Shot type: %{customdata[3]}"
                    "<br>Minute: %{customdata[4]:.0f}"
                    "<br>xG: %{customdata[5]:.3f}"
                    "<br>Post-shot xG: %{customdata[6]:.3f}"
                    "<br>Distance: %{customdata[7]:.1f}m<extra></extra>"
                ),
            )
        )
    return fig


def defensive_action_map(events: pd.DataFrame, team: str | None, title: str) -> go.Figure:
    actions = _spatial_events(events)
    if team:
        actions = actions[actions["Team"].astype(str) == str(team)]
    fig = pitch_figure(title)
    if actions.empty:
        _empty_pitch_message(fig, "No defensive locations")
        return fig

    category_col = "Defensive Category" if "Defensive Category" in actions else "Action Type"
    actions = actions.copy()
    actions[category_col] = actions[category_col].fillna("Unknown").astype(str)
    actions["Result Label"] = (
        actions["Result Label"].fillna("No Result").astype(str)
        if "Result Label" in actions
        else actions["Result"].where(actions["Result"].notna(), "No Result").astype(str)
    )
    color_map = {
        "Second Ball Win": GREEN,
        "Loose Ball Regain": RED,
        "Interception": GOLD,
        "Clearance": BLUE,
        "Block": DARK,
        "Ground Duel": DEEP_RED,
        "Referee Interception": GREY,
    }
    symbol_map = {
        "Second Ball Win": "star",
        "Loose Ball Regain": "circle",
        "Interception": "diamond",
        "Clearance": "triangle-up",
        "Block": "square",
        "Ground Duel": "x",
        "Referee Interception": "cross",
    }
    fallback_colors = [RED, DARK, GOLD, BLUE, GREEN, GREY, DEEP_RED]
    fallback_symbols = ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "hexagon"]
    category_order = (
        actions.groupby(category_col)
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    for index, action_type in enumerate(category_order):
        group = actions[actions[category_col].astype(str) == str(action_type)].copy()
        color = color_map.get(_safe_text(action_type), fallback_colors[index % len(fallback_colors)])
        symbol = symbol_map.get(_safe_text(action_type), fallback_symbols[index % len(fallback_symbols)])
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Action"].fillna(action_type),
                group["Minute"].fillna(0),
                group["Result Label"].fillna("No Result"),
                group["Action Type"].fillna("Unknown") if "Action Type" in group else group[category_col].fillna("Unknown"),
                group["Phase"].fillna("Unknown") if "Phase" in group else pd.Series("Unknown", index=group.index),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["Start X"],
                y=group["Start Y"],
                mode="markers",
                name=_safe_text(action_type).replace("_", " ").title(),
                marker=dict(
                    size=11 if _safe_text(action_type) != "Second Ball Win" else 13,
                    symbol=symbol,
                    color=color,
                    opacity=0.82,
                    line=dict(color="#ffffff", width=1.1),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]} - %{customdata[1]}"
                    "<br>Minute: %{customdata[2]:.0f}"
                    "<br>Category: " + _safe_text(action_type).replace("_", " ").title()
                    + "<br>Action type: %{customdata[4]}"
                    "<br>Phase: %{customdata[5]}"
                    "<br>Result: %{customdata[3]}<extra></extra>"
                ),
            )
        )
    return fig


def entry_zone_map(events: pd.DataFrame, team: str | None, title: str, zone: str = "Final Third") -> go.Figure:
    spatial = _spatial_events(events)
    if team:
        spatial = spatial[spatial["Team"].astype(str) == str(team)]
    entries = spatial.copy()
    for col in ["Start X", "Start Y", "End X", "End Y", "PXT Pass", "Team xT", "Pass Distance", "Minute"]:
        if col in entries:
            entries[col] = pd.to_numeric(entries[col], errors="coerce")

    fig = pitch_figure(title)
    zone_key = str(zone).strip().lower()
    if "penalty" in zone_key or "box" in zone_key:
        zone_label = "Penalty box"
        in_end_zone = entries["End X"].ge(PENALTY_BOX_X) & entries["End Y"].between(-PENALTY_BOX_Y, PENALTY_BOX_Y)
        in_start_zone = entries["Start X"].ge(PENALTY_BOX_X) & entries["Start Y"].between(-PENALTY_BOX_Y, PENALTY_BOX_Y)
        fig.add_shape(
            type="rect",
            x0=PENALTY_BOX_X,
            y0=-PENALTY_BOX_Y,
            x1=PITCH_X_MAX,
            y1=PENALTY_BOX_Y,
            line=dict(color="rgba(195,0,23,0.14)", width=0),
            fillcolor="rgba(195,0,23,0.06)",
            layer="below",
        )
    else:
        zone_label = "Final third"
        in_end_zone = entries["End X"].ge(FINAL_THIRD_X)
        in_start_zone = entries["Start X"].ge(FINAL_THIRD_X)
        fig.add_shape(
            type="rect",
            x0=FINAL_THIRD_X,
            y0=PITCH_Y_MIN,
            x1=PITCH_X_MAX,
            y1=PITCH_Y_MAX,
            line=dict(color="rgba(195,0,23,0.14)", width=0),
            fillcolor="rgba(195,0,23,0.045)",
            layer="below",
        )

    entries = entries[
        in_end_zone
        & ~in_start_zone
        & entries["Action Type"].astype(str).str.upper().ne("SHOT")
    ].copy()
    if entries.empty:
        _empty_pitch_message(fig, f"No {zone_label.lower()} entries")
        return fig

    entries["Outcome"] = entries["Result"].apply(_entry_outcome_label)
    entries["_Entry Value"] = entries[["PXT Pass", "Team xT"]].clip(lower=0).max(axis=1).fillna(0)
    max_value = max(_finite(entries["_Entry Value"].max(), 0.05), 0.05)
    colors = {"Successful entries": GREEN, "Unsuccessful entries": RED, "Other entries": GREY}
    for outcome, group in entries.groupby("Outcome", sort=False):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for _, row in group.iterrows():
            x_values += [row["Start X"], row["End X"], None]
            y_values += [row["Start Y"], row["End Y"], None]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=outcome,
                line=dict(color=colors.get(outcome, GREY), width=3.2),
                opacity=0.78,
                hoverinfo="skip",
            )
        )
        customdata = np.stack(
            [
                group["Player"].fillna("Unknown"),
                group["Receiver"].fillna(""),
                group["Action Type"].fillna("Action"),
                group["Action"].fillna("Entry"),
                group["Result"].fillna("Unknown"),
                group["Minute"].fillna(0),
                group["PXT Pass"].fillna(0),
                group["Team xT"].fillna(0),
                group["Pass Distance"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["End X"],
                y=group["End Y"],
                mode="markers",
                name=f"{outcome} end",
                marker=dict(
                    size=8 + (group["_Entry Value"] / max_value) * 15,
                    color=colors.get(outcome, GREY),
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1.1),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br>Receiver: %{customdata[1]}"
                    "<br>Action type: %{customdata[2]}"
                    "<br>Action: %{customdata[3]}"
                    "<br>Minute: %{customdata[5]:.0f}"
                    "<br>Result: %{customdata[4]}"
                    "<br>PXT pass: %{customdata[6]:.3f}"
                    "<br>Team xT: %{customdata[7]:.3f}"
                    "<br>Distance: %{customdata[8]:.1f}m<extra></extra>"
                ),
                showlegend=False,
            )
        )
    return fig


def final_third_map(events: pd.DataFrame, team: str | None, title: str) -> go.Figure:
    spatial = _spatial_events(events)
    if team:
        spatial = spatial[spatial["Team"].astype(str) == str(team)]
    passes = spatial[
        (spatial["Action Type"].astype(str).str.upper() == "PASS")
        & (pd.to_numeric(spatial["End X"], errors="coerce") >= FINAL_THIRD_X)
    ].copy()
    shots = spatial[spatial["Action Type"].astype(str).str.upper() == "SHOT"].copy()

    fig = pitch_figure(title)
    fig.add_shape(
        type="rect",
        x0=FINAL_THIRD_X,
        y0=PITCH_Y_MIN,
        x1=PITCH_X_MAX,
        y1=PITCH_Y_MAX,
        line=dict(color="rgba(195,0,23,0.14)", width=0),
        fillcolor="rgba(195,0,23,0.045)",
        layer="below",
    )
    if passes.empty and shots.empty:
        _empty_pitch_message(fig, "No final-third events")
        return fig

    if not passes.empty:
        passes["Outcome"] = passes["Result"].apply(_entry_outcome_label)
        colors = {"Successful entries": GREEN, "Unsuccessful entries": RED, "Other entries": GREY}
        for outcome, group in passes.groupby("Outcome", sort=False):
            x_values: list[float | None] = []
            y_values: list[float | None] = []
            for _, row in group.iterrows():
                x_values += [row["Start X"], row["End X"], None]
                y_values += [row["Start Y"], row["End Y"], None]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=outcome,
                    line=dict(color=colors.get(outcome, GREY), width=3.4),
                    opacity=0.82,
                    hoverinfo="skip",
                )
            )

    if not shots.empty:
        shots["Shot xG"] = pd.to_numeric(shots["Shot xG"], errors="coerce").fillna(0)
        max_xg = max(_finite(shots["Shot xG"].max(), 0.1), 0.1)
        customdata = np.stack(
            [
                shots["Player"].fillna("Unknown"),
                shots["Minute"].fillna(0),
                shots["Shot xG"].fillna(0),
                shots["Action"].fillna("Shot"),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=shots["Start X"],
                y=shots["Start Y"],
                mode="markers",
                name="Shots",
                marker=dict(
                    size=10 + (shots["Shot xG"] / max_xg) * 22,
                    color=DARK,
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1.3),
                ),
                customdata=customdata,
                hovertemplate="%{customdata[0]} - %{customdata[3]}<br>Minute: %{customdata[1]:.0f}<br>xG: %{customdata[2]:.3f}<extra></extra>",
            )
        )
    return fig


def xg_timeline(events: pd.DataFrame, title: str, end_minute: float | None = None) -> go.Figure:
    shots = events[events["Action Type"].astype(str).str.upper() == "SHOT"].copy() if not events.empty else events.copy()
    fig = go.Figure()
    if shots.empty:
        fig = charting.polish_figure(fig, title, height=500)
        fig.add_annotation(text="No xG events", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=GREY))
        return fig

    shots["Shot xG"] = pd.to_numeric(shots["Shot xG"], errors="coerce").fillna(0)
    shots["Minute"] = pd.to_numeric(shots["Minute"], errors="coerce").fillna(0)
    shots = shots.sort_values(["Team", "Second", "Event Number"])
    teams = shots["Team"].dropna().astype(str).unique().tolist()
    colors = [RED, DARK, GOLD, BLUE]
    observed_end_minute = max(96, _finite(shots["Minute"].max(), 96), _finite(end_minute, 0))
    chart_end_minute = float(np.ceil(observed_end_minute / 5) * 5)
    for index, team in enumerate(teams):
        team_shots = shots[shots["Team"].astype(str) == team].copy()
        team_shots["Cumulative xG"] = team_shots["Shot xG"].cumsum()
        x = pd.concat([pd.Series([0]), team_shots["Minute"]], ignore_index=True)
        y = pd.concat([pd.Series([0.0]), team_shots["Cumulative xG"]], ignore_index=True)
        marker_sizes = [7] * len(x)
        if len(x) and _finite(x.iloc[-1], 0) < chart_end_minute:
            x = pd.concat([x, pd.Series([chart_end_minute])], ignore_index=True)
            y = pd.concat([y, pd.Series([_finite(y.iloc[-1], 0.0)])], ignore_index=True)
            marker_sizes.append(0)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=team,
                line=dict(color=colors[index % len(colors)], width=3, shape="hv"),
                marker=dict(size=marker_sizes, line=dict(color="#ffffff", width=1)),
                hovertemplate=f"{team}<br>Minute: %{{x:.0f}}<br>Cumulative xG: %{{y:.2f}}<extra></extra>",
            )
        )
        goals = team_shots[
            team_shots["Result"].astype(str).str.upper().eq("SUCCESS")
            | team_shots["Action"].astype(str).str.upper().eq("GOAL")
        ]
        if not goals.empty:
            fig.add_trace(
                go.Scatter(
                    x=goals["Minute"],
                    y=goals["Cumulative xG"],
                    mode="markers",
                    name=f"{team} goals",
                    marker=dict(symbol="star", size=15, color=colors[index % len(colors)], line=dict(color="#ffffff", width=1)),
                    customdata=np.stack([goals["Player"].fillna("Unknown"), goals["Shot xG"].fillna(0)], axis=-1),
                    hovertemplate="%{customdata[0]} goal<br>Minute: %{x:.0f}<br>Shot xG: %{customdata[1]:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
    fig.update_layout(height=540, xaxis_title="Minute", yaxis_title="Cumulative xG")
    fig.update_xaxes(range=[0, chart_end_minute], dtick=15)
    fig.update_yaxes(tickformat=".2f", rangemode="tozero")
    fig = charting.polish_figure(fig, title)
    fig.update_layout(
        margin=dict(l=52, r=34, t=104, b=58),
        title=dict(
            text=title,
            font=dict(size=20, color=DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            title=dict(text=""),
        ),
    )
    return fig


def _timeline_minute_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    second = pd.to_numeric(frame["Second"], errors="coerce") if "Second" in frame else pd.Series(np.nan, index=frame.index)
    minute = pd.to_numeric(frame["Minute"], errors="coerce") if "Minute" in frame else pd.Series(np.nan, index=frame.index)
    period = pd.to_numeric(frame["Period"], errors="coerce") if "Period" in frame else pd.Series(np.nan, index=frame.index)
    period_base_seconds = np.select(
        [period.eq(1), period.eq(2), period.eq(3), period.eq(4)],
        [0, 45 * 60, 90 * 60, 105 * 60],
        default=np.nan,
    )
    offset_bucket = np.floor(second / 10000).clip(lower=0)
    fallback_base_seconds = offset_bucket * 45 * 60
    base_seconds = pd.Series(period_base_seconds, index=frame.index).where(pd.notna(period_base_seconds), fallback_base_seconds)
    period_seconds = second % 10000
    elapsed_seconds = second.where(second < 10000, period_seconds + base_seconds)
    derived = np.floor(elapsed_seconds / 60) + 1
    derived = derived.where(second.notna(), minute)
    return pd.to_numeric(derived, errors="coerce").clip(lower=0, upper=130)


def _timeline_card_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["Team", "Player", "Minute", "Card", "Icon"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    action_type = events["Action Type"].astype(str).str.upper() if "Action Type" in events else pd.Series("", index=events.index)
    action = events["Action"].astype(str).str.upper() if "Action" in events else pd.Series("", index=events.index)
    result = events["Result"].astype(str).str.upper() if "Result" in events else pd.Series("", index=events.index)
    combined = action_type + " " + action + " " + result
    mask = combined.str.contains("YELLOW_CARD|RED_CARD|SECOND_YELLOW|BOOKING|CARD", regex=True, na=False)
    cards = events[mask].copy()
    if cards.empty:
        return pd.DataFrame(columns=columns)

    card_text = combined.loc[cards.index]
    cards["Minute"] = _timeline_minute_series(cards)
    cards["Card"] = np.select(
        [
            card_text.str.contains("RED_CARD|SECOND_YELLOW", regex=True, na=False),
            card_text.str.contains("YELLOW_CARD|BOOKING|CARD", regex=True, na=False),
        ],
        ["Red Card", "Yellow Card"],
        default="Card",
    )
    cards["Icon"] = np.where(cards["Card"].eq("Red Card"), RED_CARD_ICON, YELLOW_CARD_ICON)
    for column in columns:
        if column not in cards:
            cards[column] = np.nan
    return cards[columns].dropna(subset=["Minute"]).reset_index(drop=True)


def _timeline_card_labels(players: pd.Series) -> list[str]:
    names = players.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    last_names = names.apply(lambda value: value.split()[-1] if value.split() else value)
    counts = last_names.value_counts()
    labels = []
    for name, last in zip(names, last_names, strict=False):
        parts = name.split()
        labels.append(f"{parts[0][0]}. {last}" if counts.get(last, 0) > 1 and len(parts) > 1 else last)
    return labels


def _timeline_card_positions(teams: list[str], scale: float) -> dict[str, float]:
    marker_scale = max(float(scale), 0.1)
    positions: dict[str, float] = {}
    for index, team in enumerate(teams):
        if index == 0:
            positions[str(team)] = marker_scale * 1.08
        elif index == 1:
            positions[str(team)] = -marker_scale * 1.08
        else:
            positions[str(team)] = marker_scale * (1.08 + 0.16 * (index - 1))
    return positions


def _add_timeline_card_traces(fig: go.Figure, cards: pd.DataFrame, team_positions: dict[str, float]) -> None:
    if cards.empty:
        return
    for team, y_base in team_positions.items():
        team_cards = cards[cards["Team"].astype(str) == str(team)]
        if team_cards.empty:
            continue
        card_names = _timeline_card_labels(team_cards["Player"])
        customdata = np.stack(
            [
                team_cards["Player"].fillna("Unknown"),
                team_cards["Card"].fillna("Card"),
                team_cards["Minute"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=team_cards["Minute"],
                y=[y_base] * len(team_cards),
                mode="text",
                name=f"{team} cards",
                text=[f"{icon} {name}" for icon, name in zip(team_cards["Icon"], card_names, strict=False)],
                textposition="middle center",
                textfont=dict(size=13, color=DARK),
                customdata=customdata,
                hovertemplate=f"{team}<br>%{{customdata[1]}}: %{{customdata[0]}}<br>Minute: %{{customdata[2]:.0f}}<extra></extra>",
                showlegend=False,
            )
        )


def threat_timeline(events: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    cards = _timeline_card_events(events)
    if events.empty:
        fig = charting.polish_figure(fig, title, height=500)
        fig.add_annotation(text="No event threat values", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=GREY))
        return fig

    values = events.copy()
    metric_cols = [col for col in ["PXT Pass", "PXT Shot", "Shot xG", "Team xT"] if col in values]
    for col in metric_cols:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    values["Threat"] = values[metric_cols].clip(lower=0).max(axis=1).fillna(0) if metric_cols else 0
    values["Minute"] = _timeline_minute_series(values).fillna(0).astype(int)
    values = values[values["Threat"] > 0]
    if values.empty:
        fig = charting.polish_figure(fig, title, height=500)
        fig.add_annotation(text="No positive threat values", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=GREY))
        if not cards.empty:
            teams = cards["Team"].dropna().astype(str).drop_duplicates().tolist()
            _add_timeline_card_traces(fig, cards, _timeline_card_positions(teams, 0.2))
            fig.update_xaxes(range=[0, max(96, _finite(cards["Minute"].max(), 96))], dtick=15)
            fig.update_yaxes(range=[-0.35, 0.35])
        return fig

    minute_values = values.groupby(["Team", "Minute"], as_index=False)["Threat"].sum().sort_values(["Team", "Minute"])
    colors = [RED, DARK, GOLD, BLUE]
    team_order = minute_values["Team"].drop_duplicates().astype(str).tolist()
    for card_team in cards["Team"].dropna().astype(str).drop_duplicates():
        if card_team not in team_order:
            team_order.append(card_team)
    max_cumulative = 0.0
    for index, (team, group) in enumerate(minute_values.groupby("Team", sort=False)):
        group = group.copy()
        group["Cumulative Threat"] = group["Threat"].cumsum()
        max_cumulative = max(max_cumulative, float(group["Cumulative Threat"].max()))
        fig.add_trace(
            go.Scatter(
                x=group["Minute"],
                y=group["Cumulative Threat"],
                mode="lines",
                name=str(team),
                line=dict(color=colors[index % len(colors)], width=3, shape="hv"),
                hovertemplate=f"{team}<br>Minute: %{{x:.0f}}<br>Cumulative threat: %{{y:.2f}}<extra></extra>",
            )
        )
    _add_timeline_card_traces(fig, cards, _timeline_card_positions(team_order, max_cumulative))
    fig.update_layout(height=540, xaxis_title="Minute", yaxis_title="Cumulative threat")
    timeline_max = max(
        96,
        _finite(values["Minute"].max(), 96),
        _finite(cards["Minute"].max(), 96) if not cards.empty else 96,
    )
    fig.update_xaxes(range=[0, timeline_max], dtick=15)
    fig.update_yaxes(tickformat=".2f", rangemode="tozero")
    fig = charting.polish_figure(fig, title)
    fig.update_layout(
        margin=dict(l=52, r=34, t=104, b=58),
        title=dict(
            text=title,
            font=dict(size=20, color=DARK),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            title=dict(text=""),
        ),
    )
    return fig


GOAL_HALF_WIDTH = 3.66
GOAL_HEIGHT = 2.44
GOALMOUTH_SIDE_PADDING = 2.0
GOALMOUTH_TOP_PADDING = 1.35
GOALMOUTH_BOTTOM_PADDING = 0.12
GOALMOUTH_TEMPLATE_PATH = ui.ASSETS_DIR / "goalmouth_template.png"
GOALMOUTH_PALETTE = [RED, DARK, GOLD, BLUE, "#15803d", "#9333ea", "#0e7490", "#b45309", "#be185d", "#4d7c0f"]


@lru_cache(maxsize=1)
def _goalmouth_template_uri() -> str:
    if not GOALMOUTH_TEMPLATE_PATH.exists():
        return ""
    encoded = base64.b64encode(GOALMOUTH_TEMPLATE_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def goalmouth_shot_map(
    shots: pd.DataFrame,
    title: str,
    group_col: str = "Outcome",
    group_order: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_symbols: dict[str, str] | None = None,
    height: int = 680,
) -> go.Figure:
    """Shots plotted against the goal face using Shot Target Y/Z, from the shooter's view.

    group_col controls what the legend/colour splits on -- 'Outcome' (goal/saved/
    blocked/...) for a single shooter's execution, or 'Player' for a whole team's
    shots in one map. Marker size is Post-Shot xG where available, else Shot xG.
    """
    fig = go.Figure()
    target_cols = {"Shot Target Y", "Shot Target Z"}
    if shots.empty or not target_cols.issubset(shots.columns):
        fig.add_annotation(text="No Shot Target Coordinates", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=height)

    target = shots.dropna(subset=["Shot Target Y", "Shot Target Z"]).copy()
    # targetPoint.y uses the same attacking-right orientation as adjusted End
    # Y. A shooter's-view goal face therefore uses -Y, matching the vertical
    # half-pitch rotation above instead of mirroring the target left-to-right.
    target["_Goalmouth X"] = -pd.to_numeric(target["Shot Target Y"], errors="coerce")
    target["_Goalmouth Z"] = pd.to_numeric(target["Shot Target Z"], errors="coerce")
    target = target.dropna(subset=["_Goalmouth X", "_Goalmouth Z"]).copy()
    if target.empty:
        fig.add_annotation(text="No Shot Target Coordinates", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return charting.polish_figure(fig, title, height=height)

    x_min = -(GOAL_HALF_WIDTH + GOALMOUTH_SIDE_PADDING)
    x_max = GOAL_HALF_WIDTH + GOALMOUTH_SIDE_PADDING
    y_min = -GOALMOUTH_BOTTOM_PADDING
    y_max = GOAL_HEIGHT + GOALMOUTH_TOP_PADDING
    template_uri = _goalmouth_template_uri()
    if template_uri:
        fig.add_layout_image(
            dict(
                source=template_uri,
                xref="x",
                yref="y",
                x=-GOAL_HALF_WIDTH,
                y=GOAL_HEIGHT,
                sizex=GOAL_HALF_WIDTH * 2,
                sizey=GOAL_HEIGHT,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )
    else:
        fig.add_shape(
            type="rect",
            x0=-GOAL_HALF_WIDTH, x1=GOAL_HALF_WIDTH, y0=0, y1=GOAL_HEIGHT,
            line=dict(color=DARK, width=6),
            fillcolor="rgba(255,255,255,0.92)",
            layer="below",
        )

    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=GOAL_HEIGHT, line=dict(color="rgba(255,255,255,0.34)", width=1.3, dash="dot"))
    fig.add_shape(type="line", x0=-GOAL_HALF_WIDTH, x1=GOAL_HALF_WIDTH, y0=GOAL_HEIGHT / 2, y1=GOAL_HEIGHT / 2, line=dict(color="rgba(255,255,255,0.26)", width=1.2, dash="dot"))
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=GOAL_HEIGHT, y1=GOAL_HEIGHT, line=dict(color="rgba(102,112,133,0.34)", width=1.2, dash="dash"))
    fig.add_shape(type="line", x0=-GOAL_HALF_WIDTH, x1=-GOAL_HALF_WIDTH, y0=y_min, y1=y_max, line=dict(color="rgba(102,112,133,0.28)", width=1.2, dash="dash"))
    fig.add_shape(type="line", x0=GOAL_HALF_WIDTH, x1=GOAL_HALF_WIDTH, y0=y_min, y1=y_max, line=dict(color="rgba(102,112,133,0.28)", width=1.2, dash="dash"))

    size_source = pd.to_numeric(target.get("Post-Shot xG"), errors="coerce") if "Post-Shot xG" in target else pd.Series(np.nan, index=target.index)
    size_source = size_source.fillna(pd.to_numeric(target.get("Shot xG"), errors="coerce"))
    target["_Size Value"] = size_source.fillna(0)
    max_size_value = max(float(target["_Size Value"].max()), 0.08)

    groups_present = target[group_col].fillna("Unknown").astype(str).unique().tolist() if group_col in target else ["Unknown"]
    ordered_groups = [g for g in (group_order or []) if g in groups_present]
    ordered_groups += [g for g in groups_present if g not in ordered_groups]
    colors = group_colors or {group: GOALMOUTH_PALETTE[index % len(GOALMOUTH_PALETTE)] for index, group in enumerate(ordered_groups)}
    symbols = group_symbols or {}

    for group in ordered_groups:
        rows = target[target[group_col].fillna("Unknown").astype(str) == group].copy() if group_col in target else target.copy()
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["_Goalmouth X"],
                y=rows["_Goalmouth Z"],
                mode="markers",
                name=charting.wrap_label(group, width=18, max_lines=2),
                cliponaxis=False,
                marker=dict(
                    size=12 + (rows["_Size Value"] / max_size_value) * 28,
                    color=colors.get(group, GREY),
                    symbol=symbols.get(group, "circle"),
                    opacity=0.88,
                    line=dict(color="#ffffff", width=1.2),
                ),
                customdata=np.stack(
                    [
                        rows["Player"].fillna("Unknown") if "Player" in rows else pd.Series("Unknown", index=rows.index),
                        rows["Minute"].fillna(0) if "Minute" in rows else pd.Series(0, index=rows.index),
                        pd.to_numeric(rows.get("Shot xG"), errors="coerce").fillna(0) if "Shot xG" in rows else pd.Series(0.0, index=rows.index),
                        pd.to_numeric(rows.get("Post-Shot xG"), errors="coerce").fillna(0) if "Post-Shot xG" in rows else pd.Series(0.0, index=rows.index),
                        rows[group_col].fillna("Unknown").astype(str) if group_col in rows else pd.Series("Unknown", index=rows.index),
                    ],
                    axis=-1,
                ),
                hovertemplate="%{customdata[0]} - %{customdata[4]}<br>Minute: %{customdata[1]:.0f}<br>xG: %{customdata[2]:.3f}<br>Post-shot xG: %{customdata[3]:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis_title="<b>Shooter's left - Target Width - Shooter's right</b>",
        yaxis_title="<b>Target Height</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[x_min, x_max], zeroline=False, tickformat=".1f", showgrid=False, fixedrange=True)
    fig.update_yaxes(range=[y_min, y_max], zeroline=False, tickformat=".1f", showgrid=False, fixedrange=True)
    fig = charting.polish_figure(fig, title)
    if group_col == "Player" and len(ordered_groups) > 4:
        fig.update_layout(
            margin=dict(l=28, r=180, t=96, b=54),
            legend=dict(
                orientation="v",
                title_text="<b>Key: colour = player</b>",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10),
            ),
        )
    else:
        legend_title = (
            "<b>Key: colour/symbol = outcome</b>"
            if group_col == "Outcome"
            else "<b>Key: colour = player</b>"
        )
        fig.update_layout(
            margin=dict(l=28, r=34, t=84, b=128),
            legend=dict(
                orientation="h",
                title_text=legend_title,
                yanchor="top",
                y=-0.12,
                xanchor="left",
                x=0,
                font=dict(size=10),
            ),
        )
    return fig


def expected_threat_timeline(events: pd.DataFrame, title: str) -> go.Figure:
    """Cumulative signed expected-threat (PXT Pass + PXT Shot) by minute, per team.

    Unlike threat_timeline (which takes the max of several threat-ish fields,
    clipped at zero, and falls back to Team xT's positional value), this uses
    only the two Impect fields that represent a specific player's marginal
    threat contribution -- PXT Pass (passes and clearances) and PXT Shot --
    summed as signed values so a misplaced pass nets against the team total
    rather than being dropped.
    """
    fig = go.Figure()
    cards = _timeline_card_events(events)
    if events.empty:
        fig = charting.polish_figure(fig, title, height=500)
        fig.add_annotation(text="No expected-threat events", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=GREY))
        return fig

    values = events.copy()
    pxt_pass = pd.to_numeric(values.get("PXT Pass"), errors="coerce") if "PXT Pass" in values else pd.Series(0.0, index=values.index)
    pxt_shot = pd.to_numeric(values.get("PXT Shot"), errors="coerce") if "PXT Shot" in values else pd.Series(0.0, index=values.index)
    values["xT Value"] = pxt_pass.fillna(0.0) + pxt_shot.fillna(0.0)
    values["Minute"] = _timeline_minute_series(values).fillna(0).astype(int)
    values = values[values["xT Value"] != 0]
    if values.empty:
        fig = charting.polish_figure(fig, title, height=500)
        fig.add_annotation(text="No expected-threat events", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=GREY))
        if not cards.empty:
            teams = cards["Team"].dropna().astype(str).drop_duplicates().tolist()
            _add_timeline_card_traces(fig, cards, _timeline_card_positions(teams, 0.2))
            fig.update_xaxes(range=[0, max(96, _finite(cards["Minute"].max(), 96))], dtick=15)
            fig.update_yaxes(range=[-0.35, 0.35])
        return fig

    minute_values = values.groupby(["Team", "Minute"], as_index=False)["xT Value"].sum().sort_values(["Team", "Minute"])
    colors = [RED, DARK, GOLD, BLUE]
    team_order = minute_values["Team"].drop_duplicates().astype(str).tolist()
    for card_team in cards["Team"].dropna().astype(str).drop_duplicates():
        if card_team not in team_order:
            team_order.append(card_team)
    max_abs_cumulative = 0.0
    for index, (team, group) in enumerate(minute_values.groupby("Team", sort=False)):
        group = group.copy()
        group["Cumulative xT"] = group["xT Value"].cumsum()
        max_abs_cumulative = max(max_abs_cumulative, float(group["Cumulative xT"].abs().max()))
        fig.add_trace(
            go.Scatter(
                x=group["Minute"],
                y=group["Cumulative xT"],
                mode="lines",
                name=str(team),
                line=dict(color=colors[index % len(colors)], width=3, shape="hv"),
                hovertemplate=f"{team}<br>Minute: %{{x:.0f}}<br>Cumulative xT: %{{y:.3f}}<extra></extra>",
            )
        )
    _add_timeline_card_traces(fig, cards, _timeline_card_positions(team_order, max_abs_cumulative))
    fig.update_layout(height=540, xaxis_title="Minute", yaxis_title="Cumulative expected threat (PXT)")
    timeline_max = max(
        96,
        _finite(values["Minute"].max(), 96),
        _finite(cards["Minute"].max(), 96) if not cards.empty else 96,
    )
    fig.update_xaxes(range=[0, timeline_max], dtick=15)
    fig.update_yaxes(tickformat=".2f")
    fig = charting.polish_figure(fig, title)
    fig.update_layout(
        margin=dict(l=52, r=34, t=104, b=58),
        title=dict(text=title, font=dict(size=20, color=DARK), x=0.01, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.01,
            title=dict(text=""),
        ),
    )
    return fig


def _empty_pitch_message(fig: go.Figure, text: str) -> None:
    fig.add_annotation(
        text=text,
        x=0,
        y=0,
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(size=16, color=GREY),
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor=LIGHT_GREY,
        borderpad=8,
    )
