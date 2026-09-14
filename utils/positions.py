"""Shared Impect position normalization and dominant-role selection."""

from __future__ import annotations

import pandas as pd


def classify_position_text(position: object) -> str | None:
    """Map one provider position label to the app's tactical role groups."""
    if position is None or pd.isna(position):
        return None
    text = f" {str(position).upper().replace('_', ' ')} "
    tokens = set(text.replace(",", " ").split())

    if "GK" in tokens or "GOALKEEPER" in tokens:
        return "Goalkeeper"
    if (
        {"CB", "LCB", "RCB", "DEF"}.intersection(tokens)
        or (
            "DEFENDER" in tokens
            and {"CENTRAL", "CENTRE", "CENTER"}.intersection(tokens)
        )
        or "CENTRE BACK" in text
        or "CENTER BACK" in text
        or "CENTRAL DEFENDER" in text
        or "CENTRE DEFENDER" in text
        or "CENTER DEFENDER" in text
    ):
        return "Centre Back"
    if (
        {"LB", "RB", "LWB", "RWB", "WB", "FB"}.intersection(tokens)
        or "FULL BACK" in text
        or "LEFT BACK" in text
        or "RIGHT BACK" in text
        or "WINGBACK" in text
        or "WING BACK" in text
    ):
        return "Full Back"
    if (
        {"DM", "DMF", "CDM", "RDMF", "LDMF"}.intersection(tokens)
        or "DEFENSIVE MIDFIELD" in text
        or "DEFENSE MIDFIELD" in text
    ):
        return "Defensive Midfielder"
    if (
        {"AM", "AMF", "CAM"}.intersection(tokens)
        or "ATTACKING MIDFIELD" in text
    ):
        return "Attacking Midfielder"
    if (
        {"CM", "CMF", "LCMF", "RCMF", "MID", "MF"}.intersection(tokens)
        or "CENTRAL MIDFIELD" in text
        or "CENTRE MIDFIELD" in text
        or "CENTER MIDFIELD" in text
    ):
        return "Central Midfielder"
    if (
        {"CF", "ST", "FW", "FWD", "LW", "RW", "LWF", "RWF", "WF"}.intersection(tokens)
        or "WINGER" in text
        or "FORWARD" in text
        or "STRIKER" in text
    ):
        return "Forward / Winger"
    return None


def position_group(position: object) -> str:
    """Classify a dominant-first position list into one tactical role group."""
    if position is None or pd.isna(position):
        return "Outfield"
    text = str(position)
    primary = text.split(",", maxsplit=1)[0]
    return (
        classify_position_text(primary)
        or classify_position_text(text)
        or "Outfield"
    )


def dominant_positions(
    position_rows: pd.DataFrame,
    group_columns: list[str],
    *,
    position_column: str = "Position",
    duration_column: str = "Play Duration Seconds",
) -> pd.DataFrame:
    """Return one duration-dominant provider position for every player group.

    Durations are first summed by canonical role, so two full-back labels can
    collectively outrank a single midfield label. Within the winning role, the
    raw Impect label with the most minutes becomes the displayed position. A
    single primary value keeps every downstream filter and peer-group classifier
    consistent; secondary stints should not redefine the player's season role.
    """
    output_columns = [*group_columns, position_column]
    required = {*group_columns, position_column, duration_column}
    if position_rows.empty or not required.issubset(position_rows.columns):
        return pd.DataFrame(columns=output_columns)

    working = position_rows[[*group_columns, position_column, duration_column]].copy()
    working[position_column] = working[position_column].astype("string").str.strip()
    invalid = (
        working[position_column].isna()
        | working[position_column].eq("")
        | working[position_column].str.casefold().isin({"nan", "none", "null"})
    )
    working = working.loc[~invalid].copy()
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    working[duration_column] = pd.to_numeric(
        working[duration_column], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    label_seconds = (
        working.groupby(
            [*group_columns, position_column],
            dropna=False,
            observed=True,
            as_index=False,
        )[duration_column]
        .sum()
        .rename(columns={duration_column: "_Position Seconds"})
    )
    label_seconds["_Role Group"] = label_seconds[position_column].map(
        lambda value: classify_position_text(value) or "Outfield"
    )
    label_seconds["_Recognized"] = label_seconds[position_column].map(
        classify_position_text
    ).notna()
    label_seconds["_Role Seconds"] = label_seconds.groupby(
        [*group_columns, "_Role Group"],
        dropna=False,
        observed=True,
    )["_Position Seconds"].transform("sum")

    sort_columns = [
        *group_columns,
        "_Role Seconds",
        "_Recognized",
        "_Position Seconds",
        position_column,
    ]
    ascending = [True] * len(group_columns) + [False, False, False, True]
    label_seconds = label_seconds.sort_values(sort_columns, ascending=ascending)

    dominant = label_seconds.drop_duplicates(group_columns)[output_columns]
    return dominant.reset_index(drop=True)
