# =============================================================================
# FINAL THIRD HELPERS - entry definitions, value and summaries
# =============================================================================
from __future__ import annotations

import pandas as pd

from utils import pitch


ZONE_OPTIONS = ["Final Third", "Penalty Box"]

UNSUCCESSFUL_RESULTS = {
    "FAIL",
    "FAILED",
    "UNSUCCESSFUL",
}


def zone_title(zone: str) -> str:
    """Return the supported zone label used across final-third pages."""
    if "penalty" in str(zone).lower() or "box" in str(zone).lower():
        return "Penalty Box"

    return "Final Third"


def entry_mask(events: pd.DataFrame, zone: str) -> pd.Series:
    """Identify actions that start outside a zone and end inside it."""
    if events.empty:
        return pd.Series(False, index=events.index)

    required_columns = {"Start X", "Start Y", "End X", "End Y"}
    if not required_columns.issubset(events.columns):
        return pd.Series(False, index=events.index)

    start_x = pd.to_numeric(events["Start X"], errors="coerce")
    start_y = pd.to_numeric(events["Start Y"], errors="coerce")
    end_x = pd.to_numeric(events["End X"], errors="coerce")
    end_y = pd.to_numeric(events["End Y"], errors="coerce")
    action_type = events.get("Action Type", pd.Series("", index=events.index)).astype(str).str.upper()

    if zone_title(zone) == "Penalty Box":
        end_zone = end_x.ge(pitch.PENALTY_BOX_X) & end_y.between(
            -pitch.PENALTY_BOX_Y,
            pitch.PENALTY_BOX_Y,
        )
        start_zone = start_x.ge(pitch.PENALTY_BOX_X) & start_y.between(
            -pitch.PENALTY_BOX_Y,
            pitch.PENALTY_BOX_Y,
        )
    else:
        end_zone = end_x.ge(pitch.FINAL_THIRD_X)
        start_zone = start_x.ge(pitch.FINAL_THIRD_X)

    return end_zone & ~start_zone & action_type.ne("SHOT")


def entry_value(events: pd.DataFrame) -> pd.Series:
    """Use the larger positive value from PXT Pass and Team xT for each entry."""
    if events.empty:
        return pd.Series(dtype="float64")

    values = pd.DataFrame(index=events.index)
    for column in ["PXT Pass", "Team xT"]:
        if column in events:
            values[column] = pd.to_numeric(events[column], errors="coerce")
        else:
            values[column] = 0

    return values.clip(lower=0).max(axis=1).fillna(0)


def result_status(results: pd.Series) -> pd.Series:
    """Group raw event result labels into analyst-readable outcome buckets."""
    clean = results.astype(str).str.strip().str.upper()

    return pd.Series(
        clean.map(
            lambda value: (
                "Successful"
                if value == "SUCCESS"
                else "Unsuccessful"
                if value in UNSUCCESSFUL_RESULTS
                else "Other"
            )
        ),
        index=results.index,
    )


def prepare_entries(
    events: pd.DataFrame,
    zone: str,
    action_types: list[str] | None = None,
    results: list[str] | None = None,
    min_value: float = 0.0,
) -> pd.DataFrame:
    """Return final-third/box entry rows with value and outcome fields added."""
    entries = events[entry_mask(events, zone)].copy()

    if entries.empty:
        entries["_Entry Value"] = pd.Series(dtype="float64")
        entries["_Outcome"] = pd.Series(dtype="object")
        return entries

    entries["_Entry Value"] = entry_value(entries)
    if "Result" in entries:
        entries["_Outcome"] = result_status(entries["Result"])
    else:
        entries["_Outcome"] = "Other"

    if action_types and "Action Type" in entries:
        entries = entries[entries["Action Type"].astype(str).isin(action_types)]

    if results and "Result" in entries:
        entries = entries[entries["Result"].astype(str).isin(results)]

    entries = entries[pd.to_numeric(entries["_Entry Value"], errors="coerce").fillna(0) >= min_value]
    return entries.copy()


def player_entry_summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Summarise entry volume, success and value by player."""
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
    values["_Outcome"] = values["_Outcome"] if "_Outcome" in values else result_status(values.get("Result", pd.Series("", index=values.index)))
    values["_Successful"] = values["_Outcome"].astype(str).eq("Successful")
    values["_Unsuccessful"] = values["_Outcome"].astype(str).eq("Unsuccessful")
    values["_Other"] = values["_Outcome"].astype(str).eq("Other")
    values["_Entry Value"] = (
        pd.to_numeric(values["_Entry Value"], errors="coerce").fillna(0)
        if "_Entry Value" in values
        else entry_value(values)
    )

    for column in ["PXT Pass", "Team xT"]:
        values[column] = pd.to_numeric(values[column], errors="coerce") if column in values else 0

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
    summary["Success %"] = summary["Successful"] / summary["Entries"].replace(0, pd.NA) * 100

    for column in [
        "Success %",
        "Entry Value",
        "Avg Entry Value",
        "PXT Pass",
        "Team xT",
    ]:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        )
    summary["Success %"] = summary["Success %"].round(1)
    summary["Entry Value"] = summary["Entry Value"].round(4)
    summary["Avg Entry Value"] = summary["Avg Entry Value"].round(5)
    summary["PXT Pass"] = summary["PXT Pass"].round(4)
    summary["Team xT"] = summary["Team xT"].round(4)

    return summary.sort_values(
        ["Entry Value", "Successful", "Entries", "Success %"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)[columns]


def lane_summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Summarise entry count and value by start/end lane."""
    columns = ["Start Lane", "End Lane", "Entries", "Entry Value", "Avg Entry Value"]
    if entries.empty or not {"Start Lane", "End Lane"}.issubset(entries.columns):
        return pd.DataFrame(columns=columns)

    values = entries.copy()
    values["Start Lane"] = values["Start Lane"].fillna("Unknown").astype(str)
    values["End Lane"] = values["End Lane"].fillna("Unknown").astype(str)
    values["_Entry Value"] = pd.to_numeric(values.get("_Entry Value", entry_value(values)), errors="coerce").fillna(0)

    summary = values.groupby(["Start Lane", "End Lane"], as_index=False).agg(
        Entries=("End Lane", "size"),
        **{
            "Entry Value": ("_Entry Value", "sum"),
            "Avg Entry Value": ("_Entry Value", "mean"),
        },
    )
    for column in ["Entry Value", "Avg Entry Value"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(2)
    return summary.sort_values(["Entries", "Entry Value"], ascending=False).reset_index(drop=True)[columns]


def outcome_summary(entries: pd.DataFrame) -> pd.DataFrame:
    """Summarise entries by outcome bucket."""
    columns = ["Outcome", "Entries", "Entry Value", "Avg Entry Value"]
    if entries.empty:
        return pd.DataFrame(columns=columns)

    values = entries.copy()
    values["_Outcome"] = values["_Outcome"] if "_Outcome" in values else result_status(values.get("Result", pd.Series("", index=values.index)))
    values["_Entry Value"] = pd.to_numeric(values.get("_Entry Value", entry_value(values)), errors="coerce").fillna(0)

    summary = values.groupby("_Outcome", as_index=False).agg(
        Entries=("_Outcome", "size"),
        **{
            "Entry Value": ("_Entry Value", "sum"),
            "Avg Entry Value": ("_Entry Value", "mean"),
        },
    )
    summary = summary.rename(columns={"_Outcome": "Outcome"})
    for column in ["Entry Value", "Avg Entry Value"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(2)
    return summary.sort_values("Entries", ascending=False).reset_index(drop=True)[columns]


def _mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[~clean.str.lower().isin(["", "nan", "none", "null"])]
    if clean.empty:
        return ""
    mode = clean.mode()
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0])
