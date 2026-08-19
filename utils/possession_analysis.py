"""Build-up possession-sequence analysis shared by the team- and player-level pages.

A build-up sequence is a possession-sequence (one Sequence Index) whose first
in-possession action starts outside the final third -- i.e. the team has to
build the attack rather than already being there. Involvement credits every
player who touches the ball in that sequence, either as the acting player or
as a pass receiver, not just the passer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import pitch


BUILD_UP_START_X = pitch.FINAL_THIRD_X  # sequences starting beyond this are already in the final third
# Below this many total build-up sequences, involvement % naturally clusters
# into a handful of 1-in-N values (e.g. 1 of 8 sequences is always 12.5%),
# which makes ranking/trend charts look flat -- not a bug, just too small a
# sample to differentiate players. Callers should warn rather than silently
# show it as a meaningful ranking.
MIN_SEQUENCES_FOR_RANKING = 30


def _player_id_key(values: pd.Series) -> pd.Series:
    """Normalise a PlayerId series to a common string key regardless of source dtype.

    Different loaders -- and even different columns within the same events
    frame -- return PlayerId as int32, float64 or str (e.g. PlayerId is
    int32 with no nulls, but ReceiverId is nullable and comes back float64;
    concatenating the two upcasts the whole column to float64). A naive
    ``.astype(str)`` comparison silently fails whenever one side is "25012"
    and the other is "25012.0".
    """
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)


def _player_id_key_scalar(player_id: object) -> str:
    """Normalise a single PlayerId value the same way as ``_player_id_key``."""
    return _player_id_key(pd.Series([player_id])).iloc[0]


def numeric(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def sequence_keys(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (MatchId, Sequence Index) with ordering, start location, minute and phase of the first event."""
    working = events.dropna(subset=["MatchId", "Sequence Index"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["MatchId", "Sequence Index", "_Order"])
    working["_Order"] = numeric(working, "Event Number")
    return working.sort_values("_Order").groupby(["MatchId", "Sequence Index"], as_index=False).first()


def buildup_sequence_keys(events: pd.DataFrame) -> pd.DataFrame:
    """Sequences whose first event is an in-possession action starting outside the final third."""
    first = sequence_keys(events)
    if first.empty:
        return first
    first["Start X"] = numeric(first, "Start X", np.nan)
    starts_deep = pd.to_numeric(first["Start X"], errors="coerce").lt(BUILD_UP_START_X)
    in_possession = first["Phase"].fillna("").astype(str).eq("IN_POSSESSION")
    return first[starts_deep & in_possession].copy()


def sequence_outcomes(events: pd.DataFrame, buildup_keys: pd.DataFrame) -> pd.DataFrame:
    """Funnel counts: build-up sequences -> reached final third -> produced a shot -> produced a goal."""
    if buildup_keys.empty:
        return pd.DataFrame(columns=["Stage", "Sequences", "Conversion %"])
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(keys, on=["MatchId", "Sequence Index"], how="inner")
    action_type = working["Action Type"].fillna("").astype(str).str.upper()
    action = working["Action"].fillna("").astype(str).str.upper()
    start_x = numeric(working, "Start X", np.nan)
    end_x = numeric(working, "End X", np.nan)
    working["_Reached Final Third"] = start_x.ge(BUILD_UP_START_X) | end_x.ge(BUILD_UP_START_X)
    working["_Shot"] = action_type.eq("SHOT")
    working["_Goal"] = action.eq("GOAL")
    working["_Shot xG"] = numeric(working, "Shot xG").where(working["_Shot"], 0)
    per_sequence = working.groupby(["MatchId", "Sequence Index"], as_index=False).agg(
        **{
            "Reached Final Third": ("_Reached Final Third", "max"),
            "Shot": ("_Shot", "max"),
            "Goal": ("_Goal", "max"),
            "Shot xG": ("_Shot xG", "sum"),
        }
    )
    per_sequence["Reached Final Third"] = per_sequence[["Reached Final Third", "Shot"]].max(axis=1)
    per_sequence["Shot"] = per_sequence[["Shot", "Goal"]].max(axis=1)
    total = len(per_sequence)
    stages = ["Build-Up Sequences", "Reached Final Third", "Produced a Shot", "Produced a Goal"]
    counts = [
        total,
        int(per_sequence["Reached Final Third"].sum()),
        int(per_sequence["Shot"].sum()),
        int(per_sequence["Goal"].sum()),
    ]
    summary = pd.DataFrame({"Stage": stages, "Sequences": counts})
    summary["Conversion %"] = summary["Sequences"].div(max(total, 1)).mul(100)
    return summary


def _successful(working: pd.DataFrame) -> pd.Series:
    """Rows where the action actually succeeded.

    Impect still populates the Receiver field on a *failed* pass with
    whoever touched the loose ball next -- which is very often the opposing
    player who intercepted it, not a teammate. Crediting that as this team's
    build-up involvement would attribute an opponent's interception to us,
    so receiver credit is restricted to successful actions only.
    """
    if "Result" not in working:
        return pd.Series(False, index=working.index)
    return working["Result"].astype(str).str.upper().eq("SUCCESS")


def _touches(events: pd.DataFrame, buildup_keys: pd.DataFrame, extra_cols: list[str]) -> pd.DataFrame:
    """Long table: one row per (sequence, extra_cols..., PlayerId) a player touched the ball in."""
    keys = buildup_keys[["MatchId", "Sequence Index", *extra_cols]].drop_duplicates(["MatchId", "Sequence Index"])
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(
        keys, on=["MatchId", "Sequence Index"], how="inner", suffixes=("", "_key")
    )
    actor = working[["MatchId", "Sequence Index", *extra_cols, "PlayerId"]].dropna(subset=["PlayerId"])
    receiver = (
        working.loc[_successful(working), ["MatchId", "Sequence Index", *extra_cols, "ReceiverId"]]
        .dropna(subset=["ReceiverId"])
        .rename(columns={"ReceiverId": "PlayerId"})
    )
    touches = pd.concat([actor, receiver], ignore_index=True)
    touches["PlayerId"] = _player_id_key(touches["PlayerId"])
    return touches.drop_duplicates(["MatchId", "Sequence Index", "PlayerId"])


def player_involvement(events: pd.DataFrame, buildup_keys: pd.DataFrame, minutes_lookup: pd.DataFrame) -> pd.DataFrame:
    """Every player's involvement % across the whole loaded window (one row per player)."""
    if buildup_keys.empty:
        return pd.DataFrame()
    total_sequences = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates().shape[0]
    if total_sequences == 0:
        return pd.DataFrame()
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(keys, on=["MatchId", "Sequence Index"], how="inner")

    actor = working[["MatchId", "Sequence Index", "PlayerId", "Player"]].dropna(subset=["Player"])
    receiver = (
        working.loc[_successful(working), ["MatchId", "Sequence Index", "ReceiverId", "Receiver"]]
        .dropna(subset=["Receiver"])
        .rename(columns={"ReceiverId": "PlayerId", "Receiver": "Player"})
    )
    touches = pd.concat([actor, receiver], ignore_index=True).drop_duplicates(["MatchId", "Sequence Index", "Player"])
    per_player = touches.groupby(["PlayerId", "Player"], as_index=False).agg(**{"Sequences Touched": ("MatchId", "size")})
    per_player["Build-Up Involvement %"] = per_player["Sequences Touched"] / total_sequences * 100

    if not minutes_lookup.empty:
        lookup = minutes_lookup.copy()
        lookup["PlayerId"] = _player_id_key(lookup["PlayerId"])
        per_player["_PlayerIdKey"] = _player_id_key(per_player["PlayerId"])
        per_player = per_player.merge(
            lookup[["PlayerId", "Minutes"]].rename(columns={"PlayerId": "_PlayerIdKey"}), on="_PlayerIdKey", how="left"
        ).drop(columns=["_PlayerIdKey"])
    return per_player.sort_values("Build-Up Involvement %", ascending=False).reset_index(drop=True)


def player_match_involvement(events: pd.DataFrame, buildup_keys: pd.DataFrame, player_id: object) -> pd.DataFrame:
    """One row per match: this player's build-up involvement % in that match, for a trend across a season."""
    columns = ["MatchId", "Sequences", "Sequences Touched", "Involvement %"]
    if buildup_keys.empty:
        return pd.DataFrame(columns=columns)
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    total_per_match = keys.groupby("MatchId").size().rename("Sequences")

    touches = _touches(events, buildup_keys, [])
    player_touches = touches[touches["PlayerId"].eq(_player_id_key_scalar(player_id))]
    touched_per_match = player_touches.groupby("MatchId").size().rename("Sequences Touched")

    result = total_per_match.to_frame().join(touched_per_match, how="left").fillna(0).reset_index()
    result["Involvement %"] = result["Sequences Touched"] / result["Sequences"].replace(0, np.nan) * 100
    return result[columns]


def player_time_window_involvement(
    events: pd.DataFrame, buildup_keys: pd.DataFrame, player_id: object, bucket_size: int = 15
) -> pd.DataFrame:
    """This player's build-up involvement %, broken into match-minute buckets across all loaded matches.

    Answers "which part of the match is this player most involved in build-up
    play" rather than a single season-aggregate number.
    """
    columns = ["Window", "_Order", "Sequences", "Sequences Touched", "Involvement %"]
    if buildup_keys.empty:
        return pd.DataFrame(columns=columns)
    keys = buildup_keys[["MatchId", "Sequence Index", "Minute"]].drop_duplicates(["MatchId", "Sequence Index"]).copy()
    keys["Minute"] = pd.to_numeric(keys["Minute"], errors="coerce").fillna(0).clip(lower=0, upper=119)
    bins = list(range(0, 121, bucket_size))
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    keys["Window"] = pd.cut(keys["Minute"], bins=bins, labels=labels, right=False, include_lowest=True)

    total_per_window = keys.groupby("Window", observed=True).size().rename("Sequences")

    touches = _touches(events, buildup_keys, [])
    player_touches = touches[touches["PlayerId"].eq(_player_id_key_scalar(player_id))]
    player_touches = player_touches.merge(keys[["MatchId", "Sequence Index", "Window"]], on=["MatchId", "Sequence Index"], how="inner")
    touched_per_window = player_touches.groupby("Window", observed=True).size().rename("Sequences Touched")

    result = total_per_window.to_frame().join(touched_per_window, how="left").fillna(0).reset_index()
    result["Involvement %"] = result["Sequences Touched"] / result["Sequences"].replace(0, np.nan) * 100
    result["_Order"] = result["Window"].map({label: index for index, label in enumerate(labels)})
    result = result[result["Sequences"] > 0].sort_values("_Order")
    return result[columns]


def player_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Each player's most common recorded position across the loaded events."""
    if events.empty or "Player" not in events:
        return pd.DataFrame(columns=["PlayerId", "Player", "Position"])
    working = events.dropna(subset=["Player"]).copy()
    working["Position"] = working["Position"].fillna("") if "Position" in working else ""

    def _mode(series: pd.Series) -> str:
        clean = series.astype(str).str.strip()
        clean = clean[clean.ne("")]
        mode = clean.mode()
        return str(mode.iloc[0]) if not mode.empty else "Unknown"

    return (
        working.groupby(["PlayerId", "Player"], as_index=False)["Position"]
        .agg(_mode)
        .rename(columns={"Position": "Position"})
    )


def player_touch_locations(events: pd.DataFrame, buildup_keys: pd.DataFrame, player_id: object) -> pd.DataFrame:
    """Start locations of this player's own actions (not receptions) within build-up sequences, for a pitch map."""
    columns = ["Start X", "Start Y", "Action", "Action Type", "Zone"]
    if buildup_keys.empty:
        return pd.DataFrame(columns=columns)
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(keys, on=["MatchId", "Sequence Index"], how="inner")
    acted = working[_player_id_key(working["PlayerId"]).eq(_player_id_key_scalar(player_id))].copy()
    acted["Start X"] = numeric(acted, "Start X", np.nan)
    acted["Start Y"] = numeric(acted, "Start Y", np.nan)
    acted = acted.dropna(subset=["Start X", "Start Y"])
    if acted.empty:
        return pd.DataFrame(columns=columns)
    acted["Zone"] = np.select(
        [acted["Start X"].lt(-BUILD_UP_START_X), acted["Start X"].lt(BUILD_UP_START_X)],
        ["Defensive Third", "Middle Third"],
        default="Final Third",
    )
    return acted[["Start X", "Start Y", "Action", "Action Type", "Zone"]]


def player_sequence_productivity(events: pd.DataFrame, buildup_keys: pd.DataFrame, player_id: object) -> pd.DataFrame:
    """Team-wide vs this-player's-touched-sequences conversion, side by side, for the same funnel stages."""
    team = sequence_outcomes(events, buildup_keys)
    if team.empty:
        return pd.DataFrame(columns=["Stage", "Team Conversion %", "Player Conversion %"])
    keys = buildup_keys[["MatchId", "Sequence Index"]].drop_duplicates()
    working = events.dropna(subset=["MatchId", "Sequence Index"]).merge(keys, on=["MatchId", "Sequence Index"], how="inner")
    pid = _player_id_key_scalar(player_id)
    acted = working[_player_id_key(working["PlayerId"]).eq(pid)][["MatchId", "Sequence Index"]].drop_duplicates()
    received = working.loc[
        _successful(working) & _player_id_key(working["ReceiverId"]).eq(pid), ["MatchId", "Sequence Index"]
    ].drop_duplicates()
    player_sequence_ids = pd.concat([acted, received], ignore_index=True).drop_duplicates()
    player_keys = buildup_keys.merge(player_sequence_ids, on=["MatchId", "Sequence Index"], how="inner")
    player = sequence_outcomes(events, player_keys)
    if player.empty:
        return pd.DataFrame(columns=["Stage", "Team Conversion %", "Player Conversion %"])
    combined = team.merge(player, on="Stage", suffixes=(" (Team)", " (Player)"))
    combined = combined.rename(
        columns={"Conversion % (Team)": "Team Conversion %", "Conversion % (Player)": "Player Conversion %"}
    )
    return combined[["Stage", "Team Conversion %", "Player Conversion %"]]
