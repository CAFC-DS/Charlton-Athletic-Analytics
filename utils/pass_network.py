"""Pure player-scope helpers for passer-to-receiver networks."""

from __future__ import annotations

from collections.abc import Iterable
import re
import unicodedata

import pandas as pd


def player_key(value: object) -> str:
    """Return a comparison key that is stable across punctuation and accents."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _name_tokens(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.findall(r"[a-z0-9]+", text)


def same_team_label(candidate: object, target: object) -> bool:
    candidate_key = player_key(candidate)
    target_key = player_key(target)
    if not candidate_key or not target_key:
        return False
    return (
        candidate_key == target_key
        or candidate_key in target_key
        or target_key in candidate_key
    )


def same_player_label(candidate: object, target: object) -> bool:
    """Safely reconcile common provider variants such as Dan/Danny McNamara.

    Exact normalized names are preferred. A relaxed match is only accepted when
    the surname is identical and one given name is a meaningful prefix of the
    other; callers additionally require that this identifies one network player.
    """
    if player_key(candidate) == player_key(target) and player_key(candidate):
        return True

    candidate_parts = _name_tokens(candidate)
    target_parts = _name_tokens(target)
    if len(candidate_parts) < 2 or len(target_parts) < 2:
        return False
    if candidate_parts[-1] != target_parts[-1]:
        return False

    candidate_given = candidate_parts[0]
    target_given = target_parts[0]
    return (
        min(len(candidate_given), len(target_given)) >= 3
        and (
            candidate_given.startswith(target_given)
            or target_given.startswith(candidate_given)
        )
    )


def network_player_names(network: pd.DataFrame) -> set[str]:
    """Return normalized names represented by either endpoint of a link."""
    names: set[str] = set()
    for column in ["Player", "Receiver"]:
        if column in network:
            names.update(key for key in network[column].map(player_key) if key)
    return names


def network_player_count(network: pd.DataFrame) -> int:
    """Count the players a network plots, by provider id rather than name.

    The plot draws one node per player id, so counting names would understate
    it whenever the dimension could not name a player.
    """
    id_columns = [column for column in ["PlayerId", "ReceiverId"] if column in network]
    if not id_columns:
        return len(network_player_names(network))
    ids = pd.concat(
        [pd.to_numeric(network[column], errors="coerce") for column in id_columns],
        ignore_index=True,
    )
    return int(ids.dropna().nunique())


def resolve_network_name_keys(
    source_names: Iterable[object],
    network: pd.DataFrame,
) -> set[str]:
    """Map external lineup names to the unique matching names in a network."""
    network_names: dict[str, str] = {}
    for column in ["Player", "Receiver"]:
        if column not in network:
            continue
        for value in network[column].dropna():
            key = player_key(value)
            if key:
                network_names.setdefault(key, str(value))

    resolved: set[str] = set()
    for source_name in source_names:
        source_key = player_key(source_name)
        if source_key in network_names:
            resolved.add(source_key)
            continue
        compatible = {
            key
            for key, network_name in network_names.items()
            if same_player_label(source_name, network_name)
        }
        if len(compatible) == 1:
            resolved.update(compatible)
    return resolved


def fallback_starting_names(team_passes: pd.DataFrame) -> set[str]:
    """Infer an XI from earliest event involvement when no lineup is available."""
    if team_passes.empty or "Player" not in team_passes:
        return set()
    candidates = team_passes.dropna(subset=["Player"]).copy()
    if candidates.empty:
        return set()
    candidates["_Second"] = pd.to_numeric(candidates.get("Second"), errors="coerce")
    grouped = candidates.groupby("Player", as_index=False).agg(
        **{
            "First Second": ("_Second", "min"),
            "Passes": ("Player", "size"),
        }
    )
    grouped["First Second"] = grouped["First Second"].fillna(999999)
    grouped = grouped.sort_values(
        ["First Second", "Passes", "Player"],
        ascending=[True, False, True],
    ).head(11)
    return {player_key(name) for name in grouped["Player"] if player_key(name)}


def top_network_passer_names(network: pd.DataFrame) -> set[str]:
    """Return the eleven highest-volume passers in the supplied network."""
    if network.empty or "Player" not in network or "Pass Count" not in network:
        return set()
    volume = network[["Player", "Pass Count"]].copy()
    volume["Pass Count"] = pd.to_numeric(
        volume["Pass Count"], errors="coerce"
    ).fillna(0)
    volume["Player Key"] = volume["Player"].map(player_key)
    volume = (
        volume[volume["Player Key"].ne("")]
        .groupby(["Player Key", "Player"], as_index=False)["Pass Count"]
        .sum()
        .sort_values(["Pass Count", "Player"], ascending=[False, True])
        .head(11)
    )
    return set(volume["Player Key"])


def filter_network(network: pd.DataFrame, selected_names: set[str]) -> pd.DataFrame:
    """Keep only links where both endpoints belong to the selected scope."""
    if network.empty or not selected_names:
        return network.iloc[0:0].copy()
    passer_keys = (
        network["Player"].map(player_key)
        if "Player" in network
        else pd.Series(False, index=network.index)
    )
    receiver_keys = (
        network["Receiver"].map(player_key)
        if "Receiver" in network
        else pd.Series(False, index=network.index)
    )
    return network[
        passer_keys.isin(selected_names) & receiver_keys.isin(selected_names)
    ].copy()
