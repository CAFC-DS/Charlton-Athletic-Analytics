"""Authoritative Snowflake relations used by the Charlton application.

The connection defaults to ``CAFC_DB.PUBLIC`` but the production data does not
live in ``PUBLIC``.  Keeping fully-qualified relation names here prevents a
connection-default change from silently redirecting application queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


CAFC_DATABASE: Final = "CAFC_DB"


@dataclass(frozen=True)
class SourceSpec:
    schema: str
    name: str
    provider: str
    layer: str

    @property
    def sql(self) -> str:
        return f'"{CAFC_DATABASE}"."{self.schema}"."{self.name}"'


SOURCES: Final[dict[str, SourceSpec]] = {
    # Typed Impect staging dimensions and facts.
    "impect_iterations": SourceSpec("IMPECT_RAW_STAGING", "STG_IMPECT__ITERATIONS", "Impect", "staging"),
    "impect_players": SourceSpec("IMPECT_RAW_STAGING", "STG_IMPECT__PLAYERS", "Impect", "staging"),
    "impect_squads": SourceSpec("IMPECT_RAW_STAGING", "STG_IMPECT__SQUADS", "Impect", "staging"),
    "impect_matches": SourceSpec("IMPECT_RAW_STAGING", "STG_IMPECT__MATCHES", "Impect", "staging"),
    "impect_iteration_player_kpis": SourceSpec(
        "IMPECT_RAW_STAGING", "STG_IMPECT__ITERATION_PLAYER_KPIS", "Impect", "staging"
    ),
    "impect_iteration_squad_kpis": SourceSpec(
        "IMPECT_RAW_STAGING", "STG_IMPECT__ITERATION_SQUAD_KPIS", "Impect", "staging"
    ),
    "impect_match_player_kpis": SourceSpec(
        "IMPECT_RAW_STAGING", "STG_IMPECT__CHAMPIONSHIP_PLAYER_KPIS", "Impect", "staging"
    ),
    "impect_match_squad_kpis": SourceSpec(
        "IMPECT_RAW_STAGING", "STG_IMPECT__CHAMPIONSHIP_SQUAD_KPIS", "Impect", "staging"
    ),
    "impect_match_info": SourceSpec(
        "IMPECT_RAW_STAGING", "STG_IMPECT__CHAMPIONSHIP_MATCH_INFO", "Impect", "staging"
    ),
    # Raw source records needed where no production flattened model exists.
    "impect_events": SourceSpec("IMPECT_RAW", "EVENTS", "Impect", "raw"),
    "impect_kpi_glossary": SourceSpec("IMPECT_RAW", "KPI_GLOSSARY", "Impect", "raw"),
    "impect_countries": SourceSpec("IMPECT_RAW", "COUNTRIES", "Impect", "raw"),
    # Canonical identities and KPI facts.
    "core_players": SourceSpec("CORE", "PLAYERS", "Canonical", "core"),
    "core_squads": SourceSpec("CORE", "CORE_SQUADS", "Canonical", "core"),
    "core_seasons": SourceSpec("CORE", "CORE_SEASONS", "Canonical", "core"),
    "core_fixtures": SourceSpec("CORE", "FIXTURES", "Canonical", "core"),
    "core_player_iteration_kpis": SourceSpec("CORE", "CORE_PLAYER_ITERATION_KPIS", "Canonical", "core"),
    "core_player_fixture_kpis": SourceSpec("CORE", "CORE_PLAYER_FIXTURE_KPIS", "Canonical", "core"),
    "core_squad_iteration_kpis": SourceSpec("CORE", "CORE_SQUAD_ITERATION_KPIS", "Canonical", "core"),
    # Immutable DVMS ingestion is the provenance source for Opta files.
    "opta_fixtures_raw": SourceSpec("DVMS_RAW", "FIXTURES", "Opta", "raw"),
    "opta_assets_raw": SourceSpec("DVMS_RAW", "ASSETS", "Opta", "raw"),
    # These CAFC_DB views parse the live DVMS F24/F7 payloads. Other objects in
    # the SCOUT_TOOL schema (including the scouting snapshot) are intentionally
    # not registered and therefore cannot be queried through this adapter.
    "opta_events_staging": SourceSpec("SCOUT_TOOL", "STG_OPTA_F24_EVENTS_V1", "Opta", "staging"),
    "opta_event_qualifiers_staging": SourceSpec(
        "SCOUT_TOOL", "STG_OPTA_F24_EVENT_QUALIFIERS_V1", "Opta", "staging"
    ),
    "opta_rosters_staging": SourceSpec("SCOUT_TOOL", "STG_OPTA_F7_ROSTERS_V1", "Opta", "staging"),
    "opta_lineups_staging": SourceSpec("SCOUT_TOOL", "STG_OPTA_F7_LINEUPS_V1", "Opta", "staging"),
    "opta_teams_staging": SourceSpec("SCOUT_TOOL", "STG_OPTA_F7_TEAMS_V1", "Opta", "staging"),
}

_ALLOWED_PRODUCTION_SCHEMAS: Final = {
    "IMPECT_RAW_STAGING",
    "IMPECT_RAW",
    "CORE",
    "DVMS_RAW",
    "SCOUT_TOOL",
}
_invalid_sources = {
    key: spec.schema for key, spec in SOURCES.items() if spec.schema not in _ALLOWED_PRODUCTION_SCHEMAS
}
if _invalid_sources:
    raise RuntimeError(f"Non-production sources registered: {_invalid_sources}")

_invalid_scout_tool_sources = {
    key: spec.name
    for key, spec in SOURCES.items()
    if spec.schema == "SCOUT_TOOL" and not spec.name.startswith("STG_OPTA_")
}
if _invalid_scout_tool_sources:
    raise RuntimeError(f"Non-staging SCOUT_TOOL sources registered: {_invalid_scout_tool_sources}")


def relation(key: str) -> str:
    """Return an allow-listed, fully-qualified Snowflake identifier."""
    try:
        return SOURCES[key].sql
    except KeyError as exc:
        raise KeyError(f"Unknown CAFC data source: {key}") from exc


def production_source_manifest() -> tuple[dict[str, str], ...]:
    """Small provenance manifest suitable for diagnostics and Data Hub pages."""
    return tuple(
        {
            "key": key,
            "provider": spec.provider,
            "layer": spec.layer,
            "relation": spec.sql,
        }
        for key, spec in SOURCES.items()
    )
