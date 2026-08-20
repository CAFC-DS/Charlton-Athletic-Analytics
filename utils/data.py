# =============================================================================
# CENTRAL DATA ADAPTER
# =============================================================================
# Every page reads through this module. Production mode uses fully-qualified,
# allow-listed CAFC_DB relations registered in utils/data_sources.py:
#
#   - IMPECT_RAW_STAGING: typed dimensions and long-form KPI facts
#   - IMPECT_RAW: provider event payloads and reference data
#   - CORE: canonical identities and curated KPI facts
#   - DVMS_RAW: immutable Opta fixture and asset provenance
#
# PUBLIC and the scouting snapshot are deliberately not production sources.
# Only SCOUT_TOOL.STG_OPTA_* parser views proven to read DVMS_RAW are allowed.
# Provider data is adapted here into the stable contracts expected by views.
#
# Production is the default. Demo data is available only when explicitly set
# with CHARLTON_DATA_MODE=demo (or the legacy CHARLTON_USE_MOCK_DATA flag).
# =============================================================================

import csv
import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_sources import relation

# ---- DATA MODE ---------------------------------------------------------------
_configured_data_mode = os.getenv("CHARLTON_DATA_MODE", "").strip().lower()
if not _configured_data_mode:
    _legacy_mock_enabled = os.getenv("CHARLTON_USE_MOCK_DATA", "").lower() in {"1", "true", "yes"}
    _configured_data_mode = "demo" if _legacy_mock_enabled else "production"
if _configured_data_mode not in {"production", "demo"}:
    raise RuntimeError("CHARLTON_DATA_MODE must be either 'production' or 'demo'.")

DATA_MODE = _configured_data_mode
USE_MOCK_DATA = DATA_MODE == "demo"

# Charlton's stable Impect squad identity. It is used only to discover the
# league iterations that belong in this Charlton application; league-wide
# loaders still return every squad in the selected competition.
CHARLTON_IMPECT_SQUAD_ID = 959

PLAYER_KPI_IDS = {
    "Bypassed Opponents /90": 0,
    "Bypassed Defenders /90": 2,
    "Receiving Progression /90": 7,
    "Receiving Defenders Bypassed /90": 9,
    "Ball Loss Threat /90": 20,
    "Team-Mates Bypassed By Losses /90": 21,
    "Ball Losses /90": 22,
    "Ball Win Value /90": 24,
    "Ball Wins /90": 27,
    "Goals /90": 28,
    "Critical Ball Losses /90": 49,
    "Assists /90": 77,
    "xG /90": 82,
    "Packing xG /90": 83,
    "Dribble Progression /90": 87,
    "_successful_passes": 90,
    "_unsuccessful_passes": 91,
    "_won_ground_duels": 94,
    "_lost_ground_duels": 95,
    "_won_aerial_duels": 96,
    "_lost_aerial_duels": 97,
    "_shots_on_target": 100,
    "_shots_off_target": 101,
    "_bypassed_by_low_pass": 106,
    "_bypassed_by_diagonal_pass": 107,
    "_bypassed_by_chipped_pass": 108,
    "_bypassed_by_low_cross": 110,
    "_bypassed_by_high_cross": 111,
    "_successful_passes_to_final_third": 331,
    "_unsuccessful_passes_to_final_third": 392,
    "Post-Shot xG /90": 1401,
    "Neutral Passes /90": 1431,
    "Goals Conceded /90": 1460,
    "Post-Shot xG Faced /90": 1462,
    "Save Actions /90": 1517,
}

# KPI ids that are only meaningful for a goalkeeper position-stint. Used by
# load_players() to decide where a missing KPI row means "zero" vs "not
# applicable to this position" -- see the comment at that loop.
_GOALKEEPING_ONLY_KPI_IDS = {1460, 1462, 1517}  # Goals Conceded, Post-Shot xG Faced, Save Actions

TEAM_KPI_IDS = {
    "Bypassed Opponents /90": 0,
    "Goals /90": 28,
    "Assists /90": 77,
    "xG /90": 82,
    "Packing xG /90": 83,
    "Ball Win Value /90": 24,
    "Ball Wins /90": 27,
    "Dribble Progression /90": 87,
    "_successful_passes": 90,
    "_unsuccessful_passes": 91,
    "_shots_on_target": 100,
    "_shots_off_target": 101,
    "_successful_passes_to_final_third": 331,
    "_unsuccessful_passes_to_final_third": 392,
}

DEFENSIVE_SQUAD_KPI_IDS = {
    "Ball Wins": 27,
    "Ball Losses": 22,
    "Opponents Removed": 24,
    "Defenders Removed": 25,
    "Ball Win Value": 1409,
    "Defensive Touches": 93,
    "Presses": 1536,
    "Counterpresses": 1539,
    "Build-Up Presses": 1537,
    "Between-Lines Presses": 1538,
    "Second Balls": 1610,
    "Second Balls Won": 1611,
    "Ground Duels Won": 94,
    "Ground Duels Lost": 95,
    "Aerial Duels Won": 96,
    "Aerial Duels Lost": 97,
    "Suffered Bypassed Opponents": 39,
    "Suffered Bypassed Defenders": 40,
    "Goals Conceded": 43,
    "xG Conceded": 1463,
    "First-Third Ball Wins": 997,
    "Middle-Third Ball Wins": 998,
    "Final-Third Ball Wins": 999,
    "Opponent-Box Ball Wins": 1000,
    "Wide-Left Ball Wins": 1005,
    "Half-Left Ball Wins": 1004,
    "Centre Ball Wins": 1003,
    "Half-Right Ball Wins": 1002,
    "Wide-Right Ball Wins": 1001,
    "Out-of-Possession Ball Wins": 1006,
    "Defensive-Transition Ball Wins": 1007,
    "Set-Piece Ball Wins": 1010,
    "Second-Ball Phase Wins": 1011,
}

DEFENSIVE_PLAYER_KPI_IDS = {
    column: DEFENSIVE_SQUAD_KPI_IDS[column]
    for column in [
        "Ball Wins",
        "Ball Losses",
        "Opponents Removed",
        "Defenders Removed",
        "Ball Win Value",
        "Defensive Touches",
        "Presses",
        "Counterpresses",
        "Second Balls",
        "Second Balls Won",
        "Ground Duels Won",
        "Ground Duels Lost",
        "Aerial Duels Won",
        "Aerial Duels Lost",
    ]
}

# Impect's ACTION values for an open-play cross, both nested under ACTION_TYPE='PASS'.
CROSS_ACTIONS = {"LOW_CROSS", "HIGH_CROSS"}

# Event action types whose EVENT_KPIS carry a per-event PXT (packing xT) value --
# PXT_PASS on PASS/CLEARANCE, PXT_SHOT on SHOT. Used to scope event loads for
# expected-threat aggregation without pulling every action type.
XT_ACTION_TYPES = ["PASS", "SHOT", "CLEARANCE"]


def is_cross(frame: pd.DataFrame) -> pd.Series:
    """True where an event row is an open-play cross (Impect ACTION LOW_CROSS/HIGH_CROSS)."""
    if frame.empty or "Action" not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["Action"].astype(str).str.upper().isin(CROSS_ACTIONS)


def event_xt_value(frame: pd.DataFrame) -> pd.Series:
    """Per-event expected-threat value: signed PXT Pass (incl. clearances) plus PXT Shot.

    Both fields are Impect's packing-xT delta for that specific action -- the
    threat a player's pass/shot/clearance added or removed -- and are mutually
    exclusive by action type, so a plain fillna(0) sum does not double-count.
    """
    if frame.empty:
        return pd.Series(dtype="float64")
    pxt_pass = pd.to_numeric(frame["PXT Pass"], errors="coerce") if "PXT Pass" in frame else pd.Series(0.0, index=frame.index)
    pxt_shot = pd.to_numeric(frame["PXT Shot"], errors="coerce") if "PXT Shot" in frame else pd.Series(0.0, index=frame.index)
    return pxt_pass.fillna(0.0) + pxt_shot.fillna(0.0)


PLAYER_METRICS = ["Goals /90", "Assists /90", "Bypassed Opponents /90", "Pass %", "Passes to Final 3rd /90"]
PLAYER_PROFILE_METRICS = [
    "Goals /90",
    "Assists /90",
    "xG /90",
    "Post-Shot xG /90",
    "Shots /90",
    "Pass %",
    "Successful Passes /90",
    "Passes to Final 3rd /90",
    "Pass Progression /90",
    "Cross Progression /90",
    "Bypassed Opponents /90",
    "Bypassed Defenders /90",
    "Receiving Progression /90",
    "Dribble Progression /90",
    "Ball Wins /90",
    "Ball Win Value /90",
    "Ground Duel Win %",
    "Aerial Duel Win %",
    "Ball Losses /90",
    "Critical Ball Losses /90",
    "Ball Loss Threat /90",
    "Team-Mates Bypassed By Losses /90",
    "Neutral Passes /90",
    "Ball Security %",
    "Losses Per 100 Actions",
    "Goals Prevented /90",
    "Save Actions /90",
    "Post-Shot xG Faced /90",
    "Goals Conceded /90",
]
TEAM_METRICS = [
    "Goals /90",
    "Assists /90",
    "xG /90",
    "Packing xG /90",
    "Shots /90",
    "Bypassed Opponents /90",
    "Pass %",
    "Passes to Final 3rd /90",
    "Ball Wins /90",
    "Ball Win Value /90",
    "Dribble Progression /90",
]
MATCH_METRICS = ["Home Goals", "Away Goals"]
MATCH_COLUMNS = [
    "MatchId",
    "Date",
    "Competition",
    "Season",
    "Home",
    "Away",
    "Home Goals",
    "Away Goals",
    "Venue Verified",
    "Match",
    "Result",
]
PLAYER_COLUMNS = list(
    dict.fromkeys(
        [
            "PlayerId",
            "Player",
            "First Name",
            "Last Name",
            "Team",
            "Position",
            "Birthdate",
            "Nationality",
            "Foot",
            "Minutes",
            "Match Share",
            "Season",
            "Competition",
            *PLAYER_PROFILE_METRICS,
            "Packing xG /90",
            "Receiving Defenders Bypassed /90",
        ]
    )
)
TEAM_COLUMNS = ["Team", "Season", "Competition", *TEAM_METRICS]
MATCH_EVENT_COLUMNS = [
    "MatchId",
    "Season",
    "Date",
    "Competition",
    "Home",
    "Away",
    "Team",
    "PlayerId",
    "Player",
    "Position",
    "Period",
    "Game Time",
    "Second",
    "Minute",
    "Event Number",
    "Sequence Index",
    "Phase",
    "Action Type",
    "Action",
    "Body Part",
    "Result",
    "Pressure",
    "Start X",
    "Start Y",
    "End X",
    "End Y",
    "Raw Start X",
    "Raw Start Y",
    "Raw End X",
    "Raw End Y",
    "Start Lane",
    "End Lane",
    "Start Pitch Position",
    "End Pitch Position",
    "ReceiverId",
    "Receiver",
    "Pass Distance",
    "Pass Angle",
    "Team xT",
    "PXT Pass",
    "PXT Shot",
    "Shot xG",
    "Post-Shot xG",
    "Packing xG",
    "Bypassed Opponents",
    "Bypassed Defenders",
    "Shot Distance",
    "Shot Angle",
    "Shot Target Y",
    "Shot Target Z",
    "Shot GK X",
    "Shot GK Y",
    "Set Piece",
    "Set Piece Category",
    "Set Piece Execution",
]
SET_PIECE_SEQUENCE_COLUMNS = [
    "MatchId",
    "Season",
    "Date",
    "Competition",
    "Home",
    "Away",
    "Game Second",
    "Event Number",
    "Team",
    "Opponent",
    "Set Piece ID",
    "Category",
    "Adjusted Category",
    "Set Piece Type",
    "Side",
    "Execution Type",
    "Start X",
    "Start Y",
    "End X",
    "End Y",
    "Start Zone",
    "Corner End Zone",
    "Corner Type",
    "Free Kick End Zone",
    "Free Kick Type",
    "Main Event Action Type",
    "Main Event Action",
    "Taker Id",
    "Taker",
    "Main Event Outcome",
    "First Touch Player Id",
    "First Touch Player",
    "First Touch Won",
    "Indirect Header",
    "Second Touch Player Id",
    "Second Touch Player",
    "Second Touch Won",
    "Next Team",
    "Long Throw",
    "Retained",
    "Shots",
    "Goals",
    "xG",
    "Second-Phase Shots",
    "Second-Phase Goals",
    "Second-Phase xG",
]
SET_PIECE_EVENT_COLUMNS = [
    "MatchId",
    "Season",
    "Date",
    "Competition",
    "Home",
    "Away",
    "Team",
    "PlayerId",
    "Player",
    "Period",
    "Game Time",
    "Second",
    "Minute",
    "Event Number",
    "Sequence Index",
    "Action Type",
    "Action",
    "Body Part",
    "Result",
    "Start X",
    "Start Y",
    "End X",
    "End Y",
    "Shot xG",
    "PXT Set Piece",
    "Opponent PXT Set Piece",
    "Defensive PXT Set Piece",
    "Set Piece ID",
    "Set Piece Phase Index",
    "Category",
    "Adjusted Category",
    "Execution Type",
    "Subphase ID",
    "Subphase Index",
    "Start Zone",
    "Corner End Zone",
    "Corner Type",
    "Free Kick End Zone",
    "Free Kick Type",
    "Main Event",
    "Main Event Player Id",
    "Main Event Player",
    "Main Event Outcome",
    "Pass Receiver Id",
    "Pass Receiver",
    "First Touch Player Id",
    "First Touch Player",
    "First Touch Won",
    "Indirect Header",
    "Second Touch Player Id",
    "Second Touch Player",
    "Second Touch Won",
]
PASS_NETWORK_COLUMNS = [
    "MatchId",
    "Team",
    "PlayerId",
    "Player",
    "ReceiverId",
    "Receiver",
    "Pass Count",
    "Passer X",
    "Passer Y",
    "Receiver X",
    "Receiver Y",
]
MATCH_PLAYER_MINUTE_COLUMNS = [
    "MatchId",
    "Season",
    "Team",
    "PlayerId",
    "Player",
    "Position",
    "Minutes",
    "Match Share",
]
OPTA_EVENT_COLUMNS = [
    "FixtureId",
    "Opta Match Id",
    "Season",
    "Date",
    "Home",
    "Away",
    "EventId",
    "Provider Event Row Id",
    "TeamId",
    "Team",
    "PlayerId",
    "Player",
    "TypeId",
    "Period",
    "Minute",
    "Second",
    "Start X",
    "Start Y",
    "Outcome",
    "Is Key Pass",
    "Is Assist",
    "Event At UTC",
]
OPTA_LINEUP_COLUMNS = [
    "FixtureId",
    "TeamId",
    "Team",
    "PlayerId",
    "Player",
    "First Name",
    "Last Name",
    "Registered Position",
    "Lineup Status",
    "Position Group",
    "Sub Position",
    "Shirt Number",
    "Formation Place",
    "Is Captain",
]
OPTA_FORMATION_COLUMNS = [
    "FixtureId",
    "TeamId",
    "Side",
    "Formation",
    "Average Age",
]
TRACKING_POSSESSION_COLUMNS = [
    "FixtureId",
    "Date",
    "Home",
    "Away",
    "Effective Playing Time",
    "Home EPT",
    "Away EPT",
    "Home Possession %",
    "Away Possession %",
    "Provider",
    "Loaded At",
]
OPTA_QUALIFIER_COLUMNS = [
    "FixtureId",
    "Opta Match Id",
    "EventId",
    "Provider Event Row Id",
    "QualifierId",
    "Provider Qualifier Row Id",
    "Qualifier Value",
]

# Trusted match-level defensive totals from Impect's squad/player KPI facts.
# These are deliberately separate from the iteration-average loaders:
# the defensive page needs what happened in each match, not an average of
# pre-aggregated season rows.
DEFENSIVE_SQUAD_MATCH_COLUMNS = [
    "MatchId",
    "Date",
    "Competition",
    "Season",
    "TeamId",
    "Team",
    "Ball Wins",
    "Ball Losses",
    "Opponents Removed",
    "Defenders Removed",
    "Ball Win Value",
    "Defensive Touches",
    "Presses",
    "Counterpresses",
    "Build-Up Presses",
    "Between-Lines Presses",
    "Second Balls",
    "Second Balls Won",
    "Ground Duels Won",
    "Ground Duels Lost",
    "Aerial Duels Won",
    "Aerial Duels Lost",
    "Suffered Bypassed Opponents",
    "Suffered Bypassed Defenders",
    "Goals Conceded",
    "xG Conceded",
    "First-Third Ball Wins",
    "Middle-Third Ball Wins",
    "Final-Third Ball Wins",
    "Opponent-Box Ball Wins",
    "Wide-Left Ball Wins",
    "Half-Left Ball Wins",
    "Centre Ball Wins",
    "Half-Right Ball Wins",
    "Wide-Right Ball Wins",
    "Out-of-Possession Ball Wins",
    "Defensive-Transition Ball Wins",
    "Set-Piece Ball Wins",
    "Second-Ball Phase Wins",
]

DEFENSIVE_PLAYER_MATCH_COLUMNS = [
    "MatchId",
    "Date",
    "Competition",
    "Season",
    "TeamId",
    "Team",
    "PlayerId",
    "Player",
    "Position",
    "Play Duration Seconds",
    "Match Share",
    "Ball Wins",
    "Ball Losses",
    "Opponents Removed",
    "Defenders Removed",
    "Ball Win Value",
    "Defensive Touches",
    "Presses",
    "Counterpresses",
    "Second Balls",
    "Second Balls Won",
    "Ground Duels Won",
    "Ground Duels Lost",
    "Aerial Duels Won",
    "Aerial Duels Lost",
]


# ---- SNOWFLAKE CONNECTION ----------------------------------------------------
def _inline_private_key_der(pem_base64: str, password: str | None) -> bytes:
    """Decode a base64-wrapped PEM private key into DER bytes.

    Lets a deployment target with no local filesystem for a .p8 file (e.g.
    Streamlit Community Cloud) supply the key inline via secrets instead of
    a file path.
    """
    import base64

    from cryptography.hazmat.primitives import serialization

    pem_bytes = base64.b64decode(pem_base64)
    password_bytes = password.encode() if password else None
    private_key = serialization.load_pem_private_key(pem_bytes, password=password_bytes)
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_connection():
    """Streamlit's built-in Snowflake connection.

    Reads its settings from .streamlit/secrets.toml under [connections.snowflake].
    st.connection caches the connection, so this is cheap to call repeatedly.

    Streamlit-in-Snowflake never has a secrets.toml (it uses the native
    session instead), and merely touching st.secrets there raises
    StreamlitSecretNotFoundError -- so that lookup is caught, not just
    guarded with hasattr. Outside it (local dev, Streamlit Community
    Cloud), [connections.snowflake] normally points private_key_file at a
    .p8 file on disk. When that file isn't available -- a secrets-only
    deployment target -- a base64-encoded PEM key can be supplied instead
    under private_key_base64, decoded here and passed straight through.
    """
    try:
        snowflake_secrets = dict(st.secrets.get("connections", {}).get("snowflake", {}))
    except Exception:
        snowflake_secrets = {}
    inline_key = snowflake_secrets.get("private_key_base64")
    if inline_key:
        der_key = _inline_private_key_der(inline_key, snowflake_secrets.get("private_key_file_pwd"))
        return st.connection("snowflake", private_key=der_key)
    return st.connection("snowflake")


def data_source_preflight() -> pd.DataFrame:
    """Verify that the connection can see the authoritative CAFC source layers.

    This deliberately checks fully-qualified relations rather than relying on
    the connection's default schema (``PUBLIC`` is empty in CAFC_DB).
    """
    if USE_MOCK_DATA:
        return pd.DataFrame(
            [{"Capability": "demo", "Available": True, "Rows Found": None}]
        )

    checks = {
        "Impect dimensions": "impect_iterations",
        "Impect events": "impect_events",
        "Canonical identities": "core_players",
        "Opta fixtures": "opta_fixtures_raw",
        "Opta assets": "opta_assets_raw",
        "Opta parsed events": "opta_events_staging",
    }
    frames: list[dict[str, object]] = []
    conn = get_connection()
    for capability, source_key in checks.items():
        try:
            if source_key == "opta_events_staging":
                probe_sql = f"""
                    SELECT TRUE AS AVAILABLE
                    FROM {relation(source_key)}
                    WHERE FIXTURE_ID = (
                        SELECT FIXTURE_ID
                        FROM {relation("opta_assets_raw")}
                        WHERE ASSET_TYPE = 2
                          AND ASSET_SUBTYPE = 20
                          AND RAW_PAYLOAD IS NOT NULL
                        LIMIT 1
                    )
                    LIMIT 1
                """
            else:
                probe_sql = f"SELECT TRUE AS AVAILABLE FROM {relation(source_key)} LIMIT 1"
            probe = conn.query(
                probe_sql,
                ttl="5m",
            )
            available = bool(probe["AVAILABLE"].iloc[0]) if not probe.empty else False
            frames.append({"Capability": capability, "Available": available, "Source": source_key})
        except Exception:
            frames.append({"Capability": capability, "Available": False, "Source": source_key})
    return pd.DataFrame(frames)


def _snowflake_param(value: object) -> object:
    """Convert Pandas/NumPy scalar values into native Python types for binding.

    The Snowflake connector rejects values such as numpy.int32 when using qmark
    parameters. Match selectors often pull IDs from Pandas rows, so normalising
    here keeps every data loader safe.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _snowflake_params(params: list[object] | tuple[object, ...] | None) -> list[object] | None:
    return [_snowflake_param(value) for value in params] if params else None


def _season_key(value: object) -> str:
    """Normalise display aliases such as 2025/26 and 25-26 to Impect's 25/26.

    Handles various formats: 2026/27, 26/27, 2026/2027, 2026-27, 26_27.
    """
    text = str(value or "").strip().replace("_", "/").replace("-", "/")
    parts = text.split("/")
    if len(parts) == 2:
        first = parts[0].lstrip("0") or "0"
        second = parts[1].lstrip("0") or "0"
        # Normalise to 2-digit format: 26/27
        if len(first) >= 2:
            first = first[-2:]
        if len(second) >= 2:
            second = second[-2:]
        return f"{first}/{second}"
    return text


def _league_contexts() -> pd.DataFrame:
    """League competitions containing Charlton, resolved from CAFC_DB itself.

    The two small queries are intentionally separate. This lets Snowflake prune
    the squad dimension before the iteration lookup and avoids mixing the many
    competitions that share a season label such as ``25/26``.
    """
    columns = ["IterationId", "Season", "Competition"]
    if USE_MOCK_DATA:
        return pd.DataFrame(
            [{"IterationId": 1410, "Season": "2025/26", "Competition": "Championship"}],
            columns=columns,
        )

    conn = get_connection()
    squad_iterations = conn.query(
        f"""
        SELECT DISTINCT ITERATION_ID AS "IterationId"
        FROM {relation("impect_squads")}
        WHERE IMPECT_SQUAD_ID = ?
        """,
        params=_snowflake_params([CHARLTON_IMPECT_SQUAD_ID]),
        ttl="6h",
    )
    if squad_iterations.empty:
        return pd.DataFrame(columns=columns)

    iteration_ids = (
        pd.to_numeric(squad_iterations["IterationId"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )
    if not iteration_ids:
        return pd.DataFrame(columns=columns)

    placeholders = ", ".join(["?"] * len(iteration_ids))
    contexts = conn.query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            SEASON AS "Season",
            COMPETITION_NAME AS "Competition",
            COMPETITION_TYPE AS "CompetitionType"
        FROM {relation("impect_iterations")}
        WHERE ITERATION_ID IN ({placeholders})
          AND (UPPER(COMPETITION_TYPE) IN ('LEAGUE', 'CHAMPIONSHIP', 'NATIONAL_LEAGUE', '')
               OR COMPETITION_TYPE IS NULL)
        ORDER BY SEASON, COMPETITION_NAME\
        """,
        params=_snowflake_params(iteration_ids),
        ttl="6h",
    )
    if contexts.empty:
        return pd.DataFrame(columns=columns)
    contexts["IterationId"] = pd.to_numeric(contexts["IterationId"], errors="coerce").astype("Int64")
    contexts["Season"] = contexts["Season"].astype(str)
    return contexts[columns].drop_duplicates().reset_index(drop=True)


def _contexts_for_season(season: str | None) -> pd.DataFrame:
    contexts = _league_contexts()
    if contexts.empty:
        return contexts
    if season is None:
        latest = contexts["Season"].astype(str).sort_values().iloc[-1]
        return contexts[contexts["Season"].astype(str).eq(latest)].reset_index(drop=True)
    wanted = _season_key(season)
    return contexts[contexts["Season"].map(_season_key).eq(wanted)].reset_index(drop=True)


def _event_iteration_ids() -> set[int]:
    if USE_MOCK_DATA:
        return {1410}
    rows = get_connection().query(
        f"""
        SELECT DISTINCT ITERATION_ID AS "IterationId"
        FROM {relation("impect_events")}
        WHERE ITERATION_ID IS NOT NULL
        """,
        ttl="6h",
    )
    if rows.empty:
        return set()
    return set(pd.to_numeric(rows["IterationId"], errors="coerce").dropna().astype(int))


def _iteration_filter(contexts: pd.DataFrame, column: str) -> tuple[str, list[int]]:
    ids = pd.to_numeric(contexts.get("IterationId"), errors="coerce").dropna().astype(int).drop_duplicates().tolist()
    if not ids:
        return "1 = 0", []
    return f'{column} IN ({", ".join(["?"] * len(ids))})', ids


def _pivot_long_kpis(
    values: pd.DataFrame,
    index_columns: list[str],
    kpi_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Pivot an allow-listed slice of Impect's long KPI facts in Pandas."""
    if values.empty:
        return pd.DataFrame(columns=index_columns)
    frame = values.copy()
    frame[kpi_column] = pd.to_numeric(frame[kpi_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    wide = (
        frame.groupby([*index_columns, kpi_column], dropna=False, observed=True)[value_column]
        .max()
        .unstack(kpi_column)
        .reset_index()
    )
    wide.columns.name = None
    return wide


def _add_pass_pct(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    """Turn raw successful/unsuccessful pass counts into a Pass % column."""
    for column in ["_successful_passes", "_unsuccessful_passes"]:
        if column not in df:
            df[column] = np.nan
    total = df["_successful_passes"] + df["_unsuccessful_passes"]
    df["Pass %"] = (df["_successful_passes"] / total.replace(0, np.nan) * 100).round(1)
    if drop:
        return df.drop(columns=["_successful_passes", "_unsuccessful_passes"])
    return df


def _add_player_profile_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derived player metrics used by the role-aware profile page."""
    df = _add_pass_pct(df, drop=False)

    def numeric(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)

    df["Shots /90"] = numeric("_shots_on_target") + numeric("_shots_off_target")
    df["Pass Progression /90"] = (
        numeric("_bypassed_by_low_pass")
        + numeric("_bypassed_by_diagonal_pass")
        + numeric("_bypassed_by_chipped_pass")
    )
    df["Cross Progression /90"] = numeric("_bypassed_by_low_cross") + numeric("_bypassed_by_high_cross")

    ground_total = numeric("_won_ground_duels") + numeric("_lost_ground_duels")
    aerial_total = numeric("_won_aerial_duels") + numeric("_lost_aerial_duels")
    df["Ground Duel Win %"] = (numeric("_won_ground_duels") / ground_total.replace(0, np.nan) * 100).round(1)
    df["Aerial Duel Win %"] = (numeric("_won_aerial_duels") / aerial_total.replace(0, np.nan) * 100).round(1)
    df["Goals Prevented /90"] = numeric("Post-Shot xG Faced /90") - numeric("Goals Conceded /90")

    retention_actions = numeric("_successful_passes") + numeric("_unsuccessful_passes") + numeric("Ball Losses /90")
    df["Ball Security %"] = (numeric("_successful_passes") / retention_actions.replace(0, np.nan) * 100).round(1)
    df["Losses Per 100 Actions"] = (numeric("Ball Losses /90") / retention_actions.replace(0, np.nan) * 100).round(1)

    drop_cols = [col for col in df.columns if col.startswith("_")]
    return df.drop(columns=drop_cols)


# ---- EVENT-BASED KPI AGGREGATION (fallback for 26/27) -----------------------
# The Impect data pipeline populates STG_IMPECT__ITERATION_SQUAD_KPIS and
# STG_IMPECT__ITERATION_PLAYER_KPIS via a scheduled job.  When a new season
# (e.g. 26/27) has events but the KPI tables haven't been backfilled yet,
# these functions compute the same KPIs directly from the raw EVENTS table.
# ----------------------------------------------------------------------------

_EVENT_KPI_TEAM_MAPPING: dict[int, str] = {
    # TEAM_KPI_IDS key ---> EVENT_KPIS JSON key (only squad-owning player elements)
    0: "BYPASSED_OPPONENTS",          # Bypassed Opponents /90
    28: "GOALS",                       # Goals /90
    77: "EXPECTED_GOAL_ASSISTS",       # Assists /90 (approximated from xA)
    82: "SHOT_XG",                     # xG /90
    83: "PACKING_XG",                  # Packing xG /90
    24: "BALL_WIN_REMOVED_OPPONENTS",  # Ball Win Value /90
    27: "BALL_WIN_NUMBER",             # Ball Wins /90
    87: "DISTANCE_TO_GOAL_COVERED_DRIBBLE",  # Dribble Progression /90
    90: "SUCCESSFUL_PASSES",           # _successful_passes
    91: "UNSUCCESSFUL_PASSES",         # _unsuccessful_passes
    100: "SHOT_AT_GOAL_NUMBER_ON_TARGET",   # _shots_on_target
    101: "SHOT_AT_GOAL_OFF_TARGET_NUMBER",  # _shots_off_target
    331: None,                         # _successful_passes_to_final_third — not in EVENT(KPIS)
    392: None,                         # _unsuccessful_passes_to_final_third — not in EVENT(KPIS)
}

_EVENT_KPI_PLAYER_MAPPING: dict[int, str] = {
    # PLAYER_KPI_IDS key ---> EVENT_KPIS JSON key
    # NOTE: For 26/27 event-based fallback aggregation, only include mappings that:
    # (1) Match the team KPI mappings for consistency
    # (2) Don't duplicate - each JSON key must map to exactly ONE KPI ID
    # Pass-type breakdowns (106-111, 331, 392) and 1460/1462 duplicates are removed
    # because they cause inflated values when multiple KPI IDs sum the same JSON field.
    0: "BYPASSED_OPPONENTS",                # Bypassed Opponents /90
    2: "BYPASSED_DEFENDERS",                # Bypassed Defenders /90
    7: "BYPASSED_OPPONENTS_RECEIVING",      # Receiving Progression /90
    9: "BYPASSED_DEFENDERS_RECEIVING",      # Receiving Defenders Bypassed /90
    20: "BALL_LOSS_ADDED_OPPONENTS",        # Ball Loss Threat /90
    21: "BALL_LOSS_REMOVED_TEAMMATES",      # Team-Mates Bypassed By Losses /90
    22: "BALL_LOSS_NUMBER",                 # Ball Losses /90
    24: "BALL_WIN_REMOVED_OPPONENTS",       # Ball Win Value /90
    27: "BALL_WIN_NUMBER",                  # Ball Wins /90
    28: "GOALS",                            # Goals /90
    77: "EXPECTED_GOAL_ASSISTS",            # Assists /90
    82: "SHOT_XG",                          # xG /90
    83: "PACKING_XG",                       # Packing xG /90
    87: "DISTANCE_TO_GOAL_COVERED_DRIBBLE", # Dribble Progression /90
    90: "SUCCESSFUL_PASSES",                # _successful_passes
    91: "UNSUCCESSFUL_PASSES",              # _unsuccessful_passes
    94: "WON_GROUND_DUELS",                 # _won_ground_duels
    95: "LOST_GROUND_DUELS",                # _lost_ground_duels
    96: "WON_AERIAL_DUELS",                 # _won_aerial_duels
    97: "LOST_AERIAL_DUELS",                # _lost_aerial_duels
    100: "SHOT_AT_GOAL_NUMBER_ON_TARGET",   # _shots_on_target
    101: "SHOT_AT_GOAL_OFF_TARGET_NUMBER",  # _shots_off_target
    1401: "POSTSHOT_XG",                    # Post-Shot xG /90
    1431: "NEUTRAL_PASSES",                 # Neutral Passes /90
    1517: "SAVE_ACTIONS",                   # Save Actions /90
}


def _compute_team_kpis_from_events(
    contexts: pd.DataFrame, iteration_ids: list[int]
) -> pd.DataFrame | None:
    """Aggregate per-squad season KPI totals from the raw EVENTS table.

    Returns a DataFrame with columns matching the ``impect_iteration_squad_kpis``
    schema: ``IterationId``, ``TeamId``, ``Matches Played``, ``KpiId``, ``KpiValue``.
    All teams in the iteration are included — squads with no events get zero
    values so the full 24-team league appears in the UI.
    Returns ``None`` if no iteration in *iteration_ids* has event data.
    """
    conn = get_connection()
    placeholders = ", ".join(["?"] * len(iteration_ids))
    event_kpi_mappings = _EVENT_KPI_TEAM_MAPPING
    # EVENT_KPIS is a VARIANT JSON array — one element per player involved in
    # the action.  `e.EVENT_KPIS[0]` points to the acting player's element, but
    # is only guaranteed to belong to the row's own PLAYER_ID for events with a
    # single involved player; the IFF guard (matching load_match_events and the
    # other seasons' KPI queries) drops the rare row where it doesn't, avoiding
    # attributing another player's KPI value to this one.
    columns = ", ".join(
        f'COALESCE(SUM(IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID, '
        f'TRY_TO_NUMBER(e.EVENT_KPIS[0]:"{json_key}"::STRING), NULL)), 0) AS "kpi_{kpi_id}"'
        for kpi_id, json_key in event_kpi_mappings.items()
        if json_key is not None
    )
    sql = f"""
        SELECT
            e.ITERATION_ID AS "IterationId",
            e.SQUAD_ID AS "TeamId",
            COUNT(DISTINCT e.MATCH_ID) AS "Matches Played",
            {columns}
        FROM {relation("impect_events")} e
        WHERE e.ITERATION_ID IN ({placeholders})
          AND e.EVENT_KPIS IS NOT NULL
          AND e.SQUAD_ID IS NOT NULL
        GROUP BY e.ITERATION_ID, e.SQUAD_ID
    """
    raw = conn.query(sql, params=_snowflake_params(iteration_ids), ttl="1h")
    if raw.empty:
        return None

    # Ensure every squad in the iteration appears — even those with no events.
    all_squads = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId"
        FROM {relation("impect_squads")}
        WHERE ITERATION_ID IN ({placeholders})
        """,
        params=_snowflake_params(iteration_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    if all_squads.empty:
        return None

    raw = all_squads.merge(raw, on=["IterationId", "TeamId"], how="left")
    for col in raw.columns:
        if col.startswith("kpi_"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
    raw["Matches Played"] = pd.to_numeric(raw["Matches Played"], errors="coerce").fillna(0).astype(int)

    long_rows: list[dict] = []
    for _, row in raw.iterrows():
        iteration_id = row["IterationId"]
        team_id = row["TeamId"]
        matches_played = int(row["Matches Played"])
        has_positive = False
        for kpi_id in event_kpi_mappings:
            value = row.get(f"kpi_{kpi_id}")
            if value is not None and pd.notna(value) and float(value) > 0:
                has_positive = True
                long_rows.append({
                    "IterationId": iteration_id,
                    "TeamId": team_id,
                    "Matches Played": matches_played,
                    "KpiId": kpi_id,
                    "KpiValue": float(value) / max(matches_played, 1),
                })
        # Ensure every team appears even if all KPIs are zero.
        if not has_positive:
            for kpi_id in event_kpi_mappings:
                long_rows.append({
                    "IterationId": iteration_id,
                    "TeamId": team_id,
                    "Matches Played": matches_played,
                    "KpiId": kpi_id,
                    "KpiValue": 0.0,
                })
    return pd.DataFrame(long_rows) if long_rows else None


def _compute_player_kpis_from_events(
    contexts: pd.DataFrame, iteration_ids: list[int]
) -> pd.DataFrame | None:
    """Aggregate per-player season KPI totals from the raw EVENTS table.

    Returns a DataFrame with columns matching the ``impect_iteration_player_kpis``
    schema: ``IterationId``, ``TeamId``, ``PlayerId``, ``Position``,
    ``Play Duration Seconds``, ``Match Share``, ``KpiId``, ``KpiValue``.
    Returns ``None`` if no iteration has event data.

    Per-90 rates need each player's *actual* on-pitch time, not an assumed
    flat 90 minutes for every match they touched the ball in — assuming a
    full match badly distorts rates for anyone substituted on or off (a
    2-minute cameo with one progressive pass previously showed the same
    "per 90" value as a full 90-minute performance, understating true rates
    for short appearances by up to ~50x). Time on pitch is estimated per
    match from the span between a player's first and last involvement,
    normalising CAFC_DB's +10000-second-per-half offset in GAME_TIME_IN_SEC
    the same way ``_event_elapsed_minutes`` does, so the half-time gap isn't
    counted as playing time.
    """
    conn = get_connection()
    placeholders = ", ".join(["?"] * len(iteration_ids))
    event_kpi_mappings = _EVENT_KPI_PLAYER_MAPPING
    kpi_cols = [f"kpi_{kpi_id}" for kpi_id in event_kpi_mappings]
    columns = ", ".join(
        f'COALESCE(SUM(IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID, '
        f'TRY_TO_DOUBLE(e.EVENT_KPIS[0]:"{json_key}"::STRING), NULL)), 0) AS "kpi_{kpi_id}"'
        for kpi_id, json_key in event_kpi_mappings.items()
        if json_key is not None
    )
    sql = f"""
        SELECT
            e.ITERATION_ID AS "IterationId",
            e.SQUAD_ID AS "TeamId",
            e.MATCH_ID AS "MatchId",
            e.PLAYER_ID AS "PlayerId",
            e.PLAYER_POSITION AS "Position",
            e.PERIOD_ID AS "PeriodId",
            MIN(e.GAME_TIME_IN_SEC) AS "FirstSec",
            MAX(e.GAME_TIME_IN_SEC) AS "LastSec",
            {columns}
        FROM {relation("impect_events")} e
        WHERE e.ITERATION_ID IN ({placeholders})
          AND e.EVENT_KPIS IS NOT NULL
          AND e.SQUAD_ID IS NOT NULL
          AND e.PLAYER_ID IS NOT NULL
        GROUP BY e.ITERATION_ID, e.SQUAD_ID, e.MATCH_ID, e.PLAYER_ID, e.PLAYER_POSITION, e.PERIOD_ID
    """
    raw = conn.query(sql, params=_snowflake_params(iteration_ids), ttl="1h")
    if raw.empty:
        return None
    for col in kpi_cols:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

    # Normalise the 10000-second-per-half offset (see _event_elapsed_minutes)
    # so a player's span isn't inflated by the half-time break between periods.
    period = pd.to_numeric(raw["PeriodId"], errors="coerce")
    period_base_seconds = np.select(
        [period.eq(1), period.eq(2), period.eq(3), period.eq(4)],
        [0, 45 * 60, 90 * 60, 105 * 60],
        default=np.nan,
    )
    for col in ("FirstSec", "LastSec"):
        second = pd.to_numeric(raw[col], errors="coerce")
        offset_bucket = np.floor(second / 10000).clip(lower=0)
        fallback_base = offset_bucket * 45 * 60
        base = pd.Series(period_base_seconds, index=raw.index).where(
            pd.notna(period_base_seconds), fallback_base
        )
        period_seconds = second % 10000
        raw[col] = second.where(second < 10000, period_seconds + base)
    raw["_PeriodSpanSeconds"] = (raw["LastSec"] - raw["FirstSec"]).clip(lower=0)

    match_keys = ["IterationId", "TeamId", "MatchId", "PlayerId", "Position"]
    per_match = (
        raw.groupby(match_keys, dropna=False, observed=True)
        .agg(**{
            "_DurationSeconds": ("_PeriodSpanSeconds", "sum"),
            **{col: (col, "sum") for col in kpi_cols if col in raw},
        })
        .reset_index()
    )
    # Floor at one minute so a single late touch doesn't produce a runaway
    # per-90 rate from a near-zero denominator.
    per_match["_DurationSeconds"] = per_match["_DurationSeconds"].clip(lower=60)

    position_keys = ["IterationId", "TeamId", "PlayerId", "Position"]
    season = (
        per_match.groupby(position_keys, dropna=False, observed=True)
        .agg(**{
            "Matches Played": ("MatchId", "nunique"),
            "_DurationSeconds": ("_DurationSeconds", "sum"),
            **{col: (col, "sum") for col in kpi_cols if col in per_match},
        })
        .reset_index()
    )

    long_rows: list[dict] = []
    for _, row in season.iterrows():
        duration_seconds = float(row.get("_DurationSeconds") or 0.0)
        if duration_seconds <= 0:
            continue
        iteration_id = row["IterationId"]
        team_id = row["TeamId"]
        player_id = row["PlayerId"]
        position = str(row["Position"] or "").strip()
        match_share = float(row["Matches Played"])  # Cumulative match count, summed across positions
        for kpi_id in event_kpi_mappings:
            value = row.get(f"kpi_{kpi_id}")
            if value is not None and pd.notna(value) and float(value) > 0:
                long_rows.append({
                    "IterationId": iteration_id,
                    "TeamId": team_id,
                    "PlayerId": player_id,
                    "Position": position,
                    "Play Duration Seconds": duration_seconds,
                    "Match Share": match_share,
                    "KpiId": kpi_id,
                    "KpiValue": float(value) / duration_seconds * 5400.0,
                })
    return pd.DataFrame(long_rows) if long_rows else None


# ---- PUBLIC LOADERS (what the pages call) ------------------------------------
# Each loader either returns mock data or runs a Snowflake query. The `ttl="1h"`
# on conn.query means results are cached for an hour before re-querying — that
# is your "data refresh" cadence, and it stops every click hammering the DB.

def _legacy_list_seasons() -> dict[str, list[str]]:
    """Compatibility alias for the current production season loader."""
    return list_seasons()


def _legacy_load_players(season: str | None = None) -> pd.DataFrame:
    """Compatibility alias for the current production player loader."""
    return load_players(season)


def _legacy_load_teams(season: str | None = None) -> pd.DataFrame:
    """Compatibility alias for the current production team loader."""
    return load_teams(season)


def _legacy_load_team_iteration_rollups(season: str | None = None) -> pd.DataFrame:
    """Compatibility alias for the current production team-rollup loader."""
    return load_team_iteration_rollups(season)


def _kpi_iteration_ids(table_key: str) -> set[int]:
    """Iteration ids that have at least one row in the given season-level KPI facts table."""
    if USE_MOCK_DATA:
        return {1410}
    rows = get_connection().query(
        f"""
        SELECT DISTINCT IMPECT_ITERATION_ID AS "IterationId"
        FROM {relation(table_key)}
        WHERE IMPECT_ITERATION_ID IS NOT NULL
        """,
        ttl="6h",
    )
    if rows.empty:
        return set()
    return set(pd.to_numeric(rows["IterationId"], errors="coerce").dropna().astype(int))


def list_seasons() -> dict[str, list[str]]:
    """League seasons available from authoritative CAFC_DB Impect sources.

    "matches"/"players"/"teams" are validated separately against where that
    kind of data actually resolves, rather than returning one undifferentiated
    season list for all three. A season can have provider KPI facts but no raw
    event coverage (e.g. Charlton's older League One seasons), or vice versa --
    offering it on a page that has nothing for it just produces a dead "no
    data available" menu option. Player/team seasons also count iterations
    with event coverage, since load_players()/load_teams() can compute from
    raw events when the KPI facts tables are empty (e.g. the newest season).
    """
    if USE_MOCK_DATA:
        return {"players": ["2025/26"], "teams": ["2025/26"], "matches": ["2025/26"]}

    contexts = _league_contexts()
    if contexts.empty:
        return {"players": [], "teams": [], "matches": []}

    event_iterations = _event_iteration_ids()
    player_kpi_iterations = _kpi_iteration_ids("impect_iteration_player_kpis")
    team_kpi_iterations = _kpi_iteration_ids("impect_iteration_squad_kpis")

    def _seasons_for(iteration_ids: set[int]) -> list[str]:
        subset = contexts[contexts["IterationId"].astype("Int64").isin(iteration_ids)]
        return subset["Season"].dropna().astype(str).drop_duplicates().sort_values().tolist()

    return {
        "matches": _seasons_for(event_iterations),
        "players": _seasons_for(player_kpi_iterations | event_iterations),
        "teams": _seasons_for(team_kpi_iterations | event_iterations),
    }


MIN_MATCHES_FOR_DEFAULT_SEASON = 5


def _charlton_match_count(season: str) -> int:
    """How many of Charlton's own matches are available for this season.

    Deliberately team-count-independent (unlike a raw league-wide match
    total, which scales with how many clubs are in the competition and can
    look "substantial" after just one round -- 11 league-wide matches in a
    24-team league is only ~1 game week per team, not 11 games of signal).
    """
    matches = load_matches(season)
    if matches.empty or "Home" not in matches or "Away" not in matches:
        return 0
    is_charlton = (
        matches["Home"].astype(str).str.contains("charlton", case=False, na=False)
        | matches["Away"].astype(str).str.contains("charlton", case=False, na=False)
    )
    return int(is_charlton.sum())


def preferred_season(seasons: list[str], match_count: "Callable[[str], int] | None" = None) -> str | None:
    """Most recent season with a substantial number of matches, else the newest available.

    Defaulting a season selector to the very latest season is wrong right
    after a new season has just started (e.g. one game week in) -- every
    page would open by default to a near-empty, unrepresentative sample.
    This only second-guesses the newest season in the list (anything else is
    already a completed past season); everything else about the selector is
    unchanged, and the user can still pick the newest season manually.

    ``match_count`` lets callers substitute a different match-count source
    (e.g. Opta fixtures instead of Impect matches, or a different team) for
    pages built on a different provider's season labels; it defaults to
    this module's own Impect-backed count of Charlton's own matches.
    """
    if not seasons:
        return None
    latest = seasons[-1]
    if len(seasons) == 1:
        return latest
    counter = match_count or _charlton_match_count
    if counter(latest) >= MIN_MATCHES_FOR_DEFAULT_SEASON:
        return latest
    return seasons[-2]


def _team_match_counts(iteration_ids: list[int]) -> pd.DataFrame:
    """Actual matches played per (IterationId, TeamId) -- the hard ceiling for any single player's Match Share.

    A player's Match Share for one position-stint can never exceed their own
    team's total match count for the season; that's a mathematical fact, not
    an estimate, so it's used to defensively cap occasional bad KPI rows
    (see load_players()) rather than guessing at a plausible-looking limit.
    """
    columns = ["IterationId", "TeamId", "TeamMatches"]
    if not iteration_ids:
        return pd.DataFrame(columns=columns)
    placeholders = ", ".join(["?"] * len(iteration_ids))
    rows = get_connection().query(
        f"""
        SELECT "IterationId", "TeamId", SUM("TeamMatches") AS "TeamMatches"
        FROM (
            SELECT ITERATION_ID AS "IterationId", HOME_SQUAD_ID AS "TeamId", COUNT(*) AS "TeamMatches"
            FROM {relation("impect_matches")}
            WHERE ITERATION_ID IN ({placeholders})
            GROUP BY ITERATION_ID, HOME_SQUAD_ID
            UNION ALL
            SELECT ITERATION_ID AS "IterationId", AWAY_SQUAD_ID AS "TeamId", COUNT(*) AS "TeamMatches"
            FROM {relation("impect_matches")}
            WHERE ITERATION_ID IN ({placeholders})
            GROUP BY ITERATION_ID, AWAY_SQUAD_ID
        )
        GROUP BY "IterationId", "TeamId"
        """,
        params=_snowflake_params([*iteration_ids, *iteration_ids]),
        ttl="6h",
    )
    return rows[columns] if not rows.empty else pd.DataFrame(columns=columns)


def load_players(season: str | None = None) -> pd.DataFrame:
    """Player season metrics reconstructed from CAFC_DB's long Impect facts."""
    if USE_MOCK_DATA:
        return _mock_players()
    contexts = _contexts_for_season(season)
    filter_sql, iteration_ids = _iteration_filter(contexts, "IMPECT_ITERATION_ID")
    if not iteration_ids:
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    kpi_ids = sorted(set(PLAYER_KPI_IDS.values()))
    kpi_placeholders = ", ".join(["?"] * len(kpi_ids))
    conn = get_connection()
    long_values = conn.query(
        f"""
        SELECT
            IMPECT_ITERATION_ID AS "IterationId",
            IMPECT_SQUAD_ID AS "TeamId",
            IMPECT_PLAYER_ID AS "PlayerId",
            POSITION_CODE AS "Position",
            PLAY_DURATION_SECONDS AS "Play Duration Seconds",
            MATCH_SHARE AS "Match Share",
            IMPECT_KPI_ID AS "KpiId",
            KPI_VALUE AS "KpiValue"
        FROM {relation("impect_iteration_player_kpis")}
        WHERE {filter_sql}
          AND IMPECT_KPI_ID IN ({kpi_placeholders})
        """,
        params=_snowflake_params([*iteration_ids, *kpi_ids]),
        ttl="1h",
    )
    if long_values.empty:
        # Fallback: compute player KPIs from the raw EVENTS table when the
        # KPI tables haven't been populated yet (e.g. 26/27 season).
        event_long = _compute_player_kpis_from_events(contexts, iteration_ids)
        if event_long is not None and not event_long.empty:
            long_values = event_long
    if long_values.empty:
        return pd.DataFrame(columns=PLAYER_COLUMNS)

    position_keys = ["IterationId", "TeamId", "PlayerId", "Position"]
    position_meta = (
        long_values.groupby(position_keys, dropna=False, observed=True)
        .agg(
            **{
                "Play Duration Seconds": ("Play Duration Seconds", "max"),
                "Match Share": ("Match Share", "max"),
            }
        )
        .reset_index()
    )

    # Defensive cap: a single position-stint's Match Share cannot exceed the
    # player's own team's actual match count for the season -- occasionally a
    # source KPI row has an implausible Match Share (seen live: 49.33 for a
    # team that played 46 matches all season), which would otherwise silently
    # inflate that player's Minutes and every per-90 rate derived from it.
    # When a row is capped, Play Duration Seconds is scaled down by the same
    # ratio so the implied average minutes-per-match is preserved rather than
    # guessed at.
    team_matches = _team_match_counts(iteration_ids)
    position_meta = position_meta.merge(team_matches, on=["IterationId", "TeamId"], how="left")
    match_share = pd.to_numeric(position_meta["Match Share"], errors="coerce")
    team_cap = pd.to_numeric(position_meta["TeamMatches"], errors="coerce")
    exceeds_cap = match_share.notna() & team_cap.notna() & match_share.gt(team_cap)
    if exceeds_cap.any():
        clip_ratio = (team_cap / match_share).where(exceeds_cap, 1.0)
        position_meta["Play Duration Seconds"] = (
            pd.to_numeric(position_meta["Play Duration Seconds"], errors="coerce") * clip_ratio
        )
        position_meta["Match Share"] = match_share.where(~exceeds_cap, team_cap)
    position_meta = position_meta.drop(columns=["TeamMatches"])

    position_rows = position_meta.merge(
        _pivot_long_kpis(long_values, position_keys, "KpiId", "KpiValue"),
        on=position_keys,
        how="left",
    )
    position_rows["Play Duration Seconds"] = pd.to_numeric(
        position_rows["Play Duration Seconds"], errors="coerce"
    ).fillna(0)

    player_keys = ["IterationId", "TeamId", "PlayerId"]
    players = (
        position_rows.groupby(player_keys, dropna=False, observed=True)
        .agg(
            **{
                "Play Duration Seconds": ("Play Duration Seconds", "sum"),
                "Match Share": ("Match Share", "sum"),
                "Position": (
                    "Position",
                    lambda values: ", ".join(
                        sorted({str(value) for value in values.dropna() if str(value).strip()})
                    ),
                ),
            }
        )
        .reset_index()
    )

    groupers = [position_rows[column] for column in player_keys]
    is_goalkeeper_stint = position_rows["Position"].astype(str).str.upper().eq("GOALKEEPER")
    for output_column, kpi_id in PLAYER_KPI_IDS.items():
        if kpi_id not in position_rows:
            players[output_column] = np.nan
            continue
        metric = pd.to_numeric(position_rows[kpi_id], errors="coerce")
        # Impect's KPI facts only store a row for a (player, position) stint
        # when that KPI's value is non-zero there -- there is no explicit
        # zero row. Treating a missing row as "exclude this stint's minutes"
        # (the old behaviour) silently shrinks the per-90 denominator to
        # whichever position happened to have a non-zero row, while the
        # "Minutes" total shown alongside it still counts every position --
        # e.g. a player's Goals /90 could be computed from a single 15-minute
        # substitute cameo while "Minutes: 620" is displayed next to it. A
        # missing row means zero for that stint, not "not applicable", with
        # one exception: goalkeeping-only KPIs (saves, goals conceded, etc.)
        # genuinely don't apply to an outfield stint, so those only get
        # zero-filled within the player's own goalkeeper stints.
        if kpi_id in _GOALKEEPING_ONLY_KPI_IDS:
            metric = metric.where(~(is_goalkeeper_stint & metric.isna()), 0.0)
        else:
            metric = metric.fillna(0.0)
        duration = position_rows["Play Duration Seconds"].where(metric.notna())
        numerator = (metric * position_rows["Play Duration Seconds"]).groupby(
            groupers, dropna=False
        ).sum(min_count=1)
        denominator = duration.groupby(groupers, dropna=False).sum(min_count=1)
        weighted = (numerator / denominator.replace(0, np.nan)).rename(output_column).reset_index()
        players = players.merge(weighted, on=player_keys, how="left")

    dimension_filter, dimension_ids = _iteration_filter(contexts, "ITERATION_ID")
    player_dimensions = conn.query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            IMPECT_PLAYER_ID AS "PlayerId",
            COALESCE(NULLIF(COMMON_NAME, ''), TRIM(CONCAT_WS(' ', FIRST_NAME, LAST_NAME))) AS "Player",
            FIRST_NAME AS "First Name",
            LAST_NAME AS "Last Name",
            BIRTH_DATE AS "Birthdate",
            STRONG_FOOT AS "Foot",
            TRY_TO_NUMBER(COUNTRY_IDS[0]::STRING) AS "CountryId"
        FROM {relation("impect_players")}
        WHERE {dimension_filter}
        """,
        params=_snowflake_params(dimension_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "PlayerId"])
    squad_dimensions = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId", SQUAD_NAME AS "Team"
        FROM {relation("impect_squads")}
        WHERE {dimension_filter}
        """,
        params=_snowflake_params(dimension_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    countries = conn.query(
        f"""
        SELECT ID AS "CountryId", COALESCE(NULLIF(FIFANAME, ''), NAME) AS "Nationality"
        FROM {relation("impect_countries")}
        """,
        ttl="24h",
    ).drop_duplicates("CountryId")

    for frame in [player_dimensions, countries]:
        frame["CountryId"] = pd.to_numeric(frame["CountryId"], errors="coerce").astype("Int64")
    player_dimensions = player_dimensions.merge(countries, on="CountryId", how="left")
    players = (
        players.merge(player_dimensions, on=["IterationId", "PlayerId"], how="left")
        .merge(squad_dimensions, on=["IterationId", "TeamId"], how="left")
        .merge(contexts, on="IterationId", how="left")
    )
    players["Minutes"] = pd.to_numeric(players["Play Duration Seconds"], errors="coerce") / 60
    players["Successful Passes /90"] = players["_successful_passes"]
    players["Passes to Final 3rd /90"] = (
        pd.to_numeric(players["_successful_passes_to_final_third"], errors="coerce")
        + pd.to_numeric(players["_unsuccessful_passes_to_final_third"], errors="coerce")
    )
    players = players.drop(
        columns=["Play Duration Seconds", "CountryId", "TeamId", "IterationId"], errors="ignore"
    )
    return _add_player_profile_metrics(players)


def load_teams(season: str | None = None) -> pd.DataFrame:
    """Provider-authored squad season metrics from CAFC_DB."""
    if USE_MOCK_DATA:
        return _mock_teams()
    contexts = _contexts_for_season(season)
    filter_sql, iteration_ids = _iteration_filter(contexts, "IMPECT_ITERATION_ID")
    if not iteration_ids:
        return pd.DataFrame(columns=TEAM_COLUMNS)

    kpi_ids = sorted(set(TEAM_KPI_IDS.values()))
    kpi_placeholders = ", ".join(["?"] * len(kpi_ids))
    conn = get_connection()
    long_values = conn.query(
        f"""
        SELECT
            IMPECT_ITERATION_ID AS "IterationId",
            IMPECT_SQUAD_ID AS "TeamId",
            MATCHES_PLAYED AS "Matches Played",
            IMPECT_KPI_ID AS "KpiId",
            KPI_VALUE AS "KpiValue"
        FROM {relation("impect_iteration_squad_kpis")}
        WHERE {filter_sql}
          AND IMPECT_KPI_ID IN ({kpi_placeholders})
        """,
        params=_snowflake_params([*iteration_ids, *kpi_ids]),
        ttl="1h",
    )
    teams = _pivot_long_kpis(long_values, ["IterationId", "TeamId"], "KpiId", "KpiValue")
    if teams.empty:
        # Fallback: compute KPIs from the raw EVENTS table when the KPI
        # tables haven't been populated yet (e.g. 26/27 season).
        event_long = _compute_team_kpis_from_events(contexts, iteration_ids)
        if event_long is not None and not event_long.empty:
            long_values = event_long
            teams = _pivot_long_kpis(long_values, ["IterationId", "TeamId"], "KpiId", "KpiValue")
    if teams.empty:
        return pd.DataFrame(columns=TEAM_COLUMNS)

    for output_column, kpi_id in TEAM_KPI_IDS.items():
        teams[output_column] = pd.to_numeric(teams[kpi_id], errors="coerce") if kpi_id in teams else np.nan
    teams["Passes to Final 3rd /90"] = (
        pd.to_numeric(teams["_successful_passes_to_final_third"], errors="coerce")
        + pd.to_numeric(teams["_unsuccessful_passes_to_final_third"], errors="coerce")
    )
    teams["Shots /90"] = (
        pd.to_numeric(teams["_shots_on_target"], errors="coerce")
        + pd.to_numeric(teams["_shots_off_target"], errors="coerce")
    )

    dimension_filter, dimension_ids = _iteration_filter(contexts, "ITERATION_ID")
    squad_dimensions = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId", SQUAD_NAME AS "Team"
        FROM {relation("impect_squads")}
        WHERE {dimension_filter}
        """,
        params=_snowflake_params(dimension_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    teams = (
        teams.merge(squad_dimensions, on=["IterationId", "TeamId"], how="left")
        .merge(contexts, on="IterationId", how="left")
        .drop(
            columns=[
                "IterationId",
                "TeamId",
                *kpi_ids,
                "_successful_passes_to_final_third",
                "_unsuccessful_passes_to_final_third",
            ],
            errors="ignore",
        )
    )
    return _add_pass_pct(teams)


def load_team_iteration_rollups(season: str | None = None) -> pd.DataFrame:
    """Return the authoritative squad KPIs used by team-style comparisons."""
    return load_teams(season)


def _legacy_load_matches_2425(season: str) -> pd.DataFrame:
    """Compatibility alias for the current production match loader."""
    return load_matches(season)


def _legacy_load_matches_2526(season: str) -> pd.DataFrame:
    """Compatibility alias for the current production match loader."""
    return load_matches(season)


def _is_event_match_season(season: str) -> bool:
    if USE_MOCK_DATA:
        return _season_key(season) == "25/26"
    contexts = _contexts_for_season(season)
    if contexts.empty:
        return False
    return bool(set(contexts["IterationId"].dropna().astype(int)) & _event_iteration_ids())


def _legacy_load_matches(season: str | None = None) -> pd.DataFrame:
    """Compatibility alias for the current production match loader."""
    return load_matches(season)


def load_matches(season: str | None = None) -> pd.DataFrame:
    """Verified fixtures and scores reconstructed from CAFC_DB Impect data."""
    if USE_MOCK_DATA:
        matches = _mock_matches()
        return matches[matches["Season"].map(_season_key) == _season_key(season)].copy() if season else matches

    contexts = _contexts_for_season(season)
    event_iterations = _event_iteration_ids()
    contexts = contexts[contexts["IterationId"].astype("Int64").isin(event_iterations)].copy()
    filter_sql, iteration_ids = _iteration_filter(contexts, "ITERATION_ID")
    if not iteration_ids:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    conn = get_connection()
    matches = conn.query(
        f"""
        SELECT
            IMPECT_MATCH_ID AS "MatchId",
            ITERATION_ID AS "IterationId",
            SCHEDULED_AT AS "Date",
            HOME_SQUAD_ID AS "HomeTeamId",
            AWAY_SQUAD_ID AS "AwayTeamId"
        FROM {relation("impect_matches")}
        WHERE {filter_sql}
        """,
        params=_snowflake_params(iteration_ids),
        ttl="1h",
    ).drop_duplicates(["IterationId", "MatchId"])
    if matches.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    # A match dimension row is a scheduled fixture, not proof that result data
    # has arrived. Keep only fixtures present in the provider event feed so an
    # unprocessed fixture can never be converted into a fabricated 0-0 draw.
    covered_matches = conn.query(
        f"""
        SELECT DISTINCT MATCH_ID AS "MatchId", ITERATION_ID AS "IterationId"
        FROM {relation("impect_events")}
        WHERE {filter_sql}
        """,
        params=_snowflake_params(iteration_ids),
        ttl="1h",
    )
    matches = matches.merge(covered_matches, on=["IterationId", "MatchId"], how="inner")
    if matches.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    dimension_filter, dimension_ids = _iteration_filter(contexts, "ITERATION_ID")
    squads = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId", SQUAD_NAME AS "Team"
        FROM {relation("impect_squads")}
        WHERE {dimension_filter}
        """,
        params=_snowflake_params(dimension_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    home = squads.rename(columns={"TeamId": "HomeTeamId", "Team": "Home"})
    away = squads.rename(columns={"TeamId": "AwayTeamId", "Team": "Away"})
    matches = (
        matches.merge(home, on=["IterationId", "HomeTeamId"], how="left")
        .merge(away, on=["IterationId", "AwayTeamId"], how="left")
        .merge(contexts, on="IterationId", how="left")
    )

    goal_filter, goal_ids = _iteration_filter(contexts, "ITERATION_ID")
    goal_events = conn.query(
        f"""
        SELECT
            MATCH_ID AS "MatchId",
            ITERATION_ID AS "IterationId",
            SQUAD_ID AS "TeamId",
            ACTION AS "Action"
        FROM {relation("impect_events")}
        WHERE {goal_filter}
          AND UPPER(ACTION) IN ('GOAL', 'OWN_GOAL')
        """,
        params=_snowflake_params(goal_ids),
        ttl="1h",
    )
    matches["Home Goals"] = 0
    matches["Away Goals"] = 0
    if not goal_events.empty:
        goal_events["Action"] = goal_events["Action"].fillna("").astype(str).str.upper()
        goal_events = goal_events.merge(
            matches[["MatchId", "IterationId", "HomeTeamId", "AwayTeamId"]],
            on=["MatchId", "IterationId"],
            how="inner",
        )
        normal_goal = goal_events["Action"].eq("GOAL")
        goal_events["Home Goal"] = (
            (normal_goal & goal_events["TeamId"].eq(goal_events["HomeTeamId"]))
            | (~normal_goal & goal_events["TeamId"].eq(goal_events["AwayTeamId"]))
        ).astype(int)
        goal_events["Away Goal"] = (
            (normal_goal & goal_events["TeamId"].eq(goal_events["AwayTeamId"]))
            | (~normal_goal & goal_events["TeamId"].eq(goal_events["HomeTeamId"]))
        ).astype(int)
        scores = goal_events.groupby(["MatchId", "IterationId"], as_index=False).agg(
            **{"Home Goals": ("Home Goal", "sum"), "Away Goals": ("Away Goal", "sum")}
        )
        matches = matches.drop(columns=["Home Goals", "Away Goals"]).merge(
            scores, on=["MatchId", "IterationId"], how="left"
        )
        matches[["Home Goals", "Away Goals"]] = matches[["Home Goals", "Away Goals"]].fillna(0)

    matches["Venue Verified"] = True
    matches["Match"] = matches["Home"].fillna("Unknown") + " vs " + matches["Away"].fillna("Unknown")
    matches["Result"] = np.select(
        [matches["Home Goals"] > matches["Away Goals"], matches["Home Goals"] < matches["Away Goals"]],
        ["Home Win", "Away Win"],
        default="Draw",
    )
    return matches.sort_values(["Date", "MatchId"]).reset_index(drop=True)[MATCH_COLUMNS]


def load_team_action_counts(season: str | None = None) -> pd.DataFrame:
    """Count event action labels by team where the event table is available.

    This powers team pages that need event categories such as shots, defensive
    actions, pass actions or set pieces. It deliberately returns counts only:
    the current app does not assume event coordinates or possession chains.
    """
    if USE_MOCK_DATA:
        return _mock_team_action_counts()

    contexts = _contexts_for_season(season)
    filter_sql, iteration_ids = _iteration_filter(contexts, "ITERATION_ID")
    if not iteration_ids:
        return pd.DataFrame(columns=["Team", "Action", "Actions"])
    conn = get_connection()
    counts = conn.query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            SQUAD_ID AS "TeamId",
            ACTION AS "Action",
            COUNT(*)    AS "Actions"
        FROM {relation("impect_events")}
        WHERE {filter_sql}
        GROUP BY ITERATION_ID, SQUAD_ID, ACTION
        """,
        params=_snowflake_params(iteration_ids),
        ttl="1h",
    )
    dimension_filter, dimension_ids = _iteration_filter(contexts, "ITERATION_ID")
    squads = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId", SQUAD_NAME AS "Team"
        FROM {relation("impect_squads")}
        WHERE {dimension_filter}
        """,
        params=_snowflake_params(dimension_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    return (
        counts.merge(squads, on=["IterationId", "TeamId"], how="left")
        .sort_values(["Team", "Action"])
        .reset_index(drop=True)[["Team", "Action", "Actions"]]
    )


def load_match_action_counts(season: str | None = None) -> pd.DataFrame:
    """Count available event action labels by match and team.

    This remains useful for aggregate tables and fallback views. Spatial match
    pages should use load_match_events instead.
    """
    if USE_MOCK_DATA:
        actions = _mock_match_action_counts()
        return actions[actions["Season"] == season].copy() if season else actions

    contexts = _contexts_for_season(season)
    filter_sql, iteration_ids = _iteration_filter(contexts, "ITERATION_ID")
    if not iteration_ids:
        return pd.DataFrame(columns=["MatchId", "Season", "Team", "Action", "Actions"])
    conn = get_connection()
    counts = conn.query(
        f"""
        SELECT
            MATCH_ID AS "MatchId",
            ITERATION_ID AS "IterationId",
            SQUAD_ID AS "TeamId",
            ACTION AS "Action",
            COUNT(*)    AS "Actions"
        FROM {relation("impect_events")}
        WHERE {filter_sql}
        GROUP BY MATCH_ID, ITERATION_ID, SQUAD_ID, ACTION
        """,
        params=_snowflake_params(iteration_ids),
        ttl="1h",
    )
    dimension_filter, dimension_ids = _iteration_filter(contexts, "ITERATION_ID")
    squads = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId", SQUAD_NAME AS "Team"
        FROM {relation("impect_squads")}
        WHERE {dimension_filter}
        """,
        params=_snowflake_params(dimension_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    return (
        counts.merge(squads, on=["IterationId", "TeamId"], how="left")
        .merge(contexts[["IterationId", "Season"]], on="IterationId", how="left")
        .sort_values(["MatchId", "Team", "Action"])
        .reset_index(drop=True)[["MatchId", "Season", "Team", "Action", "Actions"]]
    )


def _empty_defensive_squad_match_sums() -> pd.DataFrame:
    return pd.DataFrame(columns=DEFENSIVE_SQUAD_MATCH_COLUMNS)


def _empty_defensive_player_match_sums() -> pd.DataFrame:
    return pd.DataFrame(columns=DEFENSIVE_PLAYER_MATCH_COLUMNS)


def _normalise_defensive_dates(values: pd.Series) -> pd.Series:
    """Return timezone-naive UTC timestamps so date filtering is consistent."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _clean_defensive_squad_match_sums(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_defensive_squad_match_sums()

    out = df.copy()
    for column in DEFENSIVE_SQUAD_MATCH_COLUMNS:
        if column not in out:
            out[column] = np.nan
    out["Date"] = _normalise_defensive_dates(out["Date"])
    numeric_columns = [
        column
        for column in DEFENSIVE_SQUAD_MATCH_COLUMNS
        if column not in {"MatchId", "Date", "Competition", "Season", "TeamId", "Team"}
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return (
        out[DEFENSIVE_SQUAD_MATCH_COLUMNS]
        .sort_values(["Date", "MatchId", "Team"], na_position="last")
        .reset_index(drop=True)
    )


def _clean_defensive_player_match_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Combine position segments into one trusted player row per match.

    IMPECT can split one player's match across several rows when their
    position changes. Additive defensive totals and play duration therefore
    need to be summed; the position attached to the longest segment is kept as
    the player's main position for that match.
    """
    if df.empty:
        return _empty_defensive_player_match_sums()

    out = df.copy()
    for column in DEFENSIVE_PLAYER_MATCH_COLUMNS:
        if column not in out:
            out[column] = np.nan
    out["Date"] = _normalise_defensive_dates(out["Date"])
    numeric_columns = [
        column
        for column in DEFENSIVE_PLAYER_MATCH_COLUMNS
        if column
        not in {
            "MatchId",
            "Date",
            "Competition",
            "Season",
            "TeamId",
            "Team",
            "PlayerId",
            "Player",
            "Position",
        }
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    group_columns = [
        "MatchId",
        "Date",
        "Competition",
        "Season",
        "TeamId",
        "Team",
        "PlayerId",
        "Player",
    ]
    position_rows = (
        out.sort_values("Play Duration Seconds", ascending=False)
        .drop_duplicates(group_columns)
        [group_columns + ["Position"]]
    )
    totals = out.groupby(group_columns, dropna=False, as_index=False)[numeric_columns].sum()
    totals = totals.merge(position_rows, on=group_columns, how="left")
    return (
        totals[DEFENSIVE_PLAYER_MATCH_COLUMNS]
        .sort_values(["Date", "Team", "Player"], na_position="last")
        .reset_index(drop=True)
    )


def _mock_defensive_squad_match_sums(season: str | None = None) -> pd.DataFrame:
    events = _mock_match_events()
    if season:
        events = events[events["Season"].astype(str) == str(season)]
    if events.empty:
        return _empty_defensive_squad_match_sums()

    regain_types = {"LOOSE_BALL_REGAIN", "INTERCEPTION", "GK_CATCH"}
    rows: list[dict[str, object]] = []
    for (match_id, event_season, team), group in events.groupby(["MatchId", "Season", "Team"], dropna=False):
        action_type = group["Action Type"].fillna("").astype(str).str.upper()
        result = group["Result"].fillna("").astype(str).str.upper()
        phase = group["Phase"].fillna("").astype(str).str.upper()
        lane = group["Start Lane"].fillna("").astype(str).str.upper()
        start_x = pd.to_numeric(group["Start X"], errors="coerce")
        regains = group[action_type.isin(regain_types)].copy()
        regain_x = pd.to_numeric(regains.get("Start X"), errors="coerce")
        opponent = events[(events["MatchId"].astype(str) == str(match_id)) & (events["Team"].astype(str) != str(team))]
        opponent_shots = opponent[opponent["Action Type"].astype(str).str.upper().eq("SHOT")]
        ground = action_type.eq("GROUND_DUEL")
        aerial = group["Body Part"].fillna("").astype(str).str.upper().eq("HEAD") & ground
        second_ball = phase.eq("SECOND_BALL")
        rows.append(
            {
                "MatchId": match_id,
                "Date": group["Date"].iloc[0],
                "Competition": group["Competition"].iloc[0],
                "Season": event_season,
                "TeamId": team,
                "Team": team,
                "Ball Wins": len(regains),
                "Ball Losses": int(result.isin({"FAIL", "FAILED", "UNSUCCESSFUL"}).sum()),
                "Opponents Removed": pd.to_numeric(regains.get("Bypassed Opponents"), errors="coerce").fillna(0).sum(),
                "Defenders Removed": pd.to_numeric(regains.get("Bypassed Defenders"), errors="coerce").fillna(0).sum(),
                "Ball Win Value": pd.to_numeric(regains.get("Team xT"), errors="coerce").fillna(0).sum(),
                "Defensive Touches": int(action_type.isin(regain_types | {"GROUND_DUEL", "BLOCK", "CLEARANCE"}).sum()),
                "Presses": int(group["Pressure"].notna().sum()),
                "Counterpresses": int(phase.str.contains("TRANSITION").sum()),
                "Build-Up Presses": int(phase.str.contains("BUILD").sum()),
                "Between-Lines Presses": 0,
                "Second Balls": int(second_ball.sum()),
                "Second Balls Won": int((second_ball & action_type.isin(regain_types)).sum()),
                "Ground Duels Won": int((ground & result.eq("SUCCESS")).sum()),
                "Ground Duels Lost": int((ground & result.isin({"FAIL", "FAILED", "UNSUCCESSFUL"})).sum()),
                "Aerial Duels Won": int((aerial & result.eq("SUCCESS")).sum()),
                "Aerial Duels Lost": int((aerial & result.isin({"FAIL", "FAILED", "UNSUCCESSFUL"})).sum()),
                "Suffered Bypassed Opponents": 0,
                "Suffered Bypassed Defenders": 0,
                "Goals Conceded": int(opponent_shots["Action"].astype(str).str.upper().eq("GOAL").sum()),
                "xG Conceded": pd.to_numeric(opponent_shots["Shot xG"], errors="coerce").fillna(0).sum(),
                "First-Third Ball Wins": int(regain_x.lt(-17.5).sum()),
                "Middle-Third Ball Wins": int(regain_x.between(-17.5, 17.5, inclusive="left").sum()),
                "Final-Third Ball Wins": int(regain_x.ge(17.5).sum()),
                "Opponent-Box Ball Wins": int(regain_x.ge(36.0).sum()),
                "Wide-Left Ball Wins": int((action_type.isin(regain_types) & lane.eq("LEFT")).sum()),
                "Half-Left Ball Wins": int((action_type.isin(regain_types) & lane.eq("HALF_LEFT")).sum()),
                "Centre Ball Wins": int((action_type.isin(regain_types) & lane.eq("CENTER")).sum()),
                "Half-Right Ball Wins": int((action_type.isin(regain_types) & lane.eq("HALF_RIGHT")).sum()),
                "Wide-Right Ball Wins": int((action_type.isin(regain_types) & lane.eq("RIGHT")).sum()),
                "Out-of-Possession Ball Wins": 0,
                "Defensive-Transition Ball Wins": int((action_type.isin(regain_types) & phase.str.contains("TRANSITION")).sum()),
                "Set-Piece Ball Wins": int((action_type.isin(regain_types) & phase.str.contains("SET_PIECE")).sum()),
                "Second-Ball Phase Wins": int((action_type.isin(regain_types) & second_ball).sum()),
            }
        )
    return _clean_defensive_squad_match_sums(pd.DataFrame(rows))


def _mock_defensive_player_match_sums(season: str | None = None) -> pd.DataFrame:
    events = _mock_match_events()
    if season:
        events = events[events["Season"].astype(str) == str(season)]
    events = events.dropna(subset=["Player"]).copy()
    if events.empty:
        return _empty_defensive_player_match_sums()

    regain_types = {"LOOSE_BALL_REGAIN", "INTERCEPTION", "GK_CATCH"}
    rows: list[dict[str, object]] = []
    group_columns = ["MatchId", "Season", "Team", "PlayerId", "Player"]
    for (match_id, event_season, team, player_id, player), group in events.groupby(group_columns, dropna=False):
        action_type = group["Action Type"].fillna("").astype(str).str.upper()
        result = group["Result"].fillna("").astype(str).str.upper()
        phase = group["Phase"].fillna("").astype(str).str.upper()
        regains = group[action_type.isin(regain_types)]
        ground = action_type.eq("GROUND_DUEL")
        aerial = group["Body Part"].fillna("").astype(str).str.upper().eq("HEAD") & ground
        second_ball = phase.eq("SECOND_BALL")
        positions = group["Position"].dropna().astype(str)
        rows.append(
            {
                "MatchId": match_id,
                "Date": group["Date"].iloc[0],
                "Competition": group["Competition"].iloc[0],
                "Season": event_season,
                "TeamId": team,
                "Team": team,
                "PlayerId": player_id,
                "Player": player,
                "Position": positions.mode().iloc[0] if not positions.empty else "",
                "Play Duration Seconds": 90 * 60,
                "Match Share": 1.0,
                "Ball Wins": len(regains),
                "Ball Losses": int(result.isin({"FAIL", "FAILED", "UNSUCCESSFUL"}).sum()),
                "Opponents Removed": pd.to_numeric(regains.get("Bypassed Opponents"), errors="coerce").fillna(0).sum(),
                "Defenders Removed": pd.to_numeric(regains.get("Bypassed Defenders"), errors="coerce").fillna(0).sum(),
                "Ball Win Value": pd.to_numeric(regains.get("Team xT"), errors="coerce").fillna(0).sum(),
                "Defensive Touches": int(action_type.isin(regain_types | {"GROUND_DUEL", "BLOCK", "CLEARANCE"}).sum()),
                "Presses": int(group["Pressure"].notna().sum()),
                "Counterpresses": int(phase.str.contains("TRANSITION").sum()),
                "Second Balls": int(second_ball.sum()),
                "Second Balls Won": int((second_ball & action_type.isin(regain_types)).sum()),
                "Ground Duels Won": int((ground & result.eq("SUCCESS")).sum()),
                "Ground Duels Lost": int((ground & result.isin({"FAIL", "FAILED", "UNSUCCESSFUL"})).sum()),
                "Aerial Duels Won": int((aerial & result.eq("SUCCESS")).sum()),
                "Aerial Duels Lost": int((aerial & result.isin({"FAIL", "FAILED", "UNSUCCESSFUL"})).sum()),
            }
        )
    return _clean_defensive_player_match_sums(pd.DataFrame(rows))


def _legacy_load_squad_defensive_match_sums(season: str | None = None) -> pd.DataFrame:
    """Compatibility alias for the current production squad-defence loader."""
    return load_squad_defensive_match_sums(season)


def _legacy_load_player_defensive_match_sums(season: str | None = None) -> pd.DataFrame:
    """Compatibility alias for the current production player-defence loader."""
    return load_player_defensive_match_sums(season)


def _match_fact_context(season: str | None) -> tuple[pd.DataFrame, str, list[int]]:
    contexts = _contexts_for_season(season)
    contexts = contexts[contexts["IterationId"].astype("Int64").isin(_event_iteration_ids())].copy()
    filter_sql, iteration_ids = _iteration_filter(contexts, "ITERATION_ID")
    return contexts, filter_sql, iteration_ids


def _match_dimensions(contexts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return typed match, squad and player dimensions for selected iterations."""
    filter_sql, iteration_ids = _iteration_filter(contexts, "ITERATION_ID")
    if not iteration_ids:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conn = get_connection()
    matches = conn.query(
        f"""
        SELECT
            IMPECT_MATCH_ID AS "MatchId",
            ITERATION_ID AS "IterationId",
            SCHEDULED_AT AS "Date",
            HOME_SQUAD_ID AS "HomeTeamId",
            AWAY_SQUAD_ID AS "AwayTeamId"
        FROM {relation("impect_matches")}
        WHERE {filter_sql}
        """,
        params=_snowflake_params(iteration_ids),
        ttl="1h",
    ).drop_duplicates(["IterationId", "MatchId"])
    squads = conn.query(
        f"""
        SELECT ITERATION_ID AS "IterationId", IMPECT_SQUAD_ID AS "TeamId", SQUAD_NAME AS "Team"
        FROM {relation("impect_squads")}
        WHERE {filter_sql}
        """,
        params=_snowflake_params(iteration_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "TeamId"])
    players = conn.query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            IMPECT_PLAYER_ID AS "PlayerId",
            COALESCE(NULLIF(COMMON_NAME, ''), TRIM(CONCAT_WS(' ', FIRST_NAME, LAST_NAME))) AS "Player"
        FROM {relation("impect_players")}
        WHERE {filter_sql}
        """,
        params=_snowflake_params(iteration_ids),
        ttl="6h",
    ).drop_duplicates(["IterationId", "PlayerId"])
    return matches, squads, players


def load_squad_defensive_match_sums(season: str | None = None) -> pd.DataFrame:
    """One row per team-match, pivoted from authoritative match KPI facts."""
    if USE_MOCK_DATA:
        return _mock_defensive_squad_match_sums(season)
    contexts, filter_sql, iteration_ids = _match_fact_context(season)
    if not iteration_ids:
        return _empty_defensive_squad_match_sums()

    kpi_ids = sorted(set(DEFENSIVE_SQUAD_KPI_IDS.values()))
    conn = get_connection()
    long_values = conn.query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            IMPECT_MATCH_ID AS "MatchId",
            IMPECT_SQUAD_ID AS "TeamId",
            IMPECT_KPI_ID AS "KpiId",
            KPI_VALUE AS "KpiValue"
        FROM {relation("impect_match_squad_kpis")}
        WHERE {filter_sql}
          AND IMPECT_KPI_ID IN ({", ".join(["?"] * len(kpi_ids))})
        """,
        params=_snowflake_params([*iteration_ids, *kpi_ids]),
        ttl="30m",
    )
    rows = _pivot_long_kpis(long_values, ["IterationId", "MatchId", "TeamId"], "KpiId", "KpiValue")
    if rows.empty:
        return _empty_defensive_squad_match_sums()
    for output_column, kpi_id in DEFENSIVE_SQUAD_KPI_IDS.items():
        rows[output_column] = pd.to_numeric(rows[kpi_id], errors="coerce") if kpi_id in rows else np.nan

    matches, squads, _ = _match_dimensions(contexts)
    rows = (
        rows.merge(matches, on=["IterationId", "MatchId"], how="left")
        .merge(squads, on=["IterationId", "TeamId"], how="left")
        .merge(contexts, on="IterationId", how="left")
        .drop(columns=["IterationId", *kpi_ids], errors="ignore")
    )
    return _clean_defensive_squad_match_sums(rows)


def load_player_defensive_match_sums(season: str | None = None) -> pd.DataFrame:
    """Player defensive match totals reconstructed from long Impect KPI rows."""
    if USE_MOCK_DATA:
        return _mock_defensive_player_match_sums(season)
    contexts, filter_sql, iteration_ids = _match_fact_context(season)
    if not iteration_ids:
        return _empty_defensive_player_match_sums()

    kpi_ids = sorted(set(DEFENSIVE_PLAYER_KPI_IDS.values()))
    conn = get_connection()
    long_values = conn.query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            IMPECT_MATCH_ID AS "MatchId",
            IMPECT_SQUAD_ID AS "TeamId",
            IMPECT_PLAYER_ID AS "PlayerId",
            POSITION_CODE AS "Position",
            PLAY_DURATION_SECONDS AS "Play Duration Seconds",
            MATCH_SHARE AS "Match Share",
            IMPECT_KPI_ID AS "KpiId",
            KPI_VALUE AS "KpiValue"
        FROM {relation("impect_match_player_kpis")}
        WHERE {filter_sql}
          AND IMPECT_KPI_ID IN ({", ".join(["?"] * len(kpi_ids))})
        """,
        params=_snowflake_params([*iteration_ids, *kpi_ids]),
        ttl="30m",
    )
    position_keys = ["IterationId", "MatchId", "TeamId", "PlayerId", "Position"]
    rows = _pivot_long_kpis(long_values, position_keys, "KpiId", "KpiValue")
    if rows.empty:
        return _empty_defensive_player_match_sums()
    meta = (
        long_values.groupby(position_keys, dropna=False, observed=True)
        .agg(
            **{
                "Play Duration Seconds": ("Play Duration Seconds", "max"),
                "Match Share": ("Match Share", "max"),
            }
        )
        .reset_index()
    )
    rows = rows.merge(meta, on=position_keys, how="left")
    for output_column, kpi_id in DEFENSIVE_PLAYER_KPI_IDS.items():
        rows[output_column] = pd.to_numeric(rows[kpi_id], errors="coerce") if kpi_id in rows else np.nan

    matches, squads, players = _match_dimensions(contexts)
    rows = (
        rows.merge(matches, on=["IterationId", "MatchId"], how="left")
        .merge(squads, on=["IterationId", "TeamId"], how="left")
        .merge(players, on=["IterationId", "PlayerId"], how="left")
        .merge(contexts, on="IterationId", how="left")
        .drop(columns=["IterationId", *kpi_ids], errors="ignore")
    )
    return _clean_defensive_player_match_sums(rows)


def _empty_match_events() -> pd.DataFrame:
    return pd.DataFrame(columns=MATCH_EVENT_COLUMNS)


def _empty_set_piece_sequences() -> pd.DataFrame:
    return pd.DataFrame(columns=SET_PIECE_SEQUENCE_COLUMNS)


def _empty_set_piece_events() -> pd.DataFrame:
    return pd.DataFrame(columns=SET_PIECE_EVENT_COLUMNS)


def _empty_pass_network() -> pd.DataFrame:
    return pd.DataFrame(columns=PASS_NETWORK_COLUMNS)


def _empty_match_player_minutes() -> pd.DataFrame:
    return pd.DataFrame(columns=MATCH_PLAYER_MINUTE_COLUMNS)


def _clean_match_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_match_events()
    for col in MATCH_EVENT_COLUMNS:
        if col not in df:
            df[col] = np.nan
    numeric_cols = [
        "Period",
        "Second",
        "Minute",
        "Event Number",
        "Sequence Index",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
        "Raw Start X",
        "Raw Start Y",
        "Raw End X",
        "Raw End Y",
        "Pass Distance",
        "Pass Angle",
        "Team xT",
        "PXT Pass",
        "PXT Shot",
        "Shot xG",
        "Post-Shot xG",
        "Packing xG",
        "Bypassed Opponents",
        "Bypassed Defenders",
        "Shot Distance",
        "Shot Angle",
        "Shot Target Y",
        "Shot Target Z",
        "Shot GK X",
        "Shot GK Y",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Second" in df:
        df["Minute"] = _event_elapsed_minutes(df)
    return df[MATCH_EVENT_COLUMNS].reset_index(drop=True)


def _clean_optional_text(values: pd.Series) -> pd.Series:
    """Normalise provider null strings without turning real nulls into text."""
    clean = values.astype("string").str.strip()
    return clean.mask(clean.str.lower().isin(["", "nan", "none", "null", "nat"]))


def _optional_boolean(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "won": True,
            "success": True,
            "false": False,
            "0": False,
            "no": False,
            "lost": False,
            "failure": False,
        }
    ).astype("boolean")


def _clean_set_piece_sequences(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_set_piece_sequences()

    clean = df.copy()
    for column in SET_PIECE_SEQUENCE_COLUMNS:
        if column not in clean:
            clean[column] = np.nan

    numeric_columns = [
        "Game Second",
        "Event Number",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
        "Shots",
        "Goals",
        "xG",
        "Second-Phase Shots",
        "Second-Phase Goals",
        "Second-Phase xG",
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    text_columns = [
        "Season",
        "Competition",
        "Home",
        "Away",
        "Team",
        "Opponent",
        "Category",
        "Adjusted Category",
        "Execution Type",
        "Start Zone",
        "Corner End Zone",
        "Corner Type",
        "Free Kick End Zone",
        "Free Kick Type",
        "Main Event Action Type",
        "Main Event Action",
        "Taker",
        "Main Event Outcome",
        "First Touch Player",
        "Indirect Header",
        "Second Touch Player",
        "Next Team",
    ]
    for column in text_columns:
        clean[column] = _clean_optional_text(clean[column])

    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    clean["First Touch Won"] = _optional_boolean(clean["First Touch Won"])
    clean["Second Touch Won"] = _optional_boolean(clean["Second Touch Won"])

    category = clean["Category"].fillna(clean["Adjusted Category"]).astype("string").str.upper()
    action_type = clean["Main Event Action Type"].astype("string").str.upper()
    free_kick_type = clean["Free Kick Type"].astype("string").str.upper()
    direct_free_kick = category.eq("FREE_KICK") & (
        action_type.eq("SHOT") | free_kick_type.str.contains("SHOT", na=False)
    )
    clean["Set Piece Type"] = np.select(
        [
            category.str.startswith("CORNER", na=False),
            category.eq("THROW_IN"),
            direct_free_kick,
            category.eq("FREE_KICK"),
        ],
        ["Corner", "Throw-In", "Direct Free Kick", "Indirect Free Kick"],
        default=category.str.replace("_", " ", regex=False).str.title(),
    )

    start_y = pd.to_numeric(clean["Start Y"], errors="coerce")
    clean["Side"] = np.select(
        [
            category.str.endswith("_LEFT", na=False),
            category.str.endswith("_RIGHT", na=False),
            start_y.abs().le(7.5),
            start_y.gt(7.5),
            start_y.lt(-7.5),
        ],
        ["Left", "Right", "Centre", "Left", "Right"],
        default="Unknown",
    )

    forward_gain = pd.to_numeric(clean["End X"], errors="coerce") - pd.to_numeric(
        clean["Start X"], errors="coerce"
    )
    clean["Long Throw"] = category.eq("THROW_IN") & forward_gain.ge(20)
    clean["Retained"] = (
        category.eq("THROW_IN")
        & clean["Next Team"].notna()
        & clean["Next Team"].astype(str).eq(clean["Team"].astype(str))
    )

    count_columns = ["Shots", "Goals", "Second-Phase Shots", "Second-Phase Goals"]
    value_columns = ["xG", "Second-Phase xG"]
    for column in count_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0).astype(int)
    for column in value_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0.0)

    return clean[SET_PIECE_SEQUENCE_COLUMNS].sort_values(
        ["Date", "MatchId", "Game Second", "Event Number"], na_position="last"
    ).reset_index(drop=True)


def _clean_set_piece_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_set_piece_events()

    clean = df.copy()
    for column in SET_PIECE_EVENT_COLUMNS:
        if column not in clean:
            clean[column] = np.nan

    numeric_columns = [
        "Period",
        "Second",
        "Minute",
        "Event Number",
        "Sequence Index",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
        "Shot xG",
        "PXT Set Piece",
        "Opponent PXT Set Piece",
        "Defensive PXT Set Piece",
        "Set Piece Phase Index",
        "Subphase Index",
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    text_columns = [
        "Season",
        "Competition",
        "Home",
        "Away",
        "Team",
        "Player",
        "Action Type",
        "Action",
        "Body Part",
        "Result",
        "Category",
        "Adjusted Category",
        "Execution Type",
        "Start Zone",
        "Corner End Zone",
        "Corner Type",
        "Free Kick End Zone",
        "Free Kick Type",
        "Main Event Player",
        "Main Event Outcome",
        "Pass Receiver",
        "First Touch Player",
        "Indirect Header",
        "Second Touch Player",
    ]
    for column in text_columns:
        clean[column] = _clean_optional_text(clean[column])

    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    clean["Main Event"] = _optional_boolean(clean["Main Event"])
    clean["First Touch Won"] = _optional_boolean(clean["First Touch Won"])
    clean["Second Touch Won"] = _optional_boolean(clean["Second Touch Won"])
    clean["Minute"] = _event_elapsed_minutes(clean)
    return clean[SET_PIECE_EVENT_COLUMNS].sort_values(
        ["MatchId", "Second", "Event Number"], na_position="last"
    ).reset_index(drop=True)


def _clean_pass_network(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_pass_network()
    for col in PASS_NETWORK_COLUMNS:
        if col not in df:
            df[col] = np.nan
    numeric_cols = ["Pass Count", "Passer X", "Passer Y", "Receiver X", "Receiver Y"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[PASS_NETWORK_COLUMNS].reset_index(drop=True)


def _mode_string(values: pd.Series) -> str:
    clean = values.dropna().astype(str).str.strip()
    clean = clean[~clean.str.lower().isin(["", "nan", "none", "null"])]
    if clean.empty:
        return ""
    mode = clean.mode()
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0])


def _clean_match_player_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_match_player_minutes()

    required = ["MatchId", "Season", "Team", "PlayerId", "Player", "Position", "_Duration Seconds", "_Match Share"]
    for col in required:
        if col not in df:
            df[col] = np.nan

    cleaned = df.copy()
    cleaned["_Duration Seconds"] = pd.to_numeric(cleaned["_Duration Seconds"], errors="coerce").fillna(0)
    cleaned["_Match Share"] = pd.to_numeric(cleaned["_Match Share"], errors="coerce").fillna(0)
    cleaned["Minutes"] = cleaned["_Duration Seconds"] / 60
    cleaned["Match Share"] = cleaned["_Match Share"]

    group_cols = ["MatchId", "Season", "Team", "PlayerId", "Player"]
    summary = cleaned.groupby(group_cols, dropna=False, as_index=False).agg(
        Minutes=("Minutes", "sum"),
        **{"Match Share": ("Match Share", "sum")},
    )
    positions = (
        cleaned.sort_values("_Duration Seconds", ascending=False)
        .drop_duplicates(group_cols)
        [group_cols + ["Position"]]
    )
    summary = summary.merge(positions, on=group_cols, how="left")
    summary["Minutes"] = pd.to_numeric(summary["Minutes"], errors="coerce").fillna(0).round(1)
    summary["Match Share"] = pd.to_numeric(summary["Match Share"], errors="coerce").fillna(0).round(3)
    summary["Position"] = summary["Position"].fillna("")
    return (
        summary[MATCH_PLAYER_MINUTE_COLUMNS]
        .sort_values(["Team", "Minutes", "Player"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _event_elapsed_minutes(df: pd.DataFrame) -> pd.Series:
    """Convert Impect event seconds into normal match minutes.

    Some CAFC_DB Impect event rows store second-half gameTimeInSec with a
    10000-second period offset rather than true elapsed match seconds. For
    example, period 2 can start at 10000 instead of 2700. This normalises those
    values back to football minutes while leaving already-cumulative seconds
    untouched.
    """
    second = pd.to_numeric(df["Second"], errors="coerce") if "Second" in df else pd.Series(np.nan, index=df.index)
    period = pd.to_numeric(df["Period"], errors="coerce") if "Period" in df else pd.Series(np.nan, index=df.index)

    if second.empty:
        return pd.Series(np.nan, index=df.index)

    period_base_seconds = np.select(
        [
            period.eq(1),
            period.eq(2),
            period.eq(3),
            period.eq(4),
        ],
        [
            0,
            45 * 60,
            90 * 60,
            105 * 60,
        ],
        default=np.nan,
    )
    offset_bucket = np.floor(second / 10000).clip(lower=0)
    fallback_base_seconds = offset_bucket * 45 * 60
    base_seconds = pd.Series(period_base_seconds, index=df.index).where(pd.notna(period_base_seconds), fallback_base_seconds)
    period_seconds = second % 10000
    elapsed_seconds = second.where(second < 10000, period_seconds + base_seconds)
    minute = np.floor(elapsed_seconds / 60) + 1
    return pd.to_numeric(minute, errors="coerce").clip(lower=0, upper=130)


def _legacy_load_match_player_minutes(
    season: str | None = None,
    match_id: object | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Compatibility alias for the current production player-minutes loader."""
    return load_match_player_minutes(season=season, match_id=match_id, team=team)


def load_match_player_minutes(
    season: str | None = None,
    match_id: object | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Selected-match player minutes from CAFC_DB match-player facts."""
    if USE_MOCK_DATA:
        events = _mock_match_events()
        if season:
            events = events[events["Season"].map(_season_key).eq(_season_key(season))]
        if match_id is not None:
            events = events[events["MatchId"].astype(str).eq(str(match_id))]
        if team:
            events = events[events["Team"].astype(str).eq(str(team))]
        if events.empty:
            return _empty_match_player_minutes()
        values = events.dropna(subset=["Player"]).drop_duplicates(
            ["MatchId", "Team", "PlayerId", "Player"]
        ).copy()
        values["_Duration Seconds"] = 90 * 60
        values["_Match Share"] = 1.0
        return _clean_match_player_minutes(
            values[
                [
                    "MatchId",
                    "Season",
                    "Team",
                    "PlayerId",
                    "Player",
                    "Position",
                    "_Duration Seconds",
                    "_Match Share",
                ]
            ]
        )

    contexts, filter_sql, iteration_ids = _match_fact_context(season)
    if not iteration_ids:
        return _empty_match_player_minutes()
    clauses = [filter_sql]
    params: list[object] = [*iteration_ids]
    if match_id is not None:
        clauses.append("IMPECT_MATCH_ID = ?")
        params.append(match_id)
    rows = get_connection().query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            IMPECT_MATCH_ID AS "MatchId",
            IMPECT_SQUAD_ID AS "TeamId",
            IMPECT_PLAYER_ID AS "PlayerId",
            POSITION_CODE AS "Position",
            MAX(PLAY_DURATION_SECONDS) AS "_Duration Seconds",
            MAX(MATCH_SHARE) AS "_Match Share"
        FROM {relation("impect_match_player_kpis")}
        WHERE {' AND '.join(clauses)}
        GROUP BY ITERATION_ID, IMPECT_MATCH_ID, IMPECT_SQUAD_ID, IMPECT_PLAYER_ID, POSITION_CODE
        """,
        params=_snowflake_params(params),
        ttl="30m",
    )
    if rows.empty:
        return _empty_match_player_minutes()
    _, squads, players = _match_dimensions(contexts)
    rows = (
        rows.merge(squads, on=["IterationId", "TeamId"], how="left")
        .merge(players, on=["IterationId", "PlayerId"], how="left")
        .merge(contexts[["IterationId", "Season"]], on="IterationId", how="left")
    )
    if team:
        rows = rows[rows["Team"].astype(str).eq(str(team))]
    return _clean_match_player_minutes(rows)


def _legacy_load_match_events(
    season: str | None = None,
    match_id: object | None = None,
    team: str | None = None,
    action_types: list[str] | tuple[str, ...] | None = None,
    limit: int = 6000,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
) -> pd.DataFrame:
    """Compatibility alias for the current production event loader."""
    return load_match_events(
        season=season,
        match_id=match_id,
        team=team,
        action_types=action_types,
        limit=limit,
        match_ids=match_ids,
    )


def _legacy_joined_load_match_events(
    season: str | None = None,
    match_id: object | None = None,
    team: str | None = None,
    action_types: list[str] | tuple[str, ...] | None = None,
    limit: int = 6000,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
) -> pd.DataFrame:
    """Compatibility alias for the current production event loader."""
    return load_match_events(
        season=season,
        match_id=match_id,
        team=team,
        action_types=action_types,
        limit=limit,
        match_ids=match_ids,
    )


def load_match_events(
    season: str | None = None,
    match_id: object | None = None,
    team: str | None = None,
    action_types: list[str] | tuple[str, ...] | None = None,
    limit: int = 6000,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
) -> pd.DataFrame:
    """Flatten CAFC_DB events, joining lightweight dimensions in Pandas.

    Keeping the nested-event extraction separate from the dimension joins is
    materially faster on the development warehouse than joining the staging
    views while Snowflake is also scanning the VARIANT event fields.
    """
    if USE_MOCK_DATA:
        events = _mock_match_events()
        if season:
            events = events[events["Season"].map(_season_key).eq(_season_key(season))]
        if match_id is not None:
            events = events[events["MatchId"].astype(str).eq(str(match_id))]
        if match_ids:
            wanted_matches = {str(value) for value in match_ids}
            events = events[events["MatchId"].astype(str).isin(wanted_matches)]
        if team:
            events = events[events["Team"].astype(str).eq(str(team))]
        if action_types:
            wanted = {str(action).upper() for action in action_types}
            events = events[events["Action Type"].astype(str).str.upper().isin(wanted)]
        return _clean_match_events(events.head(max(int(limit), 1)).copy())

    contexts, iteration_clause, iteration_ids = _match_fact_context(season)
    if not iteration_ids:
        return _empty_match_events()
    matches, squads, players = _match_dimensions(contexts)

    clauses = [iteration_clause]
    params: list[object] = [*iteration_ids]
    if match_id is not None:
        clauses.append("MATCH_ID = ?")
        params.append(match_id)
    if match_ids:
        selected_ids = list(dict.fromkeys(match_ids))
        clauses.append(f'MATCH_ID IN ({", ".join(["?"] * len(selected_ids))})')
        params.extend(selected_ids)
    if team:
        selected_team_ids = (
            squads.loc[squads["Team"].astype(str).eq(str(team)), "TeamId"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        if not selected_team_ids:
            return _empty_match_events()
        clauses.append(f'SQUAD_ID IN ({", ".join(["?"] * len(selected_team_ids))})')
        params.extend(selected_team_ids)
    if action_types:
        clauses.append(f'ACTION_TYPE IN ({", ".join(["?"] * len(action_types))})')
        params.extend(action_types)

    maximum_limit = 120000 if match_ids else 60000
    safe_limit = max(min(int(limit), maximum_limit), 1)
    events = get_connection().query(
        f"""
        SELECT
            ITERATION_ID AS "IterationId",
            MATCH_ID AS "MatchId",
            SQUAD_ID AS "TeamId",
            PLAYER_ID AS "PlayerId",
            PLAYER_POSITION AS "Position",
            PERIOD_ID AS "Period",
            GAME_TIME AS "Game Time",
            GAME_TIME_IN_SEC AS "Second",
            FLOOR(GAME_TIME_IN_SEC / 60) + 1 AS "Minute",
            EVENT_INDEX AS "Event Number",
            SEQUENCE_INDEX AS "Sequence Index",
            PHASE AS "Phase",
            ACTION_TYPE AS "Action Type",
            ACTION AS "Action",
            BODY_PART AS "Body Part",
            RESULT AS "Result",
            PRESSURE AS "Pressure",
            START_DETAIL:"adjCoordinates":"x"::FLOAT AS "Start X",
            START_DETAIL:"adjCoordinates":"y"::FLOAT AS "Start Y",
            END_DETAIL:"adjCoordinates":"x"::FLOAT AS "End X",
            END_DETAIL:"adjCoordinates":"y"::FLOAT AS "End Y",
            START_DETAIL:"coordinates":"x"::FLOAT AS "Raw Start X",
            START_DETAIL:"coordinates":"y"::FLOAT AS "Raw Start Y",
            END_DETAIL:"coordinates":"x"::FLOAT AS "Raw End X",
            END_DETAIL:"coordinates":"y"::FLOAT AS "Raw End Y",
            START_DETAIL:"lane"::STRING AS "Start Lane",
            END_DETAIL:"lane"::STRING AS "End Lane",
            START_DETAIL:"pitchPosition"::STRING AS "Start Pitch Position",
            END_DETAIL:"pitchPosition"::STRING AS "End Pitch Position",
            PASS_DETAIL:"receiver":"playerId"::NUMBER AS "ReceiverId",
            PASS_DETAIL:"distance"::FLOAT AS "Pass Distance",
            PASS_DETAIL:"angle"::FLOAT AS "Pass Angle",
            PXT_DETAIL:"team"::FLOAT AS "Team xT",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"PXT_PASS"::STRING), NULL) AS "PXT Pass",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"PXT_SHOT"::STRING), NULL) AS "PXT Shot",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"SHOT_XG"::STRING), NULL) AS "Shot xG",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"POSTSHOT_XG"::STRING), NULL) AS "Post-Shot xG",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"PACKING_XG"::STRING), NULL) AS "Packing xG",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"BYPASSED_OPPONENTS"::STRING), NULL) AS "Bypassed Opponents",
            IFF(TRY_TO_NUMBER(EVENT_KPIS[0]:"playerId"::STRING) = PLAYER_ID,
                TRY_TO_DOUBLE(EVENT_KPIS[0]:"BYPASSED_DEFENDERS"::STRING), NULL) AS "Bypassed Defenders",
            SHOT_DETAIL:"distance"::FLOAT AS "Shot Distance",
            SHOT_DETAIL:"angle"::FLOAT AS "Shot Angle",
            SHOT_DETAIL:"targetPoint":"y"::FLOAT AS "Shot Target Y",
            SHOT_DETAIL:"targetPoint":"z"::FLOAT AS "Shot Target Z",
            SHOT_DETAIL:"gk":"adjCoordinates":"x"::FLOAT AS "Shot GK X",
            SHOT_DETAIL:"gk":"adjCoordinates":"y"::FLOAT AS "Shot GK Y",
            INFERRED_SET_PIECE AS "Set Piece",
            CASE
                WHEN ACTION_TYPE = 'CORNER' THEN 'CORNER'
                WHEN ACTION_TYPE = 'THROW_IN' THEN 'THROW_IN'
                WHEN ACTION_TYPE = 'FREE_KICK' OR ACTION = 'DIRECT_FREE_KICK' THEN 'FREE_KICK'
            END AS "Set Piece Category",
            CASE WHEN ACTION = 'DIRECT_FREE_KICK' THEN 'DIRECT' END AS "Set Piece Execution"
        FROM {relation("impect_events")}
        WHERE {' AND '.join(clauses)}
        ORDER BY GAME_TIME_IN_SEC, EVENT_INDEX
        LIMIT {safe_limit}
        """,
        params=_snowflake_params(params),
        ttl="30m",
    )
    if events.empty:
        return _empty_match_events()

    home = squads.rename(columns={"TeamId": "HomeTeamId", "Team": "Home"})
    away = squads.rename(columns={"TeamId": "AwayTeamId", "Team": "Away"})
    team_dimensions = squads.rename(columns={"Team": "Team"})
    receiver_dimensions = players.rename(columns={"PlayerId": "ReceiverId", "Player": "Receiver"})
    # Normalise dtypes so merge keys are compatible
    events["ReceiverId"] = pd.to_numeric(events["ReceiverId"], errors="coerce")
    receiver_dimensions["ReceiverId"] = pd.to_numeric(receiver_dimensions["ReceiverId"], errors="coerce")
    events = (
        events.merge(matches, on=["IterationId", "MatchId"], how="left")
        .merge(home, on=["IterationId", "HomeTeamId"], how="left")
        .merge(away, on=["IterationId", "AwayTeamId"], how="left")
        .merge(team_dimensions, on=["IterationId", "TeamId"], how="left")
        .merge(players, on=["IterationId", "PlayerId"], how="left")
        .merge(receiver_dimensions, on=["IterationId", "ReceiverId"], how="left")
        .merge(contexts, on="IterationId", how="left")
    )
    return _clean_match_events(events)


def _legacy_load_set_piece_sequences(
    season: str | None = None,
    match_id: object | None = None,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
) -> pd.DataFrame:
    """Compatibility alias for the current production set-piece loader."""
    return load_set_piece_sequences(
        season=season,
        match_id=match_id,
        match_ids=match_ids,
    )


def load_set_piece_sequences(
    season: str | None = None,
    match_id: object | None = None,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
) -> pd.DataFrame:
    """One reconstructed set-piece row per CAFC_DB event sequence.

    Raw Impect stores the provider set-piece id and main-event flag but not the
    old pre-enriched 226-column labels. Categories, delivery families and
    outcomes are therefore derived transparently from the complete sequence.
    """
    if USE_MOCK_DATA or not season or not _is_event_match_season(season):
        return _empty_set_piece_sequences()
    contexts, iteration_clause, iteration_ids = _match_fact_context(season)
    if not iteration_ids:
        return _empty_set_piece_sequences()

    clauses = [iteration_clause.replace("ITERATION_ID", "e.ITERATION_ID")]
    params: list[object] = [*iteration_ids]
    if match_id is not None:
        clauses.append("e.MATCH_ID = ?")
        params.append(match_id)
    if match_ids:
        selected_ids = list(dict.fromkeys(match_ids))
        clauses.append(f'e.MATCH_ID IN ({", ".join(["?"] * len(selected_ids))})')
        params.extend(selected_ids)

    raw = get_connection().query(
        f"""
        WITH source_events AS (
            SELECT
                e.*,
                e.SET_PIECE_DETAIL:"id"::NUMBER AS SP_ID,
                e.SET_PIECE_DETAIL:"mainEvent"::BOOLEAN AS SP_MAIN_EVENT,
                e.SET_PIECE_DETAIL:"subPhaseId"::NUMBER AS SP_SUBPHASE_ID,
                e.PASS_DETAIL:"receiver":"playerId"::NUMBER AS RECEIVER_ID,
                e.PASS_DETAIL:"receiver":"type"::STRING AS RECEIVER_TYPE,
                IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID,
                    TRY_TO_DOUBLE(e.EVENT_KPIS[0]:"SHOT_XG"::STRING), NULL) AS EVENT_SHOT_XG
            FROM {relation("impect_events")} e
            WHERE {' AND '.join(clauses)}
        ),
        anchors AS (
            SELECT *
            FROM source_events
            WHERE SP_ID IS NOT NULL
              AND (ACTION_TYPE IN ('CORNER', 'FREE_KICK', 'THROW_IN') OR ACTION = 'DIRECT_FREE_KICK')
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ITERATION_ID, MATCH_ID, SEQUENCE_INDEX
                ORDER BY IFF(SP_MAIN_EVENT, 0, 1), EVENT_INDEX
            ) = 1
        ),
        outcomes AS (
            SELECT
                a.ITERATION_ID,
                a.MATCH_ID,
                a.SEQUENCE_INDEX,
                COUNT_IF(
                    e.SQUAD_ID = a.SQUAD_ID
                    AND e.ACTION_TYPE = 'SHOT'
                    AND COALESCE(e.ACTION, '') <> 'PENALTY_KICK'
                ) AS SHOTS,
                SUM(IFF(
                    e.SQUAD_ID = a.SQUAD_ID
                    AND e.ACTION_TYPE = 'SHOT'
                    AND COALESCE(e.ACTION, '') <> 'PENALTY_KICK',
                    COALESCE(e.EVENT_SHOT_XG, 0), 0
                )) AS XG,
                COUNT_IF(
                    (
                        (e.SQUAD_ID = a.SQUAD_ID AND (e.ACTION = 'GOAL' OR e.ACTION_TYPE = 'GOAL'))
                        OR (e.SQUAD_ID <> a.SQUAD_ID AND (e.ACTION = 'OWN_GOAL' OR e.ACTION_TYPE = 'OWN_GOAL'))
                    )
                ) AS GOALS,
                COUNT_IF(
                    e.SQUAD_ID = a.SQUAD_ID
                    AND e.ACTION_TYPE = 'SHOT'
                    AND COALESCE(e.ACTION, '') <> 'PENALTY_KICK'
                    AND (
                        e.PHASE = 'SECOND_BALL'
                        OR (e.PHASE IS NOT NULL AND e.PHASE <> 'SET_PIECE')
                        OR (a.SP_SUBPHASE_ID IS NOT NULL AND e.SP_SUBPHASE_ID IS NOT NULL
                            AND e.SP_SUBPHASE_ID <> a.SP_SUBPHASE_ID)
                    )
                ) AS SECOND_PHASE_SHOTS,
                SUM(IFF(
                    e.SQUAD_ID = a.SQUAD_ID
                    AND e.ACTION_TYPE = 'SHOT'
                    AND COALESCE(e.ACTION, '') <> 'PENALTY_KICK'
                    AND (
                        e.PHASE = 'SECOND_BALL'
                        OR (e.PHASE IS NOT NULL AND e.PHASE <> 'SET_PIECE')
                        OR (a.SP_SUBPHASE_ID IS NOT NULL AND e.SP_SUBPHASE_ID IS NOT NULL
                            AND e.SP_SUBPHASE_ID <> a.SP_SUBPHASE_ID)
                    ), COALESCE(e.EVENT_SHOT_XG, 0), 0
                )) AS SECOND_PHASE_XG,
                COUNT_IF(
                    (
                        e.PHASE = 'SECOND_BALL'
                        OR (e.PHASE IS NOT NULL AND e.PHASE <> 'SET_PIECE')
                        OR (a.SP_SUBPHASE_ID IS NOT NULL AND e.SP_SUBPHASE_ID IS NOT NULL
                            AND e.SP_SUBPHASE_ID <> a.SP_SUBPHASE_ID)
                    )
                    AND (
                        (e.SQUAD_ID = a.SQUAD_ID AND (e.ACTION = 'GOAL' OR e.ACTION_TYPE = 'GOAL'))
                        OR (e.SQUAD_ID <> a.SQUAD_ID AND (e.ACTION = 'OWN_GOAL' OR e.ACTION_TYPE = 'OWN_GOAL'))
                    )
                ) AS SECOND_PHASE_GOALS
            FROM anchors a
            LEFT JOIN source_events e
              ON e.ITERATION_ID = a.ITERATION_ID
             AND e.MATCH_ID = a.MATCH_ID
             AND e.SEQUENCE_INDEX = a.SEQUENCE_INDEX
            GROUP BY a.ITERATION_ID, a.MATCH_ID, a.SEQUENCE_INDEX
        )
        SELECT
            a.ITERATION_ID AS "IterationId",
            a.MATCH_ID AS "MatchId",
            a.GAME_TIME_IN_SEC AS "Game Second",
            a.EVENT_INDEX AS "Event Number",
            a.SQUAD_ID AS "TeamId",
            a.SP_ID AS "Set Piece ID",
            CASE
                WHEN a.ACTION_TYPE = 'CORNER' THEN 'CORNER'
                WHEN a.ACTION_TYPE = 'THROW_IN' THEN 'THROW_IN'
                ELSE 'FREE_KICK'
            END AS "Category",
            CASE
                WHEN a.ACTION = 'DIRECT_FREE_KICK' THEN 'DIRECT'
                WHEN a.ACTION_TYPE = 'THROW_IN' THEN 'THROW'
                ELSE a.ACTION
            END AS "Execution Type",
            a.START_DETAIL:"adjCoordinates":"x"::FLOAT AS "Start X",
            a.START_DETAIL:"adjCoordinates":"y"::FLOAT AS "Start Y",
            a.END_DETAIL:"adjCoordinates":"x"::FLOAT AS "End X",
            a.END_DETAIL:"adjCoordinates":"y"::FLOAT AS "End Y",
            a.START_DETAIL:"pitchPosition"::STRING AS "Start Zone",
            a.END_DETAIL:"pitchPosition"::STRING AS "End Pitch Position",
            a.ACTION_TYPE AS "Main Event Action Type",
            a.ACTION AS "Main Event Action",
            a.PLAYER_ID AS "Taker Id",
            a.RESULT AS "Main Event Outcome",
            a.RECEIVER_ID AS "First Touch Player Id",
            a.RECEIVER_TYPE AS "Receiver Type",
            o.SHOTS AS "Shots",
            o.GOALS AS "Goals",
            o.XG AS "xG",
            o.SECOND_PHASE_SHOTS AS "Second-Phase Shots",
            o.SECOND_PHASE_GOALS AS "Second-Phase Goals",
            o.SECOND_PHASE_XG AS "Second-Phase xG"
        FROM anchors a
        LEFT JOIN outcomes o
          ON o.ITERATION_ID = a.ITERATION_ID
         AND o.MATCH_ID = a.MATCH_ID
         AND o.SEQUENCE_INDEX = a.SEQUENCE_INDEX
        ORDER BY a.MATCH_ID, a.EVENT_INDEX
        """,
        params=_snowflake_params(params),
        ttl="30m",
    )
    if raw.empty:
        return _empty_set_piece_sequences()

    matches, squads, players = _match_dimensions(contexts)
    home = squads.rename(columns={"TeamId": "HomeTeamId", "Team": "Home"})
    away = squads.rename(columns={"TeamId": "AwayTeamId", "Team": "Away"})
    takers = players.rename(columns={"PlayerId": "Taker Id", "Player": "Taker"})
    contacts = players.rename(
        columns={"PlayerId": "First Touch Player Id", "Player": "First Touch Player"}
    )
    # Normalise dtypes so merge keys are compatible
    raw["First Touch Player Id"] = pd.to_numeric(raw["First Touch Player Id"], errors="coerce")
    contacts["First Touch Player Id"] = pd.to_numeric(contacts["First Touch Player Id"], errors="coerce")
    raw = (
        raw.merge(matches, on=["IterationId", "MatchId"], how="left")
        .merge(home, on=["IterationId", "HomeTeamId"], how="left")
        .merge(away, on=["IterationId", "AwayTeamId"], how="left")
        .merge(squads, on=["IterationId", "TeamId"], how="left")
        .merge(takers, on=["IterationId", "Taker Id"], how="left")
        .merge(contacts, on=["IterationId", "First Touch Player Id"], how="left")
        .merge(contexts, on="IterationId", how="left")
    )
    raw["Opponent"] = np.where(raw["Team"].eq(raw["Home"]), raw["Away"], raw["Home"])
    receiver_type = raw["Receiver Type"].fillna("").astype(str).str.upper()
    raw["First Touch Won"] = receiver_type.map({"TEAMMATE": True, "OPPONENT": False})
    raw["Next Team"] = np.select(
        [receiver_type.eq("TEAMMATE"), receiver_type.eq("OPPONENT")],
        [raw["Team"], raw["Opponent"]],
        default=None,
    )
    raw["Adjusted Category"] = raw["Category"]

    category = raw["Category"].fillna("").astype(str).str.upper()
    end_y = pd.to_numeric(raw["End Y"], errors="coerce")
    start_y = pd.to_numeric(raw["Start Y"], errors="coerce")
    side = np.where(start_y.gt(0), "LEFT", "RIGHT")
    same_side = ((side == "LEFT") & end_y.gt(0)) | ((side == "RIGHT") & end_y.lt(0))
    corner_zone = np.select(
        [
            ~raw["End Pitch Position"].fillna("").astype(str).str.upper().eq("OPPONENT_BOX"),
            end_y.abs().le(7),
            same_side,
        ],
        ["Short", "Central", "Near Post"],
        default="Far Post",
    )
    raw["Corner End Zone"] = np.where(category.eq("CORNER"), corner_zone, None)
    raw["Corner Type"] = raw["Corner End Zone"]
    raw["Free Kick End Zone"] = np.where(category.eq("FREE_KICK"), corner_zone, None)
    delivered_to_box = raw["End Pitch Position"].fillna("").astype(str).str.upper().eq("OPPONENT_BOX")
    raw["Free Kick Type"] = np.where(
        raw["Main Event Action"].fillna("").astype(str).str.upper().eq("DIRECT_FREE_KICK"),
        "SHOT",
        np.where(category.eq("FREE_KICK") & delivered_to_box, "CROSS", np.where(category.eq("FREE_KICK"), "SHORT", None)),
    )
    raw["Indirect Header"] = pd.NA
    raw["Second Touch Player Id"] = pd.NA
    raw["Second Touch Player"] = pd.NA
    raw["Second Touch Won"] = pd.NA
    return _clean_set_piece_sequences(raw)


def _legacy_load_set_piece_events(
    season: str | None = None,
    match_id: object | None = None,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
    limit: int = 120000,
) -> pd.DataFrame:
    """Compatibility alias for the current production set-piece event loader."""
    return load_set_piece_events(
        season=season,
        match_id=match_id,
        match_ids=match_ids,
        limit=limit,
    )


def load_set_piece_events(
    season: str | None = None,
    match_id: object | None = None,
    match_ids: list[object] | tuple[object, ...] | set[object] | None = None,
    limit: int = 120000,
) -> pd.DataFrame:
    """Detailed rows from complete CAFC_DB set-piece possession sequences."""
    if USE_MOCK_DATA or not season or not _is_event_match_season(season):
        return _empty_set_piece_events()
    contexts, iteration_clause, iteration_ids = _match_fact_context(season)
    if not iteration_ids:
        return _empty_set_piece_events()

    clauses = [iteration_clause.replace("ITERATION_ID", "e.ITERATION_ID")]
    params: list[object] = [*iteration_ids]
    if match_id is not None:
        clauses.append("e.MATCH_ID = ?")
        params.append(match_id)
    if match_ids:
        selected_ids = list(dict.fromkeys(match_ids))
        clauses.append(f'e.MATCH_ID IN ({", ".join(["?"] * len(selected_ids))})')
        params.extend(selected_ids)
    safe_limit = max(min(int(limit), 120000), 1)

    raw = get_connection().query(
        f"""
        WITH source_events AS (
            SELECT
                e.*,
                e.SET_PIECE_DETAIL:"id"::NUMBER AS SP_ID,
                e.SET_PIECE_DETAIL:"mainEvent"::BOOLEAN AS SP_MAIN_EVENT,
                e.SET_PIECE_DETAIL:"subPhaseId"::NUMBER AS SP_SUBPHASE_ID,
                e.PASS_DETAIL:"receiver":"playerId"::NUMBER AS RECEIVER_ID,
                e.PASS_DETAIL:"receiver":"type"::STRING AS RECEIVER_TYPE
            FROM {relation("impect_events")} e
            WHERE {' AND '.join(clauses)}
        ),
        target_sequences AS (
            SELECT DISTINCT ITERATION_ID, MATCH_ID, SEQUENCE_INDEX
            FROM source_events
            WHERE SP_ID IS NOT NULL
              AND (ACTION_TYPE IN ('CORNER', 'FREE_KICK', 'THROW_IN') OR ACTION = 'DIRECT_FREE_KICK')
        ),
        anchors AS (
            SELECT *
            FROM source_events
            WHERE SP_ID IS NOT NULL
              AND (ACTION_TYPE IN ('CORNER', 'FREE_KICK', 'THROW_IN') OR ACTION = 'DIRECT_FREE_KICK')
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ITERATION_ID, MATCH_ID, SEQUENCE_INDEX
                ORDER BY IFF(SP_MAIN_EVENT, 0, 1), EVENT_INDEX
            ) = 1
        )
        SELECT
            e.ITERATION_ID AS "IterationId",
            e.MATCH_ID AS "MatchId",
            e.SQUAD_ID AS "TeamId",
            e.PLAYER_ID AS "PlayerId",
            e.PERIOD_ID AS "Period",
            e.GAME_TIME AS "Game Time",
            e.GAME_TIME_IN_SEC AS "Second",
            FLOOR(e.GAME_TIME_IN_SEC / 60) + 1 AS "Minute",
            e.EVENT_INDEX AS "Event Number",
            e.SEQUENCE_INDEX AS "Sequence Index",
            e.ACTION_TYPE AS "Action Type",
            e.ACTION AS "Action",
            e.BODY_PART AS "Body Part",
            e.RESULT AS "Result",
            e.START_DETAIL:"adjCoordinates":"x"::FLOAT AS "Start X",
            e.START_DETAIL:"adjCoordinates":"y"::FLOAT AS "Start Y",
            e.END_DETAIL:"adjCoordinates":"x"::FLOAT AS "End X",
            e.END_DETAIL:"adjCoordinates":"y"::FLOAT AS "End Y",
            IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID,
                TRY_TO_DOUBLE(e.EVENT_KPIS[0]:"SHOT_XG"::STRING), NULL) AS "Shot xG",
            IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID,
                TRY_TO_DOUBLE(e.EVENT_KPIS[0]:"PXT_SETPIECE"::STRING), NULL) AS "PXT Set Piece",
            IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID,
                TRY_TO_DOUBLE(e.EVENT_KPIS[0]:"OPP_PXT_SETPIECE"::STRING), NULL) AS "Opponent PXT Set Piece",
            IFF(TRY_TO_NUMBER(e.EVENT_KPIS[0]:"playerId"::STRING) = e.PLAYER_ID,
                TRY_TO_DOUBLE(e.EVENT_KPIS[0]:"DEF_PXT_SETPIECE"::STRING), NULL) AS "Defensive PXT Set Piece",
            a.SP_ID AS "Set Piece ID",
            IFF(
                e.PHASE = 'SECOND_BALL'
                OR (a.SP_SUBPHASE_ID IS NOT NULL AND e.SP_SUBPHASE_ID IS NOT NULL
                    AND e.SP_SUBPHASE_ID <> a.SP_SUBPHASE_ID), 1, 0
            ) AS "Set Piece Phase Index",
            CASE
                WHEN a.ACTION_TYPE = 'CORNER' THEN 'CORNER'
                WHEN a.ACTION_TYPE = 'THROW_IN' THEN 'THROW_IN'
                ELSE 'FREE_KICK'
            END AS "Category",
            CASE
                WHEN a.ACTION = 'DIRECT_FREE_KICK' THEN 'DIRECT'
                WHEN a.ACTION_TYPE = 'THROW_IN' THEN 'THROW'
                ELSE a.ACTION
            END AS "Execution Type",
            e.SP_SUBPHASE_ID AS "Subphase ID",
            IFF(
                e.PHASE = 'SECOND_BALL'
                OR (a.SP_SUBPHASE_ID IS NOT NULL AND e.SP_SUBPHASE_ID IS NOT NULL
                    AND e.SP_SUBPHASE_ID <> a.SP_SUBPHASE_ID), 1, 0
            ) AS "Subphase Index",
            a.START_DETAIL:"pitchPosition"::STRING AS "Start Zone",
            IFF(e.EVENT_ID = a.EVENT_ID, TRUE, FALSE) AS "Main Event",
            a.PLAYER_ID AS "Main Event Player Id",
            a.RESULT AS "Main Event Outcome",
            e.RECEIVER_ID AS "Pass Receiver Id",
            a.RECEIVER_ID AS "First Touch Player Id",
            a.RECEIVER_TYPE AS "Receiver Type",
            a.ACTION AS "Anchor Action",
            a.END_DETAIL:"pitchPosition"::STRING AS "Anchor End Pitch Position",
            a.END_DETAIL:"adjCoordinates":"y"::FLOAT AS "Anchor End Y",
            a.START_DETAIL:"adjCoordinates":"y"::FLOAT AS "Anchor Start Y"
        FROM source_events e
        JOIN target_sequences t
          ON t.ITERATION_ID = e.ITERATION_ID
         AND t.MATCH_ID = e.MATCH_ID
         AND t.SEQUENCE_INDEX = e.SEQUENCE_INDEX
        JOIN anchors a
          ON a.ITERATION_ID = e.ITERATION_ID
         AND a.MATCH_ID = e.MATCH_ID
         AND a.SEQUENCE_INDEX = e.SEQUENCE_INDEX
        ORDER BY e.MATCH_ID, e.EVENT_INDEX
        LIMIT {safe_limit}
        """,
        params=_snowflake_params(params),
        ttl="30m",
    )
    if raw.empty:
        return _empty_set_piece_events()

    matches, squads, players = _match_dimensions(contexts)
    home = squads.rename(columns={"TeamId": "HomeTeamId", "Team": "Home"})
    away = squads.rename(columns={"TeamId": "AwayTeamId", "Team": "Away"})
    main_players = players.rename(
        columns={"PlayerId": "Main Event Player Id", "Player": "Main Event Player"}
    )
    pass_receivers = players.rename(columns={"PlayerId": "Pass Receiver Id", "Player": "Pass Receiver"})
    first_contacts = players.rename(
        columns={"PlayerId": "First Touch Player Id", "Player": "First Touch Player"}
    )
    # Normalise dtypes so merge keys are compatible
    raw["Main Event Player Id"] = pd.to_numeric(raw["Main Event Player Id"], errors="coerce")
    main_players["Main Event Player Id"] = pd.to_numeric(main_players["Main Event Player Id"], errors="coerce")
    raw["Pass Receiver Id"] = pd.to_numeric(raw["Pass Receiver Id"], errors="coerce")
    pass_receivers["Pass Receiver Id"] = pd.to_numeric(pass_receivers["Pass Receiver Id"], errors="coerce")
    raw["First Touch Player Id"] = pd.to_numeric(raw["First Touch Player Id"], errors="coerce")
    first_contacts["First Touch Player Id"] = pd.to_numeric(first_contacts["First Touch Player Id"], errors="coerce")
    raw = (
        raw.merge(matches, on=["IterationId", "MatchId"], how="left")
        .merge(home, on=["IterationId", "HomeTeamId"], how="left")
        .merge(away, on=["IterationId", "AwayTeamId"], how="left")
        .merge(squads, on=["IterationId", "TeamId"], how="left")
        .merge(players, on=["IterationId", "PlayerId"], how="left")
        .merge(main_players, on=["IterationId", "Main Event Player Id"], how="left")
        .merge(pass_receivers, on=["IterationId", "Pass Receiver Id"], how="left")
        .merge(first_contacts, on=["IterationId", "First Touch Player Id"], how="left")
        .merge(contexts, on="IterationId", how="left")
    )
    raw["Adjusted Category"] = raw["Category"]
    receiver_type = raw["Receiver Type"].fillna("").astype(str).str.upper()
    raw["First Touch Won"] = receiver_type.map({"TEAMMATE": True, "OPPONENT": False})
    raw["Indirect Header"] = pd.NA
    raw["Second Touch Player Id"] = pd.NA
    raw["Second Touch Player"] = pd.NA
    raw["Second Touch Won"] = pd.NA

    category = raw["Category"].fillna("").astype(str).str.upper()
    end_y = pd.to_numeric(raw["Anchor End Y"], errors="coerce")
    start_y = pd.to_numeric(raw["Anchor Start Y"], errors="coerce")
    same_side = ((start_y.gt(0)) & end_y.gt(0)) | ((start_y.lt(0)) & end_y.lt(0))
    delivery_zone = np.select(
        [
            ~raw["Anchor End Pitch Position"].fillna("").astype(str).str.upper().eq("OPPONENT_BOX"),
            end_y.abs().le(7),
            same_side,
        ],
        ["Short", "Central", "Near Post"],
        default="Far Post",
    )
    raw["Corner End Zone"] = np.where(category.eq("CORNER"), delivery_zone, None)
    raw["Corner Type"] = raw["Corner End Zone"]
    raw["Free Kick End Zone"] = np.where(category.eq("FREE_KICK"), delivery_zone, None)
    delivered_to_box = raw["Anchor End Pitch Position"].fillna("").astype(str).str.upper().eq("OPPONENT_BOX")
    raw["Free Kick Type"] = np.where(
        raw["Anchor Action"].fillna("").astype(str).str.upper().eq("DIRECT_FREE_KICK"),
        "SHOT",
        np.where(category.eq("FREE_KICK") & delivered_to_box, "CROSS", np.where(category.eq("FREE_KICK"), "SHORT", None)),
    )
    return _clean_set_piece_events(raw)


def _legacy_load_pass_network(
    match_id: object | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """Compatibility alias for the current production pass-network loader."""
    return load_pass_network(match_id=match_id, team=team)


def load_pass_network(match_id: object | None = None, team: str | None = None) -> pd.DataFrame:
    """Passer-to-receiver links derived from the CAFC_DB event feed."""
    if USE_MOCK_DATA:
        network = _mock_pass_network()
        if match_id is not None:
            network = network[network["MatchId"].astype(str).eq(str(match_id))]
        if team:
            network = network[network["Team"].astype(str).eq(str(team))]
        return _clean_pass_network(network.copy())
    if match_id is None:
        return _empty_pass_network()

    passes = load_match_events(
        match_id=match_id,
        team=team,
        action_types=["PASS"],
        limit=60000,
    )
    if passes.empty:
        return _empty_pass_network()
    result = passes["Result"].fillna("").astype(str).str.upper()
    successful = passes[result.isin({"SUCCESS", "SUCCESSFUL", "WON"})].copy()
    if successful.empty:
        successful = passes[~passes["ReceiverId"].isna()].copy()
    successful = successful.dropna(subset=["PlayerId", "ReceiverId"])
    if successful.empty:
        return _empty_pass_network()

    network = (
        successful.groupby(
            ["MatchId", "Team", "PlayerId", "Player", "ReceiverId", "Receiver"],
            dropna=False,
            as_index=False,
        )
        .agg(
            **{
                "Pass Count": ("Action Type", "size"),
                "Passer X": ("Start X", "mean"),
                "Passer Y": ("Start Y", "mean"),
                "Receiver X": ("End X", "mean"),
                "Receiver Y": ("End Y", "mean"),
            }
        )
        .sort_values("Pass Count", ascending=False)
        .reset_index(drop=True)
    )
    return _clean_pass_network(network)


def load_team_match_shot_xg(season: str | None = None) -> pd.DataFrame:
    """Team-by-match shot volume and xG from CAFC_DB Impect provider events."""
    columns = ["MatchId", "Season", "Team", "Shots", "xG", "Post-Shot xG", "Goals"]
    if USE_MOCK_DATA:
        events = _mock_match_events()
        shots = events[events["Action Type"].astype(str).str.upper() == "SHOT"].copy()
        if season:
            shots = shots[shots["Season"].astype(str) == str(season)]
        if shots.empty:
            return pd.DataFrame(columns=columns)
        shots["Goal Flag"] = (
            shots["Result"].astype(str).str.upper().eq("SUCCESS")
            | shots["Action"].astype(str).str.upper().eq("GOAL")
        ).astype(int)
        out = shots.groupby(["MatchId", "Season", "Team"], as_index=False).agg(
            Shots=("Action Type", "size"),
            xG=("Shot xG", "sum"),
            **{"Post-Shot xG": ("Post-Shot xG", "sum"), "Goals": ("Goal Flag", "sum")},
        )
        return out[columns]

    df = load_match_events(season=season, action_types=["SHOT"], limit=60000)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.copy()
    df["Goal Flag"] = (
        df["Result"].fillna("").astype(str).str.upper().eq("SUCCESS")
        | df["Action"].fillna("").astype(str).str.upper().eq("GOAL")
    ).astype(int)
    df = df.groupby(["MatchId", "Season", "Team"], as_index=False).agg(
        Shots=("Action Type", "size"),
        xG=("Shot xG", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
        **{
            "Post-Shot xG": (
                "Post-Shot xG",
                lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1),
            ),
            "Goals": ("Goal Flag", "sum"),
        },
    )
    for col in ["Shots", "xG", "Post-Shot xG", "Goals"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[columns] if not df.empty else pd.DataFrame(columns=columns)


def _opta_season_labels(values: pd.Series) -> pd.Series:
    source_year = pd.to_numeric(values, errors="coerce").astype("Int64")
    return source_year.map(
        lambda value: f"{str(int(value))[-2:]}/{str(int(value) + 1)[-2:]}" if pd.notna(value) else None
    )


def load_opta_events(fixture_id: object | None = None, limit: int = 10000) -> pd.DataFrame:
    """Parsed, real Opta F24 events for one DVMS fixture.

    Event and qualifier dictionaries are currently empty in CAFC_DB, so this
    contract exposes provider ``TypeId`` values without inventing labels.
    """
    if USE_MOCK_DATA or fixture_id is None:
        return pd.DataFrame(columns=OPTA_EVENT_COLUMNS)
    safe_limit = max(min(int(limit), 50000), 1)
    events = get_connection().query(
        f"""
        SELECT
            e.FIXTURE_ID AS "FixtureId",
            e.OPTA_MATCH_ID AS "Opta Match Id",
            TO_VARCHAR(f.SEASON) AS "Source Season",
            f.MATCH_DATE AS "Date",
            f.HOME_TEAM_NAME AS "Home",
            f.AWAY_TEAM_NAME AS "Away",
            e.EVENT_ID AS "EventId",
            e.PROVIDER_EVENT_ROW_ID AS "Provider Event Row Id",
            e.OPTA_TEAM_ID AS "TeamId",
            COALESCE(NULLIF(t.OFFICIAL_TEAM_NAME, ''), NULLIF(t.TEAM_NAME, ''), e.OPTA_TEAM_ID) AS "Team",
            e.OPTA_PLAYER_ID AS "PlayerId",
            COALESCE(NULLIF(r.PLAYER_NAME, ''), e.OPTA_PLAYER_ID) AS "Player",
            e.TYPE_ID AS "TypeId",
            e.PERIOD_ID AS "Period",
            e.EVENT_MINUTE AS "Minute",
            e.EVENT_SECOND AS "Second",
            e.START_X AS "Start X",
            e.START_Y AS "Start Y",
            e.OUTCOME AS "Outcome",
            e.IS_KEY_PASS AS "Is Key Pass",
            e.IS_ASSIST AS "Is Assist",
            e.EVENT_AT_UTC AS "Event At UTC"
        FROM {relation("opta_events_staging")} e
        LEFT JOIN {relation("opta_fixtures_raw")} f
          ON f.FIXTURE_ID = e.FIXTURE_ID
        LEFT JOIN {relation("opta_teams_staging")} t
          ON t.FIXTURE_ID = e.FIXTURE_ID AND t.OPTA_TEAM_ID = e.OPTA_TEAM_ID
        LEFT JOIN {relation("opta_rosters_staging")} r
          ON r.FIXTURE_ID = e.FIXTURE_ID AND r.OPTA_PLAYER_ID = e.OPTA_PLAYER_ID
        WHERE e.FIXTURE_ID = ?
        ORDER BY e.PERIOD_ID, e.EVENT_MINUTE, e.EVENT_SECOND, e.PROVIDER_EVENT_ROW_ID
        LIMIT {safe_limit}
        """,
        params=_snowflake_params([fixture_id]),
        ttl="30m",
    )
    if events.empty:
        return pd.DataFrame(columns=OPTA_EVENT_COLUMNS)
    events["Season"] = _opta_season_labels(events.pop("Source Season"))
    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
    events["Event At UTC"] = pd.to_datetime(events["Event At UTC"], errors="coerce", utc=True)
    for column in ["TypeId", "Period", "Minute", "Second", "Start X", "Start Y"]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    return events[OPTA_EVENT_COLUMNS]


def load_opta_lineups(fixture_id: object | None = None) -> pd.DataFrame:
    """Parsed Opta F7 roster and lineup data for one DVMS fixture."""
    if USE_MOCK_DATA or fixture_id is None:
        return pd.DataFrame(columns=OPTA_LINEUP_COLUMNS)
    rows = get_connection().query(
        f"""
        SELECT
            l.FIXTURE_ID AS "FixtureId",
            l.OPTA_TEAM_ID AS "TeamId",
            COALESCE(NULLIF(t.OFFICIAL_TEAM_NAME, ''), NULLIF(t.TEAM_NAME, ''), l.OPTA_TEAM_ID) AS "Team",
            l.OPTA_PLAYER_ID AS "PlayerId",
            COALESCE(NULLIF(r.PLAYER_NAME, ''), l.OPTA_PLAYER_ID) AS "Player",
            r.FIRST_NAME AS "First Name",
            r.LAST_NAME AS "Last Name",
            r.REGISTERED_POSITION AS "Registered Position",
            l.LINEUP_STATUS AS "Lineup Status",
            l.POSITION_GROUP AS "Position Group",
            l.SUB_POSITION AS "Sub Position",
            l.SHIRT_NUMBER AS "Shirt Number",
            l.FORMATION_PLACE AS "Formation Place",
            l.IS_CAPTAIN AS "Is Captain"
        FROM {relation("opta_lineups_staging")} l
        LEFT JOIN {relation("opta_rosters_staging")} r
          ON r.FIXTURE_ID = l.FIXTURE_ID AND r.OPTA_PLAYER_ID = l.OPTA_PLAYER_ID
        LEFT JOIN {relation("opta_teams_staging")} t
          ON t.FIXTURE_ID = l.FIXTURE_ID AND t.OPTA_TEAM_ID = l.OPTA_TEAM_ID
        WHERE l.FIXTURE_ID = ?
        ORDER BY "Team", "Lineup Status", "Formation Place", "Player"
        """,
        params=_snowflake_params([fixture_id]),
        ttl="30m",
    )
    return rows[OPTA_LINEUP_COLUMNS] if not rows.empty else pd.DataFrame(columns=OPTA_LINEUP_COLUMNS)


# ---- SUBSTITUTIONS (Opta F24 events; Impect has no substitution rows) --------
# CAFC_DB does not expose a labelled Opta event-type or qualifier dictionary
# (see load_opta_events), so these TypeId values are the well-established
# public Opta F24 taxonomy, empirically verified against this database: pass
# completion rate, shots/match and goals/match (2.78 avg) all landed at
# realistic real-world figures for the mapped codes below.
OPTA_TYPE_PLAYER_OFF = 18
OPTA_TYPE_PLAYER_ON = 19
OPTA_TYPE_GOAL = 16
OPTA_TYPE_PASS = 1
_POSITION_RANK: dict[str, int] = {
    "GOALKEEPER": 0,
    "DEFENDER": 1,
    "MIDFIELDER": 2,
    "FORWARD": 3,
    "STRIKER": 3,
}
SUBSTITUTION_COLUMNS = [
    "FixtureId",
    "Season",
    "Date",
    "TeamId",
    "Team",
    "Opponent",
    "Period",
    "Minute",
    "Second",
    "Sub Number",
    "PlayerOffId",
    "Player Off",
    "Position Off",
    "PlayerOnId",
    "Player On",
    "Position On",
    "Shift Type",
    "Team Goals At Sub",
    "Opponent Goals At Sub",
    "Score State",
    "Goals After Entry",
    "Assists After Entry",
]


def _shift_type(position_off: object, position_on: object) -> str:
    off_rank = _POSITION_RANK.get(str(position_off or "").strip().upper())
    on_rank = _POSITION_RANK.get(str(position_on or "").strip().upper())
    if off_rank is None or on_rank is None:
        return "Unclear"
    if off_rank == on_rank:
        return "Like-for-Like"
    return "More Attacking" if on_rank > off_rank else "More Defensive"


def _score_state(team_goals: int, opponent_goals: int) -> str:
    if team_goals > opponent_goals:
        return "Winning"
    if team_goals < opponent_goals:
        return "Losing"
    return "Drawing"


def load_opta_goal_events(season: str | None = None, team: str | None = None) -> pd.DataFrame:
    """Lightweight season-wide goal (TypeId 16) events -- one query, no per-match loop.

    Used for team goal-timing context; a naive per-fixture loop over
    ``load_opta_events`` would need dozens of round-trips for a full season.
    """
    columns = ["FixtureId", "Date", "Team", "PlayerId", "Player", "Minute", "Second"]
    if USE_MOCK_DATA:
        return pd.DataFrame(columns=columns)

    clauses = ["e.TYPE_ID = 16"]
    params: list[object] = []
    if season:
        season_key = _season_key(season)
        start_year = season_key.split("/")[0]
        if len(start_year) == 2 and start_year.isdigit():
            start_year = f"20{start_year}"
        clauses.append("TO_VARCHAR(f.SEASON) = ?")
        params.append(start_year)
    if team:
        clauses.append("(f.HOME_TEAM_NAME = ? OR f.AWAY_TEAM_NAME = ?)")
        params.extend([team, team])

    rows = get_connection().query(
        f"""
        SELECT
            e.FIXTURE_ID AS "FixtureId",
            f.MATCH_DATE AS "Date",
            COALESCE(NULLIF(t.OFFICIAL_TEAM_NAME, ''), NULLIF(t.TEAM_NAME, ''), e.OPTA_TEAM_ID) AS "Team",
            e.OPTA_PLAYER_ID AS "PlayerId",
            COALESCE(NULLIF(r.PLAYER_NAME, ''), e.OPTA_PLAYER_ID) AS "Player",
            e.EVENT_MINUTE AS "Minute",
            e.EVENT_SECOND AS "Second"
        FROM {relation("opta_events_staging")} e
        JOIN {relation("opta_fixtures_raw")} f ON f.FIXTURE_ID = e.FIXTURE_ID
        LEFT JOIN {relation("opta_teams_staging")} t
          ON t.FIXTURE_ID = e.FIXTURE_ID AND t.OPTA_TEAM_ID = e.OPTA_TEAM_ID
        LEFT JOIN {relation("opta_rosters_staging")} r
          ON r.FIXTURE_ID = e.FIXTURE_ID AND r.OPTA_PLAYER_ID = e.OPTA_PLAYER_ID
        WHERE {' AND '.join(clauses)}
        ORDER BY e.FIXTURE_ID, e.EVENT_MINUTE, e.EVENT_SECOND
        """,
        params=_snowflake_params(params),
        ttl="30m",
    )
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    for col in ["Minute", "Second"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows[columns]


def load_opta_substitutions(season: str | None = None, team: str | None = None) -> pd.DataFrame:
    """One row per substitution, paired from Opta F24 Player Off/On events.

    Off (TypeId 18) and On (TypeId 19) events are paired by matching team,
    period, minute and second -- CAFC_DB does not expose a related-event
    qualifier for substitutions, but this pairing was verified exactly
    against real fixture data down to the second. Score state and each
    substitute's goal/assist contribution after entering are reconstructed
    from the same fixtures' goal (TypeId 16) and assisted-pass events, all
    within the Opta provider so no cross-provider player matching is needed.
    """
    if USE_MOCK_DATA:
        return pd.DataFrame(columns=SUBSTITUTION_COLUMNS)

    clauses = ["(e.TYPE_ID IN (16, 18, 19) OR (e.TYPE_ID = 1 AND e.IS_ASSIST = 1))"]
    params: list[object] = []
    if season:
        season_key = _season_key(season)
        start_year = season_key.split("/")[0]
        if len(start_year) == 2 and start_year.isdigit():
            start_year = f"20{start_year}"
        clauses.append("TO_VARCHAR(f.SEASON) = ?")
        params.append(start_year)
    if team:
        clauses.append("(f.HOME_TEAM_NAME = ? OR f.AWAY_TEAM_NAME = ?)")
        params.extend([team, team])

    raw = get_connection().query(
        f"""
        SELECT
            e.FIXTURE_ID AS "FixtureId",
            TO_VARCHAR(f.SEASON) AS "Source Season",
            f.MATCH_DATE AS "Date",
            f.HOME_TEAM_NAME AS "Home",
            f.AWAY_TEAM_NAME AS "Away",
            e.OPTA_TEAM_ID AS "TeamId",
            COALESCE(NULLIF(t.OFFICIAL_TEAM_NAME, ''), NULLIF(t.TEAM_NAME, ''), e.OPTA_TEAM_ID) AS "Team",
            e.OPTA_PLAYER_ID AS "PlayerId",
            COALESCE(NULLIF(r.PLAYER_NAME, ''), e.OPTA_PLAYER_ID) AS "Player",
            e.TYPE_ID AS "TypeId",
            e.PERIOD_ID AS "Period",
            e.EVENT_MINUTE AS "Minute",
            e.EVENT_SECOND AS "Second",
            l.POSITION_GROUP AS "Position Group",
            l.SUB_POSITION AS "Sub Position"
        FROM {relation("opta_events_staging")} e
        JOIN {relation("opta_fixtures_raw")} f ON f.FIXTURE_ID = e.FIXTURE_ID
        LEFT JOIN {relation("opta_teams_staging")} t
          ON t.FIXTURE_ID = e.FIXTURE_ID AND t.OPTA_TEAM_ID = e.OPTA_TEAM_ID
        LEFT JOIN {relation("opta_rosters_staging")} r
          ON r.FIXTURE_ID = e.FIXTURE_ID AND r.OPTA_PLAYER_ID = e.OPTA_PLAYER_ID
        LEFT JOIN {relation("opta_lineups_staging")} l
          ON l.FIXTURE_ID = e.FIXTURE_ID AND l.OPTA_PLAYER_ID = e.OPTA_PLAYER_ID
        WHERE {' AND '.join(clauses)}
        ORDER BY e.FIXTURE_ID, e.EVENT_MINUTE, e.EVENT_SECOND
        """,
        params=_snowflake_params(params),
        ttl="30m",
    )
    if raw.empty:
        return pd.DataFrame(columns=SUBSTITUTION_COLUMNS)

    raw["Season"] = _opta_season_labels(raw.pop("Source Season"))
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for col in ["TypeId", "Period", "Minute", "Second"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["_Time"] = raw["Minute"].fillna(0) * 60 + raw["Second"].fillna(0)
    raw["Opponent"] = np.where(raw["Team"].astype(str).eq(raw["Home"].astype(str)), raw["Away"], raw["Home"])

    goals = raw[raw["TypeId"] == OPTA_TYPE_GOAL][["FixtureId", "TeamId", "PlayerId", "_Time"]].copy()
    assists = raw[(raw["TypeId"] == OPTA_TYPE_PASS)][["FixtureId", "TeamId", "PlayerId", "_Time"]].copy()

    subs_off = raw[raw["TypeId"] == OPTA_TYPE_PLAYER_OFF].copy()
    subs_on = raw[raw["TypeId"] == OPTA_TYPE_PLAYER_ON].copy()
    if subs_off.empty or subs_on.empty:
        return pd.DataFrame(columns=SUBSTITUTION_COLUMNS)

    group_keys = ["FixtureId", "TeamId", "Period", "Minute", "Second"]
    subs_off["_Rank"] = subs_off.groupby(group_keys).cumcount()
    subs_on["_Rank"] = subs_on.groupby(group_keys).cumcount()
    paired = subs_off.merge(
        subs_on,
        on=[*group_keys, "_Rank"],
        suffixes=(" Off", " On"),
        how="inner",
    )
    if paired.empty:
        return pd.DataFrame(columns=SUBSTITUTION_COLUMNS)

    paired["_Time"] = paired["Minute"].fillna(0) * 60 + paired["Second"].fillna(0)
    paired = paired.sort_values(["FixtureId", "TeamId", "_Time"]).reset_index(drop=True)
    paired["Sub Number"] = paired.groupby(["FixtureId", "TeamId"]).cumcount() + 1
    # A player subbed off who was themselves a substitute earlier (a double-sub
    # chain) has "Position Group" = "Substitute" in the lineup table; fall back
    # to their own Sub Position so the shift classification still resolves.
    paired["Position Off"] = paired["Position Group Off"].where(
        ~paired["Position Group Off"].astype(str).str.upper().eq("SUBSTITUTE"), paired["Sub Position Off"]
    )
    paired["Shift Type"] = [
        _shift_type(off_pos, on_pos)
        for off_pos, on_pos in zip(paired["Position Off"], paired["Sub Position On"], strict=False)
    ]

    rows: list[dict[str, object]] = []
    for _, sub in paired.iterrows():
        fixture_id = sub["FixtureId"]
        team_id = sub["TeamId"]
        sub_time = sub["_Time"]
        team_goals = int(
            goals[(goals["FixtureId"] == fixture_id) & (goals["TeamId"] == team_id) & (goals["_Time"] <= sub_time)].shape[0]
        )
        opponent_goals = int(
            goals[(goals["FixtureId"] == fixture_id) & (goals["TeamId"] != team_id) & (goals["_Time"] <= sub_time)].shape[0]
        )
        goals_after = int(
            goals[
                (goals["FixtureId"] == fixture_id)
                & (goals["PlayerId"] == sub["PlayerId On"])
                & (goals["_Time"] > sub_time)
            ].shape[0]
        )
        assists_after = int(
            assists[
                (assists["FixtureId"] == fixture_id)
                & (assists["PlayerId"] == sub["PlayerId On"])
                & (assists["_Time"] > sub_time)
            ].shape[0]
        )
        rows.append(
            {
                "FixtureId": fixture_id,
                "Season": sub["Season Off"],
                "Date": sub["Date Off"],
                "TeamId": team_id,
                "Team": sub["Team Off"],
                "Opponent": sub["Opponent Off"],
                "Period": sub["Period"],
                "Minute": sub["Minute"],
                "Second": sub["Second"],
                "Sub Number": int(sub["Sub Number"]),
                "PlayerOffId": sub["PlayerId Off"],
                "Player Off": sub["Player Off"],
                "Position Off": sub["Position Off"],
                "PlayerOnId": sub["PlayerId On"],
                "Player On": sub["Player On"],
                "Position On": sub["Sub Position On"],
                "Shift Type": sub["Shift Type"],
                "Team Goals At Sub": team_goals,
                "Opponent Goals At Sub": opponent_goals,
                "Score State": _score_state(team_goals, opponent_goals),
                "Goals After Entry": goals_after,
                "Assists After Entry": assists_after,
            }
        )
    result = pd.DataFrame(rows)
    return result[SUBSTITUTION_COLUMNS] if not result.empty else pd.DataFrame(columns=SUBSTITUTION_COLUMNS)


def _opta_xml_id(value: object) -> str | None:
    """Return the provider ID without the F7 entity prefix (for example ``t80`` -> ``80``)."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > 1 and text[0].isalpha():
        return text[1:]
    return text or None


def load_opta_formations(fixture_id: object | None = None) -> pd.DataFrame:
    """Official starting shapes from the latest Opta F7 payload for a fixture.

    The current parsed F7 views expose each player's formation place but not
    the team-level ``Formation`` attribute, so this small adapter reads the
    same immutable F7 asset and parses that attribute locally.
    """
    if USE_MOCK_DATA or fixture_id is None:
        return pd.DataFrame(columns=OPTA_FORMATION_COLUMNS)
    payload_rows = get_connection().query(
        f"""
        SELECT RAW_PAYLOAD AS "Raw Payload"
        FROM {relation("opta_assets_raw")}
        WHERE FIXTURE_ID = ?
          AND ASSET_TYPE = 2
          AND ASSET_SUBTYPE = 21
          AND RAW_PAYLOAD IS NOT NULL
        ORDER BY LOADED_AT DESC, ASSET_ID DESC
        LIMIT 1
        """,
        params=_snowflake_params([fixture_id]),
        ttl="30m",
    )
    if payload_rows.empty:
        return pd.DataFrame(columns=OPTA_FORMATION_COLUMNS)

    try:
        root = ET.fromstring(str(payload_rows.iloc[0]["Raw Payload"]))
    except (ET.ParseError, TypeError, ValueError):
        return pd.DataFrame(columns=OPTA_FORMATION_COLUMNS)

    records: list[dict[str, object]] = []
    for team in root.findall(".//TeamData"):
        records.append(
            {
                "FixtureId": fixture_id,
                "TeamId": _opta_xml_id(team.attrib.get("TeamRef")),
                "Side": team.attrib.get("Side"),
                "Formation": team.attrib.get("Formation"),
                "Average Age": team.attrib.get("AverageAge"),
            }
        )
    rows = pd.DataFrame.from_records(records, columns=OPTA_FORMATION_COLUMNS)
    if not rows.empty:
        rows["Average Age"] = pd.to_numeric(rows["Average Age"], errors="coerce")
    return rows


def _duration_to_seconds(value: object) -> int | None:
    """Parse provider durations such as ``51:55`` without treating 51 as hours."""
    if value is None or pd.isna(value):
        return None
    parts = str(value).strip().split(":")
    if len(parts) not in {2, 3} or not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds if seconds < 60 else None
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds if minutes < 60 and seconds < 60 else None


def _physical_summary_values(payload: object) -> dict[str, str]:
    """Extract the three effective-playing-time rows from a physical-summary CSV."""
    wanted = {
        "effective playing time": "Effective Playing Time",
        "home ept": "Home EPT",
        "away ept": "Away EPT",
    }
    found: dict[str, str] = {}
    if payload is None:
        return found
    try:
        rows = csv.reader(io.StringIO(str(payload)))
        for row in rows:
            cells = [cell.strip().lstrip("\ufeff") for cell in row]
            for index, cell in enumerate(cells):
                output_name = wanted.get(cell.casefold())
                if not output_name:
                    continue
                value = next((candidate for candidate in cells[index + 1 :] if candidate), "")
                if value:
                    found[output_name] = value
    except (csv.Error, TypeError, ValueError):
        return {}
    return found


def load_fixture_effective_possession(fixture_id: object | None = None) -> pd.DataFrame:
    """Provider-delivered effective-play possession for one DVMS fixture.

    ``Home EPT`` and ``Away EPT`` come from the latest Second Spectrum physical
    summary stored in DVMS_RAW. The percentages are calculated only from those
    two durations; no pass-share or F24 event proxy is used.
    """
    if USE_MOCK_DATA or fixture_id is None:
        return pd.DataFrame(columns=TRACKING_POSSESSION_COLUMNS)
    payload_rows = get_connection().query(
        f"""
        SELECT
            f.FIXTURE_ID AS "FixtureId",
            f.MATCH_DATE AS "Date",
            f.HOME_TEAM_NAME AS "Home",
            f.AWAY_TEAM_NAME AS "Away",
            a.RAW_PAYLOAD AS "Raw Payload",
            a.LOADED_AT AS "Loaded At"
        FROM {relation("opta_assets_raw")} a
        JOIN {relation("opta_fixtures_raw")} f
          ON f.FIXTURE_ID = a.FIXTURE_ID
        WHERE a.FIXTURE_ID = ?
          AND a.ASSET_TYPE = 5
          AND a.ASSET_SUBTYPE = 43
          AND a.RAW_PAYLOAD IS NOT NULL
        ORDER BY a.LOADED_AT DESC, a.ASSET_ID DESC
        LIMIT 1
        """,
        params=_snowflake_params([fixture_id]),
        ttl="30m",
    )
    if payload_rows.empty:
        return pd.DataFrame(columns=TRACKING_POSSESSION_COLUMNS)

    source = payload_rows.iloc[0]
    values = _physical_summary_values(source.get("Raw Payload"))
    home_seconds = _duration_to_seconds(values.get("Home EPT"))
    away_seconds = _duration_to_seconds(values.get("Away EPT"))
    if home_seconds is None or away_seconds is None or home_seconds + away_seconds <= 0:
        return pd.DataFrame(columns=TRACKING_POSSESSION_COLUMNS)

    total = home_seconds + away_seconds
    record = {
        "FixtureId": source.get("FixtureId"),
        "Date": source.get("Date"),
        "Home": source.get("Home"),
        "Away": source.get("Away"),
        "Effective Playing Time": values.get("Effective Playing Time"),
        "Home EPT": values.get("Home EPT"),
        "Away EPT": values.get("Away EPT"),
        "Home Possession %": round(100 * home_seconds / total, 1),
        "Away Possession %": round(100 * away_seconds / total, 1),
        "Provider": "Second Spectrum",
        "Loaded At": source.get("Loaded At"),
    }
    rows = pd.DataFrame.from_records([record], columns=TRACKING_POSSESSION_COLUMNS)
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["Loaded At"] = pd.to_datetime(rows["Loaded At"], errors="coerce", utc=True)
    return rows


def load_opta_event_qualifiers(
    fixture_id: object | None = None,
    event_ids: list[object] | tuple[object, ...] | set[object] | None = None,
    limit: int = 50000,
) -> pd.DataFrame:
    """Parsed Opta F24 qualifiers for a fixture and optional event IDs.

    Reading the single DVMS payload and parsing it locally avoids forcing
    Snowflake to expand and sort the multi-million-row qualifier view for each
    interactive selection.
    """
    if USE_MOCK_DATA or fixture_id is None:
        return pd.DataFrame(columns=OPTA_QUALIFIER_COLUMNS)
    safe_limit = max(min(int(limit), 100000), 1)
    payload_rows = get_connection().query(
        f"""
        SELECT RAW_PAYLOAD AS "Raw Payload"
        FROM {relation("opta_assets_raw")}
        WHERE FIXTURE_ID = ?
          AND ASSET_TYPE = 2
          AND ASSET_SUBTYPE = 20
          AND RAW_PAYLOAD IS NOT NULL
        ORDER BY LOADED_AT DESC
        LIMIT 1
        """,
        params=_snowflake_params([fixture_id]),
        ttl="30m",
    )
    if payload_rows.empty:
        return pd.DataFrame(columns=OPTA_QUALIFIER_COLUMNS)

    try:
        game = ET.fromstring(str(payload_rows.iloc[0]["Raw Payload"])).find(".//Game")
    except ET.ParseError:
        return pd.DataFrame(columns=OPTA_QUALIFIER_COLUMNS)
    if game is None:
        return pd.DataFrame(columns=OPTA_QUALIFIER_COLUMNS)

    wanted = {str(value) for value in event_ids} if event_ids else None
    match_id = game.attrib.get("id")
    records: list[dict[str, object]] = []
    for event in game.findall("Event"):
        event_id = event.attrib.get("event_id")
        if wanted is not None and str(event_id) not in wanted:
            continue
        for qualifier in event.findall("Q"):
            records.append(
                {
                    "FixtureId": fixture_id,
                    "Opta Match Id": match_id,
                    "EventId": event_id,
                    "Provider Event Row Id": event.attrib.get("id"),
                    "QualifierId": qualifier.attrib.get("qualifier_id"),
                    "Provider Qualifier Row Id": qualifier.attrib.get("id"),
                    "Qualifier Value": qualifier.attrib.get("value"),
                }
            )
            if len(records) >= safe_limit:
                return pd.DataFrame.from_records(records, columns=OPTA_QUALIFIER_COLUMNS)
    return pd.DataFrame.from_records(records, columns=OPTA_QUALIFIER_COLUMNS)


def load_opta_fixtures(season: str | None = None, team: str | None = None) -> pd.DataFrame:
    """Real Opta/DVMS fixtures from CAFC_DB's immutable ingestion layer."""
    columns = [
        "FixtureId",
        "Opta Match Id",
        "Competition Id",
        "Season",
        "Date",
        "Round",
        "Home Team Id",
        "Home",
        "Away Team Id",
        "Away",
        "Home Goals",
        "Away Goals",
        "Venue",
        "Loaded At",
    ]
    if USE_MOCK_DATA:
        return pd.DataFrame(columns=columns)
    clauses: list[str] = []
    params: list[object] = []
    if season:
        season_key = _season_key(season)
        start_year = season_key.split("/")[0]
        if len(start_year) == 2 and start_year.isdigit():
            start_year = f"20{start_year}"
        clauses.append("TO_VARCHAR(SEASON) = ?")
        params.append(start_year)
    if team:
        clauses.append("(HOME_TEAM_NAME = ? OR AWAY_TEAM_NAME = ?)")
        params.extend([team, team])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    fixtures = get_connection().query(
        f"""
        SELECT
            FIXTURE_ID AS "FixtureId",
            OPTA_MATCH_ID AS "Opta Match Id",
            COMPETITION_ID AS "Competition Id",
            TO_VARCHAR(SEASON) AS "Source Season",
            MATCH_DATE AS "Date",
            ROUND AS "Round",
            OPTA_HOME_TEAM_ID AS "Home Team Id",
            HOME_TEAM_NAME AS "Home",
            OPTA_AWAY_TEAM_ID AS "Away Team Id",
            AWAY_TEAM_NAME AS "Away",
            HOME_SCORE AS "Home Goals",
            AWAY_SCORE AS "Away Goals",
            VENUE_NAME AS "Venue",
            LOADED_AT AS "Loaded At"
        FROM {relation("opta_fixtures_raw")}
        {where}
        ORDER BY MATCH_DATE, FIXTURE_ID
        """,
        params=_snowflake_params(params),
        ttl="1h",
    )
    if fixtures.empty:
        return pd.DataFrame(columns=columns)
    fixtures["Season"] = _opta_season_labels(fixtures.pop("Source Season"))
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
    for column in ["Home Goals", "Away Goals"]:
        fixtures[column] = pd.to_numeric(fixtures[column], errors="coerce")
    return fixtures[columns]


def load_opta_asset_inventory() -> pd.DataFrame:
    """Availability summary for the real Opta assets held in DVMS_RAW."""
    columns = ["Asset Type", "Asset Subtype", "Assets", "Ready Assets", "Last Loaded"]
    if USE_MOCK_DATA:
        return pd.DataFrame(columns=columns)
    return get_connection().query(
        f"""
        SELECT
            ASSET_TYPE AS "Asset Type",
            ASSET_SUBTYPE AS "Asset Subtype",
            COUNT(*) AS "Assets",
            COUNT_IF(READY) AS "Ready Assets",
            MAX(LOADED_AT) AS "Last Loaded"
        FROM {relation("opta_assets_raw")}
        GROUP BY ASSET_TYPE, ASSET_SUBTYPE
        ORDER BY ASSET_TYPE, ASSET_SUBTYPE
        """,
        ttl="1h",
    )[columns]


def all_datasets() -> dict[str, pd.DataFrame]:
    """Every top-level dataset the app knows about, keyed by display name.

    Used by pages (Data Quality Checks, Export Data) that operate generically
    across whatever datasets exist, instead of hardcoding Players/Teams/Matches.
    """
    return {
        "Players": load_players(),
        "Teams": load_teams(),
        "Matches": load_matches(),
        "Opta Fixtures": load_opta_fixtures(),
    }


def dataset_summary() -> dict:
    """Numbers for the Home page (season, refresh, counts)."""
    season_inventory = list_seasons()
    available_seasons = season_inventory.get("players", [])
    selected_season = available_seasons[-1] if available_seasons else None
    players = load_players(selected_season)
    teams = load_teams(selected_season)
    matches = load_matches(selected_season)

    if USE_MOCK_DATA:
        season = "2025/26"
        last_refreshed = "2026-07-02"
    else:
        season = selected_season or "Unknown"
        refresh = get_connection().query(
            f"SELECT MAX(LOADED_AT) AS LAST_REFRESHED FROM {relation('impect_events')}",
            ttl="1h",
        )
        last_refreshed = str(refresh["LAST_REFRESHED"].iloc[0])

    return {
        "season": season,
        "last_refreshed": last_refreshed,
        "n_players": len(players),
        "n_teams": len(teams),
        "n_matches": len(matches),
    }


# ---- MOCK DATA (development only) --------------------------------------------
# @st.cache_data builds these once and reuses them. When USE_MOCK_DATA = False
# these are simply never called.

@st.cache_data
def _mock_players() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    positions = ["GK", "CB", "LB", "DMF", "CMF", "AMF", "RW", "CF"]
    names = [f"Player {i:02d}" for i in range(1, 25)]
    teams = [
        "Charlton", "Charlton", "Charlton", "Charlton", "Charlton", "Charlton",
        "Millwall", "West Ham", "QPR", "Southampton", "Leeds", "Norwich",
        "Watford", "Sunderland", "Coventry", "Bristol City", "Hull", "Charlton",
        "Charlton", "Charlton", "Charlton", "Charlton", "Charlton", "Charlton",
    ]
    df = pd.DataFrame({
        "PlayerId": np.arange(1, len(names) + 1),
        "Player": names,
        "First Name": [name.split()[0] for name in names],
        "Last Name": [name.split()[-1] for name in names],
        "Team": teams,
        "Position": rng.choice(positions, size=len(names)),
        "Birthdate": pd.date_range("1995-01-01", periods=len(names), freq="180D").strftime("%Y-%m-%d"),
        "Nationality": rng.choice(["England", "Scotland", "Wales", "Ireland"], size=len(names)),
        "Foot": rng.choice(["left", "right", "both"], size=len(names)),
        "Minutes": rng.integers(400, 3000, size=len(names)),
        "Season": "2025/26",
        "Competition": "Mock League",
    })
    for metric in PLAYER_PROFILE_METRICS:
        if metric == "Pass %":
            df[metric] = rng.uniform(60, 92, size=len(names)).round(1)
        elif metric == "Ball Security %":
            df[metric] = rng.uniform(55, 90, size=len(names)).round(1)
        elif metric == "Losses Per 100 Actions":
            df[metric] = rng.uniform(2, 16, size=len(names)).round(1)
        elif metric in {"Ground Duel Win %", "Aerial Duel Win %"}:
            df[metric] = rng.uniform(35, 78, size=len(names)).round(1)
        elif metric == "Goals Prevented /90":
            df[metric] = rng.uniform(-0.25, 0.35, size=len(names)).round(2)
        elif metric in {"Critical Ball Losses /90", "Ball Loss Threat /90", "Team-Mates Bypassed By Losses /90"}:
            df[metric] = rng.uniform(0, 2.5, size=len(names)).round(2)
        elif metric in {"Goals Conceded /90", "Post-Shot xG Faced /90"}:
            df[metric] = rng.uniform(0.6, 2.2, size=len(names)).round(2)
        elif metric == "Neutral Passes /90":
            df[metric] = rng.uniform(4, 42, size=len(names)).round(2)
        else:
            df[metric] = rng.uniform(0, 6, size=len(names)).round(2)
    return df


@st.cache_data
def _mock_teams() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    teams = [
        "Charlton", "Millwall", "West Ham", "QPR", "Southampton", "Leeds",
        "Norwich", "Watford", "Sunderland", "Coventry", "Bristol City", "Hull",
    ]
    df = pd.DataFrame(
        {
            "Team": teams,
            "Season": "2025/26",
            "Competition": "Mock League",
        }
    )
    for metric in TEAM_METRICS:
        if metric == "Pass %":
            df[metric] = rng.uniform(60, 92, size=len(teams)).round(1)
        else:
            df[metric] = rng.uniform(0, 5, size=len(teams)).round(2)
    return df


@st.cache_data
def _mock_matches() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    opponents = [
        "Millwall", "West Ham", "QPR", "Southampton", "Leeds", "Norwich",
        "Watford", "Sunderland", "Coventry", "Bristol City", "Hull",
    ]
    rows = []
    for season, start_date, competition in [
        ("2024/25", "2024-08-10", "League One"),
        ("2025/26", "2025-08-09", "Championship"),
    ]:
        for index, opponent in enumerate(opponents, start=1):
            rows.append(
                {
                    "MatchId": f"mock-{season}-{index:02d}",
                    "Home": "Charlton" if index % 2 else opponent,
                    "Away": opponent if index % 2 else "Charlton",
                    "Date": pd.Timestamp(start_date) + pd.Timedelta(days=7 * (index - 1)),
                    "Competition": competition,
                    "Season": season,
                    "Venue Verified": True,
                }
            )
    df = pd.DataFrame(rows)
    df["Match"] = df["Home"] + " vs " + df["Away"]
    df["Home Goals"] = rng.integers(0, 4, len(df))
    df["Away Goals"] = rng.integers(0, 4, len(df))
    df["Result"] = np.select(
        [df["Home Goals"] > df["Away Goals"], df["Home Goals"] < df["Away Goals"]],
        ["Home Win", "Away Win"],
        default="Draw",
    )
    return df


@st.cache_data
def _mock_team_action_counts() -> pd.DataFrame:
    rng = np.random.default_rng(28)
    teams = _mock_teams()["Team"].tolist()
    actions = {
        "PASS": (420, 760),
        "PASS_TO_FINAL_THIRD": (28, 92),
        "SHOT": (28, 84),
        "GOAL": (8, 38),
        "CORNER": (18, 58),
        "FREE_KICK": (20, 64),
        "PENALTY": (0, 8),
        "TACKLE": (42, 118),
        "INTERCEPTION": (38, 104),
        "CLEARANCE": (48, 132),
        "DUEL": (90, 210),
        "RECOVERY": (70, 180),
    }
    rows = []
    for team in teams:
        for action, (low, high) in actions.items():
            rows.append({
                "Team": team,
                "Action": action,
                "Actions": int(rng.integers(low, high + 1)),
            })
    return pd.DataFrame(rows)


@st.cache_data
def _mock_match_action_counts() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    matches = _mock_matches()
    actions = {
        "PASS": (180, 520),
        "PASS_TO_FINAL_THIRD": (12, 58),
        "SHOT": (3, 18),
        "GOAL": (0, 4),
        "CORNER": (0, 11),
        "FREE_KICK": (4, 20),
        "PENALTY": (0, 2),
        "TACKLE": (10, 34),
        "INTERCEPTION": (8, 30),
        "CLEARANCE": (8, 42),
        "DUEL": (24, 74),
        "RECOVERY": (18, 60),
        "SUBSTITUTION": (2, 5),
    }
    rows = []
    for _, match in matches.iterrows():
        for team in [match["Home"], match["Away"]]:
            for action, (low, high) in actions.items():
                rows.append(
                    {
                        "MatchId": match["MatchId"],
                        "Season": match["Season"],
                        "Team": team,
                        "Action": action,
                        "Actions": int(rng.integers(low, high + 1)),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data
def _mock_match_events() -> pd.DataFrame:
    rng = np.random.default_rng(44)
    matches = _mock_matches()
    defensive_actions = ["LOOSE_BALL_REGAIN", "INTERCEPTION", "CLEARANCE", "BLOCK", "GROUND_DUEL"]
    rows = []
    event_number = 1
    for _, match in matches.iterrows():
        teams = [match["Home"], match["Away"]]
        for team in teams:
            opponent = match["Away"] if team == match["Home"] else match["Home"]
            players = [f"{team[:3].title()} Player {i:02d}" for i in range(1, 12)]
            for idx in range(90):
                start_x = rng.uniform(-45, 35)
                start_y = rng.uniform(-30, 30)
                end_x = np.clip(start_x + rng.normal(12, 17), -52.5, 52.5)
                end_y = np.clip(start_y + rng.normal(0, 12), -34, 34)
                second = int(rng.integers(30, 5400))
                success = rng.random() > 0.22
                forward_distance = max(0.0, float(end_x - start_x))
                bypassed_opponents = 0.0
                bypassed_defenders = 0.0
                if success and forward_distance >= 8:
                    bypassed_opponents = float(np.clip(round(forward_distance / 9 + rng.normal(0, 0.6), 1), 0, 8))
                    bypassed_defenders = float(np.clip(round(max(0.0, end_x) / 20 + rng.normal(0, 0.3), 1), 0, bypassed_opponents))
                rows.append(
                    {
                        "MatchId": match["MatchId"],
                        "Season": match["Season"],
                        "Date": match["Date"],
                        "Competition": match["Competition"],
                        "Home": match["Home"],
                        "Away": match["Away"],
                        "Team": team,
                        "PlayerId": f"{team[:3]}-{idx % 11 + 1}",
                        "Player": players[idx % 11],
                        "Position": rng.choice(["DEF", "MID", "FWD"]),
                        "Period": 1 if second < 2700 else 2,
                        "Game Time": f"{second // 60:02d}:{second % 60:02d}",
                        "Second": second,
                        "Minute": second // 60 + 1,
                        "Event Number": event_number,
                        "Sequence Index": idx,
                        "Phase": rng.choice(["BUILD_UP", "TRANSITION", "ATTACK"]),
                        "Action Type": "PASS",
                        "Action": rng.choice(["LOW_PASS", "HEADER", "CROSS"]),
                        "Body Part": rng.choice(["RIGHT_FOOT", "LEFT_FOOT", "HEAD"]),
                        "Result": "SUCCESS" if success else "FAIL",
                        "Pressure": rng.choice(["LOW", "MEDIUM", "HIGH", None]),
                        "Start X": start_x,
                        "Start Y": start_y,
                        "End X": end_x,
                        "End Y": end_y,
                        "Raw Start X": start_x,
                        "Raw Start Y": start_y,
                        "Raw End X": end_x,
                        "Raw End Y": end_y,
                        "Start Lane": rng.choice(["LEFT", "HALF_LEFT", "CENTER", "HALF_RIGHT", "RIGHT"]),
                        "End Lane": rng.choice(["LEFT", "HALF_LEFT", "CENTER", "HALF_RIGHT", "RIGHT"]),
                        "Start Pitch Position": "MIDDLE_THIRD" if start_x < 17.5 else "FINAL_THIRD",
                        "End Pitch Position": "FINAL_THIRD" if end_x >= 17.5 else "MIDDLE_THIRD",
                        "ReceiverId": f"{team[:3]}-{(idx + 1) % 11 + 1}",
                        "Receiver": players[(idx + 1) % 11],
                        "Pass Distance": float(np.hypot(end_x - start_x, end_y - start_y)),
                        "Pass Angle": rng.uniform(-180, 180),
                        "Team xT": max(0, (end_x - start_x) / 105) * rng.uniform(0.01, 0.08),
                        "PXT Pass": max(0, (end_x - start_x) / 105) * rng.uniform(0.01, 0.12),
                        "PXT Shot": np.nan,
                        "Shot xG": np.nan,
                        "Post-Shot xG": np.nan,
                        "Packing xG": np.nan,
                        "Bypassed Opponents": bypassed_opponents,
                        "Bypassed Defenders": bypassed_defenders,
                        "Shot Distance": np.nan,
                        "Shot Angle": np.nan,
                        "Shot Target Y": np.nan,
                        "Shot Target Z": np.nan,
                        "Shot GK X": np.nan,
                        "Shot GK Y": np.nan,
                        "Set Piece": False,
                        "Set Piece Category": None,
                        "Set Piece Execution": None,
                    }
                )
                event_number += 1

            for idx in range(12):
                start_x = rng.uniform(16, 51)
                start_y = rng.uniform(-28, 28)
                second = int(rng.integers(120, 5400))
                xg = float(np.clip(rng.beta(1.4, 8), 0.02, 0.75))
                goal = rng.random() < min(xg, 0.45)
                shot_bypassed_opponents = float(rng.choice([0, 0, 0, 1, 2]))
                shot_bypassed_defenders = float(min(shot_bypassed_opponents, rng.choice([0, 0, 1])))
                rows.append(
                    {
                        "MatchId": match["MatchId"],
                        "Season": match["Season"],
                        "Date": match["Date"],
                        "Competition": match["Competition"],
                        "Home": match["Home"],
                        "Away": match["Away"],
                        "Team": team,
                        "PlayerId": f"{team[:3]}-{idx % 11 + 1}",
                        "Player": players[idx % 11],
                        "Position": rng.choice(["MID", "FWD"]),
                        "Period": 1 if second < 2700 else 2,
                        "Game Time": f"{second // 60:02d}:{second % 60:02d}",
                        "Second": second,
                        "Minute": second // 60 + 1,
                        "Event Number": event_number,
                        "Sequence Index": idx,
                        "Phase": rng.choice(["ATTACK", "SET_PIECE", "TRANSITION"]),
                        "Action Type": "SHOT",
                        "Action": "GOAL" if goal else rng.choice(["MID_RANGE_SHOT", "CLOSE_RANGE_SHOT", "HEADER"]),
                        "Body Part": rng.choice(["RIGHT_FOOT", "LEFT_FOOT", "HEAD"]),
                        "Result": "SUCCESS" if goal else "FAIL",
                        "Pressure": rng.choice(["LOW", "MEDIUM", "HIGH", None]),
                        "Start X": start_x,
                        "Start Y": start_y,
                        "End X": 52.5,
                        "End Y": rng.uniform(-3.6, 3.6),
                        "Raw Start X": start_x,
                        "Raw Start Y": start_y,
                        "Raw End X": 52.5,
                        "Raw End Y": rng.uniform(-3.6, 3.6),
                        "Start Lane": rng.choice(["LEFT", "HALF_LEFT", "CENTER", "HALF_RIGHT", "RIGHT"]),
                        "End Lane": "CENTER",
                        "Start Pitch Position": "FINAL_THIRD",
                        "End Pitch Position": "GOAL",
                        "ReceiverId": None,
                        "Receiver": None,
                        "Pass Distance": np.nan,
                        "Pass Angle": np.nan,
                        "Team xT": xg,
                        "PXT Pass": np.nan,
                        "PXT Shot": xg,
                        "Shot xG": xg,
                        "Post-Shot xG": xg * rng.uniform(0.3, 1.4) if goal or rng.random() > 0.55 else np.nan,
                        "Packing xG": xg * rng.uniform(0.7, 1.1),
                        "Bypassed Opponents": shot_bypassed_opponents,
                        "Bypassed Defenders": shot_bypassed_defenders,
                        "Shot Distance": float(52.5 - start_x),
                        "Shot Angle": rng.uniform(5, 55),
                        "Shot Target Y": rng.uniform(-3.6, 3.6),
                        "Shot Target Z": rng.uniform(0, 2.4),
                        "Shot GK X": rng.uniform(49, 52.5),
                        "Shot GK Y": rng.uniform(-3.6, 3.6),
                        "Set Piece": rng.random() < 0.18,
                        "Set Piece Category": rng.choice(["CORNER", "FREE_KICK", None]),
                        "Set Piece Execution": rng.choice(["DIRECT", "SHORT", None]),
                    }
                )
                event_number += 1

            for idx in range(26):
                start_x = rng.uniform(-40, 42)
                start_y = rng.uniform(-32, 32)
                second = int(rng.integers(60, 5400))
                action = rng.choice(defensive_actions)
                rows.append(
                    {
                        "MatchId": match["MatchId"],
                        "Season": match["Season"],
                        "Date": match["Date"],
                        "Competition": match["Competition"],
                        "Home": match["Home"],
                        "Away": match["Away"],
                        "Team": team,
                        "PlayerId": f"{team[:3]}-{idx % 11 + 1}",
                        "Player": players[idx % 11],
                        "Position": rng.choice(["DEF", "MID"]),
                        "Period": 1 if second < 2700 else 2,
                        "Game Time": f"{second // 60:02d}:{second % 60:02d}",
                        "Second": second,
                        "Minute": second // 60 + 1,
                        "Event Number": event_number,
                        "Sequence Index": idx,
                        "Phase": rng.choice(["DEFENCE", "PRESS", "TRANSITION"]),
                        "Action Type": action,
                        "Action": "DUEL" if action == "GROUND_DUEL" else action,
                        "Body Part": rng.choice(["RIGHT_FOOT", "LEFT_FOOT", "HEAD", None]),
                        "Result": rng.choice(["SUCCESS", "FAIL", "NEUTRAL"]),
                        "Pressure": rng.choice(["MEDIUM", "HIGH", None]),
                        "Start X": start_x,
                        "Start Y": start_y,
                        "End X": np.nan,
                        "End Y": np.nan,
                        "Raw Start X": start_x,
                        "Raw Start Y": start_y,
                        "Raw End X": np.nan,
                        "Raw End Y": np.nan,
                        "Start Lane": rng.choice(["LEFT", "HALF_LEFT", "CENTER", "HALF_RIGHT", "RIGHT"]),
                        "End Lane": None,
                        "Start Pitch Position": "FINAL_THIRD" if start_x >= 17.5 else "MIDDLE_THIRD",
                        "End Pitch Position": None,
                        "ReceiverId": None,
                        "Receiver": None,
                        "Pass Distance": np.nan,
                        "Pass Angle": np.nan,
                        "Team xT": np.nan,
                        "PXT Pass": np.nan,
                        "PXT Shot": np.nan,
                        "Shot xG": np.nan,
                        "Post-Shot xG": np.nan,
                        "Packing xG": np.nan,
                        "Bypassed Opponents": 0.0,
                        "Bypassed Defenders": 0.0,
                        "Shot Distance": np.nan,
                        "Shot Angle": np.nan,
                        "Shot Target Y": np.nan,
                        "Shot Target Z": np.nan,
                        "Shot GK X": np.nan,
                        "Shot GK Y": np.nan,
                        "Set Piece": False,
                        "Set Piece Category": None,
                        "Set Piece Execution": None,
                    }
                )
                event_number += 1
            _ = opponent
    return _clean_match_events(pd.DataFrame(rows))


@st.cache_data
def _mock_pass_network() -> pd.DataFrame:
    events = _mock_match_events()
    passes = events[events["Action Type"] == "PASS"].dropna(subset=["Receiver"]).copy()
    if passes.empty:
        return _empty_pass_network()
    grouped = passes.groupby(["MatchId", "Team", "PlayerId", "Player", "ReceiverId", "Receiver"], as_index=False).agg(
        **{
            "Pass Count": ("Action Type", "size"),
            "Passer X": ("Start X", "mean"),
            "Passer Y": ("Start Y", "mean"),
            "Receiver X": ("End X", "mean"),
            "Receiver Y": ("End Y", "mean"),
        }
    )
    return _clean_pass_network(grouped)
