import unittest
from unittest.mock import Mock, patch

import pandas as pd

from utils import data


class MatchEventPlayerFallbackTests(unittest.TestCase):
    def test_actor_and_receiver_names_are_backfilled_without_changing_events(self) -> None:
        contexts = pd.DataFrame(
            [{"IterationId": 2114, "Season": "26/27", "Competition": "Championship"}]
        )
        matches = pd.DataFrame(
            [
                {
                    "IterationId": 2114,
                    "MatchId": 267911,
                    "Date": "2026-09-12",
                    "HomeTeamId": 959,
                    "AwayTeamId": 1001,
                }
            ]
        )
        squads = pd.DataFrame(
            [
                {"IterationId": 2114, "TeamId": 959, "Team": "Charlton Athletic"},
                {"IterationId": 2114, "TeamId": 1001, "Team": "Portsmouth"},
            ]
        )
        current_players = pd.DataFrame(
            [
                {"IterationId": 2114, "PlayerId": 32123, "Player": "Conor Coventry"},
                {"IterationId": 2114, "PlayerId": 6248, "Player": "   "},
            ]
        )
        raw_events = pd.DataFrame(
            [
                {
                    "IterationId": 2114,
                    "MatchId": 267911,
                    "TeamId": 959,
                    "PlayerId": 32123,
                    "ReceiverId": 6248,
                    "Event Number": 1,
                    "Second": 10,
                    "Action Type": "PASS",
                },
                {
                    "IterationId": 2114,
                    "MatchId": 267911,
                    "TeamId": 959,
                    "PlayerId": 6248,
                    "ReceiverId": 170059,
                    "Event Number": 2,
                    "Second": 20,
                    "Action Type": "PASS",
                },
                {
                    "IterationId": 2114,
                    "MatchId": 267911,
                    "TeamId": 959,
                    "PlayerId": 170059,
                    "ReceiverId": 32123,
                    "Event Number": 3,
                    "Second": 30,
                    "Action Type": "PASS",
                },
            ]
        )
        historical_names = pd.DataFrame(
            [
                {"PlayerId": "6248", "Player": " Oliver Skipp "},
                {"PlayerId": "170059", "Player": " Tyler Bindon "},
            ]
        )
        connection = Mock()
        connection.query.side_effect = [raw_events.copy(), historical_names.copy()]

        with (
            patch.object(data, "USE_MOCK_DATA", False),
            patch.object(
                data,
                "_match_fact_context",
                return_value=(contexts, "ITERATION_ID IN (?)", [2114]),
            ),
            patch.object(
                data,
                "_match_dimensions",
                return_value=(matches, squads, current_players),
            ),
            patch.object(data, "get_connection", return_value=connection),
        ):
            events = data.load_match_events(
                season="26/27",
                match_id=267911,
                team="Charlton Athletic",
                action_types=["PASS"],
            )

        self.assertEqual(len(events), len(raw_events))
        self.assertEqual(events["Event Number"].tolist(), [1, 2, 3])
        self.assertEqual(events["PlayerId"].tolist(), [32123, 6248, 170059])
        self.assertEqual(events["ReceiverId"].tolist(), [6248, 170059, 32123])
        self.assertEqual(
            events["Player"].tolist(),
            ["Conor Coventry", "Oliver Skipp", "Tyler Bindon"],
        )
        self.assertEqual(
            events["Receiver"].tolist(),
            ["Oliver Skipp", "Tyler Bindon", "Conor Coventry"],
        )
        self.assertEqual(connection.query.call_count, 2)
        fallback_call = connection.query.call_args_list[1]
        self.assertEqual(set(fallback_call.kwargs["params"][:-1]), {6248, 170059})
        self.assertEqual(fallback_call.kwargs["params"][-1], 2114)
        self.assertEqual(fallback_call.kwargs["ttl"], "6h")
        self.assertIn("WHERE IMPECT_PLAYER_ID IN (?, ?)", fallback_call.args[0])
        self.assertIn("IFF(ITERATION_ID IN (?), 0, 1)", fallback_call.args[0])

    def test_player_name_fallback_skips_a_query_when_no_ids_are_valid(self) -> None:
        with patch.object(data, "get_connection") as get_connection:
            names = data._load_player_name_dimensions(
                [None, "not-a-player-id", pd.NA],
                [2114],
            )

        self.assertEqual(names.columns.tolist(), ["PlayerId", "Player"])
        self.assertTrue(names.empty)
        get_connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
