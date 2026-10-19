import unittest
from unittest.mock import Mock, patch

import pandas as pd

from utils import data


class MatchPlayerRatingsInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contexts = pd.DataFrame(
            [{"IterationId": 1410, "Season": "25/26", "Competition": "Championship"}]
        )
        self.squads = pd.DataFrame([{"IterationId": 1410, "TeamId": 959, "Team": "Charlton Athletic"}])
        self.players = pd.DataFrame(
            [
                {"IterationId": 1410, "PlayerId": 1, "Player": "Starter"},
                {"IterationId": 1410, "PlayerId": 2, "Player": "Keeper"},
            ]
        )

    def _load(self, long_values: pd.DataFrame) -> pd.DataFrame:
        connection = Mock()
        connection.query.return_value = long_values
        with (
            patch.object(data, "USE_MOCK_DATA", False),
            patch.object(data, "_match_fact_context", return_value=(self.contexts, "ITERATION_ID IN (?)", [1410])),
            patch.object(data, "get_connection", return_value=connection),
            patch.object(data, "_match_dimensions", return_value=(pd.DataFrame(), self.squads, self.players)),
        ):
            return data.load_match_player_ratings_input("25/26", 999).set_index("Player")

    def test_match_totals_are_summed_across_stints_and_scaled_to_minutes_played(self) -> None:
        # Player 1 plays 45 minutes in each of two positions and scores once in each.
        long_values = pd.DataFrame(
            [
                {"IterationId": 1410, "TeamId": 959, "PlayerId": 1, "Position": "CENTER_FORWARD",
                 "Play Duration Seconds": 2700, "KpiId": 28, "KpiValue": 1.0},
                {"IterationId": 1410, "TeamId": 959, "PlayerId": 1, "Position": "LEFT_WINGER",
                 "Play Duration Seconds": 2700, "KpiId": 28, "KpiValue": 1.0},
                {"IterationId": 1410, "TeamId": 959, "PlayerId": 1, "Position": "LEFT_WINGER",
                 "Play Duration Seconds": 2700, "KpiId": 90, "KpiValue": 30.0},
                {"IterationId": 1410, "TeamId": 959, "PlayerId": 2, "Position": "GOALKEEPER",
                 "Play Duration Seconds": 5400, "KpiId": 90, "KpiValue": 20.0},
            ]
        )

        players = self._load(long_values)

        self.assertAlmostEqual(players.loc["Starter", "Minutes"], 90.0)
        self.assertAlmostEqual(players.loc["Starter", "Goals /90"], 2.0)
        self.assertAlmostEqual(players.loc["Starter", "Successful Passes /90"], 30.0)
        # A missing goalkeeping KPI row means zero for a keeper but "not applicable" outfield.
        self.assertEqual(players.loc["Keeper", "Goals Conceded /90"], 0.0)
        self.assertTrue(pd.isna(players.loc["Starter", "Goals Conceded /90"]))

    def test_substitute_rates_use_their_own_minutes(self) -> None:
        long_values = pd.DataFrame(
            [
                {"IterationId": 1410, "TeamId": 959, "PlayerId": 1, "Position": "CENTER_FORWARD",
                 "Play Duration Seconds": 1800, "KpiId": 28, "KpiValue": 1.0},
            ]
        )

        players = self._load(long_values)

        self.assertAlmostEqual(players.loc["Starter", "Minutes"], 30.0)
        self.assertAlmostEqual(players.loc["Starter", "Goals /90"], 3.0)


if __name__ == "__main__":
    unittest.main()
