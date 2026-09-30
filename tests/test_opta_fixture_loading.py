import unittest
from unittest.mock import Mock, patch

import pandas as pd

from utils import data


class OptaFixtureLoadingTests(unittest.TestCase):
    def test_latest_loaded_snapshot_is_used_once_per_fixture(self) -> None:
        raw_fixtures = pd.DataFrame(
            [
                {
                    "FixtureId": "fixture-1",
                    "Opta Match Id": "match-1",
                    "Competition Id": "competition-1",
                    "Source Season": "2026",
                    "Date": "2026-08-14 19:00:00",
                    "Round": "1",
                    "Home Team Id": "home-1",
                    "Home": "Wolverhampton Wanderers",
                    "Away Team Id": "away-1",
                    "Away": "Blackburn Rovers",
                    "Home Goals": None,
                    "Away Goals": None,
                    "Venue": "Molineux",
                    "Loaded At": "2026-08-14 18:00:00+00:00",
                },
                {
                    "FixtureId": "fixture-1",
                    "Opta Match Id": "match-1",
                    "Competition Id": "competition-1",
                    "Source Season": "2026",
                    "Date": "2026-08-14 19:00:00",
                    "Round": "1",
                    "Home Team Id": "home-1",
                    "Home": "Wolverhampton Wanderers",
                    "Away Team Id": "away-1",
                    "Away": "Blackburn Rovers",
                    "Home Goals": 2,
                    "Away Goals": 2,
                    "Venue": "Molineux",
                    "Loaded At": "2026-08-15 08:00:00+00:00",
                },
                {
                    "FixtureId": "fixture-2",
                    "Opta Match Id": "match-2",
                    "Competition Id": "competition-1",
                    "Source Season": "2026",
                    "Date": "2026-08-15 14:00:00",
                    "Round": "1",
                    "Home Team Id": "home-2",
                    "Home": "Bristol City",
                    "Away Team Id": "away-2",
                    "Away": "Millwall",
                    "Home Goals": 0,
                    "Away Goals": 2,
                    "Venue": "Ashton Gate",
                    "Loaded At": "2026-08-15 17:00:00+00:00",
                },
            ]
        )
        connection = Mock()
        connection.query.return_value = raw_fixtures

        with (
            patch.object(data, "USE_MOCK_DATA", False),
            patch.object(data, "get_connection", return_value=connection),
        ):
            fixtures = data.load_opta_fixtures("26/27")

        self.assertEqual(fixtures["FixtureId"].tolist(), ["fixture-1", "fixture-2"])
        selected = fixtures.set_index("FixtureId").loc["fixture-1"]
        self.assertEqual(selected["Home Goals"], 2)
        self.assertEqual(selected["Away Goals"], 2)
        self.assertEqual(selected["Season"], "26/27")
        self.assertIn("TO_VARCHAR(SEASON) = ?", connection.query.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
