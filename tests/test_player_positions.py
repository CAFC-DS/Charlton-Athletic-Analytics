import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from utils import player_analysis
from utils import positions


class PlayerPositionTests(unittest.TestCase):
    def test_dominant_position_uses_minutes_not_alphabetical_order(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "PlayerId": 1,
                    "Position": "CENTRAL_DEFENDER",
                    "Play Duration Seconds": 20_000,
                },
                {
                    "PlayerId": 1,
                    "Position": "ATTACKING_MIDFIELD",
                    "Play Duration Seconds": 30,
                },
                {
                    "PlayerId": 1,
                    "Position": "CENTER_FORWARD",
                    "Play Duration Seconds": 20,
                },
                {
                    "PlayerId": 1,
                    "Position": "CENTRAL_MIDFIELD",
                    "Play Duration Seconds": 10,
                },
            ]
        )

        dominant = positions.dominant_positions(position_rows, ["PlayerId"])

        self.assertEqual(dominant.loc[0, "Position"], "CENTRAL_DEFENDER")
        self.assertEqual(
            positions.position_group(dominant.loc[0, "Position"]),
            "Centre Back",
        )

    def test_role_minutes_are_summed_across_provider_aliases(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "PlayerId": 2,
                    "Position": "LEFT_WINGBACK_DEFENDER",
                    "Play Duration Seconds": 1_500,
                },
                {
                    "PlayerId": 2,
                    "Position": "RIGHT_WINGBACK_DEFENDER",
                    "Play Duration Seconds": 1_400,
                },
                {
                    "PlayerId": 2,
                    "Position": "ATTACKING_MIDFIELD",
                    "Play Duration Seconds": 2_500,
                },
            ]
        )

        dominant = positions.dominant_positions(position_rows, ["PlayerId"])

        self.assertEqual(dominant.loc[0, "Position"], "LEFT_WINGBACK_DEFENDER")
        self.assertEqual(
            positions.position_group(dominant.loc[0, "Position"]),
            "Full Back",
        )

    def test_all_live_impect_position_codes_are_classified(self) -> None:
        expected_groups = {
            "ATTACKING_MIDFIELD": "Attacking Midfielder",
            "CENTER_FORWARD": "Forward / Winger",
            "CENTRAL_DEFENDER": "Centre Back",
            "CENTRAL_MIDFIELD": "Central Midfielder",
            "DEFENSE_MIDFIELD": "Central Midfielder",
            "GOALKEEPER": "Goalkeeper",
            "LEFT_WINGBACK_DEFENDER": "Full Back",
            "LEFT_WINGER": "Forward / Winger",
            "RIGHT_WINGBACK_DEFENDER": "Full Back",
            "RIGHT_WINGER": "Forward / Winger",
        }

        for position_code, expected_group in expected_groups.items():
            with self.subTest(position_code=position_code):
                self.assertEqual(
                    positions.classify_position_text(position_code),
                    expected_group,
                )

    def test_both_impect_central_midfield_codes_share_one_peer_group(self) -> None:
        players = pd.DataFrame(
            [
                {"Player": "Six", "Position": "DEFENSE_MIDFIELD"},
                {"Player": "Eight", "Position": "CENTRAL_MIDFIELD"},
                {"Player": "Ten", "Position": "ATTACKING_MIDFIELD"},
            ]
        )

        grouped = player_analysis.add_position_groups(players)

        self.assertEqual(
            grouped.set_index("Player")["Role Group"].to_dict(),
            {
                "Six": "Central Midfielder",
                "Eight": "Central Midfielder",
                "Ten": "Attacking Midfielder",
            },
        )

    def test_split_central_midfield_labels_sum_into_one_dominant_role(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "PlayerId": 7,
                    "Position": "DEFENSE_MIDFIELD",
                    "Play Duration Seconds": 3_000,
                },
                {
                    "PlayerId": 7,
                    "Position": "CENTRAL_MIDFIELD",
                    "Play Duration Seconds": 3_400,
                },
                {
                    "PlayerId": 7,
                    "Position": "CENTER_FORWARD",
                    "Play Duration Seconds": 4_000,
                },
            ]
        )

        dominant = positions.dominant_positions(position_rows, ["PlayerId"])

        # 3,000s + 3,400s of central midfield outranks 4,000s up front, and the
        # longer of the two provider labels is the one displayed.
        self.assertEqual(dominant.loc[0, "Position"], "CENTRAL_MIDFIELD")
        self.assertEqual(
            positions.position_group(dominant.loc[0, "Position"]),
            "Central Midfielder",
        )

    def test_zero_duration_and_null_positions_do_not_displace_a_role(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "PlayerId": 3,
                    "Position": "UNMAPPED_PROVIDER_ROLE",
                    "Play Duration Seconds": 0,
                },
                {
                    "PlayerId": 3,
                    "Position": "ATTACKING_MIDFIELD",
                    "Play Duration Seconds": 0,
                },
                {
                    "PlayerId": 3,
                    "Position": "CENTRAL_DEFENDER",
                    "Play Duration Seconds": 5_000,
                },
                {
                    "PlayerId": 3,
                    "Position": None,
                    "Play Duration Seconds": 60_000,
                },
            ]
        )

        dominant = positions.dominant_positions(position_rows, ["PlayerId"])

        self.assertEqual(dominant.loc[0, "Position"], "CENTRAL_DEFENDER")

    def test_dominant_unknown_position_is_not_hidden_by_a_mapped_cameo(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "PlayerId": 6,
                    "Position": "NEW_PROVIDER_DEFENDER_ROLE",
                    "Play Duration Seconds": 50_000,
                },
                {
                    "PlayerId": 6,
                    "Position": "ATTACKING_MIDFIELD",
                    "Play Duration Seconds": 60,
                },
            ]
        )

        dominant = positions.dominant_positions(position_rows, ["PlayerId"])

        self.assertEqual(dominant.loc[0, "Position"], "NEW_PROVIDER_DEFENDER_ROLE")
        self.assertEqual(
            positions.position_group(dominant.loc[0, "Position"]),
            "Outfield",
        )

    def test_dominant_position_selection_does_not_mutate_source_totals(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "IterationId": 2114,
                    "PlayerId": 4,
                    "Position": "CENTRAL_DEFENDER",
                    "Play Duration Seconds": 4_500,
                    "Match Share": 1.0,
                },
                {
                    "IterationId": 2114,
                    "PlayerId": 4,
                    "Position": "ATTACKING_MIDFIELD",
                    "Play Duration Seconds": 90,
                    "Match Share": 0.1,
                },
            ]
        )
        original = position_rows.copy(deep=True)

        dominant = positions.dominant_positions(
            position_rows,
            ["IterationId", "PlayerId"],
        )

        assert_frame_equal(position_rows, original)
        self.assertEqual(
            dominant.columns.tolist(),
            ["IterationId", "PlayerId", "Position"],
        )
        self.assertEqual(position_rows["Play Duration Seconds"].sum(), 4_590)
        self.assertEqual(position_rows["Match Share"].sum(), 1.1)

    def test_position_rollup_integrates_with_player_role_groups(self) -> None:
        position_rows = pd.DataFrame(
            [
                {
                    "PlayerId": 5,
                    "Position": "CENTRAL_DEFENDER",
                    "Play Duration Seconds": 9_000,
                },
                {
                    "PlayerId": 5,
                    "Position": "ATTACKING_MIDFIELD",
                    "Play Duration Seconds": 60,
                },
            ]
        )
        players = positions.dominant_positions(position_rows, ["PlayerId"])
        players["Player"] = "Jannik Vestergaard"

        grouped = player_analysis.add_position_groups(players)

        self.assertEqual(grouped.loc[0, "Position"], "CENTRAL_DEFENDER")
        self.assertEqual(grouped.loc[0, "Role Group"], "Centre Back")


if __name__ == "__main__":
    unittest.main()
