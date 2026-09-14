import unittest

import pandas as pd

from utils import pass_network
from utils import pitch


def _twelve_player_network() -> pd.DataFrame:
    players = ["Danny McNamara", *[f"Starter {number}" for number in range(2, 12)], "Substitute 12"]
    rows = []
    for index, player in enumerate(players):
        receiver_index = (index + 1) % len(players)
        rows.append(
            {
                "Team": "Charlton Athletic",
                "PlayerId": index + 1,
                "Player": player,
                "ReceiverId": receiver_index + 1,
                "Receiver": players[receiver_index],
                "Pass Count": index + 2,
                "Passer X": -35 + index * 6,
                "Passer Y": -20 + (index % 4) * 12,
                "Receiver X": -35 + receiver_index * 6,
                "Receiver Y": -20 + (receiver_index % 4) * 12,
            }
        )
    return pd.DataFrame(rows)


class PassNetworkScopeTests(unittest.TestCase):
    def test_provider_given_name_variant_resolves_to_unique_network_player(self) -> None:
        network = _twelve_player_network()

        resolved = pass_network.resolve_network_name_keys(["Dan McNamara"], network)

        self.assertEqual(resolved, {pass_network.player_key("Danny McNamara")})

    def test_starting_xi_scope_retains_all_eleven_players(self) -> None:
        network = _twelve_player_network()
        starters = ["Dan McNamara", *[f"Starter {number}" for number in range(2, 12)]]

        selected = pass_network.resolve_network_name_keys(starters, network)
        visible = pass_network.filter_network(network, selected)

        self.assertEqual(len(selected), 11)
        self.assertEqual(len(pass_network.network_player_names(visible)), 11)

    def test_entire_match_scope_and_plot_have_no_player_cap(self) -> None:
        network = _twelve_player_network()
        selected = pass_network.network_player_names(network)
        visible = pass_network.filter_network(network, selected)

        figure = pitch.passing_network(
            visible,
            "Charlton Athletic",
            "Entire match",
            min_passes=1,
        )
        player_trace = next(trace for trace in figure.data if trace.name == "Players")

        self.assertEqual(len(selected), 12)
        self.assertEqual(len(player_trace.x), 12)

    def test_a_player_the_dimension_cannot_name_is_still_plotted(self) -> None:
        network = _twelve_player_network()
        network.loc[network["PlayerId"].eq(4), "Player"] = None
        network.loc[network["ReceiverId"].eq(4), "Receiver"] = None

        figure = pitch.passing_network(
            network,
            "Charlton Athletic",
            "Entire match",
            min_passes=1,
        )
        player_trace = next(trace for trace in figure.data if trace.name == "Players")

        self.assertEqual(len(player_trace.x), 12)
        self.assertIn("Unknown #4", [str(row[0]) for row in player_trace.customdata])

    def test_one_player_spelled_two_ways_is_a_single_node(self) -> None:
        network = _twelve_player_network()
        network.loc[network["ReceiverId"].eq(1), "Receiver"] = "Dan McNamara"

        figure = pitch.passing_network(
            network,
            "Charlton Athletic",
            "Entire match",
            min_passes=1,
        )
        player_trace = next(trace for trace in figure.data if trace.name == "Players")
        involvement = {
            str(row[0]): float(row[3]) for row in player_trace.customdata
        }

        self.assertEqual(len(player_trace.x), 12)
        # Both spellings' passes belong to one dot: 2 made, 13 received.
        self.assertEqual(involvement["Danny McNamara"], 15.0)

    def test_player_count_includes_a_player_the_dimension_cannot_name(self) -> None:
        network = _twelve_player_network()
        network.loc[network["PlayerId"].eq(4), "Player"] = None
        network.loc[network["ReceiverId"].eq(4), "Receiver"] = None

        self.assertEqual(len(pass_network.network_player_names(network)), 11)
        self.assertEqual(pass_network.network_player_count(network), 12)


if __name__ == "__main__":
    unittest.main()
