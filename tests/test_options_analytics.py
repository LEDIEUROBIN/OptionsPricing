import unittest
from datetime import datetime, timedelta

import pandas as pd

import options_analytics as analytics


class OptionsAnalyticsTests(unittest.TestCase):
    def test_get_option_quote_price_prefers_expected_quote_side(self):
        row = pd.Series({"bid": 1.0, "ask": 1.4, "lastPrice": 1.2})

        self.assertEqual(analytics.get_option_quote_price(row, "mid"), 1.2)
        self.assertEqual(analytics.get_option_quote_price(row, "buy"), 1.4)
        self.assertEqual(analytics.get_option_quote_price(row, "sell"), 1.0)

    def test_get_option_quote_price_falls_back_to_last_price(self):
        row = pd.Series({"bid": 0, "ask": None, "lastPrice": 2.75})

        self.assertEqual(analytics.get_option_quote_price(row, "mid"), 2.75)
        self.assertEqual(analytics.get_option_quote_price(row, "buy"), 2.75)
        self.assertEqual(analytics.get_option_quote_price(row, "sell"), 2.75)

    def test_compute_implied_move_uses_atm_straddle(self):
        calls = pd.DataFrame(
            [
                {"strike": 95, "bid": 3.0, "ask": 3.4, "lastPrice": 3.2},
                {"strike": 100, "bid": 2.0, "ask": 2.4, "lastPrice": 2.2},
                {"strike": 105, "bid": 1.0, "ask": 1.4, "lastPrice": 1.2},
            ]
        )
        puts = pd.DataFrame(
            [
                {"strike": 95, "bid": 0.9, "ask": 1.1, "lastPrice": 1.0},
                {"strike": 100, "bid": 1.8, "ask": 2.2, "lastPrice": 2.0},
                {"strike": 105, "bid": 3.5, "ask": 3.9, "lastPrice": 3.7},
            ]
        )

        move_usd, move_pct, straddle = analytics.compute_implied_move(calls, puts, 101)

        self.assertEqual(straddle, 4.2)
        self.assertEqual(move_usd, 3.57)
        self.assertEqual(move_pct, 3.53)

    def test_compute_pop_keeps_long_short_relationship_consistent(self):
        pop_long, price_long, edge_long, fair_value_long = analytics.compute_pop(
            100, 100, 30 / 365, 0.04, 0.25, "call", "long", premium=3.5
        )
        pop_short, price_short, edge_short, fair_value_short = analytics.compute_pop(
            100, 100, 30 / 365, 0.04, 0.25, "call", "short", premium=3.5
        )

        self.assertAlmostEqual(pop_long + pop_short, 100.0, places=1)
        self.assertEqual(price_long, price_short)
        self.assertEqual(fair_value_long, fair_value_short)
        self.assertAlmostEqual(edge_long, -edge_short, places=2)

    def test_compute_roll_metrics_handles_long_and_short_positions(self):
        current_expiry = (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
        new_expiry = (datetime.now() + timedelta(days=55)).strftime("%Y-%m-%d")
        curr_row = pd.Series({"bid": 1.0, "ask": 1.2, "lastPrice": 1.1, "impliedVolatility": 0.20})
        new_row = pd.Series({"bid": 2.0, "ask": 2.2, "lastPrice": 2.1, "impliedVolatility": 0.25})

        long_roll = analytics.compute_roll_metrics(
            curr_row, new_row, current_expiry, new_expiry, 100, "call", 100, 0.04, "long"
        )
        short_roll = analytics.compute_roll_metrics(
            curr_row, new_row, current_expiry, new_expiry, 100, "call", 100, 0.04, "short"
        )

        self.assertIsNotNone(long_roll)
        self.assertIsNotNone(short_roll)
        self.assertEqual(long_roll["curr_price"], 1.0)
        self.assertEqual(long_roll["new_price"], 2.2)
        self.assertAlmostEqual(long_roll["roll_cost"], 1.2, places=6)
        self.assertEqual(short_roll["curr_price"], 1.2)
        self.assertEqual(short_roll["new_price"], 2.0)
        self.assertAlmostEqual(short_roll["roll_cost"], -0.8, places=6)
        self.assertEqual(long_roll["days_gained"], 35)
        self.assertEqual(short_roll["days_gained"], 35)


if __name__ == "__main__":
    unittest.main()
