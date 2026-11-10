from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ultimateorb_paper.paper_engine import PaperEngine, TradingHalted


class PaperEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.engine = PaperEngine(base_dir=self.base, starting_capital=10_000)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_halt_blocks_new_entries_but_allows_marks(self) -> None:
        self.engine.open_trade("long", 100.0)
        self.engine.halt("test halt")
        with self.assertRaises(TradingHalted):
            self.engine.open_trade("long", 101.0)
        result = self.engine.mark(high=101, low=99, close=100.5)
        self.assertEqual(result["positions"][0]["unrealized_points"], 0.5)
        self.engine.resume()
        self.engine.close_manual(100.5)
        self.assertFalse(self.engine.is_halted())

    def test_long_target_and_commission(self) -> None:
        self.engine.open_trade("long", 100.0, target_points=6.6, stop_points=10.0)
        result = self.engine.mark(high=107, low=99, close=106)
        self.assertEqual(len(result["closed"]), 1)
        trade = result["closed"][0]
        self.assertEqual(trade["reason"], "target")
        self.assertAlmostEqual(trade["gross_points"], 6.6)
        self.assertAlmostEqual(trade["net_points"], 6.6 - 0.95)
        self.assertAlmostEqual(trade["net_usd"], (6.6 - 0.95) * 2, places=2)
        self.assertEqual(len(self.engine.state["positions"]), 0)

    def test_same_bar_stop_and_target_is_stop_first(self) -> None:
        self.engine.open_trade("short", 100.0, target_points=6.6, stop_points=10.0)
        result = self.engine.mark(high=111, low=92, close=100)
        trade = result["closed"][0]
        self.assertEqual(trade["reason"], "stop")
        self.assertAlmostEqual(trade["gross_points"], -10.0)

    def test_halt_does_not_close_existing_position(self) -> None:
        self.engine.open_trade("long", 100.0)
        self.engine.halt("manual")
        self.assertEqual(len(self.engine.state["positions"]), 1)
        self.assertEqual(self.engine.snapshot()["stats"]["open_positions"], 1)

    def test_trade_csv_export(self) -> None:
        self.engine.open_trade("long", 100.0)
        self.engine.mark(high=107, low=99, close=106)
        path = self.base / "state" / "paper_trades.csv"
        self.assertTrue(path.exists())
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "target")

    def test_control_file_is_persisted(self) -> None:
        self.engine.halt("persisted")
        reloaded = PaperEngine(base_dir=self.base)
        self.assertTrue(reloaded.is_halted())
        control = json.loads((self.base / "state" / "control.json").read_text(encoding="utf-8"))
        self.assertEqual(control["reason"], "persisted")

    def test_paper_only_guard(self) -> None:
        import ultimateorb_paper.paper_engine as module
        self.assertTrue(module.PAPER_ONLY)
        self.assertFalse(self.engine.config["allow_live_orders"])


if __name__ == "__main__":
    unittest.main()
