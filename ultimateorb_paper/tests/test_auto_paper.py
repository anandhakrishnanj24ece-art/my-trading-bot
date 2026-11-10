from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from ultimateorb_paper.auto_paper import _decision
from ultimateorb_paper.paper_engine import PaperEngine

CT = ZoneInfo("America/Chicago")


def _session_bars() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2026-11-09 08:30",
            "2026-11-09 08:45",
            "2026-11-09 09:00",
            "2026-11-09 09:15",
            "2026-11-09 09:30",
            "2026-11-09 09:45",
        ]
    ).tz_localize(CT)
    return pd.DataFrame(
        {
            "Open": [100, 100.5, 101, 101.5, 102, 102.5],
            "High": [101, 102, 102.2, 102.5, 102.8, 103.5],
            "Low": [99, 98, 100.8, 101.2, 101.8, 102.4],
            "Close": [100.5, 101, 101.5, 102, 102.5, 103],
        },
        index=index,
    )


class AutoPaperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.engine = PaperEngine(base_dir=self.base, starting_capital=10_000)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, now: datetime) -> dict:
        with patch("ultimateorb_paper.auto_paper._load_nq", return_value=_session_bars()):
            return _decision(self.engine, now, dry_run=False)

    def test_opening_range_breakout_opens_one_paper_trade(self) -> None:
        result = self._run(datetime(2026, 11, 9, 10, 0, tzinfo=CT))
        self.assertEqual(result["action"], "open")
        self.assertEqual(result["trade"]["direction"], "long")
        self.assertAlmostEqual(result["trade"]["entry_price"], 103.0)
        self.assertEqual(self.engine.state["last_entry_session"], "2026-11-09")
        self.assertEqual(len(self.engine.state["positions"]), 1)

    def test_second_run_same_session_does_not_add_a_trade(self) -> None:
        self._run(datetime(2026, 11, 9, 10, 0, tzinfo=CT))
        result = self._run(datetime(2026, 11, 9, 10, 6, tzinfo=CT))
        self.assertIn(result["action"], {"mark", "time_exit"})
        self.assertEqual(len(self.engine.state["positions"]), 1)

    def test_halt_blocks_auto_entry(self) -> None:
        self.engine.halt("test")
        result = self._run(datetime(2026, 11, 9, 10, 0, tzinfo=CT))
        self.assertEqual(result["action"], "none")
        self.assertIn("halted", result["reason"])
        self.assertEqual(len(self.engine.state["positions"]), 0)

    def test_session_close_exits_open_position(self) -> None:
        self._run(datetime(2026, 11, 9, 10, 0, tzinfo=CT))
        result = self._run(datetime(2026, 11, 9, 15, 0, tzinfo=CT))
        self.assertEqual(result["action"], "time_exit")
        self.assertEqual(result["trade"]["reason"], "session close")
        self.assertEqual(len(self.engine.state["positions"]), 0)


if __name__ == "__main__":
    unittest.main()
