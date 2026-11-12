"""Unit tests for the paper-only FVG Duo runner (no network)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fvg_duo_paper.paper_engine import PaperEngine
from fvg_duo_paper.signals import (
    build_signals, completed_5m, normalize_5m, resample_complete,
)


def frame(start: str, periods: int, *, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="5min",
                        tz="America/New_York").tz_convert("UTC")
    return pd.DataFrame(
        {
            "open": [price] * periods,
            "high": [price + 0.5] * periods,
            "low": [price - 0.5] * periods,
            "close": [price] * periods,
            "volume": [100] * periods,
        },
        index=idx,
    )


class SignalTests(unittest.TestCase):
    def test_partial_high_timeframe_bucket_is_dropped(self) -> None:
        bars = normalize_5m(frame("2026-11-12 09:30", 3))
        m10 = resample_complete(bars, 10)
        self.assertEqual(len(m10), 1)
        self.assertEqual(m10.iloc[0]["hm"], "09:30")

    def test_5m_bucket_with_missing_1m_bars_is_dropped(self) -> None:
        bars = normalize_5m(frame("2026-11-12 09:30", 4))
        bars["n_1m"] = 5
        bars.iloc[-1, bars.columns.get_loc("n_1m")] = 3
        now = pd.Timestamp("2026-11-12 09:50",
                           tz="America/New_York").tz_convert("UTC")
        out = completed_5m(bars, now)
        self.assertEqual(len(out), 3)
        self.assertNotIn(bars.index[-1], out.index)

    def test_opening_drive_fvg_signal(self) -> None:
        bars = frame("2026-11-12 09:00", 8, price=100.0)
        # 09:10 and 09:20 buckets: high 101; 09:30 bucket: low 102.
        for ts in bars.index:
            ny = ts.tz_convert("America/New_York")
            hm = ny.strftime("%H:%M")
            if hm in ("09:10", "09:15", "09:20", "09:25"):
                bars.loc[ts, ["high", "low", "close"]] = [101.0, 99.0, 100.0]
            elif hm in ("09:30", "09:35"):
                bars.loc[ts, ["high", "low", "close"]] = [103.0, 102.0, 102.5]
            elif hm in ("09:40", "09:45"):
                bars.loc[ts, ["high", "low", "close"]] = [104.0, 102.0, 103.0]
        now = pd.Timestamp("2026-11-12 09:40:30", tz="America/New_York")
        sig = build_signals(completed_5m(bars, now.tz_convert("UTC")))
        self.assertIn(
            pd.Timestamp("2026-11-12 09:40", tz="America/New_York")
            .tz_convert("UTC"),
            sig[0],
        )
        self.assertEqual(
            sig[0][pd.Timestamp("2026-11-12 09:40",
                                tz="America/New_York").tz_convert("UTC")],
            1,
        )


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = PaperEngine(base_dir=Path(self.tmp.name),
                                  starting_capital=50_000)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _flat_day(self) -> pd.DataFrame:
        return frame("2026-11-12 09:30", 79, price=102.6)

    def test_od_entry_stop_and_no_duplicate(self) -> None:
        bars = self._flat_day()
        now = pd.Timestamp("2026-11-12 09:40:30",
                           tz="America/New_York").tz_convert("UTC")
        sig_ts = pd.Timestamp("2026-11-12 09:40",
                              tz="America/New_York").tz_convert("UTC")
        result = self.engine.tick(now, bars, signals=({sig_ts: 1}, {}))
        self.assertEqual(result["action"], "open")
        pos = self.engine.state["positions"][0]
        self.assertEqual(pos["source"], "od")
        self.assertEqual(pos["direction"], "long")
        self.assertAlmostEqual(
            float(pos["stop_price"]),
            float(pos["entry_price"]) * 0.985,
            places=3,
        )
        # A second tick with the same signal must not add a trade.
        self.engine.tick(now + pd.Timedelta(minutes=1), bars,
                         signals=({sig_ts: 1}, {}))
        self.assertEqual(len(self.engine.state["trades"]), 0)
        self.assertEqual(len(self.engine.state["positions"]), 1)

    def test_stop_exit_uses_entry_bar(self) -> None:
        bars = self._flat_day()
        sig_ts = pd.Timestamp("2026-11-12 09:40",
                              tz="America/New_York").tz_convert("UTC")
        now1 = sig_ts + pd.Timedelta(seconds=30)
        self.engine.tick(now1, bars, signals=({sig_ts: 1}, {}))
        entry = float(self.engine.state["positions"][0]["entry_price"])
        stop = float(self.engine.state["positions"][0]["stop_price"])
        # Make the 09:40 bar trade through the stop.
        bars.loc[sig_ts, "low"] = stop - 2.0
        bars.loc[sig_ts, "high"] = entry + 1.0
        now2 = sig_ts + pd.Timedelta(minutes=5, seconds=30)
        result = self.engine.tick(now2, bars, signals=({sig_ts: 1}, {}))
        self.assertEqual(result["action"], "exit")
        self.assertEqual(result["trade"]["reason"], "stop")
        self.assertEqual(len(self.engine.state["positions"]), 0)

    def test_eod_close(self) -> None:
        bars = self._flat_day()
        sig_ts = pd.Timestamp("2026-11-12 09:40",
                              tz="America/New_York").tz_convert("UTC")
        now1 = sig_ts + pd.Timedelta(seconds=30)
        self.engine.tick(now1, bars, signals=({sig_ts: 1}, {}))
        now2 = pd.Timestamp("2026-11-12 15:56:00",
                            tz="America/New_York").tz_convert("UTC")
        result = self.engine.tick(now2, bars, signals=({sig_ts: 1}, {}))
        self.assertEqual(result["action"], "exit")
        self.assertEqual(result["trade"]["reason"], "eod")

    def test_halt_blocks_new_entry(self) -> None:
        self.engine.control = {"halted": True, "reason": "test"}
        bars = self._flat_day()
        sig_ts = pd.Timestamp("2026-11-12 09:40",
                              tz="America/New_York").tz_convert("UTC")
        now = sig_ts + pd.Timedelta(seconds=30)
        result = self.engine.tick(now, bars, signals=({sig_ts: 1}, {}))
        self.assertEqual(result["action"], "none")
        self.assertEqual(len(self.engine.state["positions"]), 0)


if __name__ == "__main__":
    unittest.main()
