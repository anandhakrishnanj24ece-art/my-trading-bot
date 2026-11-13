from __future__ import annotations

import json
import tempfile
import unittest
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from trend_trio_paper.paper_engine import (
    CONTRACT_LABEL, FEE_PER_SIDE, POINT_VALUE, PaperEngine,
)
from trend_trio_paper.signals import (
    TZ, Entry, Exit, adx, alpha_trend, build_entries, build_exits,
    chandelier_exit, completed_1m, scan_entry, supertrend, ultimate_ma,
)


def make_bars(n: int = 400, start: str = "2026-11-13 09:30") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz=TZ).tz_convert("UTC")
    x = np.arange(n, dtype=float)
    close = 100.0 + 0.02 * x + 2.0 * np.sin(x / 11.0)
    open_ = close - 0.05
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15
    vol = 100.0 + (x % 7) * 10.0
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    }, index=idx)


class SignalTests(unittest.TestCase):
    def test_supertrend_has_both_directions(self):
        bars = make_bars(500)
        st = supertrend(bars, 10, 3.0)
        self.assertIn(1, set(st["trend"].dropna().astype(int)))
        self.assertIn(-1, set(st["trend"].dropna().astype(int)))
        self.assertTrue(bool(st["flip_up"].any()))
        self.assertTrue(bool(st["flip_down"].any()))

    def test_ultimate_ma_turns_green(self):
        idx = pd.date_range("2026-11-13 00:00", periods=80, freq="4min",
                            tz=TZ).tz_convert("UTC")
        close = np.r_[np.linspace(120, 100, 40), np.linspace(100, 130, 40)]
        bars = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 1.0,
        }, index=idx)
        uma = ultimate_ma(bars, 20, 2)
        self.assertTrue(bool(uma["up"].iloc[-1]))
        self.assertTrue(bool(uma["turn_green"].any()))

    def test_chandelier_turn_down(self):
        idx = pd.date_range("2026-11-13 00:00", periods=120, freq="4min",
                            tz=TZ).tz_convert("UTC")
        close = np.r_[np.linspace(100, 140, 60), np.linspace(140, 100, 60)]
        bars = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 1.0,
        }, index=idx)
        ce = chandelier_exit(bars, 22, 3.0)
        self.assertTrue(bool(ce["turn_down"].any()))

    def test_alpha_trend_buy_and_sell(self):
        bars = make_bars(500)
        at = alpha_trend(bars)
        self.assertTrue(bool(at["buy"].any()))
        self.assertTrue(bool(at["sell"].any()))

    def test_adx_is_positive_after_warmup(self):
        bars = make_bars(200)
        vals = adx(bars, 14).dropna()
        self.assertFalse(vals.empty)
        self.assertTrue((vals >= 0).all())

    def test_completed_1m_drops_partial_bar(self):
        bars = make_bars(5)
        now = bars.index[2] + pd.Timedelta(seconds=30)
        done = completed_1m(bars, now)
        self.assertEqual(len(done), 2)

    def test_scan_entry_rejects_stale_signal(self):
        now = pd.Timestamp("2026-11-13 10:00:00", tz=TZ).tz_convert("UTC")
        old = Entry(now - pd.Timedelta(minutes=6), "st", 1, 1.0)
        fresh = Entry(now - pd.Timedelta(minutes=2), "st", 1, 1.0)
        self.assertIsNone(scan_entry([old], now))
        self.assertEqual(scan_entry([old, fresh], now), fresh)

    def test_build_entries_and_exits_do_not_crash(self):
        bars = make_bars(1200, start="2026-11-13 00:00")
        entries = build_entries(bars)
        exits = build_exits(bars)
        self.assertIsInstance(entries, list)
        self.assertIsInstance(exits, list)


class EngineTests(unittest.TestCase):
    def test_deployed_model_is_a(self):
        """Lock the deployed paper model: 1 MNQ-equivalent, $2/pt, $2/side."""
        self.assertEqual(POINT_VALUE, 2.0)
        self.assertEqual(FEE_PER_SIDE, 2.0)
        self.assertIn("MNQ", CONTRACT_LABEL)

    def _bars(self):
        return make_bars(30, start="2026-11-13 09:30")

    def test_open_then_stop(self):
        bars = self._bars()
        sig_time = bars.index[5]
        entries = [Entry(sig_time, "st", 1, 1.0, "test")]
        with tempfile.TemporaryDirectory() as td:
            eng = PaperEngine(base_dir=td)
            now = bars.index[6] + pd.Timedelta(seconds=1)
            res = eng.tick(now, bars, entries=entries, exits=[])
            self.assertEqual(res["action"], "open")
            self.assertEqual(len(eng.state["positions"]), 1)
            pos = eng.state["positions"][0]
            self.assertAlmostEqual(pos["stop_price"], pos["entry_price"] - 1.0)

            # Force the next completed bar through the stop.
            bars2 = bars.copy()
            bars2.iloc[6, bars2.columns.get_loc("low")] = (
                float(pos["stop_price"]) - 1.0
            )
            now2 = bars.index[7] + pd.Timedelta(seconds=1)
            res2 = eng.tick(now2, bars2, entries=[], exits=[])
            self.assertEqual(res2["action"], "exit")
            self.assertEqual(res2["trade"]["reason"], "stop")

    def test_uma_no_stop_and_signal_exit(self):
        bars = self._bars()
        sig_time = bars.index[5]
        entries = [Entry(sig_time, "uma", 1, 0.0, "test")]
        exits = [Exit(bars.index[6], "uma", 1, "chandelier")]
        with tempfile.TemporaryDirectory() as td:
            eng = PaperEngine(base_dir=td)
            now = bars.index[6] + pd.Timedelta(seconds=1)
            res = eng.tick(now, bars, entries=entries, exits=[])
            self.assertEqual(res["action"], "open")
            self.assertIsNone(eng.state["positions"][0]["stop_price"])

            now2 = bars.index[7] + pd.Timedelta(seconds=1)
            res2 = eng.tick(now2, bars, entries=[], exits=exits)
            self.assertEqual(res2["action"], "exit")
            self.assertEqual(res2["trade"]["reason"], "chandelier")

    def test_halt_blocks_new_entries(self):
        bars = self._bars()
        entries = [Entry(bars.index[5], "st", 1, 1.0, "test")]
        with tempfile.TemporaryDirectory() as td:
            ctrl = Path(td) / "control.json"
            ctrl.write_text(json.dumps({"halted": True}), encoding="utf-8")
            eng = PaperEngine(base_dir=td)
            now = bars.index[6] + pd.Timedelta(seconds=1)
            res = eng.tick(now, bars, entries=entries, exits=[])
            self.assertEqual(res["action"], "none")
            self.assertIn("halted", res["reason"])
            self.assertEqual(eng.state["positions"], [])

    def test_eod_exit_for_supertrend(self):
        bars = self._bars()
        sig_time = bars.index[5]
        entries = [Entry(sig_time, "st", 1, 100.0, "test")]
        with tempfile.TemporaryDirectory() as td:
            eng = PaperEngine(base_dir=td)
            now = bars.index[6] + pd.Timedelta(seconds=1)
            eng.tick(now, bars, entries=entries, exits=[])
            # 15:59 New York
            eod = pd.Timestamp("2026-11-13 15:59:00", tz=TZ).tz_convert("UTC")
            bars2 = bars.copy()
            bars2.loc[eod] = {
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0, "volume": 1.0,
            }
            bars2 = bars2.sort_index()
            res = eng.tick(eod + pd.Timedelta(minutes=1, seconds=1), bars2,
                           entries=[], exits=[])
            self.assertEqual(res["action"], "exit")
            self.assertEqual(res["trade"]["reason"], "eod")


if __name__ == "__main__":
    unittest.main()
