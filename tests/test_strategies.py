"""Regression tests for the live strategy accounting and signal guards.

Run from the repository root:

    python -m unittest discover -s tests -v

The tests use synthetic prices and temporary state files, so they never touch
the deployed paper-trading accounts or the real price snapshots.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / 'nosharashi_1'
sys.path.insert(0, str(LIVE_DIR))

from neural_flow.alpha.model import LiquidityAlpha  # noqa: E402
from neural_flow.alpha.v3_core import btc_regime_series  # noqa: E402
from neural_flow.execution.model import StealthExecution  # noqa: E402
from neural_flow.risk.model import VolatilitySwitch, drawdown_multiplier  # noqa: E402
import neural_flow.run_adaptive as adaptive_runner  # noqa: E402
import neural_flow.run_adaptive_v3 as v3_runner  # noqa: E402
import neural_flow.run_trend as trend_runner  # noqa: E402

END = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=1)
DATES = pd.date_range(end=END, periods=420, freq='D')
TICKERS_9 = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
             'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
TICKERS_15 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
              'DOT-USD', 'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD',
              'ATOM-USD', 'NEAR-USD', 'ALGO-USD', 'FIL-USD']
CLOSES = {}


def make_close(ticker):
    seed = sum(ord(ch) for ch in ticker)
    rng = np.random.default_rng(seed)
    drift = 0.0008 if seed % 2 else -0.0002
    rets = drift + rng.normal(0.0, 0.022, len(DATES))
    return 100.0 * np.exp(np.cumsum(rets))


def fake_download(tickers, period=None, interval=None, progress=False,
                  threads=False, auto_adjust=True, **kwargs):
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    frames = []
    for ticker in tickers:
        close = make_close(ticker)
        CLOSES[ticker] = close
        df = pd.DataFrame({
            'Open': close * 0.999,
            'High': close * 1.01,
            'Low': close * 0.99,
            'Close': close,
            'Volume': np.full(len(DATES), 1_000_000.0),
        }, index=DATES)
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        frames.append(df)
    raw = pd.concat(frames, axis=1)
    return raw.swaplevel(0, 1, axis=1).sort_index(axis=1)


def run_quiet(func):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func()
    return buffer.getvalue()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


class SignalComponentTests(unittest.TestCase):
    def test_cost_model(self):
        # 0.05% fee + 0.014% effective slippage = 0.064% per unit turnover.
        self.assertAlmostEqual(StealthExecution().calculate_cost(1.0), 0.00064, places=8)

    def test_liquidity_alpha_does_not_trade_during_sma_warmup(self):
        close = pd.Series(np.linspace(100.0, 220.0, 250), index=DATES[:250])
        volume = pd.Series(np.linspace(1e6, 2e6, 250), index=DATES[:250])
        signals = LiquidityAlpha().generate_signal(
            pd.DataFrame({'Close': close, 'Volume': volume})
        )
        self.assertTrue((signals['signal'].iloc[:199] == 0.0).all())

    def test_volatility_switch_does_not_lever_flat_data(self):
        engine = VolatilitySwitch(base_leverage=4.0)
        self.assertEqual(engine.get_leverage(pd.Series([0.0] * 30)), 1.0)
        self.assertEqual(engine.get_leverage(pd.Series([np.nan] * 30)), 1.0)

    def test_drawdown_multiplier(self):
        # Below caution: full exposure.
        self.assertEqual(drawdown_multiplier(10000.0, 9900.0), 1.0)
        # In the decay band: linear between 1.0 and floor.
        mult = drawdown_multiplier(10000.0, 9400.0)  # 6% drawdown
        self.assertAlmostEqual(mult, 1.0 - (0.01 / 0.10) * 0.75, places=6)
        # At/above halt: floor exposure.
        self.assertEqual(drawdown_multiplier(10000.0, 8400.0), 0.25)
        self.assertEqual(drawdown_multiplier(10000.0, 7000.0), 0.25)
        # Non-positive peak: full exposure.
        self.assertEqual(drawdown_multiplier(0.0, 100.0), 1.0)

    def test_btc_regime_fallback_without_btc(self):
        close = pd.DataFrame(
            {'ETH-USD': pd.Series(np.linspace(100.0, 220.0, 250), index=DATES[:250])}
        )
        regime = btc_regime_series(close)
        self.assertEqual(len(regime), len(close))
        self.assertTrue(set(regime.dropna().unique()).issubset({0.0, 1.0}))


class LiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='nosharashi_test_'))
        self.original_download = yf.download
        yf.download = fake_download
        for ticker in set(TICKERS_9 + TICKERS_15):
            CLOSES[ticker] = make_close(ticker)

        self.original_paths = {}
        path_map = {
            adaptive_runner: ('STATE_FILE', 'BALANCE_FILE', 'SNAPSHOT_DIR'),
            v3_runner: ('STATE_FILE', 'BALANCE_FILE', 'LEGACY_BALANCE_FILE', 'SNAPSHOT_DIR'),
            trend_runner: ('STATE_FILE', 'BALANCE_FILE', 'SNAPSHOT_DIR'),
        }
        names = {
            adaptive_runner: 'adaptive',
            v3_runner: 'v3',
            trend_runner: 'trend',
        }
        for module, attributes in path_map.items():
            self.original_paths[module] = {
                attr: getattr(module, attr) for attr in attributes
            }
            name = names[module]
            module.STATE_FILE = str(self.tmp / f'{name}_state.json')
            module.BALANCE_FILE = str(self.tmp / f'{name}_balance.txt')
            module.SNAPSHOT_DIR = self.tmp / f'{name}_snapshots'
            if hasattr(module, 'LEGACY_BALANCE_FILE'):
                module.LEGACY_BALANCE_FILE = str(self.tmp / 'legacy_balance.txt')

    def tearDown(self):
        yf.download = self.original_download
        for module, attributes in self.original_paths.items():
            for attr, value in attributes.items():
                setattr(module, attr, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def seed_state(self, path, tickers, weights, extra=None, days_ago=1):
        state = {
            'portfolio_value': 10000.0,
            'peak_equity': 10000.0,
            'max_dd': 0.0,
            'weights': dict(weights),
            'last_prices': {t: float(CLOSES[t][-1 - days_ago]) for t in tickers},
            'last_bar_date': str(DATES[-1 - days_ago].date()),
            'trade_history': [],
        }
        state.update(extra or {})
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(state, handle)
        return state

    def test_adaptive_accounting_and_stale_close(self):
        weights = {t: (-0.10 if i % 2 == 0 else 0.10)
                   for i, t in enumerate(TICKERS_9)}
        self.seed_state(
            adaptive_runner.STATE_FILE,
            TICKERS_9,
            weights,
            extra={'universe': 'adaptive_9'},
            days_ago=2,
        )

        run_quiet(adaptive_runner.run_adaptive_weighted_strategy)
        state = load_json(adaptive_runner.STATE_FILE)
        self.assertEqual(state['universe'], 'adaptive_9')
        self.assertGreater(state['last_costs'], 0.0)
        self.assertGreater(state['last_short_financing'], 0.0)
        self.assertGreaterEqual(state['max_dd'], 0.0)
        self.assertIn('peak_equity', state)

        # Same bar for the same book must be idempotent.
        output = run_quiet(adaptive_runner.run_adaptive_weighted_strategy)
        self.assertIn('Skipping', output)

        # A stale ticker must be closed and charged, not silently dropped.
        state['weights']['FAKE-USD'] = -0.10
        state['last_prices']['FAKE-USD'] = 50.0
        state['last_bar_date'] = str((END - pd.Timedelta(days=2)).date())
        with open(adaptive_runner.STATE_FILE, 'w', encoding='utf-8') as handle:
            json.dump(state, handle)
        run_quiet(adaptive_runner.run_adaptive_weighted_strategy)
        state = load_json(adaptive_runner.STATE_FILE)
        self.assertNotIn('FAKE-USD', state['weights'])
        self.assertGreater(state['last_costs'], 0.0)
        self.assertTrue(any(adaptive_runner.SNAPSHOT_DIR.glob('adaptive_*.json')))

    def test_v3_accounting(self):
        weights = {t: (-0.05 if i < 5 else 0.05)
                   for i, t in enumerate(TICKERS_15)}
        self.seed_state(
            v3_runner.STATE_FILE,
            TICKERS_15,
            weights,
            extra={
                'universe': 'v3_15',
                'signal_model': 'trend50',
                'weights_raw': dict(weights),
            },
            days_ago=2,
        )

        run_quiet(v3_runner.run_adaptive_v3)
        state = load_json(v3_runner.STATE_FILE)
        self.assertEqual(state['universe'], 'v3_15')
        self.assertGreater(state['last_costs'], 0.0)
        self.assertGreater(state['last_short_financing'], 0.0)
        self.assertIn('leverage', state)
        self.assertIn('max_dd', state)

        output = run_quiet(v3_runner.run_adaptive_v3)
        self.assertIn('Skipping', output)
        self.assertTrue(any(v3_runner.SNAPSHOT_DIR.glob('v3_*.json')))

    def test_trend_accounting(self):
        weights = {'BTC-USD': 0.01, 'ETH-USD': 0.01}
        self.seed_state(
            trend_runner.STATE_FILE,
            ['BTC-USD', 'ETH-USD'],
            weights,
            days_ago=2,
        )

        run_quiet(trend_runner.main)
        state = load_json(trend_runner.STATE_FILE)
        self.assertGreaterEqual(state['last_costs'], 0.0)
        self.assertIn('max_dd', state)
        self.assertIn('last_gross_return_pct', state)

        output = run_quiet(trend_runner.main)
        self.assertIn('Skipping', output)
        self.assertTrue(any(trend_runner.SNAPSHOT_DIR.glob('trend_*.json')))


if __name__ == '__main__':
    unittest.main()
