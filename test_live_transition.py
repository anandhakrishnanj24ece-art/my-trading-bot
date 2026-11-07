"""Dry-run the live v3 runner against a copied old state.

This does not touch the real paper account: the state, balance and snapshot
paths are redirected to a temp directory, and yfinance is replaced by the
cached 7-year data.

Run:  python test_live_transition.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'nosharashi_1'))

import neural_flow.run_adaptive_v3 as r  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix='v3_transition_'))
TMP.mkdir(parents=True, exist_ok=True)

# Redirect every write away from the real package.
r.STATE_FILE = str(TMP / 'paper_state_v3.json')
r.BALANCE_FILE = str(TMP / 'paper_balance_v3.txt')
r.LEGACY_BALANCE_FILE = str(TMP / 'paper_balance.txt')
r.__file__ = str(TMP / 'pkg' / 'neural_flow' / 'run_adaptive_v3.py')

shutil.copy(ROOT / 'nosharashi_1' / 'paper_state_v3.json', r.STATE_FILE)

# Use the cached data as a one-year download.
close, high, low, volume = pd.read_pickle(ROOT / 'trend_data_7y.pkl')
idx = close.index[-380:]
fake = pd.concat({'Close': close.loc[idx], 'Volume': volume.loc[idx],
                  'High': high.loc[idx], 'Low': low.loc[idx]}, axis=1)


def fake_download(*args, **kwargs):
    return fake


r.yf.download = fake_download


class FakeDateTime:
    """Pretend it is 02:00 UTC on the day after the last bar."""

    @staticmethod
    def now(tz=None):
        return datetime(2026, 11, 8, 2, 0, 0, tzinfo=timezone.utc)


r.datetime = FakeDateTime

print('temp state:', TMP)
print('old state:', json.loads(Path(r.STATE_FILE).read_text()))
r.run_adaptive_v3()

new = json.loads(Path(r.STATE_FILE).read_text())
print('\n--- NEW STATE ---')
print('portfolio_value:', new['portfolio_value'])
print('universe:', new.get('universe'))
print('signal_model:', new.get('signal_model'))
print('last_bar_date:', new.get('last_bar_date'))
print('regime_on:', new.get('regime_on'))
print('weights:')
for t, w in new['weights'].items():
    if abs(w) > 1e-6:
        print(f'  {t:<10} {w*100:+.2f}%')
print('turnover-related last action:', new.get('last_action'))
print('trade history tail:', new.get('trade_history', [])[-1])
print('\nbalance file:', Path(r.BALANCE_FILE).read_text())
