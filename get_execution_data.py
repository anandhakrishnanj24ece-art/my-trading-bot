"""Download daily opens and hourly bars for execution-timing audits."""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from backtest_trend import UNIVERSE

DAILY_OPEN = 'open_data.pkl'
HOURLY = 'hourly_data.pkl'


def download_daily_open():
    last = None
    for _ in range(5):
        try:
            raw = yf.download(UNIVERSE, period='6y', interval='1d',
                              progress=False, threads=True, auto_adjust=True)
            if raw is None or len(raw) == 0 or 'Open' not in raw:
                raise RuntimeError('empty')
            op = raw['Open'].ffill()
            pd.to_pickle(op, DAILY_OPEN)
            print('daily open:', op.shape, op.index[0].date(), '->', op.index[-1].date())
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3)
    raise RuntimeError(f'daily open download failed: {last}')


def download_hourly():
    last = None
    for _ in range(5):
        try:
            raw = yf.download(UNIVERSE, period='730d', interval='1h',
                              progress=False, threads=True, auto_adjust=True)
            if raw is None or len(raw) == 0 or 'Open' not in raw:
                raise RuntimeError('empty')
            # Keep both open and close for execution simulation.
            out = {'open': raw['Open'].ffill(), 'close': raw['Close'].ffill()}
            pd.to_pickle(out, HOURLY)
            print('hourly:', raw['Open'].shape, raw.index[0], '->', raw.index[-1])
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3)
    raise RuntimeError(f'hourly download failed: {last}')


if __name__ == '__main__':
    download_daily_open()
    download_hourly()
