"""Download 7 years of daily data, keeping the first year as warm-up.

The live runner downloads one year of history, so the 50-day trend and the
150-day BTC regime are always fully warm.  A backtest that starts cold in
2020-11 unfairly misses the 2021 rally; this cache fixes that.

Run:  python get_7y_data.py
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from backtest_trend import UNIVERSE

OUT = 'trend_data_7y.pkl'


def main():
    last = None
    for _ in range(6):
        try:
            raw = yf.download(UNIVERSE, period='7y', interval='1d',
                              progress=False, threads=True, auto_adjust=True)
            if raw is None or len(raw) == 0 or 'Close' not in raw:
                raise RuntimeError('empty download')
            close, high, low, volume = (raw[k].ffill() for k in
                                        ('Close', 'High', 'Low', 'Volume'))
            valid = [c for c in close.columns
                     if close[c].notna().sum() > 300 and c in volume.columns]
            close, high, low, volume = (x[valid] for x in
                                        (close, high, low, volume))
            if 'BTC-USD' not in close.columns:
                raise RuntimeError('BTC missing')
            pd.to_pickle((close, high, low, volume), OUT)
            print('saved raw:', close.shape, close.index[0].date(), '->',
                  close.index[-1].date())
            # Remove pre-launch / stale placeholder history before saving the
            # cache the backtest actually uses.
            from clean_7y_cache import main as clean_cache
            clean_cache()
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(4)
    raise RuntimeError(f'7y download failed: {last}')


if __name__ == '__main__':
    main()
