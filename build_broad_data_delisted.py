"""Add available collapsed/delisted assets to the broad PIT universe.

The existing broad_data.pkl is missing FTT, CEL, LUNC and USTC because the
batch yfinance download drops dead tickers.  Individual Ticker.history calls
still work for those names.  LUNA/UST/ANC/MIR are no longer available; LUNC and
USTC are the post-collapse continuations of LUNA and UST.

Run:  python build_broad_data_delisted.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SRC = Path('broad_data.pkl')
DST = Path('broad_data_delisted.pkl')

CANDIDATES = ['LUNA-USD', 'LUNC-USD', 'UST-USD', 'USTC-USD',
              'FTT-USD', 'CEL-USD', 'ANC-USD', 'MIR-USD']


def fetch_one(ticker):
    h = yf.Ticker(ticker).history(period='6y', interval='1d',
                                  auto_adjust=True)
    if h is None or len(h) == 0 or 'Close' not in h:
        return None
    s = h['Close'].astype(float).copy()
    v = h['Volume'].astype(float).copy()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
        v.index = v.index.tz_localize(None)
    s.index = s.index.normalize()
    v.index = v.index.normalize()
    s = s[~s.index.duplicated(keep='last')].sort_index()
    v = v[~v.index.duplicated(keep='last')].sort_index()
    return s, v


def main():
    close0, volume0 = pd.read_pickle(SRC)
    new = {}
    for t in CANDIDATES:
        if t in close0.columns:
            continue
        try:
            got = fetch_one(t)
        except Exception as exc:  # noqa: BLE001
            print(t, 'ERR', type(exc).__name__, str(exc)[:70])
            continue
        if got is None:
            print(t, 'EMPTY')
            continue
        s, v = got
        new[t] = (s, v)
        print(f"{t:<10} rows={len(s):<5} first={s.index[0].date()} "
              f"last={s.index[-1].date()} last_px={float(s.iloc[-1]):.6g}")

    if not new:
        raise SystemExit('no new tickers downloaded')

    idx = close0.index
    for s, _ in new.values():
        idx = idx.union(s.index)
    idx = idx.sort_values()

    close = close0.reindex(idx).ffill()
    volume = volume0.reindex(idx).ffill()
    for t, (s, v) in new.items():
        close[t] = s.reindex(idx).ffill()
        volume[t] = v.reindex(idx).ffill()

    close = close.dropna(axis=1, how='all')
    volume = volume[close.columns]
    pd.to_pickle((close, volume), DST)
    print('\nsaved:', DST, close.shape, close.index[0].date(), '->',
          close.index[-1].date())
    print('\nnew column coverage:')
    for t in new:
        s = close[t]
        zeros = int((s <= 0).sum())
        print(f"  {t:<10} first={str(s.first_valid_index())[:10]} "
              f"last={str(s.last_valid_index())[:10]} rows={int(s.notna().sum()):<5} "
              f"zero/neg={zeros}")


if __name__ == '__main__':
    main()
