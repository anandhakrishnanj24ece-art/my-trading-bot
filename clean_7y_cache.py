"""Clean the 7y cache: remove pre-launch and stale placeholder history.

Why:
  * FIL-USD carries a synthetic/IOU series from 2019-11; Filecoin spot did not
    launch until 2020-10-15 (the price jumps from ~$30 to ~$59 at launch).
  * AVAX-USD is flat at $4.986754 with tiny volume from 2020-07-13 until the
    real launch on 2020-09-22.

The cleaned cache keeps leading NaNs.  The strategy's 50d/150d signals then
warm up only from the real launch, instead of trading a placeholder series.

Run:  python clean_7y_cache.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path('trend_data_7y.pkl')
DST = Path('trend_data_7y_clean.pkl')

# First date of tradable spot history.  Only names with a known synthetic /
# placeholder prefix need an entry.
LAUNCH = {
    'FIL-USD': '2020-10-15',
    'AVAX-USD': '2020-09-22',
    'NEAR-USD': '2020-10-14',
    'SOL-USD': '2020-04-10',
    'DOT-USD': '2020-08-20',
}
STALE_RUN_DAYS = 10


def trim_column(frames, ticker, first_date):
    mask = frames[0].index < pd.Timestamp(first_date)
    for df in frames:
        df.loc[mask, ticker] = np.nan


def main():
    close, high, low, volume = pd.read_pickle(SRC)
    frames = (close, high, low, volume)

    for t, d in LAUNCH.items():
        if t in close.columns:
            trim_column(frames, t, d)

    # Generic leading flat-run trim (catches placeholder IOUs without a map).
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 2:
            continue
        first = s.index[0]
        p0 = s.iloc[0]
        after = s[s.index > first]
        moved = after[after != p0]
        if len(moved) and (moved.index[0] - first).days >= STALE_RUN_DAYS:
            trim_column(frames, t, moved.index[0])

    pd.to_pickle(frames, DST)
    print('saved:', DST, close.shape, close.index[0].date(), '->',
          close.index[-1].date())
    print(f"\n{'asset':<10}{'first':>12}{'rows':>7}{'note':>18}")
    for t in close.columns:
        s = close[t]
        note = ''
        if t in LAUNCH:
            note = 'launch ' + LAUNCH[t]
        elif s.notna().sum() and (s.dropna().iloc[0] == s.dropna().iloc[1]
                                  if s.notna().sum() > 1 else False):
            note = 'flat start'
        print(f"{t:<10}{str(s.first_valid_index())[:10]:>12}"
              f"{int(s.notna().sum()):>7}{note:>18}")


if __name__ == '__main__':
    main()
