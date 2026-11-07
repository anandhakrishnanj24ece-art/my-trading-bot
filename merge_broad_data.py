"""Merge broad universe with the original 15-asset cache."""
from __future__ import annotations

import pandas as pd

from backtest_trend import load_data

close_b, vol_b = pd.read_pickle('broad_data.pkl')
close_15, high, low, vol_15 = load_data('6y')

for t in close_15.columns:
    if t not in close_b.columns:
        close_b[t] = close_15[t]
        vol_b[t] = vol_15[t]

close_b = close_b.sort_index().ffill()
vol_b = vol_b.reindex(close_b.index).ffill()
# Drop columns with no usable history.
keep = [c for c in close_b.columns if close_b[c].notna().sum() > 200]
close_b = close_b[keep]
vol_b = vol_b[keep]

pd.to_pickle((close_b, vol_b), 'broad_data.pkl')
print('merged:', close_b.shape, close_b.index[0].date(), '->', close_b.index[-1].date())
print('tickers:', ', '.join(close_b.columns))
