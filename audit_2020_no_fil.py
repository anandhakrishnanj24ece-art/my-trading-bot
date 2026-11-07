"""Check how much the suspicious FIL history changes the 2020 result."""
import pandas as pd

from backtest_trend import metrics, slice_s
from backtest_v3_improved import load_7y, run_v3_live_logic, TICKERS, COST, TEST_END

close, high, low, vol = load_7y()
for excl in ([], ['FIL-USD'], ['FIL-USD', 'AVAX-USD']):
    cols = [t for t in TICKERS if t not in excl and t in close.columns]
    c = close[cols]
    fwd = (c.shift(-1) / c - 1.0).fillna(0.0)
    df, tg = run_v3_live_logic(c, fwd, cost=COST, short_cost_bps=3.0,
                               start=pd.Timestamp('2020-01-01'))
    r = slice_s(df['net'], '2020-01-01', '2021-01-01')
    r2 = slice_s(df['net'], '2020-01-01', TEST_END)
    r3 = slice_s(df['net'], '2024-01-01', TEST_END)
    print('exclude:', excl or 'none', '| assets:', len(cols))
    for label, rr in [('2020', r), ('2020-26', r2), ('2024-26', r3)]:
        m = metrics(rr)
        print(f'  {label:<8} {m["ann"]*100:7.1f}%  Sharpe {m["sharpe"]:5.2f}  '
              f'MDD {m["mdd"]*100:5.1f}%')
