"""
Audit 6: statistical honesty.

  * standard Sharpe (mean/std) vs the reported CAGR/vol Sharpe
  * t-stat and block-bootstrap confidence intervals
  * year-by-year stability for the fixed 15 and the PIT top 15
  * how much the multiple-testing / universe selection costs

Run:  python audit_v3_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_trend import metrics, yearly, slice_s
from research_v3_improved import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from backtest_v3_improved import (load_7y, run_v3_live_logic, TICKERS,
                                  SHORT_COST_BPS, TRAIN_START as BT_TRAIN)
from audit_v3_survivorship import run_pit


def block_bootstrap(rets, block=20, n_boot=2000, seed=42):
    r = np.asarray(rets, dtype=float)
    n = len(r)
    if n <= block:
        return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty((n_boot, 2))
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        m = sample.mean()
        s = sample.std()
        cagr = (1.0 + sample).prod() ** (365.0 / n) - 1.0
        sharpe = m / (s + 1e-12) * np.sqrt(365.0)
        out[i] = (cagr, sharpe)
    return np.percentile(out, [2.5, 50, 97.5], axis=0)


def report(name, net, start, end):
    r = slice_s(net, start, end)
    if len(r) == 0:
        return
    m = metrics(r)
    std_sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(365)
    tstat = r.mean() / (r.std() + 1e-12) * np.sqrt(len(r))
    ci = block_bootstrap(r)
    print(f"\n{name}  ({r.index[0].date()} -> {r.index[-1].date()}, {len(r)} days)")
    print(f"  total={m['total']*100:6.1f}%  CAGR={m['ann']*100:6.1f}%  "
          f"vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%")
    print(f"  reported Sharpe(CAGR/vol)={m['sharpe']:5.2f}  "
          f"standard Sharpe={std_sharpe:5.2f}  t-stat={tstat:5.2f}")
    print(f"  block-bootstrap 95%: CAGR [{ci[0,0]*100:6.1f}%, {ci[2,0]*100:6.1f}%]  "
          f"Sharpe [{ci[0,1]:5.2f}, {ci[2,1]:5.2f}]")


def main():
    close, high, low, volume = load_7y()
    close = close[[t for t in TICKERS if t in close.columns]]
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)
    fixed, _ = run_v3_live_logic(close, fwd, short_cost_bps=SHORT_COST_BPS,
                                 start=BT_TRAIN)

    broad_path = (Path('broad_data_delisted.pkl')
                  if Path('broad_data_delisted.pkl').exists()
                  else Path('broad_data.pkl'))
    close_b, vol_b = pd.read_pickle(broad_path)
    pit, ns = run_pit(close_b, vol_b, top_n=15, min_dv=1e6,
                      short_cost_bps=SHORT_COST_BPS)

    print('=' * 100)
    print('STATISTICAL AUDIT (3 bp/day short financing included)')
    print('=' * 100)

    print('\n--- Fixed 15 (current live universe, survivorship-biased) ---')
    report('test', fixed['net'], TEST_START, TEST_END)
    report('full', fixed['net'], TRAIN_START, TEST_END)

    print('\n--- Point-in-time top 15 by 60d $volume (no survivorship) ---')
    report('test', pit['net'], TEST_START, TEST_END)
    report('full', pit['net'], TRAIN_START, TEST_END)

    print('\n--- Year-by-year net returns ---')
    fy = yearly(slice_s(fixed['net'], TRAIN_START, TEST_END)) * 100
    py = yearly(slice_s(pit['net'], TRAIN_START, TEST_END)) * 100
    table = pd.DataFrame({'fixed 15': fy, 'PIT top 15': py}).round(1)
    print(table.to_string())

    print('\n--- Selection / multiple-testing note ---')
    print('The test window was observed during tuning, so it is not a pristine')
    print('holdout.  Rough family-wise check: for a single strategy the test')
    print('t-stat is about 2.6-2.7; the expected maximum t of 100 independent')
    print('zero-edge trials is about 2.5.  The fixed-15 test edge is therefore')
    print('positive but not overwhelming after selection; the PIT test edge is')
    print('not statistically significant.  The full-period t-stat is ~4.1, but')
    print('it includes the in-sample/tuning period.')


if __name__ == '__main__':
    main()
