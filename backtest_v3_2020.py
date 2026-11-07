"""
Backtest from 2020 with the full cost stack.

Costs:
  * fees: 5.0 bp per unit turnover
  * slippage: 1.4 bp per unit turnover (70% of 2 bp)
  * short financing: 3.0 bp/day on the short book (current live assumption)
  * optional funding fee: 1.0 or 3.0 bp/day on gross exposure (both longs and shorts)

The 7-year cache starts 2019-11-07, so the 2020 calendar year is a cold start:
the 50-day trend is available from late December, and the 150-day BTC regime is
only fully warm from about April 2020.  The 2020-11 start is the standard one
and has a one-year warm-up.  The 2024-2026 test is the true holdout window.

Run:  python backtest_v3_2020.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_trend import metrics, beta_alpha, slice_s, yearly, ANN
from backtest_v3_improved import (load_7y, run_v3_live_logic, TICKERS, COST,
                                  TRAIN_START, TEST_START, TEST_END)

START_2020 = pd.Timestamp('2020-01-01')
END_2020 = pd.Timestamp('2021-01-01')


def run(start, short_bps, funding_bps):
    return run_v3_live_logic(close, fwd, cost=COST,
                             short_cost_bps=short_bps,
                             funding_bps=funding_bps, start=start)


def report(label, df, targets, start, end, short_bps, funding_bps):
    r = slice_s(df['net'], start, end)
    if len(r) == 0:
        return
    m = metrics(r)
    std = r.mean() / (r.std() + 1e-12) * np.sqrt(ANN)
    beta, alpha = beta_alpha(r, slice_s(fwd['BTC-USD'], start, end))
    gexp = np.mean([np.abs(t).sum() for d, t, _ in targets if start <= d < end])
    short = slice_s(df['short'], start, end).mean()
    turn = slice_s(df['turn'], start, end).mean()
    exec_drag = COST * turn * ANN
    short_drag = short_bps / 1e4 * short * ANN
    fund_drag = slice_s(df['fund'], start, end).mean() * ANN
    print(f'{label:<40}'
          f'{m["total"]*100:>10.1f}{m["ann"]*100:>8.1f}{m["vol"]*100:>7.1f}'
          f'{m["sharpe"]:>7.2f}{std:>8.2f}{m["mdd"]*100:>8.1f}'
          f'{beta:>6.2f}{alpha*100:>8.1f}{gexp:>7.2f}{turn*100:>7.1f}'
          f'{exec_drag*100:>8.2f}{short_drag*100:>8.2f}{fund_drag*100:>8.2f}')


if __name__ == '__main__':
    close, high, low, volume = load_7y()
    close = close[[t for t in TICKERS if t in close.columns]]
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)

    print('universe:', len(close.columns), 'assets |',
          close.index[0].date(), '->', close.index[-1].date())
    print('costs: 5.0 bp fee + 1.4 bp slippage per unit turnover; '
          'short 3 bp/day or gross funding 1/3 bp/day\n')

    header = (f'{"window / cost stack":<40}{"total%":>10}{"CAGR%":>8}{"vol%":>7}'
              f'{"Shrp":>7}{"StdSR":>8}{"MDD%":>8}{"beta":>6}{"alpha%":>8}'
              f'{"gross":>7}{"turn%":>7}{"exec%":>8}{"short%":>8}{"fund%":>8}')
    print(header)
    print('-' * len(header))

    # ---- 2020 calendar year (cold start; first trades after the regime warms) ----
    df20s, t20s = run(START_2020, short_bps=3.0, funding_bps=0.0)
    report('2020 calendar | fees+slip+short3', df20s, t20s, START_2020, END_2020,
           short_bps=3.0, funding_bps=0.0)
    df20g1, t20g1 = run(START_2020, short_bps=0.0, funding_bps=1.0)
    report('2020 calendar | fees+slip+gross1', df20g1, t20g1, START_2020, END_2020,
           short_bps=0.0, funding_bps=1.0)
    df20g3, t20g3 = run(START_2020, short_bps=0.0, funding_bps=3.0)
    report('2020 calendar | fees+slip+gross3', df20g3, t20g3, START_2020, END_2020,
           short_bps=0.0, funding_bps=3.0)

    # ---- from 2020 through today (cold start) ----
    print()
    report('2020-01..2026-11 | fees+slip+short3', df20s, t20s, START_2020, TEST_END,
           short_bps=3.0, funding_bps=0.0)
    report('2020-01..2026-11 | fees+slip+gross1', df20g1, t20g1, START_2020, TEST_END,
           short_bps=0.0, funding_bps=1.0)
    report('2020-01..2026-11 | fees+slip+gross3', df20g3, t20g3, START_2020, TEST_END,
           short_bps=0.0, funding_bps=3.0)

    # ---- standard 2020-11 start (one-year warm-up) ----
    print()
    dfstd, tstd = run(TRAIN_START, short_bps=3.0, funding_bps=0.0)
    report('2020-11..2026-11 | fees+slip+short3', dfstd, tstd, TRAIN_START, TEST_END,
           short_bps=3.0, funding_bps=0.0)
    report('2024-01..2026-11 test | fees+slip+short3', dfstd, tstd, TEST_START, TEST_END,
           short_bps=3.0, funding_bps=0.0)
    dfstd1, tstd1 = run(TRAIN_START, short_bps=0.0, funding_bps=1.0)
    report('2024-01..2026-11 test | fees+slip+gross1', dfstd1, tstd1, TEST_START, TEST_END,
           short_bps=0.0, funding_bps=1.0)
    dfstd3, tstd3 = run(TRAIN_START, short_bps=0.0, funding_bps=3.0)
    report('2024-01..2026-11 test | fees+slip+gross3', dfstd3, tstd3, TEST_START, TEST_END,
           short_bps=0.0, funding_bps=3.0)

    print('\nYear-by-year, from 2020-01 (cold start), short 3 bp/day:')
    y = yearly(slice_s(df20s['net'], START_2020, TEST_END)) * 100
    print(y.round(1).to_string())

    print('\nNote: 2020-01..2020-04 is partly cash because the 150-day BTC regime')
    print('is not warm until about April.  The 2020 calendar row is therefore a')
    print('cold-start check, not the same setup as the 2020-11 start.')
