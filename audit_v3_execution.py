"""
Audit 3: execution timing.

The live runner runs at 01:00 UTC, one hour after the daily candle close.
The backtest currently assumes execution at the close. This script tests:

  1. close-to-close (current backtest)
  2. daily open-to-open (execution at the next daily open)
  3. hourly 01:00-to-01:00 execution (the actual live schedule)

All use the same signal, optimizer, regime, band, and costs.

Run:  python audit_v3_execution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_trend import load_data, metrics, COST
from research_v3_improved import TRAIN_START, TEST_END, slice_s
from audit_v3_backtest_fix import run_fixed
from neural_flow.alpha.v3_core import (AdaptiveWeightedV3, ProRiskManagerV3,
                                       trend_score, btc_regime_series)
from backtest_v3_improved import (TICKERS, LOOKBACK, REGIME_LOOKBACK,
                                  TARGET_VOL, BASE_LEVERAGE, COV_WINDOW,
                                  LAMBDA_BASE, TAU_BASE, MAX_ASSET_WEIGHT,
                                  NO_TRADE_BAND)


def run_execution(close, exec_open, signal_dates, price_offset_days,
                  cost=COST, band=NO_TRADE_BAND, short_cost_bps=0.0):
    tickers = close.columns.tolist()
    score, conf = trend_score(close, LOOKBACK)
    regime = btc_regime_series(close, REGIME_LOOKBACK)
    rets = close.pct_change()
    tz = exec_open.index.tz

    opt = AdaptiveWeightedV3(lambda_base=LAMBDA_BASE, tau_base=TAU_BASE,
                             critical_threshold=3.0)
    risk = ProRiskManagerV3(target_vol=TARGET_VOL,
                            base_leverage=BASE_LEVERAGE,
                            cov_window=COV_WINDOW, ann=365)

    rows, targets = [], []
    current = np.zeros(len(tickers))
    for d in signal_dates:
        sig = score.loc[d].reindex(tickers).fillna(0.0).values
        cf = conf.loc[d].reindex(tickers).fillna(0.0).values
        mkt_vol = rets.loc[:d].tail(COV_WINDOW).std().mean() * np.sqrt(365)
        raw = opt.optimize(sig, cf, market_vol=mkt_vol)
        lev = risk.get_portfolio_leverage(raw, rets.loc[:d])
        desired = np.clip(raw * lev, -MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT)
        if regime.loc[d] == 0:
            desired = np.zeros_like(desired)

        if band > 0:
            target = np.where(np.abs(desired - current) >= band, desired, current)
        else:
            target = desired
        if regime.loc[d] == 0:
            target = np.zeros_like(target)

        t0 = pd.Timestamp(d, tz=tz) + pd.Timedelta(days=price_offset_days)
        t1 = t0 + pd.Timedelta(days=1)
        if t0 not in exec_open.index or t1 not in exec_open.index:
            continue
        p0 = exec_open.loc[t0].values.astype(float)
        p1 = exec_open.loc[t1].values.astype(float)
        r = np.where(p0 > 0, p1 / p0 - 1.0, 0.0)
        r = np.nan_to_num(r)

        turnover = float(np.abs(target - current).sum())
        gross = float(np.sum(target * r))
        short_gross = float(np.abs(np.minimum(target, 0.0)).sum())
        short_cost = short_cost_bps / 1e4 * short_gross
        rows.append((gross - cost * turnover - short_cost, gross,
                     turnover, lev, short_gross))
        targets.append((d, target.copy(), lev))

        port_ret = gross
        denom = 1.0 + port_ret
        current = (target * (1.0 + r) / denom if denom > 1e-9
                   else np.zeros_like(target))

    idx = [x[0] for x in targets]
    return pd.DataFrame(rows, index=idx,
                        columns=['net', 'gross', 'turn', 'lev', 'short']), targets


def show(name, df, start, end):
    r = slice_s(df['net'], start, end)
    m = metrics(r)
    std = r.mean() / (r.std() + 1e-12) * np.sqrt(365)
    print(f"{name:<40} CAGR={m['ann']*100:6.1f}%  Sharpe={m['sharpe']:5.2f}  "
          f"stdSharpe={std:5.2f}  vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%  "
          f"turn={df['turn'].mean()*100:4.1f}%")


def main():
    close_all, high, low, volume = load_data('6y')
    close = close_all[[t for t in TICKERS if t in close_all.columns]]
    fwd = close.shift(-1) / close - 1.0

    print('=' * 110)
    print('EXECUTION TIMING AUDIT')
    print('=' * 110)

    fixed, _ = run_fixed(close, fwd, band=NO_TRADE_BAND)
    print('\nFull 6y (2020-11 -> 2026-11):')
    show('close-to-close (current)', fixed, TRAIN_START, TEST_END)

    op = pd.read_pickle('open_data.pkl')
    op = op[[t for t in TICKERS if t in op.columns]].ffill()
    op = op.reindex(close.index).ffill()
    dates = [d for d in close.index
             if d >= pd.Timestamp('2020-11-09')
             and d <= pd.Timestamp('2026-11-05')]
    op_to_op, _ = run_execution(close, op, dates, price_offset_days=1,
                                band=NO_TRADE_BAND)
    print('\nFull 6y, daily open-to-open (signal close d -> open d+1 -> open d+2):')
    show('open-to-open', op_to_op, TRAIN_START, TEST_END)

    h = pd.read_pickle('hourly_data.pkl')
    h_open = h['open'][[t for t in TICKERS if t in h['open'].columns]].ffill()
    h_shift = h_open.copy()
    h_shift.index = h_shift.index - pd.Timedelta(hours=1)
    hourly_dates = [d for d in close.index
                    if pd.Timestamp(d, tz='UTC') + pd.Timedelta(days=1)
                    in h_shift.index
                    and pd.Timestamp(d, tz='UTC') + pd.Timedelta(days=2)
                    in h_shift.index]
    print(f'\nHourly period: {hourly_dates[0].date()} -> {hourly_dates[-1].date()} '
          f'({len(hourly_dates)} signal dates)')

    hourly, _ = run_execution(close, h_shift, hourly_dates,
                              price_offset_days=1, band=NO_TRADE_BAND)
    print('\nHourly 01:00 UTC execution:')
    show('01:00 execution (live schedule)', hourly,
         hourly_dates[0], hourly_dates[-1] + pd.Timedelta(days=1))

    print('\nSame period, close-to-close:')
    show('close-to-close', fixed, hourly_dates[0],
         hourly_dates[-1] + pd.Timedelta(days=1))

    hourly_sc, _ = run_execution(close, h_shift, hourly_dates,
                                 price_offset_days=1, band=NO_TRADE_BAND,
                                 short_cost_bps=3)
    print('\nHourly 01:00 + 3bp/day short cost:')
    show('01:00 + 3bp/d short', hourly_sc, hourly_dates[0],
         hourly_dates[-1] + pd.Timedelta(days=1))


if __name__ == '__main__':
    main()
