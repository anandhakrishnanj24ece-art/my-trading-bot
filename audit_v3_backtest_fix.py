"""
Audit 2: is the no-trade band modelled correctly?

The current backtest uses `prev_target` both as the trading reference and as the
weights that earn the next return. When the band says "hold", the real portfolio
has drifted, so:

  * P&L should use the drifted weights,
  * turnover should be measured from the drifted weights,
  * the band should compare the new target with the drifted weights.

This script implements the drift-aware version and compares it with the current
backtest, then adds short-financing cost and a one-bar execution delay.

Run:  python audit_v3_backtest_fix.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_trend import load_data, metrics, COST
from research_v3_improved import (TRAIN_START, TRAIN_END, TEST_START, TEST_END,
                                  slice_s)
from neural_flow.alpha.v3_core import (AdaptiveWeightedV3, ProRiskManagerV3,
                                       trend_score, btc_regime_series)
from backtest_v3_improved import (TICKERS, LOOKBACK, REGIME_LOOKBACK,
                                  TARGET_VOL, BASE_LEVERAGE, COV_WINDOW,
                                  LAMBDA_BASE, TAU_BASE, MAX_ASSET_WEIGHT,
                                  NO_TRADE_BAND, run_v3_live_logic)


def run_fixed(close, fwd, cost=COST, band=NO_TRADE_BAND, short_cost_bps=0.0,
              lag=0, lambda_base=LAMBDA_BASE, target_vol=TARGET_VOL,
              base_leverage=BASE_LEVERAGE, cov_window=COV_WINDOW,
              max_asset=MAX_ASSET_WEIGHT, regime_off=0.0):
    tickers = close.columns.tolist()
    score, conf = trend_score(close, LOOKBACK)
    regime = btc_regime_series(close, REGIME_LOOKBACK)
    rets = close.pct_change()

    opt = AdaptiveWeightedV3(lambda_base=lambda_base, tau_base=TAU_BASE,
                             critical_threshold=3.0)
    risk = ProRiskManagerV3(target_vol=target_vol,
                            base_leverage=base_leverage,
                            cov_window=cov_window, ann=365)

    current = np.zeros(len(tickers))
    pending = None
    rows, targets = [], []
    for d in close.index[:-1]:
        sig = score.loc[d].reindex(tickers).fillna(0.0).values
        cf = conf.loc[d].reindex(tickers).fillna(0.0).values
        mkt_vol = rets.loc[:d].tail(cov_window).std().mean() * np.sqrt(365)
        raw = opt.optimize(sig, cf, market_vol=mkt_vol)
        lev = risk.get_portfolio_leverage(raw, rets.loc[:d])
        desired = np.clip(raw * lev, -max_asset, max_asset)
        if regime.loc[d] == 0:
            desired = desired * regime_off

        if lag == 0:
            base_target = desired
        else:
            base_target = pending if pending is not None else current
            pending = desired

        if band > 0:
            target = np.where(np.abs(base_target - current) >= band,
                              base_target, current)
        else:
            target = base_target

        # A regime exit must always go fully to cash, regardless of the band.
        if regime.loc[d] == 0:
            target = np.zeros_like(target)

        turnover = float(np.abs(target - current).sum())
        r = fwd.loc[d].values
        gross = float(np.sum(target * r))
        short_gross = float(np.abs(np.minimum(target, 0.0)).sum())
        short_cost = short_cost_bps / 1e4 * short_gross
        rows.append((gross - cost * turnover - short_cost, gross,
                     turnover, lev, short_gross))

        targets.append((d, target.copy(), lev))
        port_ret = float(np.sum(target * r))
        denom = 1.0 + port_ret
        current = (target * (1.0 + r) / denom if denom > 1e-9
                   else np.zeros_like(target))

    return pd.DataFrame(rows, index=close.index[:-1],
                        columns=['net', 'gross', 'turn', 'lev', 'short']), targets


def summarise(name, df):
    line = f"{name:<42}"
    for wl, s, e in [('train', TRAIN_START, TRAIN_END),
                     ('test', TEST_START, TEST_END),
                     ('full', TRAIN_START, TEST_END)]:
        m = metrics(slice_s(df['net'], s, e))
        line += f" {wl[:2]}={m['ann']*100:5.1f}%/{m['sharpe']:4.2f}/{m['mdd']*100:4.1f}%"
    avg_short = df['short'].mean() if 'short' in df else float('nan')
    print(line + f"  turn={df['turn'].mean()*100:4.1f}%"
          f"  avgLev={df['lev'].mean():.2f}  avgShort={avg_short:.2f}")


def main():
    close_all, high, low, volume = load_data('6y')
    close = close_all[[t for t in TICKERS if t in close_all.columns]]
    fwd = close.shift(-1) / close - 1.0

    print('=' * 130)
    print('TURNOVER / BAND / COST AUDIT')
    print('=' * 130)
    print(f"{'variant':<42} {'train CAGR/Sharpe/MDD':>28}"
          f" {'test CAGR/Sharpe/MDD':>28} {'full CAGR/Sharpe/MDD':>28}")
    print('-' * 130)

    old_df, _ = run_v3_live_logic(close, fwd)
    summarise('OLD backtest (target weights every day)', old_df)

    fixed0, _ = run_fixed(close, fwd, band=0.0)
    summarise('FIXED, no band (daily rebalance)', fixed0)

    fixed, _ = run_fixed(close, fwd, band=0.02)
    summarise('FIXED, band 2%', fixed)

    for bps in (1, 3, 5, 10):
        d, _ = run_fixed(close, fwd, band=0.02, short_cost_bps=bps)
        summarise(f'FIXED, band 2%, short cost {bps}bp/d', d)

    lag1, _ = run_fixed(close, fwd, band=0.02, lag=1)
    summarise('FIXED, band 2%, 1-bar execution lag', lag1)

    combo, _ = run_fixed(close, fwd, band=0.02, lag=1, short_cost_bps=3)
    summarise('FIXED, band 2%, lag 1, short 3bp/d', combo)

    print('\nNote: train/test/full columns are CAGR / Sharpe / MaxDD.')


if __name__ == '__main__':
    main()
