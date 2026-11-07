"""
Audit 4: survivorship bias and point-in-time universe.

The 15-asset universe is chosen today. A fair historical test must include the
assets that were liquid back then, including ones that later collapsed.

Tests:
  * baseline: fixed 15 assets
  * broad fixed: all 34 assets with full history (includes SRM, WAVES, etc.)
  * point-in-time top-N by 60-day dollar volume (N = 10, 15, 20)
  * liquidity threshold variants
  * short-financing and cost stress on the point-in-time book

Run:  python audit_v3_survivorship.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_trend import load_data, metrics, COST
from research_v3_improved import (TRAIN_START, TRAIN_END, TEST_START, TEST_END,
                                  slice_s)
from audit_v3_backtest_fix import run_fixed
from neural_flow.alpha.v3_core import (AdaptiveWeightedV3, ProRiskManagerV3,
                                       trend_score, btc_regime_series)
from backtest_v3_improved import (TICKERS, LOOKBACK, REGIME_LOOKBACK,
                                  TARGET_VOL, BASE_LEVERAGE, COV_WINDOW,
                                  LAMBDA_BASE, TAU_BASE, MAX_ASSET_WEIGHT,
                                  NO_TRADE_BAND)


def run_pit(close, volume, top_n=15, min_dv=1e6, cost=COST,
            band=NO_TRADE_BAND, short_cost_bps=0.0, hist_min=61):
    tickers = close.columns.tolist()
    n = len(tickers)
    score, conf = trend_score(close, LOOKBACK)
    regime = btc_regime_series(close, REGIME_LOOKBACK)
    rets = close.pct_change()
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)
    dv60 = (close * volume).rolling(60, min_periods=20).median()
    hist = (close > 0).cumsum()
    risk = ProRiskManagerV3(target_vol=TARGET_VOL,
                            base_leverage=BASE_LEVERAGE,
                            cov_window=COV_WINDOW, ann=365)

    current = np.zeros(n)
    prev_raw = np.zeros(n)
    rows, nsel = [], []
    for d in close.index[:-1]:
        elig = ((hist.loc[d] >= hist_min) & (close.loc[d] > 0)
                & (dv60.loc[d] > min_dv))
        cand = dv60.loc[d][elig].sort_values(ascending=False)
        sel = cand.index[:top_n].tolist()
        nsel.append(len(sel))

        if not sel:
            turnover = float(np.abs(current).sum())
            rows.append((-cost * turnover, 0.0, turnover, 0.0, 0.0))
            current = np.zeros(n)
            prev_raw = np.zeros(n)
            continue

        sel_idx = np.array([tickers.index(t) for t in sel])
        sig = score.loc[d].reindex(sel).fillna(0.0).values
        cf = conf.loc[d].reindex(sel).fillna(0.0).values
        rsel = rets.loc[:d][sel].fillna(0.0)
        mkt_vol = rsel.tail(COV_WINDOW).std().mean() * np.sqrt(365)

        opt = AdaptiveWeightedV3(lambda_base=LAMBDA_BASE, tau_base=TAU_BASE,
                                 critical_threshold=3.0,
                                 prev_weights=prev_raw[sel_idx])
        raw_sel = opt.optimize(sig, cf, market_vol=mkt_vol)
        raw = np.zeros(n)
        raw[sel_idx] = raw_sel
        lev = risk.get_portfolio_leverage(raw_sel, rsel)

        desired = np.zeros(n)
        desired[sel_idx] = np.clip(raw_sel * lev, -MAX_ASSET_WEIGHT,
                                   MAX_ASSET_WEIGHT)
        if regime.loc[d] == 0:
            desired[:] = 0.0

        if band > 0:
            target = np.where(np.abs(desired - current) >= band,
                              desired, current)
        else:
            target = desired

        # Forced exits: dropped assets and regime-off periods always go to cash.
        keep = np.zeros(n, dtype=bool)
        keep[sel_idx] = True
        target = np.where(keep, target, 0.0)
        if regime.loc[d] == 0:
            target[:] = 0.0

        turnover = float(np.abs(target - current).sum())
        r = fwd.loc[d].values
        gross = float(np.sum(target * r))
        short_gross = float(np.abs(np.minimum(target, 0.0)).sum())
        short_cost = short_cost_bps / 1e4 * short_gross
        rows.append((gross - cost * turnover - short_cost, gross,
                     turnover, lev, short_gross))

        port_ret = gross
        denom = 1.0 + port_ret
        current = (target * (1.0 + r) / denom if denom > 1e-9
                   else np.zeros(n))
        prev_raw = raw

    df = pd.DataFrame(rows, index=close.index[:-1],
                      columns=['net', 'gross', 'turn', 'lev', 'short'])
    return df, nsel


def show(name, df, nsel=None):
    line = f"{name:<40}"
    for wl, s, e in [('train', TRAIN_START, TRAIN_END),
                     ('test', TEST_START, TEST_END),
                     ('full', TRAIN_START, TEST_END)]:
        r = slice_s(df['net'], s, e)
        m = metrics(r)
        line += f" {wl[:2]}={m['ann']*100:5.1f}%/{m['sharpe']:4.2f}/{m['mdd']*100:4.1f}%"
    avg_n = f"  n={np.mean(nsel):4.1f}" if nsel is not None else ''
    print(line + f"  turn={df['turn'].mean()*100:4.1f}%{avg_n}")


def main():
    close_15, high, low, vol_15 = load_data('6y')
    close_15 = close_15[[t for t in TICKERS if t in close_15.columns]]
    fwd_15 = close_15.shift(-1) / close_15 - 1.0

    broad_path = (Path('broad_data_delisted.pkl')
                  if Path('broad_data_delisted.pkl').exists()
                  else Path('broad_data.pkl'))
    close_b, vol_b = pd.read_pickle(broad_path)
    # Full-history assets only for the fixed broad test.
    full = [c for c in close_b.columns
            if close_b[c].first_valid_index() <= pd.Timestamp('2020-11-07')]
    close_full = close_b[full]
    fwd_full = close_full.shift(-1) / close_full - 1.0

    print('=' * 140)
    print('SURVIVORSHIP / POINT-IN-TIME UNIVERSE AUDIT')
    print('=' * 140)
    print(f"{'variant':<40} {'train CAGR/Sharpe/MDD':>28}"
          f" {'test CAGR/Sharpe/MDD':>28} {'full CAGR/Sharpe/MDD':>28}")
    print('-' * 140)

    base, _ = run_fixed(close_15, fwd_15, band=NO_TRADE_BAND)
    show('fixed 15 (current live universe)', base)

    broad, _ = run_fixed(close_full, fwd_full, band=NO_TRADE_BAND)
    show(f'fixed broad ({len(full)} full-history)', broad)

    for n in (10, 15, 20):
        df, ns = run_pit(close_b, vol_b, top_n=n, min_dv=1e6)
        show(f'PIT top {n} by 60d $volume', df, ns)

    for mdv in (1e7, 5e7):
        df, ns = run_pit(close_b, vol_b, top_n=15, min_dv=mdv)
        show(f'PIT top 15, min $vol {mdv/1e6:.0f}m', df, ns)

    df, ns = run_pit(close_b, vol_b, top_n=15, min_dv=1e6, short_cost_bps=3)
    show('PIT top 15 + 3bp/d short', df, ns)

    df, ns = run_pit(close_b, vol_b, top_n=15, min_dv=1e6, cost=COST * 2)
    show('PIT top 15 + 2x costs', df, ns)

    df, ns = run_pit(close_b, vol_b, top_n=15, min_dv=1e6, cost=COST * 5)
    show('PIT top 15 + 5x costs', df, ns)

    # Which names were selected most often?
    _, ns = run_pit(close_b, vol_b, top_n=15, min_dv=1e6)
    print(f'\nAverage selected count: {np.mean(ns):.1f} | '
          f'min {np.min(ns)} | max {np.max(ns)}')


if __name__ == '__main__':
    main()
