"""
Audit 5: what drives the point-in-time (PIT) result?

For the PIT top-15 universe, record:
  * which assets are selected, by year
  * average absolute weight in the test period
  * per-asset contribution to returns (target weight x next-day return)
  * the same for the fixed 15-asset universe

Also test quality filters on the PIT universe:
  * only assets above their 200-day SMA
  * only assets not down more than 50% over 200 days

Run:  python audit_v3_pit_detail.py
"""
from __future__ import annotations

from collections import Counter
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
            band=NO_TRADE_BAND, short_cost_bps=0.0, hist_min=61,
            above_sma200=False, min_ret200=None, record=False):
    tickers = close.columns.tolist()
    n = len(tickers)
    score, conf = trend_score(close, LOOKBACK)
    regime = btc_regime_series(close, REGIME_LOOKBACK)
    rets = close.pct_change()
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)
    dv60 = (close * volume).rolling(60, min_periods=20).median()
    hist = close.notna().cumsum()
    sma200 = close.rolling(200, min_periods=200).mean()
    ret200 = close.pct_change(200)
    risk = ProRiskManagerV3(target_vol=TARGET_VOL,
                            base_leverage=BASE_LEVERAGE,
                            cov_window=COV_WINDOW, ann=365)

    current = np.zeros(n)
    prev_raw = np.zeros(n)
    rows, nsel, records = [], [], []
    for d in close.index[:-1]:
        elig = ((hist.loc[d] >= hist_min) & close.loc[d].notna()
                & (dv60.loc[d] > min_dv))
        if above_sma200:
            elig &= close.loc[d] > sma200.loc[d]
        if min_ret200 is not None:
            elig &= ret200.loc[d] > min_ret200

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

        if record:
            records.append((d, sel, target.copy(), r.copy()))

        port_ret = gross
        denom = 1.0 + port_ret
        current = (target * (1.0 + r) / denom if denom > 1e-9
                   else np.zeros(n))
        prev_raw = raw

    df = pd.DataFrame(rows, index=close.index[:-1],
                      columns=['net', 'gross', 'turn', 'lev', 'short'])
    return df, nsel, records


def show(name, df, nsel=None):
    line = f"{name:<44}"
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

    print('=' * 150)
    print('PIT DETAIL / QUALITY FILTERS')
    print('=' * 150)
    print(f"{'variant':<44} {'train CAGR/Sharpe/MDD':>28}"
          f" {'test CAGR/Sharpe/MDD':>28} {'full CAGR/Sharpe/MDD':>28}")
    print('-' * 150)

    base15, _ = run_fixed(close_15, fwd_15, band=NO_TRADE_BAND)
    show('fixed 15 (survivorship-biased)', base15)

    pit, ns, recs = run_pit(close_b, vol_b, top_n=15, min_dv=1e6, record=True)
    show('PIT top 15 by 60d $volume', pit, ns)

    pit_q, ns_q, _ = run_pit(close_b, vol_b, top_n=15, min_dv=1e6,
                             above_sma200=True)
    show('PIT top 15, only above 200d SMA', pit_q, ns_q)

    pit_r, ns_r, _ = run_pit(close_b, vol_b, top_n=15, min_dv=1e6,
                             min_ret200=-0.50)
    show('PIT top 15, 200d return > -50%', pit_r, ns_r)

    # Selection frequency and contribution in the test period.
    test_start, test_end = TEST_START, TEST_END
    test_recs = [x for x in recs if test_start <= x[0] <= test_end]
    freq = Counter()
    for d, sel, w, r in test_recs:
        for t in sel:
            freq[t] += 1
    contrib = pd.DataFrame(0.0, index=close_b.columns,
                           columns=['sum_contrib', 'sum_abs_weight', 'days'])
    for d, sel, w, r in test_recs:
        for i, t in enumerate(close_b.columns):
            if abs(w[i]) > 1e-12:
                contrib.loc[t, 'sum_contrib'] += w[i] * r[i]
                contrib.loc[t, 'sum_abs_weight'] += abs(w[i])
                contrib.loc[t, 'days'] += 1
    print('\nTop 15 by selection frequency (test period):')
    for t, c in freq.most_common(15):
        print(f"  {t:<12} {c:>4} / {len(test_recs)} days  "
              f"avg|w|={contrib.loc[t,'sum_abs_weight']/max(contrib.loc[t,'days'],1):.3f}  "
              f"contrib={contrib.loc[t,'sum_contrib']*100:+.1f}%")

    print('\nTop 10 contributors (test period):')
    cc = contrib.sort_values('sum_contrib', ascending=False)
    for t in cc.index[:10]:
        print(f"  {t:<12} {cc.loc[t,'sum_contrib']*100:+6.1f}%  "
              f"avg|w|={contrib.loc[t,'sum_abs_weight']/max(contrib.loc[t,'days'],1):.3f}")

    print('\nBottom 10 contributors (test period):')
    for t in cc.index[-10:]:
        print(f"  {t:<12} {cc.loc[t,'sum_contrib']*100:+6.1f}%  "
              f"avg|w|={contrib.loc[t,'sum_abs_weight']/max(contrib.loc[t,'days'],1):.3f}")

    # Fixed-15 contributions for comparison.
    _, tgt = run_fixed(close_15, fwd_15, band=NO_TRADE_BAND)
    fc = pd.Series(0.0, index=close_15.columns)
    for d, w, lev in tgt:
        if test_start <= d <= test_end:
            fc += pd.Series(w * fwd_15.loc[d].values, index=close_15.columns)
    print('\nFixed-15 contributions (test period):')
    for t, v in fc.sort_values(ascending=False).items():
        print(f"  {t:<12} {v*100:+6.1f}%")


if __name__ == '__main__':
    main()
