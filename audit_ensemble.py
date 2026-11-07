"""Does a parameter ensemble improve out-of-sample robustness?

The single chosen config is lambda=0.05, lookback=50.  The grid shows a
lookback-50 plateau but unstable train/test ranking.  This script averages the
daily net returns of neighbouring plateau configs (equal capital per sleeve)
and compares the single vs ensemble test/full performance.

This is an exploratory robustness check, not a new tuned production config.
The 2024-2026 test is already observed, so the ensemble test is another trial.

Run:  python audit_ensemble.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_v3_improved as bt
import audit_v3_survivorship as avs
from backtest_trend import metrics, slice_s, ANN
from audit_v3_stats import block_bootstrap

TRAIN = (bt.TRAIN_START, bt.TRAIN_END)
TEST = (bt.TEST_START, bt.TEST_END)
FULL = (bt.TRAIN_START, bt.TEST_END)


def ann_stats(r):
    m = metrics(r)
    std = float(r.mean() / (r.std() + 1e-12) * np.sqrt(ANN))
    t = float(r.mean() / (r.std() + 1e-12) * np.sqrt(len(r)))
    ci = block_bootstrap(r, block=20, n_boot=2000)
    return dict(cagr=float(m['ann']), vol=float(m['vol']),
                sharpe=float(m['sharpe']), std_sharpe=std, t=t,
                mdd=float(m['mdd']), total=float(m['total']),
                bootstrap_cagr=[float(ci[0, 0]), float(ci[2, 0])],
                bootstrap_sharpe=[float(ci[0, 1]), float(ci[2, 1])])


def window(r, se):
    return slice_s(r, se[0], se[1])


def run_fixed(close, fwd, lookback, lam):
    saved = {k: getattr(bt, k) for k in ('LOOKBACK', 'LAMBDA_BASE')}
    try:
        bt.LOOKBACK, bt.LAMBDA_BASE = lookback, lam
        df, _ = bt.run_v3_live_logic(close, fwd, short_cost_bps=bt.SHORT_COST_BPS,
                                     start=bt.TRAIN_START)
        return df
    finally:
        for k, v in saved.items():
            setattr(bt, k, v)


def main():
    close, high, low, volume = bt.load_7y()
    tickers = [t for t in bt.TICKERS if t in close.columns]
    close = close[tickers]
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)

    # ---- fixed universe: plateau components ----
    fixed = {}
    for lb in (30, 50, 70):
        fixed[(lb, 0.05)] = run_fixed(close, fwd, lb, 0.05)['net']
    for lam in (0.04, 0.06):
        fixed[(50, lam)] = run_fixed(close, fwd, 50, lam)['net']
    for lam in (0.04, 0.06):
        fixed[(30, lam)] = run_fixed(close, fwd, 30, lam)['net']

    def avg(keys):
        return pd.concat([fixed[k] for k in keys], axis=1).mean(axis=1)

    sleeves = {
        'single 50/0.05': fixed[(50, 0.05)],
        'lookback 30/50': avg([(30, 0.05), (50, 0.05)]),
        'lookback 30/50/70': avg([(30, 0.05), (50, 0.05), (70, 0.05)]),
        'lambda 0.04/0.05/0.06 (50)': avg([(50, 0.04), (50, 0.05), (50, 0.06)]),
        'both 30/50 x 0.04/0.05/0.06': avg([(30, 0.04), (30, 0.05), (30, 0.06),
                                            (50, 0.04), (50, 0.05), (50, 0.06)]),
    }
    fixed_out = {}
    print('=' * 118)
    print('FIXED 15 — single vs parameter ensembles (equal capital per sleeve)')
    print('=' * 118)
    print(f"{'variant':<38}{'train':>16}{'test':>16}{'full':>16}")
    print(f"{'':<38}{'CAGR/Sh/MDD':>16}{'CAGR/Sh/MDD':>16}{'CAGR/Sh/MDD':>16}")
    print('-' * 118)
    for name, r in sleeves.items():
        tr, te, fu = (ann_stats(window(r, w)) for w in (TRAIN, TEST, FULL))
        fixed_out[name] = dict(train=tr, test=te, full=fu)
        print(f"{name:<38}"
              f"{tr['cagr']*100:>7.1f}/{tr['sharpe']:>4.2f}/{tr['mdd']*100:>4.1f}"
              f"{te['cagr']*100:>7.1f}/{te['sharpe']:>4.2f}/{te['mdd']*100:>4.1f}"
              f"{fu['cagr']*100:>7.1f}/{fu['sharpe']:>4.2f}/{fu['mdd']*100:>4.1f}")

    print('\nTest details (single vs best ensemble):')
    for name in ('single 50/0.05', 'lookback 30/50',
                 'lambda 0.04/0.05/0.06 (50)', 'both 30/50 x 0.04/0.05/0.06'):
        te = fixed_out[name]['test']
        print(f"  {name:<38} stdSh={te['std_sharpe']:4.2f} t={te['t']:4.2f} "
              f"CI(CAGR)={te['bootstrap_cagr'][0]*100:5.1f}..{te['bootstrap_cagr'][1]*100:5.1f} "
              f"CI(Sh)={te['bootstrap_sharpe'][0]:4.2f}..{te['bootstrap_sharpe'][1]:4.2f}")

    # component correlations on the test window
    comp = pd.DataFrame({f'{k[0]}/{k[1]}': v for k, v in fixed.items()})
    print('\nTest-window correlation of component daily net returns:')
    print(window(comp, TEST).corr().round(2).to_string())

    # ---- PIT universe: lookback 30/50 ensemble ----
    broad_path = (Path('broad_data_delisted.pkl')
                  if Path('broad_data_delisted.pkl').exists()
                  else Path('broad_data.pkl'))
    close_b, vol_b = pd.read_pickle(broad_path)
    saved = {k: getattr(avs, k) for k in ('LOOKBACK', 'LAMBDA_BASE')}
    pit_net = {}
    try:
        for lb in (30, 50):
            avs.LOOKBACK, avs.LAMBDA_BASE = lb, 0.05
            df, _ = avs.run_pit(close_b, vol_b, top_n=15, min_dv=1e6,
                                short_cost_bps=bt.SHORT_COST_BPS)
            pit_net[lb] = df['net']
        pit_sleeves = {
            'single PIT 50/0.05': pit_net[50],
            'PIT lookback 30/50': (pit_net[30] + pit_net[50]) / 2.0,
        }
    finally:
        for k, v in saved.items():
            setattr(avs, k, v)

    print('\n' + '=' * 118)
    print('PIT TOP 15 — single vs lookback ensemble')
    print('=' * 118)
    print(f"{'variant':<38}{'test CAGR/Sh/MDD':>24}{'full CAGR/Sh/MDD':>24}")
    print('-' * 118)
    pit_out = {}
    for name, r in pit_sleeves.items():
        te, fu = ann_stats(window(r, TEST)), ann_stats(window(r, FULL))
        pit_out[name] = dict(test=te, full=fu)
        print(f"{name:<38}"
              f"{te['cagr']*100:>8.1f}/{te['sharpe']:>4.2f}/{te['mdd']*100:>4.1f}"
              f"{fu['cagr']*100:>8.1f}/{fu['sharpe']:>4.2f}/{fu['mdd']*100:>4.1f}")
    for name in pit_out:
        te = pit_out[name]['test']
        print(f"  {name:<38} stdSh={te['std_sharpe']:4.2f} t={te['t']:4.2f} "
              f"CI(Sh)={te['bootstrap_sharpe'][0]:4.2f}..{te['bootstrap_sharpe'][1]:4.2f}")

    out = dict(fixed=fixed_out, pit=pit_out, broad=str(broad_path))
    with open('ensemble_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('\nsaved ensemble_results.json')


if __name__ == '__main__':
    main()
