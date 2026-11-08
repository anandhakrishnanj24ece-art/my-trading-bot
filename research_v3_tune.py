"""
Tuning sweep for the improved v3.

Builds on research_v3_improved.py but explores:
  * universe: 9 vs 15 assets
  * signal: trend / EMA / breakout / slope / momentum / cross-sectional blend
  * regime: BTC150 / BTC150 hysteresis / breadth / combo / none
  * risk: vol target, covariance window, leverage, per-asset cap
  * execution: rebalance frequency, no-trade band, inverse-vol sizing
  * bear-regime behaviour: cash vs partial/full flip

Everything is causal and net of costs.

Run:  python research_v3_tune.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, COST, ANN
from research_v3_improved import (V3Optimizer, ProRiskManagerV3,
                                  TRAIN_START, TRAIN_END, TEST_START, TEST_END,
                                  slice_s, yearly)

warnings.filterwarnings('ignore')

U9 = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
      'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
U15 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
       'DOT-USD', 'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD',
       'NEAR-USD', 'ALGO-USD', 'FIL-USD']

DEFAULTS = dict(lambda_base=0.02, tau_base=0.02, critical=3.0, tau_mode='vol',
                target_vol=0.15, base_leverage=2.0, cov_window=60, risk_ann=365,
                max_asset=1.0, regime_off=0.0, rebalance_days=1,
                no_trade_band=0.0, inverse_vol=False, cost=COST)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def _pack(score):
    score = score.clip(-5, 5)
    return score, score.abs().clip(0, 3)


def sig_trend(close, lb=50):
    vol = close.pct_change().rolling(20).std()
    sma = close.rolling(lb, min_periods=lb).mean()
    return _pack((close / sma - 1.0) / (vol + 1e-9))


def sig_ema(close, lb=50):
    vol = close.pct_change().rolling(20).std()
    ema = close.ewm(span=lb, adjust=False, min_periods=lb).mean()
    return _pack((close / ema - 1.0) / (vol + 1e-9))


def sig_breakout(close, lb=50):
    vol = close.pct_change().rolling(20).std()
    hi = close.rolling(lb, min_periods=lb).max()
    lo = close.rolling(lb, min_periods=lb).min()
    mid = (hi + lo) / 2.0
    return _pack((close / mid - 1.0) / (vol + 1e-9))


def sig_slope(close, lb=50):
    logp = np.log(close)
    x = np.arange(lb)
    xm = x.mean()
    xv = ((x - xm) ** 2).sum()
    slope = logp.rolling(lb).apply(
        lambda y: float(((x - xm) * (y - y.mean())).sum() / xv), raw=True)
    vol = close.pct_change().rolling(20).std()
    return _pack(slope / (vol + 1e-9))


def sig_mom(close, lb=50, skip=0):
    vol = close.pct_change().rolling(20).std()
    if skip:
        raw = close.shift(skip) / close.shift(lb) - 1.0
        n = lb - skip
    else:
        raw = close.pct_change(lb)
        n = lb
    return _pack(raw / (vol * np.sqrt(n) + 1e-9))


def sig_xs_blend(close, lb=50, w_xs=0.3):
    s, _ = sig_trend(close, lb)
    xs = close.pct_change(lb).rank(axis=1, pct=True) - 0.5
    xs = xs * 4.0
    return _pack((1.0 - w_xs) * s + w_xs * xs)


def sig_dual(close, fast=20, slow=50):
    vol = close.pct_change().rolling(20).std()
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    return _pack((f / s - 1.0) / (vol + 1e-9))


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def reg_none(close):
    return pd.Series(1.0, index=close.index)


def reg_btc(close, lb=150):
    btc = close['BTC-USD']
    sma = btc.rolling(lb, min_periods=1).mean()
    return (btc > sma).astype(float)


def reg_btc_hyst(close, lb=150, band=0.01):
    btc = close['BTC-USD']
    sma = btc.rolling(lb, min_periods=1).mean()
    state, out = 0.0, []
    for b, s in zip(btc.values, sma.values):
        if b > s * (1.0 + band):
            state = 1.0
        elif b < s * (1.0 - band):
            state = 0.0
        out.append(state)
    return pd.Series(out, index=close.index)


def reg_breadth(close, lb=50, threshold=0.5):
    above = (close > close.rolling(lb, min_periods=1).mean()).mean(axis=1)
    return (above > threshold).astype(float)


def reg_combo(close, lb=150, breadth_lb=50):
    a = reg_btc(close, lb).astype(bool)
    b = (close > close.rolling(breadth_lb, min_periods=1).mean()).mean(axis=1) > 0.5
    return (a & b).astype(float)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_config(close, fwd, sig_df, conf_df, regime, **overrides):
    p = dict(DEFAULTS)
    p.update(overrides)
    tickers = close.columns.tolist()
    rets = close.pct_change()
    opt = V3Optimizer(p['lambda_base'], p['tau_base'], p['critical'], p['tau_mode'])
    risk = ProRiskManagerV3(p['target_vol'], p['base_leverage'],
                            p['cov_window'], p['risk_ann'])

    prev_target = np.zeros(len(tickers))
    last_lev = 1.0
    rows = []
    for i, d in enumerate(close.index[:-1]):
        fr = fwd.loc[d].values

        if i % p['rebalance_days'] == 0:
            sig = sig_df.loc[d].reindex(tickers).fillna(0.0).values
            conf = conf_df.loc[d].reindex(tickers).fillna(0.0).values
            mkt_vol = rets.loc[:d].tail(p['cov_window']).std().mean() * np.sqrt(ANN)

            raw = opt.optimize(sig, conf, market_vol=mkt_vol)

            if p['inverse_vol']:
                iv = 1.0 / (rets.loc[:d].tail(p['cov_window']).std().values + 1e-6)
                iv = np.nan_to_num(iv)
                raw = raw * iv
                if np.abs(raw).sum() > 0:
                    raw = raw / np.abs(raw).sum()

            lev = risk.leverage(raw, rets.loc[:d])
            target = np.clip(raw * lev, -p['max_asset'], p['max_asset'])

            if regime.loc[d] == 0:
                target = target * p['regime_off']

            if p['no_trade_band'] > 0:
                delta = target - prev_target
                target = np.where(np.abs(delta) >= p['no_trade_band'],
                                  target, prev_target)

            last_lev = lev
        else:
            target = prev_target

        turnover = float(np.abs(target - prev_target).sum())
        gross = float(np.sum(target * fr))
        rows.append((gross - p['cost'] * turnover, gross, turnover, last_lev))
        prev_target = target

    return pd.DataFrame(rows, index=close.index[:-1],
                        columns=['net', 'gross', 'turn', 'lev'])


def evaluate(close, fwd, sig, conf, regime, **ov):
    df = run_config(close, fwd, sig, conf, regime, **ov)
    tr = metrics(slice_s(df['net'], TRAIN_START, TRAIN_END))
    te = metrics(slice_s(df['net'], TEST_START, TEST_END))
    fu = metrics(slice_s(df['net'], TRAIN_START, TEST_END))
    turn = slice_s(df['turn'], TRAIN_START, TEST_END).mean()
    return dict(tr=tr, te=te, fu=fu, turn=turn, df=df)


def main():
    close_all, high, low, volume = load_data('6y')

    # Precompute signals per universe.
    cache = {}
    for tag, uni in [('9', U9), ('15', U15)]:
        cols = [t for t in uni if t in close_all.columns]
        c = close_all[cols]
        cache[tag] = {
            'close': c,
            'fwd': c.shift(-1) / c - 1.0,
            'sigs': {
                'trend50': sig_trend(c, 50),
                'ema50': sig_ema(c, 50),
                'breakout50': sig_breakout(c, 50),
                'slope50': sig_slope(c, 50),
                'mom50': sig_mom(c, 50),
                'mom50skip5': sig_mom(c, 50, 5),
                'xs_blend30': sig_xs_blend(c, 50, 0.3),
                'dual20_50': sig_dual(c, 20, 50),
            },
            'regs': {
                'btc150': reg_btc(c, 150),
                'btc150_hyst': reg_btc_hyst(c, 150, 0.01),
                'breadth50': reg_breadth(c, 50, 0.5),
                'combo': reg_combo(c, 150, 50),
                'none': reg_none(c),
            },
        }

    hdr = (f"{'config':<46}{'trSh':>7}{'teSh':>7}{'fuSh':>7}"
           f"{'teCAGR':>8}{'teMDD':>8}{'teTot':>7}{'turn':>7}")
    print('=' * 97)
    print('V3 TUNING SWEEP')
    print('=' * 97)
    print(hdr)
    print('-' * 97)

    results = []

    def add(name, tag, sig_name, reg_name, **ov):
        u = cache[tag]
        sig, conf = u['sigs'][sig_name]
        r = evaluate(u['close'], u['fwd'], sig, conf, u['regs'][reg_name], **ov)
        results.append((name, r))
        print(f"{name:<46}{r['tr']['sharpe']:>7.2f}{r['te']['sharpe']:>7.2f}"
              f"{r['fu']['sharpe']:>7.2f}{r['te']['ann']*100:>8.1f}"
              f"{r['te']['mdd']*100:>8.1f}{r['te']['total']*100:>7.1f}"
              f"{r['turn']*100:>7.1f}")
        return r

    # 1. Baseline
    add('BASE: 9a trend50 btc150', '9', 'trend50', 'btc150')

    # 2. Universe
    add('15 assets', '15', 'trend50', 'btc150')

    # 3. Signals (15 assets)
    for s in ['ema50', 'breakout50', 'slope50', 'mom50', 'mom50skip5',
              'xs_blend30', 'dual20_50']:
        add(f'15a {s}', '15', s, 'btc150')

    # 4. Regime variants
    add('15a trend50 btc150_hyst', '15', 'trend50', 'btc150_hyst')
    add('15a trend50 breadth50', '15', 'trend50', 'breadth50')
    add('15a trend50 combo', '15', 'trend50', 'combo')
    add('15a trend50 no regime', '15', 'trend50', 'none')

    # 5. Bear-regime behaviour
    add('15a trend50 btc150 off=-0.5', '15', 'trend50', 'btc150', regime_off=-0.5)
    add('15a trend50 btc150 off=-1.0', '15', 'trend50', 'btc150', regime_off=-1.0)
    add('15a trend50 btc150 off=0.5', '15', 'trend50', 'btc150', regime_off=0.5)

    # 6. Risk scaling
    for vt in (0.20, 0.25):
        add(f'15a trend50 btc150 vt={vt:.0%}', '15', 'trend50', 'btc150',
            target_vol=vt)
    for cw in (40, 90):
        add(f'15a trend50 btc150 cov={cw}', '15', 'trend50', 'btc150',
            cov_window=cw)
    add('15a trend50 btc150 max_asset=.75', '15', 'trend50', 'btc150',
        max_asset=0.75)

    # 7. Execution
    for rd in (2, 5):
        add(f'15a trend50 btc150 rebal={rd}d', '15', 'trend50', 'btc150',
            rebalance_days=rd)
    for band in (0.01, 0.02):
        add(f'15a trend50 btc150 band={band}', '15', 'trend50', 'btc150',
            no_trade_band=band)
    add('15a trend50 btc150 inverse-vol', '15', 'trend50', 'btc150',
        inverse_vol=True)

    print('-' * 97)
    print('\nTop 10 by TEST Sharpe (check train too):')
    for name, r in sorted(results, key=lambda x: x[1]['te']['sharpe'],
                          reverse=True)[:10]:
        print(f"  {name:<44} tr={r['tr']['sharpe']:.2f}  te={r['te']['sharpe']:.2f}"
              f"  full={r['fu']['sharpe']:.2f}  teCAGR={r['te']['ann']*100:.1f}%"
              f"  teMDD={r['te']['mdd']*100:.1f}%")

    print('\nTop 10 by FULL Sharpe:')
    for name, r in sorted(results, key=lambda x: x[1]['fu']['sharpe'],
                          reverse=True)[:10]:
        print(f"  {name:<44} tr={r['tr']['sharpe']:.2f}  te={r['te']['sharpe']:.2f}"
              f"  full={r['fu']['sharpe']:.2f}  teCAGR={r['te']['ann']*100:.1f}%"
              f"  teMDD={r['te']['mdd']*100:.1f}%")

    # Yearly for the best full-Sharpe config.
    best_name, best_r = max(results, key=lambda x: x[1]['fu']['sharpe'])
    print(f'\nBest full-Sharpe config: {best_name}')
    y = yearly(slice_s(best_r['df']['net'], TRAIN_START, TEST_END)) * 100
    btc = yearly(slice_s(cache['15']['fwd']['BTC-USD'], TRAIN_START, TEST_END)) * 100
    print(pd.DataFrame({'best': y, 'BTC B&H': btc}).round(1).to_string())


if __name__ == '__main__':
    main()
