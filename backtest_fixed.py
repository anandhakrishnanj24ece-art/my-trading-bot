"""
Corrected backtest harness for the Adaptive Weighted and Adaptive v3 strategies.

Key fix vs. the original scripts: a signal formed at bar i can only earn the
return of bar i+1. The original code multiplied the signal by the SAME bar's
return, which is look-ahead bias. This harness reports both alignments so the
inflation is visible.

Run:  python backtest_fixed.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
           'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']


def load_data(period='2y'):
    import yfinance as yf
    raw = yf.download(TICKERS, period=period, interval='1d', progress=False, threads=False)
    close = raw['Close'].ffill().dropna()
    volume = raw['Volume'].ffill().dropna()
    tickers = close.columns.tolist()
    rets = close.pct_change().dropna()
    idx = rets.index.intersection(close.index).intersection(volume.index)
    return close.loc[idx], volume.loc[idx], rets.loc[idx], tickers


def compute_signals(close, volume, tickers):
    alpha = LiquidityAlpha()
    sig, conf = {}, {}
    for t in tickers:
        r = alpha.generate_signal(pd.DataFrame({'Close': close[t], 'Volume': volume[t]}))
        sig[t], conf[t] = r['signal'], r['confidence']
    return pd.DataFrame(sig), pd.DataFrame(conf)


# ------------------------- strategy implementations -------------------------
def base_weights(signals, confs, btc_rets):
    s = np.asarray(signals, dtype=float)
    c = np.asarray(confs, dtype=float)
    if np.abs(s).sum() > 0:
        w = np.sign(s) * (np.abs(s) / np.abs(s).sum())
    else:
        w = np.zeros(len(s))
    caps = np.where(c < 1.0, 0.05, np.where(c < 2.0, 0.20, 0.50))
    w = np.clip(w, -caps, caps)
    if np.abs(w).sum() > 1.0:
        w = w / np.abs(w).sum()
    if len(btc_rets) < 20:
        lev = 1.0
    else:
        rv = btc_rets.tail(20).std() * np.sqrt(252)
        if not np.isfinite(rv) or rv <= 0:
            lev = 1.0
        else:
            lev = 1.0 if rv > 0.40 else 4.0
    return w * lev


def v3_weights(signals, confs, market_vol, prev):
    s = np.asarray(signals, dtype=float)
    c = np.asarray(confs, dtype=float)
    snr = np.mean(np.abs(s)) / (np.std(s) + 1e-6)
    lam = 0.005 / (1.0 + snr)
    abs_s = np.abs(s)
    if abs_s.sum() == 0:
        return np.zeros(len(s))
    w = np.sign(s) * (abs_s / abs_s.sum())
    w = np.where(np.abs(w) > lam, w, 0.0)
    if np.abs(w).sum() > 0:
        w = w / np.abs(w).sum()
    else:
        return np.zeros(len(s))
    caps = 0.05 + (np.clip(c, 0, 3) / 3.0) * (0.50 - 0.05)
    w = np.clip(w, -caps, caps)
    if np.abs(w).sum() > 1.0:
        w = w / np.abs(w).sum()
    if prev is not None and len(prev) == len(w):
        new = np.copy(prev)
        vol_scale = 1.0 / (1.0 + market_vol * 10)
        tau = 0.01 * vol_scale
        for i in range(len(w)):
            if abs(s[i]) > 3.0:
                new[i] = w[i]
                continue
            d = w[i] - prev[i]
            if d < 0 and abs(d) > 0.5 * tau:
                new[i] = w[i]
            elif d > 0 and abs(d) > 1.5 * tau:
                new[i] = w[i]
            else:
                new[i] = prev[i]
        w = new
    return w


def v3_leverage(weights, returns_upto):
    if len(returns_upto) < 20:
        return 1.0
    recent = returns_upto.tail(20)
    try:
        cov = LedoitWolf().fit(recent).covariance_ * 252
    except Exception:
        cov = recent.cov() * 252
    pvar = float(np.dot(weights.T, np.dot(cov, weights)))
    pvol = np.sqrt(max(0.0, pvar))
    if not np.isfinite(pvol) or pvol <= 0:
        return 1.0
    return min(0.15 / pvol, 4.0)


def run(mode, sig_df, conf_df, rets, tickers, shift):
    """shift=0 -> buggy same-bar; shift=1 -> correct next-bar."""
    engine = StealthExecution()
    pv = [1.0]
    prev_w = np.zeros(len(tickers))
    prev_raw = None
    daily, turnovers = [], []
    n = len(rets) - shift

    for i in range(n):
        s = sig_df.iloc[i].values
        c = conf_df.iloc[i].values
        if mode == 'base':
            w = base_weights(s, c, rets.iloc[:i + 1, 0])
        else:
            mkt_vol = rets.iloc[:i + 1].std().mean() * np.sqrt(252) if i > 20 else 0.15
            raw = v3_weights(s, c, mkt_vol, prev_raw)
            prev_raw = raw
            w = raw * v3_leverage(raw, rets.iloc[:i + 1])
        r = rets.iloc[i + shift].values
        turnover = float(np.sum(np.abs(w - prev_w)))
        cost = engine.calculate_cost(turnover)
        net = float(np.sum(w * r)) - cost
        daily.append(net)
        pv.append(pv[-1] * (1.0 + net))
        prev_w = w
        turnovers.append(turnover)

    daily = np.asarray(daily)
    total = float(np.prod(1.0 + daily) - 1.0)
    ann = (1.0 + total) ** (252.0 / len(daily)) - 1.0 if len(daily) else 0.0
    vol = float(daily.std() * np.sqrt(252))
    sharpe = ann / vol if vol else 0.0
    eq = np.asarray(pv)
    mdd = float((1 - eq / np.maximum.accumulate(eq)).max())

    # Hit rate: signal at bar i vs return at i+1 (always the honest pairing)
    hits = tot = 0
    for i in range(len(rets) - 1):
        s = sig_df.iloc[i].values
        r = rets.iloc[i + 1].values
        for si, ri in zip(s, r):
            if si != 0:
                tot += 1
                hits += int(np.sign(si) == np.sign(ri))
    hit_rate = 100.0 * hits / tot if tot else 0.0

    return {
        'Total': total * 100,
        'CAGR': ann * 100,
        'Vol': vol * 100,
        'Sharpe': sharpe,
        'MaxDD': mdd * 100,
        'HitRate': hit_rate,
        'Turnover': float(np.mean(turnovers)) * 100,
    }


def main():
    print('Downloading 2y daily data for', len(TICKERS), 'assets...')
    close, volume, rets, tickers = load_data('2y')
    print(f'{len(rets)} return bars | {rets.index[0].date()} -> {rets.index[-1].date()}\n')

    sig, conf = compute_signals(close, volume, tickers)

    rows = []
    for mode, label in [('base', 'Adaptive Weighted'), ('v3', 'Adaptive v3')]:
        buggy = run(mode, sig, conf, rets, tickers, shift=0)
        fixed = run(mode, sig, conf, rets, tickers, shift=1)
        rows.append((label + ' [OLD same-bar]', buggy))
        rows.append((label + ' [FIXED next-bar]', fixed))

    hdr = f"{'Strategy':<32}{'Total%':>9}{'CAGR%':>9}{'Vol%':>8}{'Sharpe':>9}{'MaxDD%':>9}{'Hit%':>8}{'Turn%':>8}"
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for name, m in rows:
        print(f"{name:<32}{m['Total']:>9.2f}{m['CAGR']:>9.2f}{m['Vol']:>8.2f}"
              f"{m['Sharpe']:>9.2f}{m['MaxDD']:>9.2f}{m['HitRate']:>8.2f}{m['Turnover']:>8.2f}")
    print('=' * len(hdr))
    print('\nNote: SAME-BAR is the original (look-ahead) result; NEXT-BAR is the honest result.')


if __name__ == '__main__':
    main()
