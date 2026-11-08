"""Parameter robustness for the only candidate with positive test performance.

Sweeps lookback, vol target, and rebalance frequency for BTC+ETH trend.
A real edge should be positive across a plateau, not one lucky combination.
"""
import pandas as pd
import research_edge as re

close, volume = re.load_data()
ind = re.make_indicators(close, volume)


def btc_eth_fn(lookback, target_vol):
    def f(i):
        t = ind['close'].index[i]
        ok = re.eligible(ind, t)
        sma = ind['close'].rolling(lookback).mean().loc[t]
        sel = [a for a in ['BTC-USD', 'ETH-USD']
               if ok.get(a, False) and ind['close'].loc[t, a] > sma[a]]
        return re.inv_vol_weights(ind, t, sel, target_vol=target_vol, max_weight=1.0)
    return f


def show(label, fn, rebal=7):
    d, _ = re.backtest(ind, fn, cost=re.COST_BASE, rebal=rebal)
    tr = re.metrics(re.slice_s(d, re.TRAIN_START, re.TRAIN_END))
    te = re.metrics(re.slice_s(d, re.TEST_START, re.TEST_END))
    print(f"{label:<28}{tr['cagr']*100:>8.1f}{tr['sharpe']:>8.2f}"
          f"{te['cagr']*100:>9.1f}{te['sharpe']:>8.2f}{te['mdd']*100:>8.1f}")


print(f"\n{'lookback / target vol':<28}{'trCAGR':>8}{'trSh':>8}{'teCAGR':>9}{'teSh':>8}{'teMDD':>8}")
print('-' * 69)
for lb in [30, 50, 75, 100, 150]:
    for tv in [0.10, 0.15, 0.20]:
        show(f"lb={lb} vol={tv:.2f}", btc_eth_fn(lb, tv))

print(f"\n{'rebalance days':<28}{'trCAGR':>8}{'trSh':>8}{'teCAGR':>9}{'teSh':>8}{'teMDD':>8}")
print('-' * 69)
for rb in [1, 7, 14, 30]:
    show(f"rebal={rb}d", btc_eth_fn(50, 0.15), rebal=rb)
