"""Old vs new v3 on the exact same out-of-sample window."""
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha
from research_v3_improved import (V3_TICKERS, TEST_START, TEST_END, run_v3,
                                  liq_signals, trend_signals, slice_s)

CFG = dict(tau_mode='vol', tau_base=0.02, lambda_base=0.02,
           base_leverage=2.0, cov_window=60, risk_ann=365,
           max_asset=1.0, target_vol=0.15)


def main():
    close, high, low, volume = load_data('6y')
    tickers = [t for t in V3_TICKERS if t in close.columns]
    close, volume = close[tickers], volume[tickers]
    fwd = close.shift(-1) / close - 1.0

    liq_sig, liq_conf = liq_signals(close, volume)
    old = run_v3(close, fwd, liq_sig, liq_conf)

    tr_sig, tr_conf = trend_signals(close, 50)
    new = run_v3(close, fwd, tr_sig, tr_conf, regime='btc150',
                 regime_off=0.0, **CFG)

    for name, df in [('OLD v3 (LiquidityAlpha, 4x)', old),
                     ('NEW v3 (trend50+BTC150, 2x)', new)]:
        r = slice_s(df['net'], TEST_START, TEST_END)
        m = metrics(r)
        _, a = beta_alpha(r, slice_s(fwd['BTC-USD'], TEST_START, TEST_END))
        print(f"{name:<34} total={m['total']*100:7.1f}%  "
              f"CAGR={m['ann']*100:6.1f}%  vol={m['vol']*100:5.1f}%  "
              f"Sharpe={m['sharpe']:5.2f}  MDD={m['mdd']*100:5.1f}%  "
              f"alpha={a*100:5.1f}%")


if __name__ == '__main__':
    main()
