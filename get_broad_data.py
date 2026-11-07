"""Download a broad, survivorship-inclusive crypto universe for audit."""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

BROAD = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
    'DOT-USD', 'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD',
    'NEAR-USD', 'ALGO-USD', 'FIL-USD',
    # collapsed / delisted / distressed names
    'LUNA1-USD', 'LUNC-USD', 'UST-USD', 'FTT-USD', 'CEL-USD', 'SRM-USD',
    'ANC-USD', 'MIR-USD',
    # other liquid names from the period
    'XLM-USD', 'TRX-USD', 'ETC-USD', 'BCH-USD', 'UNI-USD', 'AAVE-USD',
    'MKR-USD', 'SUSHI-USD', 'COMP-USD', 'YFI-USD', 'SAND-USD', 'MANA-USD',
    'AXS-USD', 'THETA-USD', 'VET-USD', 'ICP-USD', 'HBAR-USD', 'EOS-USD',
    'XTZ-USD', 'NEO-USD', 'IOTA-USD', 'DASH-USD', 'ZEC-USD', 'XMR-USD',
    'WAVES-USD', 'KSM-USD', 'CRV-USD', 'SNX-USD', 'GRT-USD', 'ENJ-USD',
    'CHZ-USD', 'BAT-USD', 'ZIL-USD', 'ONE-USD', 'CELO-USD',
]


def main():
    last = None
    for _ in range(5):
        try:
            raw = yf.download(BROAD, period='6y', interval='1d',
                              progress=False, threads=True, auto_adjust=True)
            if raw is None or len(raw) == 0 or 'Close' not in raw:
                raise RuntimeError('empty')
            close = raw['Close'].ffill()
            volume = raw['Volume'].ffill()
            close = close.dropna(axis=1, how='all')
            volume = volume[close.columns]
            pd.to_pickle((close, volume), 'broad_data.pkl')
            print('saved:', close.shape, close.index[0].date(), '->',
                  close.index[-1].date())
            print('\ncoverage:')
            for t in close.columns:
                s = close[t]
                print(f"  {t:<12} first={str(s.first_valid_index())[:10]} "
                      f"last={str(s.last_valid_index())[:10]} "
                      f"rows={int(s.notna().sum()):<5} "
                      f"last_px={float(s.dropna().iloc[-1]):.4g}")
            # The batch call drops dead tickers; fetch the ones Yahoo still
            # serves individually so the PIT universe is less survivorship-biased.
            try:
                from build_broad_data_delisted import main as add_delisted
                add_delisted()
            except Exception as exc:  # noqa: BLE001
                print('delisted add-on failed:', type(exc).__name__, exc)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3)
    raise RuntimeError(f'broad download failed: {last}')


if __name__ == '__main__':
    main()
