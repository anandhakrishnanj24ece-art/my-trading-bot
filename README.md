# No-Sharashi — Multi-Strategy Paper Trading Bot

Four paper-trading strategies run on GitHub Actions and publish a static dashboard:

| # | Strategy | Runner | State |
|---|----------|--------|-------|
| 1 | Adaptive Weighted (LiquidityAlpha) | `nosharashi_1/neural_flow/run_adaptive.py` | `nosharashi_1/paper_state_adaptive.json` |
| 2 | Adaptive v3 (trend + regime + Lasso) | `nosharashi_1/neural_flow/run_adaptive_v3.py` | `nosharashi_1/paper_state_v3.json` |
| 3 | TrendVol (50d time-series momentum) | `nosharashi_1/neural_flow/run_trend.py` | `nosharashi_1/paper_state_trend.json` |
| 4 | AMD Smart Money (PO3) | `amd_paper_trader.py` | `paper_state.json` |

## Workflows

| File | Schedule | Purpose |
|------|----------|---------|
| `.github/workflows/adaptive_daily.yml` | Daily at 01:00 UTC | Runs strategies 1–3 after the crypto daily candle closes |
| `.github/workflows/live_trade.yml` | Every 6 minutes | Runs strategy 4 (AMD) |

Both workflows share the `paper-trading-state` concurrency group, so they cannot
push state at the same time. They commit their own state files back to `main`.
No Personal Access Token is required: the workflows use the repository
`GITHUB_TOKEN` with `contents: write`.

## Dashboard

`index.html` is a static dashboard that reads `dashboard_data.json`. Regenerate
the feed after running any strategy:

```bash
python build_dashboard.py
python nosharashi_1/update_history_md.py
```

The dashboard shows all four strategies, a union of monitored assets, and the
last strategy comparison cards from `dashboard_data.json`.

## Tests

The live runners have synthetic-data regression tests. They use temporary state
files and never touch the deployed accounts:

```bash
python -m unittest discover -s tests -v
```

## Running locally

```bash
pip install -r requirements.txt

cd nosharashi_1
export PYTHONPATH=$(pwd)          # PowerShell: $env:PYTHONPATH = (Get-Location).Path
python neural_flow/run_adaptive.py
python neural_flow/run_adaptive_v3.py
python neural_flow/run_trend.py

cd ..
python amd_paper_trader.py --live --capital 10000 --symbols BTC-USD ETH-USD SOL-USD BNB-USD XRP-USD
python build_dashboard.py
python nosharashi_1/update_history_md.py
```

## Notes

- Market data comes from yfinance (free, no API key) and can be rate-limited or
  return incomplete data. Runners forward-fill gaps and never back-fill, so
  missing history cannot leak future prices into a signal.
- Entry costs, turnover costs, and short financing are charged in the live
  accounting. Period returns shown on the dashboard are net of those costs.
- Strategies 1 and 3 use a 2% no-trade band and compare targets with the actual
  price-drifted book. Small same-direction rebalances are skipped, but full
  exits and long/short reversals always execute.
- Crypto volatility is annualised with 365 days, not 252. Zero or non-finite
  volatility returns 1x leverage instead of levering up into bad data.
- LiquidityAlpha requires a full 200-day SMA warm-up before it can signal, and
  the runners refuse to trade if too little common history is available.
- Local caches, model pickles, account artefacts, and `config.json` are ignored
  by git; see `.gitignore`.
- `TRADE_HISTORY.md` is generated from the live state files by
  `nosharashi_1/update_history_md.py`.

## Backtest validation (honest, post-cost)

All figures below include execution costs (0.05% fee + 0.014% effective slippage
per unit turnover, 6.4 bps total) and short financing where shorting is used.
They are point-in-time estimates, not guarantees.

| Strategy | Window | CAGR/yr | Sharpe | Max DD | Alpha vs BTC | t-stat | Assessment |
|----------|--------|---------|--------|--------|--------------|--------|------------|
| Adaptive Weighted | 2024-11 → 2026-11 | -11.1% | -0.53 | 39.0% | — | — | **No edge.** Negative after look-ahead fix + costs. Do not trade. |
| Adaptive v3 | 2024-11 → 2026-11 (2y) | -5.0% | -0.43 | 20.7% | — | — | **No robust edge.** Bootstrap interval includes zero. |
| TrendVol (15 assets) | 2020-11 → 2026-11 | +58.3% | 2.55 | 17.4% | +39.6%/yr | 5.19 | **Best candidate.** Strong post-cost backtest, incl. out-of-sample. Survivorship-biased universe — treat as upper bound. |
| TrendVol (15 assets) | 2024-01 → 2026-11 (test) | +33.0% | 1.57 | 13.1% | +24.1%/yr | — | Out-of-sample test confirms edge, but window is short. |

Interpretation:
- **Adaptive Weighted** loses money once look-ahead bias and transaction costs
  are removed (-11% CAGR). It should not be traded expecting positive edge.
- **Adaptive v3** shows no robust edge (bootstrap interval includes zero; 2y
  window is negative). Do not trade expecting positive returns.
- **TrendVol (15 assets)** is the only strategy with a demonstrated post-cost
  edge (+58% CAGR, Sharpe 2.55 full period; +33% CAGR, Sharpe 1.57 test
  period). However, the 15-asset universe is survivorship-biased — it only
  includes assets that exist today. Treat backtest figures as upper bounds,
  not guarantees. Forward paper-trading evidence is still limited.

None of the strategies demonstrates a statistically robust edge after all
costs with sufficient live evidence. Position sizes should be kept small and
expectations modest. TrendVol is the best candidate but should be traded small
until it builds a longer live track record.

## Risk controls (live)

- **Drawdown overlay:** exposure is scaled from 1.0x down to a 0.25x floor as
  drawdown moves from 5% to 15% of the high-water mark.
- **No-trade band (2%):** small same-direction rebalances are skipped; full
  exits and reversals always execute.
- **Short financing:** 3 bp/day charged on short exposure (Adaptive Weighted
  and v3).
- **Data quality:** full SMA warm-up required; minimum common history enforced;
  stale/missing tickers are closed rather than silently dropped.
- **Idempotency:** the same daily bar is never processed twice for the same
  book.

## Manual trigger

Open the **Actions** tab, select either workflow, and choose **Run workflow**.
