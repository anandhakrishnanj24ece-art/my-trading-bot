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
- Local caches, model pickles, account artefacts, and `config.json` are ignored
  by git; see `.gitignore`.
- `TRADE_HISTORY.md` is generated from the live state files by
  `nosharashi_1/update_history_md.py`.

## Manual trigger

Open the **Actions** tab, select either workflow, and choose **Run workflow**.
