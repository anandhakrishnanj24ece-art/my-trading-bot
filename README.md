# AMD Quant Trading Bot — Paper Trading on GitHub Actions

## Overview

The AMD (Accumulation / Manipulation / Distribution) Smart Money Concepts bot runs as a **paper trader** on GitHub Actions every 6 minutes. It fetches live OHLCV data via yfinance (free, no API key), simulates trades with fees/slippage/funding, and persists state between runs via `paper_state.json`.

## Files

| File | Purpose |
|------|---------|
| `.github/workflows/paper_trade.yml` | GitHub Actions cron (every 6 min) |
| `amd_paper_trader.py` | Paper trading engine with state persistence |
| `amd_futures_backtest.py` | Backtest engine + signal logic |
| `amd_full_validation_oos.py` | Quantitative validation suite |
| `paper_state.json` | Persisted state (auto-generated) |
| `equity_history.json` | Equity curve history (auto-generated) |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Create a GitHub repository

Push this code to a GitHub repo.

### 2. Create a Personal Access Token (PAT)

The workflow needs a PAT to commit `paper_state.json` back to the repo after each run.

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Generate a new token with these permissions:
   - **Repository permissions**: Contents (read & write)
   - **Repository permissions**: Issues (read & write) — optional
3. Copy the token.

### 3. Add the PAT as a GitHub secret

1. In your GitHub repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PAPER_TRADER_PAT`
4. Value: paste your PAT token
5. Click "Add secret"

### 4. Enable Actions

In your repo → Settings → Actions → General → Allow all actions and reusable workflows.

### 5. Initialize state (optional)

Create an empty `paper_state.json` and `equity_history.json` in the repo root so the first commit has something to work with:

```json
{}
```

## How It Works

1. **Cron trigger**: Every 6 minutes (`*/6 * * * *`), GitHub Actions starts a new job
2. **Checkout**: The repo is checked out with write permissions (using the PAT)
3. **Python setup**: Python 3.11 is installed, dependencies from `requirements.txt` are installed
4. **Paper trade**: `amd_paper_trader.py` runs for each symbol (BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD)
   - Loads persisted state from `paper_state.json` (equity, open positions, stops, etc.)
   - Fetches last 60 days of 1h OHLCV data from yfinance
   - Processes only the past 30 days as the "live" paper window
   - Executes entries/exits with fees (0.05% taker), slippage (1 bp), funding (0.01%/8h)
   - Applies drawdown kill switch (20% halt, 15% → 50% size reduction)
5. **Commit**: Updated `paper_state.json` and `equity_history.json` are committed back to the repo

## Viewing Results

- Check the **Actions tab** in your GitHub repo for run logs
- `paper_state.json` shows current equity, open positions, stops, and trade history per symbol
- `equity_history.json` shows the equity curve over time

## Manual Trigger

You can also run the trader manually from the Actions tab → "AMD Paper Trader" → "Run workflow".

## Important Notes

- **yfinance is free but unreliable** — it may return empty data or rate-limit. For production paper trading, consider a paid market data API.
- **GitHub Actions free tier**: 2,000 minutes/month. A 6-minute cron running 5 symbols should stay well within limits.
- **Each job has a 10-minute timeout** — if data fetching takes longer, the job will fail.
- **State persistence**: `paper_state.json` is committed back after each run, so positions, stops, and equity persist across cron executions.
- **Capital**: Each symbol starts with `$10,000` (set via `--capital 10000` in the workflow). The state persists per-symbol.

## Customizing

Edit `.github/workflows/paper_trade.yml` to:
- Change symbols: modify `--symbols` argument
- Change capital: modify `--capital` argument
- Change frequency: modify the cron expression (minimum 5 minutes on GitHub Actions)

## Stopping the Bot

To stop, either:
1. Disable the workflow in the Actions tab, or
2. Remove the `.github/workflows/paper_trade.yml` file and push
