"""
================================================================================
SNAPSHOT-BASED BACKTEST ENGINE
================================================================================
Reads historical price & volume data from persisted live runner snapshots
(in `price_snapshots/`) rather than fetching fresh data from yfinance.

This guarantees:
1. Zero lookahead bias.
2. Identical close and volume figures seen by the live runner in production.
3. Accurate evaluation of signals, target weights, turnover, and execution slippage.
================================================================================
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path so neural_flow modules are available
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch


def load_snapshots(snapshot_dir: Path, strategy_filter: str = None) -> list[dict]:
    """Load all JSON snapshots in chronological order."""
    if not snapshot_dir.exists():
        print(f"Directory {snapshot_dir} does not exist.")
        return []

    files = sorted(snapshot_dir.glob("*.json"))
    snapshots = []
    for f in files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                if strategy_filter:
                    strat = data.get("strategy")
                    if strat and strat != strategy_filter:
                        continue
                snapshots.append((f.name, data))
        except Exception as e:
            print(f"Warning: Failed to parse snapshot {f.name}: {e}")

    return snapshots


def reconstruct_dataframe_from_latest_snapshot(snapshot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct full historical (Close, Volume) DataFrames from a snapshot.
    Each snapshot contains the entire historical price array seen during that execution.
    """
    tickers = snapshot["tickers"]
    dates = pd.to_datetime(snapshot["dates"])
    
    close_dict = {t: snapshot["close"][t] for t in tickers}
    volume_dict = {t: snapshot["volume"][t] for t in tickers}
    
    close_df = pd.DataFrame(close_dict, index=dates)
    volume_df = pd.DataFrame(volume_dict, index=dates)
    
    return close_df, volume_df


def backtest_from_snapshot_archive(snapshot_dir: Path, initial_capital: float = 10000.0, strategy: str = "adaptive"):
    """
    Execute backtest across snapshot files.
    - If multiple run snapshots exist: evaluates sequence of live executions across time.
    - If a single snapshot exists: executes full bar-by-bar backtest using the snapshot's exact historical OHLCV.
    """
    print("=" * 70)
    print(f" RUNNING BACKTEST FROM PRICE SNAPSHOTS ({strategy.upper()})")
    print("=" * 70)

    snapshots = load_snapshots(snapshot_dir, strategy_filter=strategy)

    if not snapshots:
        print(f"No price snapshots found in {snapshot_dir.resolve()}.")
        print("Note: Snapshots are generated automatically whenever run_adaptive.py or run_adaptive_v3.py executes.")
        return None

    print(f"Found {len(snapshots)} snapshot file(s) for strategy '{strategy}'.\n")

    # Mode A: If multiple periodic snapshots exist, evaluate step-by-step telemetry
    if len(snapshots) >= 2:
        print(f"Mode: Multi-snapshot chronological playback ({len(snapshots)} live steps)")
        capital = initial_capital
        history = []

        prev_weights = {}
        prev_prices = {}

        for fname, snap in snapshots:
            ts = snap.get("timestamp")
            tickers = snap.get("tickers", [])
            last_prices = snap.get("last_prices", {})
            target_weights = snap.get("weights", {})
            saved_pv = snap.get("portfolio_value", capital)

            period_return = 0.0
            if prev_prices and prev_weights:
                for t in tickers:
                    w = prev_weights.get(t, 0.0)
                    if t in prev_prices and prev_prices[t] > 0 and t in last_prices:
                        ret = (last_prices[t] - prev_prices[t]) / prev_prices[t]
                        period_return += w * ret

                execution_engine = StealthExecution()
                turnover = sum(abs(target_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in tickers)
                costs = execution_engine.calculate_cost(turnover)
                capital = capital * (1.0 + period_return - costs)

            history.append({
                "snapshot_file": fname,
                "timestamp": ts,
                "simulated_capital": capital,
                "live_runner_capital": saved_pv,
                "discrepancy": capital - saved_pv
            })

            prev_weights = target_weights
            prev_prices = last_prices

        hist_df = pd.DataFrame(history)
        print(hist_df.to_string(index=False))
        return hist_df

    # Mode B: Single snapshot bar-by-bar backtest using exact live runner data
    fname, snap = snapshots[-1]
    print(f"Using exact price matrix from snapshot: {fname}")
    close_df, volume_df = reconstruct_dataframe_from_latest_snapshot(snap)

    tickers = close_df.columns.tolist()
    returns = close_df.pct_change().dropna()
    common_idx = returns.index.intersection(close_df.index).intersection(volume_df.index)
    returns = returns.loc[common_idx]
    close_df = close_df.loc[common_idx]
    volume_df = volume_df.loc[common_idx]

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    portfolio_values = [initial_capital]
    prev_weights = np.zeros(len(tickers))

    for i in range(len(returns)):
        # Compute signals on expanding historical window up to bar i
        sub_close = close_df.iloc[:i + 1]
        sub_vol = volume_df.iloc[:i + 1]

        signals = []
        confidences = []
        for t in tickers:
            df_t = pd.DataFrame({'Close': sub_close[t], 'Volume': sub_vol[t]})
            sig_df = alpha_engine.generate_signal(df_t)
            if not sig_df.empty:
                signals.append(sig_df['signal'].iloc[-1])
                confidences.append(sig_df['confidence'].iloc[-1])
            else:
                signals.append(0.0)
                confidences.append(0.0)

        signals_arr = np.array(signals)
        conf_arr = np.array(confidences)

        if np.abs(signals_arr).sum() > 0:
            raw_weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
        else:
            raw_weights = np.zeros(len(tickers))

        clipped = []
        for j in range(len(raw_weights)):
            c = conf_arr[j]
            cap = 0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50)
            clipped.append(np.clip(raw_weights[j], -cap, cap))

        clipped = np.array(clipped)
        tot = np.sum(np.abs(clipped))
        if tot > 1.0:
            clipped = clipped / tot

        bench_ret = returns.iloc[:i + 1, 0]
        lev = risk_engine.get_leverage(bench_ret)
        target_weights = clipped * lev

        # Calculate period return if not first bar
        if i < len(returns) - 1:
            next_bar_ret = returns.iloc[i + 1].values
            p_ret = np.sum(prev_weights * next_bar_ret)
            turnover = np.sum(np.abs(target_weights - prev_weights))
            cost = execution_engine.calculate_cost(turnover)
            new_val = portfolio_values[-1] * (1.0 + p_ret - cost)
            portfolio_values.append(new_val)

        prev_weights = target_weights

    final_val = portfolio_values[-1]
    total_ret = ((final_val - initial_capital) / initial_capital) * 100

    print("\n" + "=" * 60)
    print(" SNAPSHOT BACKTEST COMPLETED")
    print("=" * 60)
    print(f" Bars Tested     : {len(common_idx)}")
    print(f" Initial Capital : ${initial_capital:,.2f}")
    print(f" Final Capital   : ${final_val:,.2f}")
    print(f" Total Return    : {total_ret:+.2f}%")
    print("=" * 60 + "\n")

    return {
        "initial_capital": initial_capital,
        "final_capital": final_val,
        "total_return_pct": total_ret,
        "bars": len(common_idx)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest using live runner price snapshots")
    parser.add_argument("--snapshots", type=str, default="price_snapshots", help="Path to snapshot directory")
    parser.add_argument("--strategy", type=str, default="adaptive", choices=["adaptive", "adaptive_v3"], help="Strategy")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    args = parser.parse_args()

    snap_path = Path(args.snapshots)
    if not snap_path.is_absolute():
        snap_path = CURRENT_DIR / args.snapshots

    backtest_from_snapshot_archive(snap_path, initial_capital=args.capital, strategy=args.strategy)
