"""
Best-available past-year proxy for UltimateORB.

The dated trade CSV is missing. The PDF scatter groups winners and losers, but
within each group the marker order follows the plotted trade order. The report's
2026 slice has 365 trades and a 94% win rate, so this proxy uses:

    last 343 winners + last 22 losers = 365 trades

Validation against the report's 2026 slice:
  - proxy win rate 93.97% vs report 94%
  - proxy avg win 8.61 pts vs OOS report ~8 pts
  - proxy avg loss -82.87 pts vs OOS report ~-72 pts
  - proxy current net 3.09 pts/trade ($2,257) vs reported 2026 $2,704

This is a reconstruction, not the original NinjaTrader CSV. Treat it as a
past-year proxy, not an exact backtest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from extract_and_replay import COMMISSION_PTS, CONTRACT_MULT, extract_excursions

ROOT = Path(__file__).resolve().parent
EXCURSIONS = ROOT / "ultimateorb_excursions.csv"
TRADES_OUT = ROOT / "past_year_proxy_trades.csv"
RESULTS_OUT = ROOT / "past_year_proxy_results.csv"

PROXY_TRADES = 365
PROXY_WINNERS = 343
PROXY_LOSERS = 22


def load_proxy() -> pd.DataFrame:
    source = pd.read_csv(EXCURSIONS) if EXCURSIONS.exists() else extract_excursions()
    winners = source[source["outcome"] == "winner"].tail(PROXY_WINNERS)
    losers = source[source["outcome"] == "loser"].tail(PROXY_LOSERS)
    proxy = pd.concat([winners, losers], ignore_index=True)
    if len(proxy) != PROXY_TRADES:
        raise RuntimeError(f"Expected {PROXY_TRADES} proxy trades, got {len(proxy)}")
    proxy.to_csv(TRADES_OUT, index=False)
    return proxy


def stop_only(proxy: pd.DataFrame, stop_pts: float) -> np.ndarray:
    winners = proxy["outcome"].to_numpy() == "winner"
    mae = proxy["mae_pts"].to_numpy()
    mfe = proxy["mfe_pts"].to_numpy()
    gross = np.where(mae > stop_pts, -stop_pts, np.where(winners, mfe, -mae))
    return gross - COMMISSION_PTS


def target_scenario(
    proxy: pd.DataFrame,
    target_pts: float,
    stop_pts: float,
    continuation_boost: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted outcomes for a fixed target/stop with a driftless extension."""
    winners = proxy["outcome"].to_numpy() == "winner"
    mae = proxy["mae_pts"].to_numpy()
    mfe = proxy["mfe_pts"].to_numpy()

    outcomes: list[float] = []
    weights: list[float] = []

    for is_winner, trade_mae, trade_mfe in zip(winners, mae, mfe):
        if trade_mae > stop_pts:
            outcomes.append(-stop_pts)
            weights.append(1.0)
        elif trade_mfe >= target_pts:
            outcomes.append(target_pts)
            weights.append(1.0)
        elif is_winner:
            p = (trade_mfe + stop_pts) / (target_pts + stop_pts)
            p = min(1.0, max(0.0, p + continuation_boost))
            outcomes.extend([target_pts, -stop_pts])
            weights.extend([p, 1.0 - p])
        else:
            outcomes.append(-trade_mae)
            weights.append(1.0)

    return np.asarray(outcomes, dtype=float) - COMMISSION_PTS, np.asarray(weights, dtype=float)


def summarize(label: str, target: float | None, stop: float | None, net: np.ndarray, weights: np.ndarray) -> dict:
    wins = net > 0
    total_weight = weights.sum()
    avg_net = float((net * weights).sum() / total_weight)
    win_rate = float((weights * wins).sum() / total_weight)
    avg_win = float((net[wins] * weights[wins]).sum() / weights[wins].sum())
    avg_loss = float((net[~wins] * weights[~wins]).sum() / weights[~wins].sum())
    pf = float(
        (net[wins] * weights[wins]).sum()
        / abs((net[~wins] * weights[~wins]).sum())
    )
    return {
        "setting": label,
        "target_pts": target,
        "stop_pts": stop,
        "win_rate_pct": win_rate * 100,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "avg_net_pts": avg_net,
        "profit_factor": pf,
        "loss_rate_pct": (1.0 - win_rate) * 100,
        "total_net_usd_1_mnq": avg_net * total_weight * CONTRACT_MULT,
        "worst_trade_pts": float(net.min()),
    }


def main() -> None:
    proxy = load_proxy()
    n = len(proxy)
    winners = (proxy["outcome"] == "winner").sum()
    losers = n - winners

    rows: list[dict] = []
    # Current: dynamic target, losers at -MAE.
    current_net = np.where(
        proxy["outcome"].to_numpy() == "winner",
        proxy["mfe_pts"].to_numpy(),
        -proxy["mae_pts"].to_numpy(),
    ) - COMMISSION_PTS
    rows.append(summarize("current", None, None, current_net, np.ones(n)))

    for stop in [10, 20, 30, 40, 60.7]:
        net = stop_only(proxy, stop)
        rows.append(summarize(f"6.6/{stop:g}", 6.6, stop, net, np.ones(n)))

    for target, stop in [(10, 30), (12, 30)]:
        net, weights = target_scenario(proxy, target, stop, 0.0)
        rows.append(summarize(f"{target}/{stop}", target, stop, net, weights))

    results = pd.DataFrame(rows).round(3)
    results.to_csv(RESULTS_OUT, index=False)

    print(f"Proxy: {n} trades ({winners} winners / {losers} losers), win rate {winners/n*100:.2f}%")
    print(f"Winner MFE mean: {proxy.loc[proxy.outcome == 'winner', 'mfe_pts'].mean():.2f} pts")
    print(f"Loser MAE mean: {proxy.loc[proxy.outcome == 'loser', 'mae_pts'].mean():.2f} pts")
    print("\nPast-year proxy results:")
    print(results.to_string(index=False))
    print(f"\nWrote:\n  {TRADES_OUT}\n  {RESULTS_OUT}")


if __name__ == "__main__":
    main()
