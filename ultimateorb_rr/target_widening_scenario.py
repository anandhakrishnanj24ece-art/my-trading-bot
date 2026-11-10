"""
Target-widening scenario for UltimateORB.

This is NOT an exact backtest. The PDF's MFE is censored at the original target,
so there is no direct evidence of how far a winner would have run after its
original exit. This script makes the smallest possible extension assumption and
labels it clearly:

  After a winner reaches its original target M, the remaining path is a
  driftless random walk between the new target T and the stop -S.

Under that neutral assumption, the probability of reaching T before -S is the
gambler's-ruin probability:

    p = (M + S) / (T + S)

If the trade's MAE already exceeded S, it is stopped before it ever reaches M,
so it is a loss at -S. This prevents the usual look-ahead error of giving the
trade credit for reaching M first.

The results are scenario evidence, not proof. A positive continuation edge is
tested separately as a +pp boost to p; that boost is an assumption, not a
calibrated parameter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from extract_and_replay import COMMISSION_PTS, CONTRACT_MULT, extract_excursions

ROOT = Path(__file__).resolve().parent
GRID_CSV = ROOT / "target_widening_grid.csv"
CONT_CSV = ROOT / "target_widening_continuation.csv"


def target_scenario(
    df: pd.DataFrame,
    target_pts: float,
    stop_pts: float,
    continuation_boost: float = 0.0,
) -> dict[str, float]:
    """Expected-outcome replay for a fixed target/stop pair.

    For M < T we keep two weighted outcomes, T and -S, instead of collapsing
    them to the expected value. That preserves the win rate, PF, and loss rate.
    """
    winners = df["outcome"].to_numpy() == "winner"
    mae = df["mae_pts"].to_numpy()
    mfe = df["mfe_pts"].to_numpy()

    outcomes: list[float] = []
    weights: list[float] = []

    for is_winner, trade_mae, trade_mfe in zip(winners, mae, mfe):
        # The stop is checked first. A trade whose MAE exceeds S never gets to
        # its original target, let alone a wider one.
        if trade_mae > stop_pts:
            outcomes.append(-stop_pts)
            weights.append(1.0)
            continue

        if trade_mfe >= target_pts:
            outcomes.append(target_pts)
            weights.append(1.0)
            continue

        if is_winner:
            # Extend beyond the observed original target with a driftless walk.
            p = (trade_mfe + stop_pts) / (target_pts + stop_pts)
            p = min(1.0, max(0.0, p + continuation_boost))
            outcomes.extend([target_pts, -stop_pts])
            weights.extend([p, 1.0 - p])
        else:
            # A loser never reached its original target; it cannot reach a
            # wider target in this model.
            outcomes.append(-trade_mae)
            weights.append(1.0)

    outcomes_arr = np.asarray(outcomes, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    net = outcomes_arr - COMMISSION_PTS
    wins = net > 0

    total_weight = weights_arr.sum()
    avg_net = float((net * weights_arr).sum() / total_weight)
    win_rate = float((weights_arr * wins).sum() / total_weight)
    avg_win = float((net[wins] * weights_arr[wins]).sum() / weights_arr[wins].sum())
    avg_loss = float((net[~wins] * weights_arr[~wins]).sum() / weights_arr[~wins].sum())
    pf = float(
        (net[wins] * weights_arr[wins]).sum()
        / abs((net[~wins] * weights_arr[~wins]).sum())
    )
    loss_rate = float((weights_arr * ~wins).sum() / total_weight)
    break_even = (stop_pts + COMMISSION_PTS) / (stop_pts + target_pts)

    return {
        "target_pts": target_pts,
        "stop_pts": stop_pts,
        "continuation_boost": continuation_boost,
        "break_even_win_pct": break_even * 100,
        "win_rate_pct": win_rate * 100,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "avg_net_pts": avg_net,
        "profit_factor_net": pf,
        "loss_rate_pct": loss_rate * 100,
        "total_net_usd_1_mnq": avg_net * total_weight * CONTRACT_MULT,
    }


def main() -> None:
    df = extract_excursions()

    rows = []
    for target, stop in [(10, 30), (12, 30), (10, 60.7), (12, 60.7)]:
        rows.append(target_scenario(df, target, stop, 0.0))
    grid = pd.DataFrame(rows).round(3)
    grid.to_csv(GRID_CSV, index=False)

    cont_rows = []
    for target in [10, 12]:
        for boost in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
            cont_rows.append(target_scenario(df, target, 30, boost))
    cont = pd.DataFrame(cont_rows).round(3)
    cont.to_csv(CONT_CSV, index=False)

    print("Neutral, no post-target drift assumption:")
    print(grid.to_string(index=False))
    print("\nContinuation-edge sensitivity (10/30 and 12/30):")
    print(cont.to_string(index=False))

    # Reference rows from the stop-only analysis.
    current = (df["replay_result_pts"].mean() - COMMISSION_PTS)
    stop30 = df.copy()
    stop30["replay_result_pts"] = np.where(
        stop30["mae_pts"] > 30,
        -30.0,
        np.where(stop30["outcome"] == "winner", stop30["mfe_pts"], -stop30["mae_pts"]),
    )
    stop30_net = stop30["replay_result_pts"].mean() - COMMISSION_PTS
    print(f"\nReference current replay: {current:.3f} pts/trade")
    print(f"Reference 6.6/30 stop-only: {stop30_net:.3f} pts/trade")
    print(f"\nWrote:\n  {GRID_CSV}\n  {CONT_CSV}")


if __name__ == "__main__":
    main()
