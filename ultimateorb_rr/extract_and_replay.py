"""
Extract UltimateORB excursion points from the PDF report and replay fixed-stop
counterfactuals.

Why this exists
---------------
The report references `UltimateORB_NT_2026-11-07.csv`, but that trade file is not
on disk. Page 3 of the PDF contains a vector scatter of every trade's MAE/MFE,
so we can recover the excursion pairs and replay stop changes without pretending
we have the original trade list.

What can be estimated
---------------------
* Tightening the stop, with the strategy's target rule unchanged.
* Commission and stop-slippage stress.
* Break-even win rates for target changes.

What cannot be estimated
------------------------
* Raising the target. Winners' MFE is censored at the original target (100% MFE
  capture), so the PDF does not tell us how much further a winner would have run.
* Chronological max drawdown, because the scatter has no trade order.
* Exact intrabar fill sequencing, latency, gaps through stops, or contract rolls.

The fixed-stop replay is therefore evidence, not a substitute for the trade CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
PDF = ROOT.parent / "UltimateORB-Backtest-Report.pdf"
OUT_CSV = ROOT / "ultimateorb_excursions.csv"
STOP_CSV = ROOT / "stop_grid.csv"
SLIP_CSV = ROOT / "stop_slippage.csv"
BE_CSV = ROOT / "break_even.csv"

COMMISSION_PTS = 0.95  # $1.90 round turn / $2 per point per MNQ
CONTRACT_MULT = 2.0     # $ per point per MNQ
CURRENT_STOP = 60.7     # report's average stop-loss exit, points/contract
CURRENT_WIN = 95.2      # report's headline win rate, percent

GREEN = (0.18, 0.49, 0.357)
RED = (0.698, 0.227, 0.282)


def extract_excursions(pdf_path: Path = PDF) -> pd.DataFrame:
    """Read page 3's scatter markers and map them to MAE/MFE in points."""
    doc = fitz.open(pdf_path)
    page = doc[2]
    green: list[tuple[float, float]] = []
    red: list[tuple[float, float]] = []

    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        rect = drawing["rect"]
        width = rect.x1 - rect.x0
        height = rect.y1 - rect.y0
        if not fill or not (1.5 <= width <= 2.5 and 1.5 <= height <= 2.5):
            continue

        fill = tuple(round(channel, 3) for channel in fill)
        x = (rect.x0 + rect.x1) / 2
        y = (rect.y0 + rect.y1) / 2

        # Legend markers sit at a fixed x and are not trade points.
        if abs(x - 247.8) < 0.5 and (abs(y - 471.4) < 0.5 or abs(y - 481.4) < 0.5):
            continue

        # Keep only the data area of the excursion chart.
        if not (62.5 <= x <= 281.0 and 480.0 <= y <= 577.5):
            continue

        # Axis calibration from the chart ticks:
        # x: 64.4 -> 0 pts, 94.7 -> 20 pts; y: 575.4 -> 0 pts, 550.2 -> 10 pts.
        mae = max(0.0, (x - 64.4) / (94.7 - 64.4) * 20.0)
        mfe = max(0.0, (575.4 - y) / (575.4 - 550.2) * 10.0)

        if fill == GREEN:
            green.append((mae, mfe))
        elif fill == RED:
            red.append((mae, mfe))

    if not (1_400 <= len(green) <= 1_500 and 60 <= len(red) <= 90):
        raise RuntimeError(f"Unexpected marker counts: {len(green)} green, {len(red)} red")

    rows = []
    for outcome, points in (("winner", green), ("loser", red)):
        for mae, mfe in points:
            rows.append(
                {
                    "outcome": outcome,
                    "mae_pts": mae,
                    "mfe_pts": mfe,
                    # Winners captured 100% of MFE; losers are replayed at -MAE.
                    "replay_result_pts": mfe if outcome == "winner" else -mae,
                }
            )

    df = pd.DataFrame(rows)
    df.insert(0, "trade_id", np.arange(1, len(df) + 1))
    df = df.round({"mae_pts": 3, "mfe_pts": 3, "replay_result_pts": 3})
    df.to_csv(OUT_CSV, index=False)
    return df


def replay_stop(
    df: pd.DataFrame,
    stop_pts: float,
    stop_slip_pts: float = 0.0,
    commission_pts: float = COMMISSION_PTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gross P&L, net P&L, and stop-trigger mask for a fixed stop."""
    winners = df["outcome"].to_numpy() == "winner"
    mae = df["mae_pts"].to_numpy()
    mfe = df["mfe_pts"].to_numpy()
    stopped = mae > stop_pts

    gross = np.where(
        stopped,
        -(stop_pts + stop_slip_pts),
        np.where(winners, mfe, -mae),
    )
    net = gross - commission_pts
    return gross, net, stopped


def build_stop_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stop in [5, 10, 15, 20, 25, 30, 40, 50, CURRENT_STOP, 80, 100, 120]:
        gross, net, stopped = replay_stop(df, stop)
        win = net > 0
        avg_win = gross[win].mean()
        avg_loss = gross[~win].mean()
        pf = net[win].sum() / abs(net[~win].sum())
        se = net.std(ddof=1) / np.sqrt(len(net))
        rows.append(
            {
                "stop_pts": stop,
                "rr_avg_win_over_avg_loss": avg_win / abs(avg_loss),
                "win_rate_pct": win.mean() * 100,
                "avg_win_pts": avg_win,
                "avg_loss_pts": avg_loss,
                "avg_net_pts": net.mean(),
                "profit_factor_net": pf,
                "total_net_usd_1_mnq": net.mean() * len(net) * CONTRACT_MULT,
                "net_after_3pt_uniform_slip_pts": net.mean() - 3.0,
                "stop_rate_pct": stopped.mean() * 100,
                "worst_trade_pts": gross.min(),
                "ci95_low_pts": net.mean() - 1.96 * se,
                "ci95_high_pts": net.mean() + 1.96 * se,
            }
        )
    result = pd.DataFrame(rows).round(3)
    result.to_csv(STOP_CSV, index=False)
    return result


def build_slippage_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stop in [5, 10, 20, 30, CURRENT_STOP]:
        for slip in [0, 1, 2, 5, 10, 20, 30]:
            _, net, _ = replay_stop(df, stop, stop_slip_pts=slip)
            rows.append(
                {
                    "stop_pts": stop,
                    "stop_slip_pts": slip,
                    "avg_net_pts": net.mean(),
                    "total_net_usd_1_mnq": net.mean() * len(net) * CONTRACT_MULT,
                }
            )
    result = pd.DataFrame(rows).round(3)
    result.to_csv(SLIP_CSV, index=False)
    return result


def build_break_even(df: pd.DataFrame, stop_grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = 6.6  # report's average profit target
    for stop in [5, 10, 20, 30, 40, 50, CURRENT_STOP]:
        break_even = (stop + COMMISSION_PTS) / (stop + target)
        est_win = float(stop_grid.loc[stop_grid["stop_pts"] == stop, "win_rate_pct"].iloc[0])
        rows.append(
            {
                "kind": "tighten_stop",
                "target_pts": target,
                "stop_pts": stop,
                "break_even_win_pct": break_even * 100,
                "estimated_win_pct": est_win,
                "win_buffer_pct": est_win - break_even * 100,
            }
        )

    for target in [8, 10, 12, 14, 20]:
        break_even = (CURRENT_STOP + COMMISSION_PTS) / (CURRENT_STOP + target)
        rows.append(
            {
                "kind": "widen_target_unestimable",
                "target_pts": target,
                "stop_pts": CURRENT_STOP,
                "break_even_win_pct": break_even * 100,
                "estimated_win_pct": np.nan,
                "win_buffer_pct": CURRENT_WIN - break_even * 100,
            }
        )

    result = pd.DataFrame(rows).round(3)
    result.to_csv(BE_CSV, index=False)
    return result


def main() -> None:
    df = extract_excursions()
    stop_grid = build_stop_grid(df)
    slippage = build_slippage_grid(df)
    break_even = build_break_even(df, stop_grid)

    n = len(df)
    winners = (df["outcome"] == "winner").sum()
    losers = n - winners
    gross_per_trade = df["replay_result_pts"].mean()
    net_per_trade = gross_per_trade - COMMISSION_PTS

    print(f"Extracted {n} trades: {winners} winners, {losers} losers")
    print(f"Winner MFE mean: {df.loc[df.outcome == 'winner', 'mfe_pts'].mean():.2f} pts (report 6.6)")
    print(f"Loser MAE mean: {df.loc[df.outcome == 'loser', 'mae_pts'].mean():.2f} pts (report 54)")
    print(f"Replay gross: {gross_per_trade:.2f} pts/trade (report 3.8)")
    print(f"Replay net after commission: {net_per_trade:.2f} pts/trade (report 2.88)")
    print("\nFixed-stop grid:")
    print(stop_grid.to_string(index=False))
    print("\nStop-slippage grid (avg net points):")
    pivot = slippage.pivot(index="stop_pts", columns="stop_slip_pts", values="avg_net_pts")
    print(pivot.to_string())
    print("\nBreak-even table:")
    print(break_even.to_string(index=False))
    print(f"\nWrote:\n  {OUT_CSV}\n  {STOP_CSV}\n  {SLIP_CSV}\n  {BE_CSV}")


if __name__ == "__main__":
    main()
