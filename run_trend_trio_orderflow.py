"""Test Trend Trio as a confirmation filter on the crypto LVN-FVG strategy.

Base strategy
-------------
`multi_tf_strategy.py`: 10m LTF, 1D -> 12H -> 8H -> 4H -> 2H -> 1H -> 30m
hierarchy, LVN-aligned FVG/IFVG zones, orderflow confirmation, 2R target,
2.5 x 10m-ATR minimum stop, 0.5% risk, 5 bps fee + 1 bp slippage per side.

Trend Trio signal
-----------------
The deployed Trend Trio rules are rebuilt on the same crypto 1-minute data.
Only the direction is used here: the base orderflow entry is allowed only when
the Trend Trio signal agrees. Two definitions are tested:

  active : a Trend Trio trade is open in the same direction at the base entry
  latest : the most recent Trend Trio entry is in the same direction

Paper/research only. No broker, API key, or order path.
"""
from __future__ import annotations

import argparse
import copy
import json
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace
from datetime import time

import numpy as np
import pandas as pd

from multi_tf_strategy import build_zones, load_footprint, simulate
from lvn_fvg_strategy import Config
from trend_trio_paper.backtest import run_backtest as run_trend_trio
from trend_trio_paper.signals import normalize_1m
import trend_trio_paper.signals as tt_signals

START = "2025-11-11"
END = "2026-10-31"
SYMS = [("ETHUSDT", 1.0), ("SOLUSDT", 0.05), ("XRPUSDT", 0.0001)]
DEFAULT_TFS = "1d,12h,8h,4h,2h,1h,30m"
LTF = "10m"
START_EQUITY = 10_000.0
OUT = Path("data") / "results" / "trend_trio_orderflow_backtest.json"


def load_1m(root: Path) -> pd.DataFrame:
    raw = pd.read_parquet(root / "bars_1m.parquet")
    return normalize_1m(raw)


def trend_trades(root: Path) -> list[dict]:
    df1 = load_1m(root)
    # Direction-only use: costs are irrelevant here.
    result = run_trend_trio(df1, point_value=1.0, fee_per_side=0.0)
    trades = []
    for t in result.get("trades", []):
        entry = pd.Timestamp(t["entry_time"])
        exit_ = pd.Timestamp(t["exit_time"])
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        if exit_.tzinfo is None:
            exit_ = exit_.tz_localize("UTC")
        trades.append({
            "entry_time": entry.tz_convert("UTC"),
            "exit_time": exit_.tz_convert("UTC"),
            "direction": 1 if t["direction"] == "long" else -1,
            "source": t["source"],
        })
    trades.sort(key=lambda x: x["entry_time"])
    return trades


def make_filter(trades: list[dict], mode: str):
    entries = [t["entry_time"] for t in trades]
    exits = [t["exit_time"] for t in trades]
    dirs = [t["direction"] for t in trades]

    if mode == "active":
        def keep(ts: pd.Timestamp, direction: int) -> bool:
            for e, x, d in zip(entries, exits, dirs):
                if e <= ts < x and d == direction:
                    return True
            return False
    elif mode == "latest":
        def keep(ts: pd.Timestamp, direction: int) -> bool:
            i = bisect_right(entries, ts) - 1
            return i >= 0 and dirs[i] == direction
    else:
        raise ValueError(f"unknown filter mode: {mode}")
    return keep


def trade_stats(trades: list[dict], label: str) -> dict:
    rows = [t for t in trades
            if isinstance(t.get("pnl"), (int, float)) and np.isfinite(t["pnl"])]
    if not rows:
        return {
            "label": label, "trades": 0, "gross": 0.0, "fees": 0.0,
            "net": 0.0, "return_pct": 0.0, "win_rate_pct": 0.0,
            "profit_factor": None, "avg_r": 0.0, "max_dd_pct": 0.0,
            "pnls": [],
        }
    pnl = np.array([float(t["pnl"]) for t in rows])
    gross = 0.0
    fees = 0.0
    for t in rows:
        d = 1 if t["direction"] == "long" else -1
        qty = float(t["qty"])
        gross += (float(t["exit"]) - float(t["entry"])) * d * qty
        fees += (float(t["entry"]) + float(t["exit"])) * qty * 6.0 / 1e4
    wins = pnl > 0
    losses = pnl < 0
    gp = float(pnl[wins].sum())
    gl = float(abs(pnl[losses].sum()))
    equity = START_EQUITY + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return {
        "label": label,
        "trades": len(rows),
        "gross": round(gross, 2),
        "fees": round(fees, 2),
        "net": round(float(pnl.sum()), 2),
        "return_pct": round(float(pnl.sum() / START_EQUITY * 100), 2),
        "win_rate_pct": round(float(wins.mean() * 100), 2),
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "avg_r": round(float(np.mean([t["r_multiple"] for t in rows])), 3),
        "max_dd_pct": round(float(dd.max() * 100), 2),
        "pnls": [float(x) for x in pnl],
    }


def run_symbol(sym: str, tick: float, tfs: list[str], trend_window: str) -> dict:
    root = Path("data") / f"{sym}_{START}_{END}"
    print(f"\n== {sym} ==", flush=True)
    cfg = Config(
        tf=LTF,
        profile_window=720,
        profile_tick=tick,
        min_score=12.0,
        min_div_stages=0,
        require_retest_div=False,
        confirm_from_retest=False,
        tp_mode="r",
        tp_r=2.0,
        min_stop_atr=2.5,
        time_stop_bars=96,
        confirm_window=5,
        fvg_max_age=500,
        use_footprint=True,
        use_heatmap=True,
        starting_equity=START_EQUITY,
    )
    args = SimpleNamespace(
        nested_mode="impulse", nested_overlap=0.5, nested_time_pad=1.0,
        max_zone_age_days=30,
    )
    print("building base zones ...", flush=True)
    ltf_df, zones, _ = build_zones(str(root), tfs, LTF, cfg, args)
    fp_map = load_footprint(str(root), LTF)
    print(f"footprint bars: {len(fp_map)}", flush=True)

    print("building Trend Trio direction timeline ...", flush=True)
    if trend_window == "all":
        tt_signals._WINDOW = {k: (time(0, 0), time(23, 59)) for k in tt_signals._WINDOW}
    tt = trend_trades(root)
    print(f"Trend Trio trades: {len(tt)}", flush=True)

    base = simulate(ltf_df, copy.deepcopy(zones), tfs, cfg, args, fp_map)
    active = simulate(ltf_df, copy.deepcopy(zones), tfs, cfg, args, fp_map,
                      entry_filter=make_filter(tt, "active"))
    latest = simulate(ltf_df, copy.deepcopy(zones), tfs, cfg, args, fp_map,
                      entry_filter=make_filter(tt, "latest"))

    return {
        "symbol": sym,
        "trend_trio_trades": len(tt),
        "base": trade_stats(base.get("trades_detail", []), "Base orderflow"),
        "filter_active": trade_stats(active.get("trades_detail", []),
                                     "Base + Trend Trio active"),
        "filter_latest": trade_stats(latest.get("trades_detail", []),
                                     "Base + Trend Trio latest"),
        "base_funnel": base.get("funnel", {}),
        "filtered_funnel": latest.get("funnel", {}),
    }


def aggregate(rows: list[dict], key: str) -> dict:
    all_trades = []
    for row in rows:
        all_trades.append(row[key])
    n = sum(x["trades"] for x in all_trades)
    gross = sum(x["gross"] for x in all_trades)
    fees = sum(x["fees"] for x in all_trades)
    net = sum(x["net"] for x in all_trades)
    pnls = [p for x in all_trades for p in x.get("pnls", [])]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    return {
        "trades": n,
        "gross": round(gross, 2),
        "fees": round(fees, 2),
        "net": round(net, 2),
        "return_pct": round(net / (START_EQUITY * len(rows)) * 100, 2),
        "avg_symbol_return_pct": round(
            float(np.mean([x["return_pct"] for x in all_trades])), 2),
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "worst_symbol_dd_pct": round(
            max(x["max_dd_pct"] for x in all_trades), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfs", default=DEFAULT_TFS)
    parser.add_argument("--trend-window", default="ny", choices=["ny", "all"])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]

    rows = [run_symbol(sym, tick, tfs, args.trend_window) for sym, tick in SYMS]
    payload = {
        "paper_only": True,
        "assumptions": {
            "period": f"{START} -> {END}",
            "symbols": [s for s, _ in SYMS],
            "base": "multi_tf_strategy.py LVN-FVG orderflow",
            "hierarchy": tfs,
            "ltf": LTF,
            "trend_trio": f"deployed rules, direction-only, {args.trend_window} session windows",
            "sizing": "$10,000 per symbol, 0.5% risk, 5 bps fee + 1 bp slip/side",
            "note": "3M/1M are not available in the one-year crypto dataset; "
                    "the test starts at 1D.",
        },
        "symbols": {r["symbol"]: r for r in rows},
        "aggregate": {
            "base": aggregate(rows, "base"),
            "filter_active": aggregate(rows, "filter_active"),
            "filter_latest": aggregate(rows, "filter_latest"),
        },
    }
    for r in rows:
        for key in ("base", "filter_active", "filter_latest"):
            r[key].pop("pnls", None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== RESULTS ===")
    print(f"{'symbol':9} {'case':14} {'n':>4} {'gross':>9} {'fees':>7} "
          f"{'net':>9} {'ret%':>7} {'PF':>6} {'DD%':>6}")
    for r in rows:
        for key in ("base", "filter_active", "filter_latest"):
            s = r[key]
            print(f"{r['symbol']:9} {key:14} {s['trades']:>4} "
                  f"{s['gross']:>9.0f} {s['fees']:>7.0f} {s['net']:>9.0f} "
                  f"{s['return_pct']:>7.2f} {str(s['profit_factor']):>6} "
                  f"{s['max_dd_pct']:>6.2f}")
    print("\n=== AGGREGATE (equal-weight, 3 x $10k) ===")
    for key in ("base", "filter_active", "filter_latest"):
        s = payload["aggregate"][key]
        print(f"{key:14} n={s['trades']:>4} gross={s['gross']:>9.0f} "
              f"fees={s['fees']:>7.0f} net={s['net']:>9.0f} "
              f"return={s['return_pct']:>7.2f}% PF={s['profit_factor']} "
              f"worstDD={s['worst_symbol_dd_pct']:.2f}%")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
