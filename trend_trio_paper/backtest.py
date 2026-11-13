"""Research backtest for the video-derived Trend Trio rules.

This is intentionally separate from the paper runner.  It uses the same
signal functions, fills at the next 1-minute bar open, and applies the paper
cost model.  It is not a claim that the video's 16-year result reproduces.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .signals import (
    TZ, Entry, Exit, PRIORITY, build_entries, build_exits, normalize_1m,
)

STARTING_CAPITAL = 50_000.0
POINT_VALUE = 2.0       # 1 MNQ-equivalent
FEE_PER_SIDE = 2.0      # USD per side


@dataclass
class Trade:
    source: str
    direction: str
    entry_time: str
    signal_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_price: float | None
    gross_points: float
    net_points: float
    gross_usd: float
    net_usd: float
    risk_points: float
    r_gross: float | None
    r_net: float | None
    reason: str


def load_nq_1m(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(
        p,
        usecols=["timestamp ET", "open", "high", "low", "close", "volume"],
    )
    ts = pd.to_datetime(df["timestamp ET"], format="%m/%d/%Y %H:%M")
    idx = ts.dt.tz_localize(TZ).dt.tz_convert("UTC")
    df.index = idx
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def _summarise(trades: List[Trade], point_value: float) -> dict:
    if not trades:
        return {"trades": 0}
    net = np.array([t.net_usd for t in trades], dtype=float)
    gross = np.array([t.gross_usd for t in trades], dtype=float)
    wins = net > 0
    losses = net < 0
    gp = float(net[wins].sum())
    gl = float(abs(net[losses].sum()))
    r_net = [t.r_net for t in trades if t.r_net is not None]
    return {
        "trades": len(trades),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "win_rate_pct": round(float(wins.mean() * 100), 2),
        "gross_usd": round(float(gross.sum()), 2),
        "net_usd": round(float(net.sum()), 2),
        "gross_points": round(sum(t.gross_points for t in trades), 2),
        "net_points": round(sum(t.net_points for t in trades), 2),
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "avg_r_net": round(float(np.mean(r_net)), 4) if r_net else None,
        "risk_trades": len(r_net),
    }


def run_backtest(df1: pd.DataFrame,
                 point_value: float = POINT_VALUE,
                 fee_per_side: float = FEE_PER_SIDE) -> dict:
    df1 = normalize_1m(df1)
    entries = build_entries(df1)
    exits = build_exits(df1)
    entry_map: Dict[pd.Timestamp, List[Entry]] = defaultdict(list)
    for e in entries:
        entry_map[e.signal_time].append(e)
    for ts in entry_map:
        entry_map[ts].sort(key=lambda e: PRIORITY.get(e.source, 9))
    exit_map: Dict[tuple, Exit] = {
        (e.signal_time, e.source, e.for_direction): e for e in exits
    }

    idx = df1.index
    o = df1["open"].to_numpy(dtype=float)
    h = df1["high"].to_numpy(dtype=float)
    l = df1["low"].to_numpy(dtype=float)
    c = df1["close"].to_numpy(dtype=float)

    cash = STARTING_CAPITAL
    peak = cash
    max_dd = 0.0
    max_dd_pct = 0.0
    curve: List[dict] = []
    trades: List[Trade] = []
    pos = None
    last_day = None

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal cash, pos, peak, max_dd, max_dd_pct
        p = pos
        d = p["direction"]
        pts = (price - p["entry_price"]) * d
        net_pts = pts - 2 * fee_per_side / point_value
        gross_usd = pts * point_value
        net_usd = gross_usd - 2 * fee_per_side
        risk = float(p["risk_points"])
        trades.append(Trade(
            source=p["source"],
            direction="long" if d == 1 else "short",
            entry_time=p["entry_time"].isoformat(),
            signal_time=p["signal_time"].isoformat(),
            exit_time=idx[i].isoformat(),
            entry_price=round(float(p["entry_price"]), 4),
            exit_price=round(float(price), 4),
            stop_price=(round(float(p["stop_price"]), 4)
                        if p.get("stop_price") is not None else None),
            gross_points=round(pts, 4),
            net_points=round(net_pts, 4),
            gross_usd=round(gross_usd, 2),
            net_usd=round(net_usd, 2),
            risk_points=round(risk, 4),
            r_gross=(round(pts / risk, 4) if risk > 1e-9 else None),
            r_net=(round(net_pts / risk, 4) if risk > 1e-9 else None),
            reason=reason,
        ))
        cash += net_usd
        equity = cash
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd / peak if peak > 0 else 0.0)
        pos = None

    for i, ts in enumerate(idx):
        bar_et = ts.tz_convert(TZ)
        hm = bar_et.time()
        weekday = bar_et.weekday()

        if pos is not None:
            d = pos["direction"]
            exit_price = None
            reason = ""

            if pos.get("stop_price") is not None:
                stop = float(pos["stop_price"])
                if d == 1 and l[i] <= stop:
                    exit_price = min(stop, o[i])
                    reason = "stop"
                elif d == -1 and h[i] >= stop:
                    exit_price = max(stop, o[i])
                    reason = "stop"

            if exit_price is None:
                ev = exit_map.get((ts, pos["source"], d))
                if ev is not None:
                    exit_price = c[i]
                    reason = ev.reason

            if exit_price is None:
                if pos["source"] == "st" and hm >= time(15, 59):
                    exit_price, reason = c[i], "eod"
                elif pos["source"] in ("at2", "at1h") and hm >= time(16, 0):
                    exit_price, reason = c[i], "eod"
                elif (pos["source"] == "uma" and weekday == 4
                      and hm >= time(16, 45)):
                    exit_price, reason = c[i], "weekend_flat"

            if exit_price is not None:
                close_position(i, exit_price, reason)
            else:
                equity = cash + (c[i] - pos["entry_price"]) * d * point_value
                peak = max(peak, equity)
                dd = peak - equity
                max_dd = max(max_dd, dd)
                max_dd_pct = max(max_dd_pct, dd / peak if peak > 0 else 0.0)

        else:
            cands = entry_map.get(ts, [])
            if cands:
                sig = cands[0]
                d = sig.direction
                entry = o[i]
                risk = float(sig.risk_points)
                stop = entry - d * risk if risk > 1e-9 else None
                pos = {
                    "source": sig.source,
                    "direction": d,
                    "entry_price": entry,
                    "stop_price": stop,
                    "risk_points": risk,
                    "signal_time": sig.signal_time,
                    "entry_time": ts,
                }
                equity = cash + (c[i] - entry) * d * point_value
                peak = max(peak, equity)
                dd = peak - equity
                max_dd = max(max_dd, dd)
                max_dd_pct = max(max_dd_pct, dd / peak if peak > 0 else 0.0)

        day = bar_et.date()
        if day != last_day:
            curve.append({"timestamp": ts.isoformat(),
                          "equity": round(float(cash if pos is None else (
                              cash + (c[i] - pos["entry_price"])
                              * pos["direction"] * point_value)), 2)})
            last_day = day

    if pos is not None:
        close_position(len(idx) - 1, c[-1], "end_of_data")
        curve.append({"timestamp": idx[-1].isoformat(),
                      "equity": round(float(cash), 2)})

    by_source = {}
    for source in ("st", "uma", "at2", "at1h"):
        subset = [t for t in trades if t.source == source]
        by_source[source] = _summarise(subset, point_value)

    years = {}
    for t in trades:
        year = pd.Timestamp(t.exit_time).tz_convert(TZ).year
        years.setdefault(year, []).append(t)
    by_year = {str(y): _summarise(v, point_value) for y, v in sorted(years.items())}

    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]

    return {
        "assumptions": {
            "point_value_usd": point_value,
            "fee_per_side_usd": fee_per_side,
            "starting_capital_usd": STARTING_CAPITAL,
            "fill": "next 1-minute bar open; stop at stop price or gap open",
            "indicator_defaults": {
                "supertrend": "ATR10 x3.0, RMA",
                "ultimate_ma": "SMA20, colour smoothing 2",
                "sling_shot": "EMA38/EMA62",
                "chandelier": "22 x3.0, closes",
                "alphatrend": "period 14, coefficient 1.0, MFI",
                "adx": "14, >=20",
                "ema_filter": "1h EMA200",
            },
        },
        "start": idx[0].isoformat() if len(idx) else None,
        "end": idx[-1].isoformat() if len(idx) else None,
        "total": _summarise(trades, point_value),
        "long": _summarise(long_trades, point_value),
        "short": _summarise(short_trades, point_value),
        "by_source": by_source,
        "by_year": by_year,
        "max_drawdown_usd": round(float(max_dd), 2),
        "max_drawdown_pct": round(float(max_dd_pct * 100), 2),
        "ending_equity_usd": round(float(cash), 2),
        "trades": [asdict(t) for t in trades],
        "equity_curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend Trio research backtest")
    parser.add_argument("--csv", default=r"D:\diamondhead_data\Dataset_NQ_1min_2022_2025.csv")
    parser.add_argument("--out", default="data/results/trend_trio_backtest.json")
    parser.add_argument("--point-value", type=float, default=POINT_VALUE)
    parser.add_argument("--fee-per-side", type=float, default=FEE_PER_SIDE)
    args = parser.parse_args()

    df = load_nq_1m(args.csv)
    result = run_backtest(df, point_value=args.point_value,
                          fee_per_side=args.fee_per_side)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    total = result["total"]
    print(f"bars={len(df):,}  {result['start']} -> {result['end']}")
    print(f"trades={total.get('trades', 0)}  net=${total.get('net_usd', 0):,.2f}"
          f"  gross=${total.get('gross_usd', 0):,.2f}"
          f"  PF={total.get('profit_factor')}"
          f"  win={total.get('win_rate_pct')}%")
    print(f"maxDD=${result['max_drawdown_usd']:,.2f} "
          f"({result['max_drawdown_pct']:.2f}%)  "
          f"ending=${result['ending_equity_usd']:,.2f}")
    for src, s in result["by_source"].items():
        print(f"  {src:4s} n={s.get('trades', 0):4d} "
              f"net=${s.get('net_usd', 0):,.2f} "
              f"PF={s.get('profit_factor')} "
              f"win={s.get('win_rate_pct')}%")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
