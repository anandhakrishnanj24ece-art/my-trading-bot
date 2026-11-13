"""Backtest Trend Trio as an extra signal stream inside FVG Duo and UltimateORB.

Deployment interpretation
-------------------------
* Trend Trio keeps exactly the rules used by the paper runner.
* The base strategy keeps its own rules.
* One position at a time, one $50,000 paper account.
* If both want to enter at the same time, the base strategy wins the tie.
* A signal whose entry occurs while the account is already in a trade is
  skipped.  This is a portfolio overlay, not a rewrite of either strategy's
  internal state machine.

Costs
-----
* Point value: $2 per index point (1 MNQ-equivalent).
* Trend Trio: $2 per side.
* FVG Duo: $2 per side plus 0.25 point entry slippage.
* UltimateORB proxy: 0.95 points round-turn commission.

The backtest is chronological and uses the same 2023-01-03 -> 2025-12-11
NQ 1-minute history.  It is research only; no live order path exists.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from run_fvg_duo_multi import TZ as FVG_TZ
from run_fvg_duo_multi import build_signals as build_fvg_signals
from run_fvg_duo_multi import load_nq_1m as load_nq_5m
from run_fvg_duo_multi import run as run_fvg
from trend_trio_paper.backtest import load_nq_1m as load_nq_1m
from trend_trio_paper.backtest import run_backtest as run_trend_trio

START = pd.Timestamp("2023-01-03", tz=FVG_TZ)
END = pd.Timestamp("2025-12-11", tz=FVG_TZ)
START_CAPITAL = 50_000.0
POINT_VALUE = 2.0
OUT_DIR = Path("data") / "results"
OUT_JSON = OUT_DIR / "trend_trio_combo_backtest.json"
OUT_TRADES = OUT_DIR / "trend_trio_combo_trades.csv"

TREND_COST_PER_SIDE = 2.0
FVG_COST_PER_SIDE = 2.0
FVG_SLIP_POINTS = 0.25
ORB_COMMISSION_POINTS = 0.95


def _to_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _in_period(trade: dict, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    t = _to_utc(trade["entry_time"])
    ny = t.tz_convert(FVG_TZ)
    return start <= ny < end


def _record(
    source: str,
    direction: str,
    entry_time,
    exit_time,
    entry: float,
    exit: float,
    point_value: float,
    net_usd: float,
    reason: str,
) -> dict:
    d = 1 if direction == "long" else -1
    gross_usd = (float(exit) - float(entry)) * d * point_value
    return {
        "source": source,
        "direction": direction,
        "entry_time": _to_utc(entry_time),
        "exit_time": _to_utc(exit_time),
        "entry": float(entry),
        "exit": float(exit),
        "gross_usd": round(float(gross_usd), 2),
        "net_usd": round(float(net_usd), 2),
        "reason": reason,
    }


def fvg_trades() -> list[dict]:
    df5 = load_nq_5m()
    sig = build_fvg_signals(df5, tick=0.25)
    raw = run_fvg(
        df5,
        "duo",
        point_value=POINT_VALUE,
        fee=FVG_COST_PER_SIDE,
        slip=FVG_SLIP_POINTS,
        tick=0.25,
        signals=sig,
    )
    out = []
    for t in raw:
        if not _in_period(t, START, END):
            continue
        out.append(_record(
            "fvg_duo",
            t["dir"],
            t["entry_time"],
            t["exit_time"],
            t["entry"],
            t["exit"],
            POINT_VALUE,
            t["pnl"],
            t["reason"],
        ))
    return out


def trend_trio_trades() -> list[dict]:
    df1 = load_nq_1m(r"D:\diamondhead_data\Dataset_NQ_1min_2022_2025.csv")
    result = run_trend_trio(
        df1,
        point_value=POINT_VALUE,
        fee_per_side=TREND_COST_PER_SIDE,
    )
    out = []
    for t in result["trades"]:
        if not _in_period(t, START, END):
            continue
        out.append(_record(
            "trend_trio",
            t["direction"],
            t["entry_time"],
            t["exit_time"],
            t["entry_price"],
            t["exit_price"],
            POINT_VALUE,
            t["net_usd"],
            t["reason"],
        ))
    return out


def ultimate_orb_trades(df5: pd.DataFrame) -> list[dict]:
    """Proxy used by ultimateorb_paper: 30m opening range, 6.6/10, 1 trade/day."""
    out: list[dict] = []
    for day, bars in df5.groupby("day", sort=True):
        opening = bars[(bars["hm"] >= "09:30") & (bars["hm"] < "10:00")]
        if len(opening) < 6:
            continue
        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        entry_window = bars[(bars["hm"] >= "10:00") & (bars["hm"] <= "15:35")]
        if entry_window.empty:
            continue

        pos = None
        traded = False
        for ts, bar in entry_window.iterrows():
            if pos is not None:
                d = pos["dir"]
                stop = pos["stop"]
                target = pos["target"]
                stop_hit = bar["low"] <= stop if d == 1 else bar["high"] >= stop
                target_hit = bar["high"] >= target if d == 1 else bar["low"] <= target
                exit_price = None
                reason = ""
                if stop_hit:
                    exit_price, reason = stop, "stop"
                elif target_hit:
                    exit_price, reason = target, "target"
                elif bar["hm"] >= "15:40":
                    exit_price, reason = float(bar["close"]), "session_close"

                if exit_price is not None:
                    gross_pts = (exit_price - pos["entry"]) * d
                    net_pts = gross_pts - ORB_COMMISSION_POINTS
                    out.append(_record(
                        "ultimate_orb",
                        "long" if d == 1 else "short",
                        pos["entry_time"],
                        ts,
                        pos["entry"],
                        exit_price,
                        POINT_VALUE,
                        net_pts * POINT_VALUE,
                        reason,
                    ))
                    pos = None
                    break
                continue

            if traded:
                break
            close = float(bar["close"])
            if close > or_high:
                d = 1
            elif close < or_low:
                d = -1
            else:
                continue
            entry = close
            pos = {
                "dir": d,
                "entry": entry,
                "entry_time": ts,
                "stop": entry - d * 10.0,
                "target": entry + d * 6.6,
            }
            traded = True

        if pos is not None:
            # Should not happen because the session-close branch above exits,
            # but keep the ledger complete if a day is truncated.
            last_ts = entry_window.index[-1]
            last_close = float(entry_window.iloc[-1]["close"])
            d = pos["dir"]
            gross_pts = (last_close - pos["entry"]) * d
            net_pts = gross_pts - ORB_COMMISSION_POINTS
            out.append(_record(
                "ultimate_orb",
                "long" if d == 1 else "short",
                pos["entry_time"],
                last_ts,
                pos["entry"],
                last_close,
                POINT_VALUE,
                net_pts * POINT_VALUE,
                "end_of_session",
            ))

    return [t for t in out if _in_period(t, START, END)]


def overlay(base: list[dict], trend: list[dict]) -> tuple[list[dict], int]:
    """Chronological one-position overlay. Base wins exact-time ties."""
    events = [(t["entry_time"], 0, t) for t in base]
    events += [(t["entry_time"], 1, t) for t in trend]
    events.sort(key=lambda x: (x[0], x[1]))
    taken: list[dict] = []
    free_at: pd.Timestamp | None = None
    skipped = 0
    for entry_time, _, trade in events:
        if free_at is not None and entry_time < free_at:
            skipped += 1
            continue
        taken.append(trade)
        free_at = trade["exit_time"]
    return taken, skipped


def filter_open(base: list[dict], trend: list[dict]) -> list[dict]:
    """Keep base entries only while a same-direction Trend Trio trade is open."""
    out = []
    for b in base:
        t0 = b["entry_time"]
        keep = False
        for t in trend:
            if t["entry_time"] <= t0 < t["exit_time"] and t["direction"] == b["direction"]:
                keep = True
                break
        if keep:
            out.append(b)
    return out


def filter_last(base: list[dict], trend: list[dict]) -> list[dict]:
    """Keep base entries only when the latest Trend Trio entry agrees."""
    ordered = sorted(trend, key=lambda t: t["entry_time"])
    times = [t["entry_time"] for t in ordered]
    out = []
    for b in base:
        i = int(np.searchsorted(times, b["entry_time"], side="right")) - 1
        if i >= 0 and ordered[i]["direction"] == b["direction"]:
            out.append(b)
    return out


def stats(trades: list[dict], label: str) -> dict:
    if not trades:
        return {
            "label": label,
            "trades": 0,
            "gross_usd": 0.0,
            "net_usd": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_trade_usd": 0.0,
        }
    net = np.array([t["net_usd"] for t in trades], dtype=float)
    gross = np.array([t["gross_usd"] for t in trades], dtype=float)
    wins = net > 0
    losses = net < 0
    gp = float(net[wins].sum())
    gl = float(abs(net[losses].sum()))
    equity = START_CAPITAL + np.cumsum(net)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(dd.max()) if len(dd) else 0.0
    max_dd_pct = float((dd / peak).max() * 100) if len(dd) else 0.0
    return {
        "label": label,
        "trades": len(trades),
        "gross_usd": round(float(gross.sum()), 2),
        "net_usd": round(float(net.sum()), 2),
        "return_pct": round(float(net.sum() / START_CAPITAL * 100), 2),
        "win_rate_pct": round(float(wins.mean() * 100), 2),
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_trade_usd": round(float(net.mean()), 2),
    }


def by_year(trades: list[dict]) -> dict:
    buckets: dict[int, list[dict]] = {}
    for t in trades:
        year = _to_utc(t["entry_time"]).tz_convert(FVG_TZ).year
        buckets.setdefault(year, []).append(t)
    out = {}
    for year, rows in sorted(buckets.items()):
        s = stats(rows, str(year))
        out[str(year)] = {
            "trades": s["trades"],
            "gross_usd": s["gross_usd"],
            "net_usd": s["net_usd"],
            "return_pct": s["return_pct"],
            "profit_factor": s["profit_factor"],
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading NQ 1m and building FVG Duo trades ...")
    fvg = fvg_trades()
    print(f"FVG Duo: {len(fvg)} trades")

    print("building Trend Trio trades ...")
    trend = trend_trio_trades()
    print(f"Trend Trio: {len(trend)} trades")

    print("building UltimateORB proxy trades ...")
    orb = ultimate_orb_trades(load_nq_5m())
    print(f"UltimateORB: {len(orb)} trades")

    fvg_combo, fvg_skipped = overlay(fvg, trend)
    orb_combo, orb_skipped = overlay(orb, trend)

    fvg_filter_open = filter_open(fvg, trend)
    fvg_filter_last = filter_last(fvg, trend)
    orb_filter_open = filter_open(orb, trend)
    orb_filter_last = filter_last(orb, trend)

    payload = {
        "paper_only": True,
        "assumptions": {
            "period": f"{START.date()} -> {END.date()}",
            "starting_capital_usd": START_CAPITAL,
            "point_value_usd": POINT_VALUE,
            "contract": "1 MNQ-equivalent",
            "trend_trio_fee_per_side_usd": TREND_COST_PER_SIDE,
            "fvg_fee_per_side_usd": FVG_COST_PER_SIDE,
            "fvg_entry_slippage_points": FVG_SLIP_POINTS,
            "ultimate_orb_commission_points": ORB_COMMISSION_POINTS,
            "combination": "one position at a time; base wins exact-time ties; "
                           "overlapping signals skipped",
            "trend_trio_rules": "exactly the deployed paper runner rules",
        },
        "results": {
            "fvg_duo": {
                "base": stats(fvg, "FVG Duo only"),
                "trend_trio": stats(trend, "Trend Trio only"),
                "combined": stats(fvg_combo, "FVG Duo + Trend Trio"),
                "filter_open": stats(fvg_filter_open, "FVG Duo, Trend open agrees"),
                "filter_last": stats(fvg_filter_last, "FVG Duo, latest Trend agrees"),
                "skipped_overlaps": fvg_skipped,
                "by_year": by_year(fvg_combo),
            },
            "ultimate_orb": {
                "base": stats(orb, "UltimateORB only"),
                "trend_trio": stats(trend, "Trend Trio only"),
                "combined": stats(orb_combo, "UltimateORB + Trend Trio"),
                "filter_open": stats(orb_filter_open, "UltimateORB, Trend open agrees"),
                "filter_last": stats(orb_filter_last, "UltimateORB, latest Trend agrees"),
                "skipped_overlaps": orb_skipped,
                "by_year": by_year(orb_combo),
            },
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = []
    for name, combo in (("fvg_duo", fvg_combo), ("ultimate_orb", orb_combo)):
        for t in combo:
            row = dict(t)
            row["portfolio"] = name
            row["entry_time"] = _to_utc(row["entry_time"]).isoformat()
            row["exit_time"] = _to_utc(row["exit_time"]).isoformat()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_TRADES, index=False)

    for name, key in (("FVG Duo", "fvg_duo"), ("UltimateORB", "ultimate_orb")):
        block = payload["results"][key]
        print(f"\n=== {name} ===")
        for case in ("base", "trend_trio", "combined", "filter_open", "filter_last"):
            s = block[case]
            print(f"{s['label']:32} n={s['trades']:5d}  "
                  f"gross=${s['gross_usd']:>10,.0f}  net=${s['net_usd']:>10,.0f}  "
                  f"return={s['return_pct']:>7.2f}%  PF={s['profit_factor']}  "
                  f"maxDD={s['max_drawdown_pct']:.2f}%")
        print(f"skipped overlapping entries: {block['skipped_overlaps']}")

    print(f"\nsaved {OUT_JSON}")
    print(f"saved {OUT_TRADES}")


if __name__ == "__main__":
    main()
