"""Scheduled paper runner for the UltimateORB proxy.

This is a paper-only proxy. It is NOT the original UltimateORB strategy: the
original entry/exit rules and trade CSV are still missing. It is a simple
opening-range breakout on NQ futures, using the 6.6-point target / 10-point
stop settings from the UltimateORB research.

It only writes local JSON/CSV state. There is no broker adapter and no live
order path.

Schedule-friendly behavior:
* one paper trade per US cash session;
* checks the halt flag before opening anything;
* still marks/exits an open position while halted;
* exits at the cash-session close.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .paper_engine import PaperEngine

CT = ZoneInfo("America/Chicago")
BAR_MINUTES = 5
SESSION_CLOSE = time(14, 45)


def _load_nq() -> pd.DataFrame:
    raw = yf.download(
        "NQ=F",
        period="3d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        prepost=False,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].copy().dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df.index = df.index.tz_convert(CT)
    return df


def _completed_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = now - timedelta(minutes=BAR_MINUTES)
    return df[df.index <= cutoff]


def _decision(engine: PaperEngine, now: datetime, dry_run: bool) -> dict:
    result = {
        "time_ct": now.isoformat(),
        "halted": engine.is_halted(),
        "action": "none",
        "reason": "",
        "trade": None,
    }

    if now.weekday() >= 5:
        result["reason"] = "weekend"
        return result

    raw = _load_nq()
    bars = _completed_bars(raw, now)
    if bars.empty:
        result["reason"] = "no data"
        return result

    session = now.date()
    day = bars[bars.index.date == session]
    if len(day) < 2:
        result["reason"] = "not enough session bars"
        return result

    # Opening range: first 30 minutes (08:30 and 08:45 CT bars).
    opening = day[
        (day.index.time >= time(8, 30)) & (day.index.time < time(9, 0))
    ]
    if len(opening) < 2:
        result["reason"] = "opening range incomplete"
        return result
    or_high = float(opening["High"].max())
    or_low = float(opening["Low"].min())
    latest = day.iloc[-1]
    latest_close = float(latest["Close"])
    result["opening_range"] = [or_high, or_low]
    result["latest_bar"] = str(latest.name)
    result["latest_close"] = latest_close

    # Manage an existing position first; halt must not prevent exits.
    if engine.state["positions"]:
        position = engine.state["positions"][0]
        entry_time = pd.Timestamp(position["opened_at"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        entry_time = entry_time.tz_convert(CT)
        after_entry = day[day.index > entry_time]
        for ts, bar in after_entry.iterrows():
            engine.mark(
                high=float(bar["High"]),
                low=float(bar["Low"]),
                close=float(bar["Close"]),
                timestamp=ts.isoformat(),
            )
            if not engine.state["positions"]:
                break
        if engine.state["positions"] and now.time() >= SESSION_CLOSE:
            closed = engine.close_manual(
                latest_close,
                reason="session close",
                timestamp=latest.name.isoformat(),
            )
            result["action"] = "time_exit"
            result["trade"] = closed
        else:
            result["action"] = "mark"
            result["position"] = engine.state["positions"][0] if engine.state["positions"] else None
        result["reason"] = "managed position"
        return result

    if engine.is_halted():
        result["reason"] = "halted: new entries blocked"
        return result

    if not (time(9, 0) <= now.time() <= time(14, 40)):
        result["reason"] = "outside entry window"
        return result

    if engine.state.get("last_entry_session") == str(session):
        result["reason"] = "already traded this session"
        return result

    if latest_close > or_high:
        direction = "long"
    elif latest_close < or_low:
        direction = "short"
    else:
        result["reason"] = "inside opening range"
        return result

    if dry_run:
        result["action"] = "would_open"
        result["direction"] = direction
        result["entry"] = latest_close
        result["reason"] = "dry run"
        return result

    position = engine.open_trade(
        direction=direction,
        entry_price=latest_close,
        timestamp=latest.name.isoformat(),
    )
    engine.state["last_entry_session"] = str(session)
    engine._save_state()
    result["action"] = "open"
    result["trade"] = position
    result["reason"] = "opening-range breakout"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="UltimateORB scheduled paper runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base", default=None)
    parser.add_argument("--now", default=None, help="ISO timestamp, for local testing only")
    args = parser.parse_args()

    engine = PaperEngine(base_dir=args.base)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(CT)
    if now.tzinfo is None:
        now = now.replace(tzinfo=CT)
    now = now.astimezone(CT)
    result = _decision(engine, now, args.dry_run)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
