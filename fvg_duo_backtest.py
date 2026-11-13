"""FVG Duo (Opening Drive + Trend FVG) — faithful port from the video rules.

Source: "FVG Trading Strategy: Stop Waiting for the Retest (10 Years Tested)"
        Technical Trading, https://youtu.be/71QsMhpv0FI

Rules encoded (all confirmed from the transcript):
  Common: one NQ contract, one trade at a time, entries at next candle open,
          fees $2.14/side, optional 1 tick slippage, flat by 16:00 NY.

  Opening Drive (OD):
    - 5m execution, FVG read on 10m candles.
    - At 09:40 NY check the 10m candles 09:10/09:20/09:30; if no FVG,
      at 09:50 check 09:20/09:30/09:40.
    - Bull FVG: third low > first high.  Bear FVG: third high < first low.
    - Enter at next 5m open; stop 1.5% from entry; no target; exit 16:00.
    - At most one OD trade per day, and once it trades no Trend entry that day.

  Trend FVG:
    - 5m execution, FVG / EMA / ATR from 15m candles.
    - Long only; bullish FVG only.
    - 15m EMA50 > EMA100.
    - Middle candle range >= 1.5 x 15m ATR14 (ATR at the middle candle).
    - Entry at next 5m open, no retest wait, entries 09:30-15:30.
    - Stop 1 tick below the first FVG candle low; target = 3R.
    - Exit 16:00.

  Interaction:
    - If Trend is already long when OD signals long, keep it and switch to
      OD rules (1.5% stop from the OD signal close, exit 16:00).
    - If OD signals short, close the long and go short under OD rules.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.join("data", "NASDAQ_60d_NQF")
BARS = os.path.join(ROOT, "bars_5m.parquet")
OUT = os.path.join("data", "results")
TZ = "America/New_York"
TICK = 0.25
POINT = 20.0
FEE = 2.14          # per contract per side
SLIP_TICKS = 1      # per side, sensitivity run
OD_STOP = 0.015
RR = 3.0


def ny(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ny"] = df["timestamp"].dt.tz_convert(TZ)
    df["day"] = df["ny"].dt.date
    df["hm"] = df["ny"].dt.strftime("%H:%M")
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index("timestamp") if "timestamp" in df.columns else df
    x = x[["open", "high", "low", "close", "volume"]]
    x = x.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    x["ny"] = x.index.tz_convert(TZ)
    x["hm"] = x["ny"].dt.strftime("%H:%M")
    return x


def atr_rma(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def fvg(df: pd.DataFrame, i: int):
    """Return ('bull'|'bear', first_low, first_high, mid_range) at candle i."""
    if i < 2:
        return None
    first, mid, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    mid_range = float(mid["high"] - mid["low"])
    if third["low"] > first["high"]:
        return ("bull", float(first["low"]), float(first["high"]), mid_range)
    if third["high"] < first["low"]:
        return ("bear", float(first["low"]), float(first["high"]), mid_range)
    return None


def build_signals(df5: pd.DataFrame):
    m10 = resample(df5, "10min")
    m15 = resample(df5, "15min")
    m15["ema50"] = m15["close"].ewm(span=50, adjust=False).mean()
    m15["ema100"] = m15["close"].ewm(span=100, adjust=False).mean()
    m15["atr"] = atr_rma(m15, 14)

    od = {}      # 5m open timestamp -> direction
    for i in range(len(m10)):
        r = m10.iloc[i]
        close_hm = (r["ny"] + pd.Timedelta(minutes=10)).strftime("%H:%M")
        if close_hm not in ("09:40", "09:50"):
            continue
        g = fvg(m10, i)
        if g:
            t = m10.index[i] + pd.Timedelta(minutes=10)   # next 5m open
            od[t] = 1 if g[0] == "bull" else -1

    trend = {}   # 5m open timestamp -> (dir, stop)
    for i in range(2, len(m15)):
        r = m15.iloc[i]
        close_hm = (r["ny"] + pd.Timedelta(minutes=15)).strftime("%H:%M")
        if not ("09:30" <= close_hm <= "15:30"):
            continue
        if not (r["ema50"] > r["ema100"]):
            continue
        g = fvg(m15, i)
        if not g or g[0] != "bull":
            continue
        if not np.isfinite(r["atr"]) or g[3] < 1.5 * r["atr"]:
            continue
        t = m15.index[i] + pd.Timedelta(minutes=15)
        stop = g[1] - TICK
        if stop <= 0:
            continue
        trend[t] = (1, stop)

    return od, trend


def hit(pos, row):
    if pos["dir"] == 1:
        if row["low"] <= pos["stop"]:
            return pos["stop"]
        if pos["target"] is not None and row["high"] >= pos["target"]:
            return pos["target"]
    else:
        if row["high"] >= pos["stop"]:
            return pos["stop"]
        if pos["target"] is not None and row["low"] <= pos["target"]:
            return pos["target"]
    return None


def run(df5: pd.DataFrame, mode: str = "duo", slip_ticks: int = SLIP_TICKS):
    od, trend = build_signals(df5)
    trades = []
    pos = None
    od_day = set()
    n = len(df5)

    for i in range(n):
        r = df5.iloc[i]
        t = df5.index[i]
        day = r["day"]
        hm = r["hm"]
        exited = False

        # ---- manage open position on this bar
        if pos is not None:
            level = hit(pos, r)
            if level is not None:
                trades.append(close(pos, level, t, "stop/target"))
                pos = None
                exited = True

        if pos is not None and hm >= "15:55":
            trades.append(close(pos, float(r["close"]), t, "eod"))
            pos = None
            exited = True

        if exited:
            continue

        # ---- OD signal
        if mode in ("duo", "od") and t in od and day not in od_day:
            d = od[t]
            if pos is not None:
                if pos["src"] == "trend" and d == 1:
                    # keep the long, switch to OD rules
                    ref = float(df5.iloc[i - 1]["close"])  # signal candle close
                    pos["stop"] = ref * (1 - OD_STOP)
                    pos["target"] = None
                    pos["src"] = "od"
                    pos["switched"] = True
                elif pos["src"] == "trend" and d == -1:
                    trades.append(close(pos, float(r["open"]), t, "od_reverse"))
                    pos = None
            if pos is None:
                entry = float(r["open"]) + d * slip_ticks * TICK
                pos = dict(dir=d, entry=entry, stop=entry * (1 - d * OD_STOP),
                           target=None, src="od", entry_time=t)
                lvl = hit(pos, r)
                if lvl is not None:
                    trades.append(close(pos, lvl, t, "stop/target"))
                    pos = None
            od_day.add(day)

        # ---- Trend FVG signal (only if no OD day and no position)
        if (mode in ("duo", "trend") and pos is None and day not in od_day
                and t in trend and "09:30" <= hm <= "15:30"):
            d, stop = trend[t]
            entry = float(r["open"]) + slip_ticks * TICK
            risk = entry - stop
            if risk > 0 and entry > stop:
                pos = dict(dir=1, entry=entry, stop=stop,
                           target=entry + RR * risk, src="trend",
                           entry_time=t)
                lvl = hit(pos, r)
                if lvl is not None:
                    trades.append(close(pos, lvl, t, "stop/target"))
                    pos = None

    if pos is not None:
        trades.append(close(pos, float(df5.iloc[-1]["close"]),
                            df5.index[-1], "end"))
    return trades


def close(pos, price, t, reason):
    gross = (price - pos["entry"]) * pos["dir"] * POINT
    pnl = gross - 2 * FEE
    return dict(entry_time=str(pos["entry_time"]), exit_time=str(t),
                dir="long" if pos["dir"] == 1 else "short",
                entry=pos["entry"], exit=price, reason=reason,
                src=pos["src"], pnl=pnl,
                r=((price - pos["entry"]) * pos["dir"]) /
                  max(abs(pos["entry"] - pos["stop"]), 1e-9))


def summarize(trades, label):
    if not trades:
        print(f"{label}: 0 trades")
        return
    pnl = np.array([x["pnl"] for x in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    pf = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
    print(f"{label:16} n={len(trades):>3} net=${pnl.sum():>9,.0f} "
          f"wr={len(wins)/len(pnl):>5.1%} pf={pf:>4.2f} "
          f"avgR={np.mean([x['r'] for x in trades]):>5.2f} maxDD=${dd:>8,.0f}")
    for src in ("od", "trend"):
        s = [x for x in trades if x["src"] == src]
        if s:
            p = sum(x["pnl"] for x in s)
            print(f"  {src:6} n={len(s):>3} net=${p:>9,.0f} "
                  f"wr={sum(1 for x in s if x['pnl'] > 0)/len(s):>5.1%}")


def main():
    os.makedirs(OUT, exist_ok=True)
    df5 = ny(pd.read_parquet(BARS))
    df5 = df5.set_index("timestamp").sort_index()
    span = (df5["ny"].min(), df5["ny"].max())
    print(f"5m data: {len(df5):,} bars | {span[0]} -> {span[1]} NY\n")

    for mode in ("od", "trend", "duo"):
        trades = run(df5, mode=mode)
        summarize(trades, mode.upper())

    trades = run(df5, mode="duo", slip_ticks=0)
    summarize(trades, "DUO no-slip")
    pd.DataFrame(trades).to_csv(os.path.join(OUT, "fvg_duo_trades.csv"),
                                index=False)


if __name__ == "__main__":
    main()
