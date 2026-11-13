"""Multi-year, multi-feed validation of the FVG Duo rules.

Feeds
  * NQ futures 1-minute (2022-12 -> 2025-12) resampled to 5m.
  * Nasdaq-100 CFD 5-minute from Dukascopy (2021-10 -> 2026-11).

Rules are exactly those in fvg_duo_backtest.py:
  OD    : 09:40 / 09:50 NY checks on 10m candles, 1.5% stop, hold to 16:00.
  Trend : 15m bullish FVG, EMA50>EMA100, mid range >= 1.5*ATR14,
          stop 1 tick under first candle low, 3R target, flat 16:00.
  Duo   : OD overrides Trend on the day; if Trend is already long and OD is
          long, keep it and switch to OD rules; if OD is short, reverse.

Costs
  NQ    : $2.14/side + 1 tick slippage/side, $20 per point.
  CFD   : 1.0 index point/side (spread+slippage), $1 per point.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from fvg_duo_backtest import resample, atr_rma, fvg

TZ = "America/New_York"
OD_STOP = 0.015
RR = 3.0

NQ_CSV = r"D:\diamondhead_data\Dataset_NQ_1min_2022_2025.csv"
DUKA = os.path.join("data", "NASDAQ_DUKA_5y", "bars_5m.parquet")
RECENT = os.path.join("data", "NASDAQ_60d_NQF", "bars_5m.parquet")
OUT = os.path.join("data", "results")


# ---------------------------------------------------------------- loading
def decorate(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    nyidx = x.index.tz_convert(TZ)
    x["ny"] = nyidx
    x["day"] = nyidx.date
    x["hm"] = nyidx.strftime("%H:%M")
    return x


def load_nq_1m() -> pd.DataFrame:
    d = pd.read_csv(NQ_CSV, usecols=["timestamp ET", "open", "high", "low",
                                     "close", "volume"])
    t = pd.to_datetime(d["timestamp ET"], format="%m/%d/%Y %H:%M")
    t = t.dt.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    d.index = pd.DatetimeIndex(t).tz_convert("UTC")
    d = d[["open", "high", "low", "close", "volume"]]
    d = d[~d.index.duplicated()].sort_index()
    x = d.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"})
    x = x.dropna(subset=["open"])
    return decorate(x)


def load_5m(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp").sort_index()
    d = d[["open", "high", "low", "close", "volume"]]
    d = d[~d.index.duplicated()]
    return decorate(d)


# ---------------------------------------------------------------- signals
def build_signals(df5: pd.DataFrame, tick: float):
    m10 = resample(df5, "10min")
    m15 = resample(df5, "15min")
    m15["ema50"] = m15["close"].ewm(span=50, adjust=False).mean()
    m15["ema100"] = m15["close"].ewm(span=100, adjust=False).mean()
    m15["atr"] = atr_rma(m15, 14)

    od = {}
    for i in range(len(m10)):
        r = m10.iloc[i]
        close_hm = (r["ny"] + pd.Timedelta(minutes=10)).strftime("%H:%M")
        if close_hm not in ("09:40", "09:50"):
            continue
        g = fvg(m10, i)
        if g:
            od[m10.index[i] + pd.Timedelta(minutes=10)] = (
                1 if g[0] == "bull" else -1)

    trend = {}
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
        stop = g[1] - tick
        if stop <= 0:
            continue
        trend[m15.index[i] + pd.Timedelta(minutes=15)] = (1, stop)
    return od, trend


# ---------------------------------------------------------------- engine
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


def close(pos, price, t, reason, point_value, fee):
    gross = (price - pos["entry"]) * pos["dir"] * point_value
    cost = pos.get("cost_frac", 0.0) * pos["entry"] * point_value
    pnl = gross - 2 * fee - cost
    return dict(entry_time=pos["entry_time"], exit_time=t,
                dir="long" if pos["dir"] == 1 else "short",
                entry=pos["entry"], exit=price, reason=reason,
                src=pos["src"], pnl=pnl,
                points=(price - pos["entry"]) * pos["dir"]
                       - pos.get("cost_frac", 0.0) * pos["entry"],
                r=((price - pos["entry"]) * pos["dir"]
                   - pos.get("cost_frac", 0.0) * pos["entry"]) / pos["risk"])


def run(df5: pd.DataFrame, mode: str, *, point_value: float, fee: float,
        slip: float, tick: float, signals=None, slip_frac: float = 0.0,
        cost_frac: float = 0.0):
    od, trend = signals if signals is not None else build_signals(df5, tick)
    rth = df5[(df5["hm"] >= "09:00") & (df5["hm"] <= "16:00")]
    trades = []
    pos = None
    od_day = set()

    for i in range(len(rth)):
        r = rth.iloc[i]
        t = rth.index[i]
        day = r["day"]
        hm = r["hm"]
        exited = False

        if pos is not None:
            lvl = hit(pos, r)
            if lvl is not None:
                trades.append(close(pos, lvl, t, "stop/target",
                                    point_value, fee))
                pos = None
                exited = True

        if pos is not None and hm >= "15:55":
            trades.append(close(pos, float(r["close"]), t, "eod",
                                point_value, fee))
            pos = None
            exited = True

        if exited:
            continue

        # Opening Drive
        if mode in ("duo", "od") and t in od and day not in od_day:
            d = od[t]
            if pos is not None:
                if pos["src"] == "trend" and d == 1:
                    ref = float(rth.iloc[i - 1]["close"])
                    pos["stop"] = ref * (1 - OD_STOP)
                    pos["target"] = None
                    pos["src"] = "od"
                elif pos["src"] == "trend" and d == -1:
                    trades.append(close(pos, float(r["open"]), t,
                                        "od_reverse", point_value, fee))
                    pos = None
            if pos is None:
                entry = (float(r["open"]) + d * slip
                         + d * float(r["open"]) * slip_frac)
                pos = dict(dir=d, entry=entry, stop=entry * (1 - d * OD_STOP),
                           target=None, src="od",
                           risk=entry * OD_STOP, entry_time=t,
                           cost_frac=cost_frac)
                lvl = hit(pos, r)
                if lvl is not None:
                    trades.append(close(pos, lvl, t, "stop/target",
                                        point_value, fee))
                    pos = None
            od_day.add(day)

        # Trend FVG
        if (mode in ("duo", "trend") and pos is None and day not in od_day
                and t in trend and "09:30" <= hm <= "15:30"):
            _, stop = trend[t]
            entry = (float(r["open"]) + slip
                     + float(r["open"]) * slip_frac)
            risk = entry - stop
            if risk > 0:
                pos = dict(dir=1, entry=entry, stop=stop,
                           target=entry + RR * risk, src="trend",
                           risk=risk, entry_time=t, cost_frac=cost_frac)
                lvl = hit(pos, r)
                if lvl is not None:
                    trades.append(close(pos, lvl, t, "stop/target",
                                        point_value, fee))
                    pos = None

    if pos is not None:
        trades.append(close(pos, float(rth.iloc[-1]["close"]),
                            rth.index[-1], "end", point_value, fee))
    return trades


# ---------------------------------------------------------------- stats
def _metrics(g: pd.DataFrame):
    if g.empty:
        return dict(n=0, net=0.0, netR=0.0, wr=0.0, pf=0.0, ddR=0.0, avgR=0.0)
    pnl = g["pnl"].to_numpy()
    r = g["r"].to_numpy()
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    return dict(n=len(g), net=float(pnl.sum()), netR=float(r.sum()),
                wr=float((pnl > 0).mean()),
                pf=float(pos / neg) if neg > 0 else float("inf"),
                ddR=dd, avgR=float(r.mean()))


def report(trades, label: str, start=None, end=None):
    df = pd.DataFrame(trades)
    if df.empty:
        print(f"{label}: 0 trades")
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    ny = df["entry_time"].dt.tz_convert(TZ)
    if start is not None:
        df = df[ny >= pd.Timestamp(start, tz=TZ)]
        ny = df["entry_time"].dt.tz_convert(TZ)
    if end is not None:
        df = df[ny < pd.Timestamp(end, tz=TZ)]
        ny = df["entry_time"].dt.tz_convert(TZ)

    print(f"\n=== {label} ===")
    print(f"period {ny.min().date()} -> {ny.max().date()}  "
          f"({ny.dt.year.nunique()} calendar years)")
    m = _metrics(df)
    print(f"TOTAL  n={m['n']:>4}  net=${m['net']:>10,.0f}  "
          f"netR={m['netR']:>8.1f}  wr={m['wr']:>5.1%}  "
          f"pf={m['pf']:>4.2f}  avgR={m['avgR']:>6.3f}  "
          f"maxDD={m['ddR']:>6.1f}R")

    by_src = df.groupby("src")
    for src, g in by_src:
        s = _metrics(g)
        print(f"  {src:6} n={s['n']:>4}  net=${s['net']:>10,.0f}  "
              f"netR={s['netR']:>8.1f}  wr={s['wr']:>5.1%}  "
              f"pf={s['pf']:>4.2f}  avgR={s['avgR']:>6.3f}")

    df["year"] = df["entry_time"].dt.tz_convert(TZ).dt.year
    rows = []
    for year, g in df.groupby("year"):
        s = _metrics(g)
        rows.append(dict(year=year, n=s["n"], net=round(s["net"], 0),
                         netR=round(s["netR"], 1), wr=round(s["wr"], 3),
                         pf=round(s["pf"], 2), avgR=round(s["avgR"], 3),
                         maxDD_R=round(s["ddR"], 1)))
    y = pd.DataFrame(rows)
    print(y.to_string(index=False))
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    nq = load_nq_1m()
    duka = load_5m(DUKA)
    print("NQ 5m bars", len(nq), nq.index.min(), "->", nq.index.max())
    print("Duka 5m bars", len(duka), duka.index.min(), "->", duka.index.max())

    specs = {
        "NQ futures": dict(df=nq, point_value=20.0, fee=2.14,
                           slip=0.25, tick=0.25),
        "NDX CFD": dict(df=duka, point_value=1.0, fee=0.0,
                        slip=1.0, tick=0.01),
    }

    all_trades = []
    for name, sp in specs.items():
        print(f"\n\n########## {name} ##########")
        sig = build_signals(sp["df"], sp["tick"])
        for mode in ("od", "trend", "duo"):
            trades = run(sp["df"], mode, point_value=sp["point_value"],
                         fee=sp["fee"], slip=sp["slip"], tick=sp["tick"],
                         signals=sig)
            if name == "NQ futures":
                start, end = "2023-01-03", "2025-12-11"
            else:
                start, end = "2021-11-15", "2026-11-13"
            df = report(trades, f"{name} / {mode.upper()}", start, end)
            if not df.empty:
                df["feed"] = name
                df["mode"] = mode
                all_trades.append(df)

    if all_trades:
        pd.concat(all_trades).to_csv(
            os.path.join(OUT, "fvg_duo_multiyear_trades.csv"), index=False)

    # recent 60d feed: confirm the refactor reproduces the earlier result
    recent = load_5m(RECENT)
    print("\n\n########## recent 5m (refactor check) ##########")
    rsig = build_signals(recent, 0.25)
    for mode in ("od", "duo"):
        t = run(recent, mode, point_value=20.0, fee=2.14, slip=0.25,
                tick=0.25, signals=rsig)
        report(t, f"recent / {mode.upper()}", "2026-09-03", "2026-11-13")


if __name__ == "__main__":
    main()
