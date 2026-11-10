"""Multi-Filter Opening Range Breakout — paper-only runner.

Port of the Pine Script v6 strategy to a live paper runner:
- 5-minute NQ=F bars, America/New_York session
- Opening range 09:30-10:00 ET
- Entry window 10:00-15:55 ET
- Max 2 entries per NY day
- Filters: HTF 60m EMA(200), local EMA(50), VWAP, RSI(14), ADX(14), rel volume
- Stop = 1.5 x ATR(14); target = 2.5 x stop (configurable)
- Daily loss limit 2% of day-start equity
- Commission 0.05% of trade value; slippage 2 ticks per side

Paper-only. No broker adapter, no API keys, no live orders.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

CT = ZoneInfo("America/New_York")
POINT_VALUE = 20.0  # USD per NQ point per contract
STARTING_CAPITAL = 10_000.0
COMMISSION_PCT = 0.0005  # 0.05%
SLIPPAGE_POINTS = 0.5  # 2 ticks of 0.25 per side
REWARD_RISK = 2.5  # target = 2.5 x stop
MAX_TRADES_PER_DAY = 2
DAILY_LOSS_PCT = 0.02
ATR_LENGTH = 14
STOP_ATR_MULT = 1.5

STATE_DIR = Path(__file__).resolve().parent / "state"
STATE_PATH = STATE_DIR / "paper_state.json"
CONTROL_PATH = STATE_DIR / "control.json"
TRADES_CSV_PATH = STATE_DIR / "paper_trades.csv"


def _now_iso() -> str:
    return datetime.now(CT).isoformat()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(default)


class PaperState:
    def __init__(self) -> None:
        self.state = _load_json(
            STATE_PATH,
            {
                "capital": STARTING_CAPITAL,
                "cash": STARTING_CAPITAL,
                "equity": STARTING_CAPITAL,
                "realized_pnl": 0.0,
                "positions": [],
                "trades": [],
                "equity_curve": [
                    {"timestamp": _now_iso(), "equity": STARTING_CAPITAL}
                ],
                "last_update": None,
                "last_price": None,
                "last_entry_session": None,
            },
        )
        self.control = _load_json(
            CONTROL_PATH, {"halted": False, "reason": "", "updated_at": None}
        )
        self.state.setdefault("positions", [])
        self.state.setdefault("trades", [])
        self.state.setdefault("equity_curve", [])
        self.state.setdefault("cash", self.state.get("capital", STARTING_CAPITAL))
        self.state.setdefault("equity", self.state.get("cash", STARTING_CAPITAL))
        self.state.setdefault("realized_pnl", 0.0)
        self.state.setdefault("last_entry_session", None)

    def save(self) -> None:
        _save_json(STATE_PATH, self.state)
        _save_json(CONTROL_PATH, self.control)

    def is_halted(self) -> bool:
        return bool(self.control.get("halted", False))

    def set_halted(self, halted: bool, reason: str = "") -> None:
        self.control = {
            "halted": bool(halted),
            "reason": reason if halted else "",
            "updated_at": _now_iso(),
        }
        self.save()

    def stats(self) -> dict:
        trades = self.state["trades"]
        wins = [t for t in trades if float(t.get("net_usd", 0)) > 0]
        losses = [t for t in trades if float(t.get("net_usd", 0)) < 0]
        gross_profit = sum(float(t.get("net_usd", 0)) for t in wins)
        gross_loss = abs(sum(float(t.get("net_usd", 0)) for t in losses))
        peak = -np.inf
        max_dd = 0.0
        for p in self.state.get("equity_curve", []):
            e = float(p.get("equity", 0))
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak)
        return {
            "equity": float(self.state.get("equity", STARTING_CAPITAL)),
            "cash": float(self.state.get("cash", STARTING_CAPITAL)),
            "realized_pnl": float(self.state.get("realized_pnl", 0.0)),
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
            "max_drawdown_pct": max_dd * 100,
            "open_positions": len(self.state["positions"]),
            "halted": self.is_halted(),
        }


def download() -> pd.DataFrame:
    raw = yf.download(
        "NQ=F",
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        prepost=False,
    )
    if raw.empty:
        raise RuntimeError("no data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(CT)
    df = df.between_time("09:30", "16:00")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema50"] = d["Close"].ewm(span=50, adjust=False).mean()

    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    d["vwap"] = (
        (tp * d["Volume"]).groupby(d.index.date).cumsum()
        / d["Volume"].groupby(d.index.date).cumsum()
    )

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    d["rsi"] = 100 - (100 / (1 + rs))

    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / ATR_LENGTH, adjust=False).mean()

    d["vol_avg20"] = d["Volume"].rolling(20).mean()
    d["rel_vol"] = d["Volume"] / d["vol_avg20"]

    prev_high = d["High"].shift(1)
    prev_low = d["Low"].shift(1)
    up_move = d["High"] - prev_high
    down_move = prev_low - d["Low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_tr = d["atr"]
    plus_di = (
        100
        * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean()
        / atr_tr
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean()
        / atr_tr
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    htf = (
        d[["Open", "High", "Low", "Close", "Volume"]]
        .resample("60min", label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    htf["ema200"] = htf["Close"].ewm(span=200, adjust=False).mean()
    htf_close = htf["Close"]
    htf_ema = htf["ema200"]
    htf_index = htf.index
    d["htf_close"] = np.nan
    d["htf_ema"] = np.nan
    pos = htf_index.searchsorted(d.index, side="right") - 1
    valid = pos >= 0
    d.loc[valid, "htf_close"] = htf_close.values[pos[valid]]
    d.loc[valid, "htf_ema"] = htf_ema.values[pos[valid]]
    return d


def _completed_bars(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = now - timedelta(minutes=5)
    return df[df.index <= cutoff]


def run_tick(dry_run: bool = False, now: datetime | None = None) -> dict:
    state = PaperState()
    result = {
        "time_et": None,
        "halted": state.is_halted(),
        "action": "none",
        "reason": "",
        "trade": None,
        "stats": state.stats(),
    }

    now = now or datetime.now(CT)
    if now.tzinfo is None:
        now = now.replace(tzinfo=CT)
    now = now.astimezone(CT)
    result["time_et"] = now.isoformat()

    if now.weekday() >= 5:
        result["reason"] = "weekend"
        result["stats"] = state.stats()
        return result

    # Outside regular session: still manage exits
    if not (dtime(9, 30) <= now.time() <= dtime(16, 0)):
        result["reason"] = "outside session"
        result["stats"] = state.stats()
        return result

    df = download()
    df = compute_indicators(df)
    bars = _completed_bars(df, now)
    if bars.empty:
        result["reason"] = "no data"
        result["stats"] = state.stats()
        return result

    session = now.date()
    day = bars[bars.index.date == session]
    if len(day) < 3:
        result["reason"] = "not enough session bars"
        result["stats"] = state.stats()
        return result

    day = day.sort_index()

    # Opening range: 09:30-09:55 ET
    opening = day[
        (day.index.time >= dtime(9, 30)) & (day.index.time < dtime(10, 0))
    ]
    if len(opening) < 2:
        result["reason"] = "opening range incomplete"
        result["stats"] = state.stats()
        return result
    orb_high = float(opening["High"].max())
    orb_low = float(opening["Low"].min())

    day_start_equity = float(state.state.get("cash", STARTING_CAPITAL))
    for p in state.state.get("positions", []):
        day_start_equity += float(p.get("unrealized_usd", 0))
    daily_loss_limit = day_start_equity * (1 - DAILY_LOSS_PCT)

    latest = day.iloc[-1]
    latest_close = float(latest["Close"])

    # Manage open position
    if state.state["positions"]:
        position = state.state["positions"][0]
        entry_price = float(position["entry_price"])
        stop = float(position["stop_price"])
        target = float(position["target_price"])
        direction = position["direction"]
        opened_at = pd.Timestamp(position["opened_at"])
        if opened_at.tzinfo is None:
            opened_at = opened_at.tz_localize("UTC")
        opened_at = opened_at.tz_convert(CT)
        after_entry = day[day.index > opened_at]

        exit_price = None
        reason = ""
        for ts, bar in after_entry.iterrows():
            high = float(bar["High"])
            low = float(bar["Low"])
            if direction == "long":
                if low <= stop:
                    exit_price, reason = stop, "stop"
                    break
                if high >= target:
                    exit_price, reason = target, "target"
                    break
            else:
                if high >= stop:
                    exit_price, reason = stop, "stop"
                    break
                if low <= target:
                    exit_price, reason = target, "target"
                    break

        if exit_price is None and now.time() >= dtime(15, 55):
            exit_price, reason = latest_close, "window_close"

        if exit_price is not None:
            gross_pts = (
                (exit_price - entry_price)
                if direction == "long"
                else (entry_price - exit_price)
            )
            commission_usd = (entry_price + exit_price) * COMMISSION_PCT
            slippage_usd = SLIPPAGE_POINTS * 2 * POINT_VALUE
            net_pts = gross_pts - (slippage_usd / POINT_VALUE)
            net_usd = net_pts * POINT_VALUE - commission_usd

            trade = {
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": float(exit_price),
                "stop_price": stop,
                "target_price": target,
                "entry_time": str(position["opened_at"]),
                "exit_time": str(ts if "ts" in locals() else latest.name),
                "gross_points": round(gross_pts, 4),
                "net_points": round(net_pts, 4),
                "net_usd": round(net_usd, 2),
                "reason": reason,
            }
            state.state["trades"].append(trade)
            state.state["positions"] = []
            state.state["cash"] = round(
                float(state.state.get("cash", 0)) + net_usd, 2
            )
            state.state["realized_pnl"] = round(
                float(state.state.get("realized_pnl", 0)) + net_usd, 2
            )
            state.state["equity"] = round(float(state.state["cash"]), 2)
            state.state["last_price"] = round(float(exit_price), 4)
            state.state["last_update"] = _now_iso()
            state.state["equity_curve"].append(
                {"timestamp": _now_iso(), "equity": float(state.state["equity"])}
            )
            state.save()
            result["action"] = "exit"
            result["trade"] = trade
            result["stats"] = state.stats()
            return result

        # Update unrealized
        position["last_price"] = round(latest_close, 4)
        position["unrealized_points"] = round(
            (latest_close - entry_price)
            if direction == "long"
            else (entry_price - latest_close),
            4,
        )
        position["unrealized_usd"] = round(
            position["unrealized_points"] * POINT_VALUE, 2
        )
        state.state["last_price"] = round(latest_close, 4)
        state.state["last_update"] = _now_iso()
        state.save()
        result["action"] = "mark"
        result["position"] = position
        result["stats"] = state.stats()
        return result

    # Flat: check entry
    if state.is_halted():
        result["reason"] = "halted: new entries blocked"
        result["stats"] = state.stats()
        return result

    if not (dtime(10, 0) <= now.time() <= dtime(15, 55)):
        result["reason"] = "outside entry window"
        result["stats"] = state.stats()
        return result

    if str(state.state.get("last_entry_session")) == str(session):
        result["reason"] = "already traded this session"
        result["stats"] = state.stats()
        return result

    trades_today = len(
        [
            t
            for t in state.state.get("trades", [])
            if str(t.get("entry_time", "")).startswith(str(session))
        ]
    )
    if trades_today >= MAX_TRADES_PER_DAY:
        result["reason"] = "max trades today reached"
        result["stats"] = state.stats()
        return result

    if day_start_equity <= daily_loss_limit:
        result["reason"] = "daily loss limit reached"
        result["stats"] = state.stats()
        return result

    prev_close = float(day.iloc[-2]["Close"])
    bar = latest
    atr = float(bar["atr"]) if pd.notna(bar["atr"]) else np.nan
    rsi = float(bar["rsi"]) if pd.notna(bar["rsi"]) else np.nan
    adx = float(bar["adx"]) if pd.notna(bar["adx"]) else np.nan
    rel_vol = float(bar["rel_vol"]) if pd.notna(bar["rel_vol"]) else np.nan
    htf_close = float(bar["htf_close"]) if pd.notna(bar["htf_close"]) else np.nan
    htf_ema = float(bar["htf_ema"]) if pd.notna(bar["htf_ema"]) else np.nan
    vwap = float(bar["vwap"]) if pd.notna(bar["vwap"]) else np.nan
    ema50 = float(bar["ema50"]) if pd.notna(bar["ema50"]) else np.nan

    long_breakout = latest_close > orb_high and prev_close <= orb_high
    short_breakout = latest_close < orb_low and prev_close >= orb_low

    long_ok = bool(long_breakout)
    short_ok = bool(short_breakout)

    if not np.isnan(htf_close) and not np.isnan(htf_ema):
        long_ok = long_ok and htf_close > htf_ema
        short_ok = short_ok and htf_close < htf_ema
    if not np.isnan(ema50):
        long_ok = long_ok and latest_close > ema50
        short_ok = short_ok and latest_close < ema50
    if not np.isnan(vwap):
        long_ok = long_ok and latest_close > vwap
        short_ok = short_ok and latest_close < vwap
    if not np.isnan(rsi):
        long_ok = long_ok and rsi > 50
        short_ok = short_ok and rsi < 50
    if not np.isnan(adx):
        long_ok = long_ok and adx >= 18
        short_ok = short_ok and adx >= 18
    if not np.isnan(rel_vol):
        long_ok = long_ok and rel_vol >= 1.0
        short_ok = short_ok and rel_vol >= 1.0

    if not (long_ok or short_ok):
        result["reason"] = "no qualifying breakout"
        result["stats"] = state.stats()
        return result

    stop_pts = max(1.0, STOP_ATR_MULT * atr) if not np.isnan(atr) else 10.0
    target_pts = REWARD_RISK * stop_pts
    entry = latest_close
    direction = "long" if long_ok else "short"
    stop = entry - stop_pts if direction == "long" else entry + stop_pts
    target = entry + target_pts if direction == "long" else entry - target_pts

    if dry_run:
        result["action"] = "would_open"
        result["direction"] = direction
        result["entry"] = entry
        result["stop"] = stop
        result["target"] = target
        result["reason"] = "dry run"
        result["stats"] = state.stats()
        return result

    position = {
        "direction": direction,
        "entry_price": round(entry, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "target_points": round(target_pts, 4),
        "stop_points": round(stop_pts, 4),
        "contracts": 1,
        "opened_at": str(latest.name),
        "last_price": round(entry, 4),
        "unrealized_points": 0.0,
        "unrealized_usd": 0.0,
    }
    state.state["positions"].append(position)
    state.state["last_entry_session"] = str(session)
    state.state["last_price"] = round(entry, 4)
    state.state["last_update"] = _now_iso()
    state.state["equity"] = round(float(state.state.get("cash", 0)), 2)
    state.save()
    result["action"] = "open"
    result["trade"] = position
    result["reason"] = "filtered opening-range breakout"
    result["stats"] = state.stats()
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now", default=None)
    args = parser.parse_args()

    now = None
    if args.now:
        now = datetime.fromisoformat(args.now)
        if now.tzinfo is None:
            now = now.replace(tzinfo=CT)
        now = now.astimezone(CT)

    result = run_tick(dry_run=args.dry_run, now=now)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
