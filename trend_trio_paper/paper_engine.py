"""Paper-only ledger and execution for Trend Trio.

One contract (MNQ-equivalent) at a time.  No broker adapter, API key, or
live-order code exists in this package.
"""
from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from .signals import (
    TZ, Entry, Exit, build_entries, build_exits, completed_1m,
    normalize_1m, scan_entry,
)

PAPER_ONLY = True
STARTING_CAPITAL = 50_000.0
POINT_VALUE = 2.0          # 1 MNQ-equivalent on the Nasdaq-100 CFD feed
FEE_PER_SIDE = 2.0         # ~1 index point spread + commission, USD per side
SLIPPAGE_TICKS = 0
CONTRACT_LABEL = "USATECH.IDX/USD (MNQ-equivalent)"

DEFAULT_STATE_DIR = Path(__file__).resolve().parent / "state"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(default)


class PaperEngine:
    def __init__(self, base_dir: str | Path | None = None,
                 starting_capital: float = STARTING_CAPITAL) -> None:
        if not PAPER_ONLY:
            raise RuntimeError("paper-only guard failed")
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_STATE_DIR
        self.state_path = self.base_dir / "paper_state.json"
        self.control_path = self.base_dir / "control.json"
        self.trades_csv_path = self.base_dir / "paper_trades.csv"
        self.state = _load_json(
            self.state_path,
            {
                "capital": starting_capital,
                "cash": starting_capital,
                "equity": starting_capital,
                "realized_pnl": 0.0,
                "positions": [],
                "trades": [],
                "equity_curve": [],
                "last_update": None,
                "last_price": None,
                "last_decision": None,
                "last_reason": None,
                "last_exit_time": None,
            },
        )
        self.control = _load_json(
            self.control_path,
            {"halted": False, "reason": "", "updated_at": None},
        )
        for key, default in (
            ("capital", starting_capital), ("cash", starting_capital),
            ("equity", starting_capital), ("realized_pnl", 0.0),
            ("positions", []), ("trades", []), ("equity_curve", []),
            ("last_exit_time", None),
        ):
            self.state.setdefault(key, default)
        self._dry_run = False
        if not self.state["equity_curve"]:
            self.state["equity_curve"].append(
                {"timestamp": _now_iso(),
                 "equity": float(self.state["equity"])}
            )

    def save(self) -> None:
        if self._dry_run:
            return
        _save_json(self.state_path, self.state)
        _save_json(self.control_path, self.control)
        if self.state["trades"]:
            pd.DataFrame(self.state["trades"]).to_csv(
                self.trades_csv_path, index=False
            )

    def is_halted(self) -> bool:
        return bool(self.control.get("halted", False))

    def stats(self) -> dict:
        trades = self.state["trades"]
        wins = [t for t in trades if float(t.get("net_usd", 0)) > 0]
        losses = [t for t in trades if float(t.get("net_usd", 0)) < 0]
        gp = sum(float(t.get("net_usd", 0)) for t in wins)
        gl = abs(sum(float(t.get("net_usd", 0)) for t in losses))
        peak = -np.inf
        max_dd = 0.0
        for p in self.state.get("equity_curve", []):
            e = float(p.get("equity", 0))
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak)
        unrealized = sum(float(p.get("unrealized_usd", 0))
                         for p in self.state.get("positions", []))
        rs = [float(t["r_net"]) for t in trades
              if t.get("r_net") is not None]
        return {
            "equity": float(self.state.get("equity", STARTING_CAPITAL)),
            "cash": float(self.state.get("cash", STARTING_CAPITAL)),
            "realized_pnl": float(self.state.get("realized_pnl", 0.0)),
            "unrealized_pnl": unrealized,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
            "gross_profit_usd": gp,
            "gross_loss_usd": gl,
            "profit_factor": (gp / gl) if gl > 0 else None,
            "avg_r_net": (sum(rs) / len(rs)) if rs else None,
            "max_drawdown_pct": max_dd * 100,
            "open_positions": len(self.state.get("positions", [])),
            "halted": self.is_halted(),
        }

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _utc(now: datetime | pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(now)
        if t.tzinfo is None:
            t = t.tz_localize(TZ)
        return t.tz_convert("UTC")

    def _append_equity(self, when: pd.Timestamp) -> None:
        eq = float(self.state.get("equity", 0.0))
        curve = self.state.setdefault("equity_curve", [])
        if curve and abs(float(curve[-1].get("equity", 0)) - eq) < 1e-9:
            return
        curve.append({"timestamp": when.isoformat(), "equity": eq})
        if len(curve) > 5000:
            del curve[: len(curve) - 5000]

    def _close(self, price: float, when: pd.Timestamp, reason: str) -> dict:
        pos = self.state["positions"][0]
        d = int(pos["direction_sign"])
        pts = (price - float(pos["entry_price"])) * d
        net_pts = pts - (2 * FEE_PER_SIDE / POINT_VALUE)
        gross_usd = pts * POINT_VALUE
        net_usd = gross_usd - 2 * FEE_PER_SIDE
        risk_pts = max(float(pos.get("risk_points", 0.0)), 1e-9)
        trade = {
            "strategy": "Trend Trio",
            "source": pos.get("source", ""),
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "signal_time": pos.get("signal_time", pos["entry_time"]),
            "closed_at": when.isoformat(),
            "entry_price": round(float(pos["entry_price"]), 4),
            "exit_price": round(float(price), 4),
            "stop_price": (round(float(pos["stop_price"]), 4)
                           if pos.get("stop_price") is not None else None),
            "target_price": (round(float(pos["target_price"]), 4)
                             if pos.get("target_price") is not None else None),
            "gross_points": round(pts, 4),
            "net_points": round(net_pts, 4),
            "gross_usd": round(gross_usd, 2),
            "net_usd": round(net_usd, 2),
            "r_gross": (round(pts / risk_pts, 4)
                        if float(pos.get("risk_points", 0.0)) > 1e-9
                        else None),
            "r_net": (round(net_pts / risk_pts, 4)
                      if float(pos.get("risk_points", 0.0)) > 1e-9
                      else None),
            "reason": reason,
        }
        self.state["positions"] = []
        self.state["trades"].append(trade)
        self.state["cash"] = round(float(self.state["cash"]) + net_usd, 2)
        self.state["realized_pnl"] = round(
            float(self.state["realized_pnl"]) + net_usd, 2
        )
        self.state["equity"] = round(float(self.state["cash"]), 2)
        self.state["last_price"] = round(float(price), 4)
        self.state["last_exit_time"] = when.isoformat()
        self._append_equity(when)
        return trade

    def _open(self, sig: Entry, entry: float, when: pd.Timestamp) -> dict:
        d = int(sig.direction)
        risk = float(sig.risk_points)
        stop = entry - d * risk if risk > 1e-9 else None
        pos = {
            "direction": "long" if d == 1 else "short",
            "direction_sign": d,
            "entry_price": round(float(entry), 4),
            "stop_price": round(float(stop), 4) if stop is not None else None,
            "target_price": None,
            "risk_points": round(risk, 4),
            "contracts": 1,
            "contract": CONTRACT_LABEL,
            "source": sig.source,
            "source_detail": sig.detail,
            "entry_time": when.isoformat(),
            "signal_time": pd.Timestamp(sig.signal_time).isoformat(),
            "last_check_time": (pd.Timestamp(when)
                                + pd.Timedelta(minutes=1)).isoformat(),
            "last_price": round(float(entry), 4),
            "unrealized_points": 0.0,
            "unrealized_usd": 0.0,
        }
        self.state["positions"] = [pos]
        self.state["last_price"] = round(float(entry), 4)
        return pos

    def _mark(self, price: float, when: pd.Timestamp) -> None:
        pos = self.state["positions"][0]
        d = int(pos["direction_sign"])
        pts = (price - float(pos["entry_price"])) * d
        pos["last_price"] = round(float(price), 4)
        pos["unrealized_points"] = round(pts, 4)
        pos["unrealized_usd"] = round(pts * POINT_VALUE, 2)
        self.state["equity"] = round(
            float(self.state["cash"]) + float(pos["unrealized_usd"]), 2
        )
        self.state["last_price"] = round(float(price), 4)
        self._append_equity(when)

    @staticmethod
    def _parse_time(value: str | None) -> pd.Timestamp | None:
        if not value:
            return None
        t = pd.Timestamp(value)
        if t.tzinfo is None:
            t = t.tz_localize(TZ)
        return t.tz_convert("UTC")

    def _exit_for_bar(self, exits: list[Exit], source: str,
                      direction: int, ts: pd.Timestamp) -> Exit | None:
        for e in exits:
            if (e.signal_time == ts and e.source == source
                    and e.for_direction == direction):
                return e
        return None

    # ---------------------------------------------------------------- tick
    def tick(self, now: datetime | pd.Timestamp, bars: pd.DataFrame,
             entries: list[Entry] | None = None,
             exits: list[Exit] | None = None,
             dry_run: bool = False) -> dict:
        now_utc = self._utc(now)
        now_et = now_utc.tz_convert(TZ)
        bars1 = normalize_1m(bars)
        completed = completed_1m(bars1, now_utc)
        if entries is None:
            entries = build_entries(completed)
        if exits is None:
            exits = build_exits(completed)
        self._dry_run = bool(dry_run)

        result = {
            "time_et": now_et.isoformat(),
            "action": "none",
            "reason": "",
            "trade": None,
            "position": None,
            "stats": self.stats(),
        }

        # Materialise an initial state directory on the first run.
        self.save()

        if bars1.empty or completed.empty:
            result["reason"] = "no completed data"
            result["stats"] = self.stats()
            return result

        latest_ts = completed.index[-1]
        latest = float(completed.iloc[-1]["close"])

        # ---- manage an existing position
        if self.state["positions"]:
            pos = self.state["positions"][0]
            source = pos.get("source", "")
            d = int(pos["direction_sign"])
            last_check = self._parse_time(
                pos.get("last_check_time") or pos["entry_time"]
            )
            if last_check is None:
                last_check = pd.Timestamp(pos["entry_time"])
                if last_check.tzinfo is None:
                    last_check = last_check.tz_localize(TZ)
                last_check = last_check.tz_convert("UTC")
            after = completed[completed.index >= last_check]

            for ts, bar in after.iterrows():
                bar_et = ts.tz_convert(TZ)
                hm = bar_et.time()
                exit_price = None
                reason = ""

                # Stops are intrabar and take priority over signal exits.
                if pos.get("stop_price") is not None:
                    stop = float(pos["stop_price"])
                    if d == 1 and float(bar["low"]) <= stop:
                        exit_price = min(stop, float(bar["open"]))
                        reason = "stop"
                    elif d == -1 and float(bar["high"]) >= stop:
                        exit_price = max(stop, float(bar["open"]))
                        reason = "stop"

                if exit_price is None:
                    ev = self._exit_for_bar(exits, source, d, ts)
                    if ev is not None:
                        exit_price = float(bar["close"])
                        reason = ev.reason

                if exit_price is None:
                    if source == "st" and hm >= time(15, 59):
                        exit_price, reason = float(bar["close"]), "eod"
                    elif source in ("at2", "at1h") and hm >= time(16, 0):
                        exit_price, reason = float(bar["close"]), "eod"
                    elif (source == "uma" and bar_et.weekday() == 4
                          and hm >= time(16, 45)):
                        exit_price, reason = float(bar["close"]), "weekend_flat"

                if exit_price is not None:
                    trade = self._close(exit_price, ts, reason)
                    self.state["last_update"] = _now_iso()
                    self.state["last_decision"] = "exit"
                    self.state["last_reason"] = reason
                    self.save()
                    result.update(action="exit", reason=reason,
                                  trade=trade, stats=self.stats())
                    return result

                pos["last_check_time"] = ts.isoformat()

            self._mark(latest, latest_ts)
            self.state["last_update"] = _now_iso()
            self.state["last_decision"] = "mark"
            self.state["last_reason"] = "open position"
            self.save()
            result.update(action="mark", position=self.state["positions"][0],
                          stats=self.stats())
            return result

        # ---- flat: no new entries while halted
        if self.is_halted():
            result["reason"] = "halted: new paper entries blocked"
            result["stats"] = self.stats()
            return result

        after = self._parse_time(self.state.get("last_exit_time"))
        sig = scan_entry(entries, now_utc, after=after)
        if sig is None:
            result["reason"] = "no fresh entry signal"
            result["stats"] = self.stats()
            return result

        pos = self._open(sig, latest, latest_ts)
        self.state["last_update"] = _now_iso()
        self.state["last_decision"] = "open"
        self.state["last_reason"] = (
            f"{sig.source} {'long' if sig.direction == 1 else 'short'} "
            f"signal {pd.Timestamp(sig.signal_time).tz_convert(TZ).strftime('%H:%M')}"
        )
        self.save()
        result.update(action="open", reason=self.state["last_reason"],
                      position=pos, stats=self.stats())
        return result
