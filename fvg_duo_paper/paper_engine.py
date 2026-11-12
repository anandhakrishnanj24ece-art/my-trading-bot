"""Paper-only ledger and one-tick execution for FVG Duo.

There is no broker adapter, API key, or live-order code in this package.  The
runner reads public NQ futures candles, writes a local/committed paper ledger,
and never sends an order.
"""
from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from .signals import (
    OD_STOP, RR, TICK, TZ, build_signals, completed_5m, normalize_5m,
    signal_day,
)

PAPER_ONLY = True
STARTING_CAPITAL = 50_000.0
POINT_VALUE = 1.0          # USATECH Nasdaq-100 CFD, $1 per index point
FEE_PER_SIDE = 1.0         # all-in spread + slippage, USD per side
SLIPPAGE_TICKS = 0         # fee already includes execution cost
MAX_ENTRY_DELAY_SEC = 330  # one 5m bar + scheduler jitter

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
        self.state_dir = self.base_dir
        self.state_path = self.state_dir / "paper_state.json"
        self.control_path = self.state_dir / "control.json"
        self.trades_csv_path = self.state_dir / "paper_trades.csv"
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
                "od_days": [],
            },
        )
        self.control = _load_json(
            self.control_path,
            {"halted": False, "reason": "", "updated_at": None},
        )
        self.state.setdefault("capital", starting_capital)
        self.state.setdefault("cash", starting_capital)
        self.state.setdefault("equity", starting_capital)
        self.state.setdefault("realized_pnl", 0.0)
        self.state.setdefault("positions", [])
        self.state.setdefault("trades", [])
        self.state.setdefault("equity_curve", [])
        self.state.setdefault("od_days", [])
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
        positions = self.state.get("positions", [])
        unrealized = sum(float(p.get("unrealized_usd", 0))
                         for p in positions)
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
            "max_drawdown_pct": max_dd * 100,
            "open_positions": len(positions),
            "halted": self.is_halted(),
        }

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _utc(now: datetime | pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(now)
        if t.tzinfo is None:
            t = t.tz_localize(TZ)
        return t.tz_convert("UTC")

    def _latest_close(self, bars: pd.DataFrame,
                      now: pd.Timestamp) -> float | None:
        usable = bars[bars.index <= now]
        if usable.empty:
            return None
        return float(usable.iloc[-1]["close"])

    def _previous_close(self, bars: pd.DataFrame,
                        ts: pd.Timestamp) -> float | None:
        earlier = bars[bars.index < ts]
        if earlier.empty:
            return None
        return float(earlier.iloc[-1]["close"])

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
            "strategy": "FVG Duo",
            "source": pos.get("source", ""),
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "signal_time": pos.get("signal_time", pos["entry_time"]),
            "closed_at": when.isoformat(),
            "entry_price": round(float(pos["entry_price"]), 4),
            "exit_price": round(float(price), 4),
            "stop_price": round(float(pos["stop_price"]), 4),
            "target_price": (round(float(pos["target_price"]), 4)
                             if pos.get("target_price") is not None else None),
            "gross_points": round(pts, 4),
            "net_points": round(net_pts, 4),
            "gross_usd": round(gross_usd, 2),
            "net_usd": round(net_usd, 2),
            "r_gross": round(pts / risk_pts, 4),
            "r_net": round(net_pts / risk_pts, 4),
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
        self._append_equity(when)
        return trade

    def _open(self, direction: int, entry: float, stop: float,
              target: float | None, source: str, when: pd.Timestamp,
              risk_points: float, signal_time: pd.Timestamp | None = None) -> dict:
        pos = {
            "direction": "long" if direction == 1 else "short",
            "direction_sign": direction,
            "entry_price": round(float(entry), 4),
            "stop_price": round(float(stop), 4),
            "target_price": (round(float(target), 4)
                             if target is not None else None),
            "risk_points": round(float(risk_points), 4),
            "contracts": 1,
            "contract": "USATECH.IDX/USD",
            "source": source,
            "entry_time": when.isoformat(),
            "signal_time": (pd.Timestamp(signal_time).isoformat()
                            if signal_time is not None else when.isoformat()),
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

    # ---------------------------------------------------------------- tick
    def tick(self, now: datetime | pd.Timestamp, bars: pd.DataFrame,
             signals=None, dry_run: bool = False) -> dict:
        now_utc = self._utc(now)
        now_et = now_utc.tz_convert(TZ)
        bars = normalize_5m(bars)
        if signals is None:
            signals = build_signals(completed_5m(bars, now_utc))
        od, trend = signals

        result = {
            "time_et": now_et.isoformat(),
            "action": "none",
            "reason": "",
            "trade": None,
            "position": None,
            "stats": self.stats(),
        }
        self._dry_run = bool(dry_run)

        # Always persist initial files so GitHub has a state directory.
        self.save()

        if bars.empty:
            result["reason"] = "no data"
            result["stats"] = self.stats()
            self.save()
            return result

        latest = self._latest_close(bars, now_utc)
        if latest is None:
            result["reason"] = "no completed price"
            self.save()
            return result

        # ---- manage an existing position
        completed = completed_5m(bars, now_utc)
        if self.state["positions"]:
            pos = self.state["positions"][0]
            entry_ts = pd.Timestamp(pos.get("signal_time", pos["entry_time"]))
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize(TZ)
            entry_ts = entry_ts.tz_convert("UTC")
            after = completed[(completed.index >= entry_ts)
                              & (completed.index <= now_utc)]
            exit_price = None
            reason = ""
            exit_ts = now_utc
            for ts, bar in after.iterrows():
                if pos["direction"] == "long":
                    if float(bar["low"]) <= float(pos["stop_price"]):
                        exit_price, reason, exit_ts = (
                            float(pos["stop_price"]), "stop", ts
                        )
                        break
                    if (pos.get("target_price") is not None
                            and float(bar["high"]) >= float(pos["target_price"])):
                        exit_price, reason, exit_ts = (
                            float(pos["target_price"]), "target", ts
                        )
                        break
                else:
                    if float(bar["high"]) >= float(pos["stop_price"]):
                        exit_price, reason, exit_ts = (
                            float(pos["stop_price"]), "stop", ts
                        )
                        break
                    if (pos.get("target_price") is not None
                            and float(bar["low"]) <= float(pos["target_price"])):
                        exit_price, reason, exit_ts = (
                            float(pos["target_price"]), "target", ts
                        )
                        break

            if exit_price is None and now_et.time() >= time(15, 55):
                exit_price, reason, exit_ts = latest, "eod", now_utc

            if exit_price is not None:
                trade = self._close(exit_price, exit_ts, reason)
                self.state["last_update"] = _now_iso()
                self.state["last_decision"] = "exit"
                self.state["last_reason"] = reason
                self.save()
                result.update(action="exit", reason=reason, trade=trade,
                              stats=self.stats())
                return result

            # OD interaction: a Trend long can be switched to OD rules, or
            # reversed if OD is short.
            today = now_et.date()
            od_days = set(str(x) for x in self.state.get("od_days", []))
            if (pos.get("source") == "trend"
                    and str(today) not in od_days
                    and time(9, 40) <= now_et.time() <= time(10, 0)):
                cands = sorted(
                    t for t in od
                    if signal_day(t) == today and t <= now_utc
                )
                if cands:
                    sig_ts = cands[0]
                    age = (now_utc - sig_ts).total_seconds()
                    if age <= MAX_ENTRY_DELAY_SEC:
                        d = int(od[sig_ts])
                        if d == 1:
                            ref = self._previous_close(completed, sig_ts)
                            if ref is None:
                                ref = latest
                            pos["stop_price"] = round(ref * (1 - OD_STOP), 4)
                            pos["target_price"] = None
                            pos["source"] = "od"
                            pos["switched"] = True
                            self.state["od_days"].append(str(today))
                            self.state["od_days"] = self.state["od_days"][-60:]
                            self._mark(latest, now_utc)
                            self.state["last_update"] = _now_iso()
                            self.state["last_decision"] = "switch_od"
                            self.state["last_reason"] = (
                                "Trend long switched to OD rules"
                            )
                            self.save()
                            result.update(action="switch",
                                          reason=self.state["last_reason"],
                                          position=self.state["positions"][0],
                                          stats=self.stats())
                            return result
                        else:
                            trade = self._close(latest, now_utc,
                                                "od_reverse")
                            entry = latest - SLIPPAGE_TICKS * TICK
                            stop = entry * (1 + OD_STOP)
                            self._open(-1, entry, stop, None, "od",
                                       now_utc, abs(entry - stop),
                                       signal_time=sig_ts)
                            self.state["od_days"].append(str(today))
                            self.state["od_days"] = self.state["od_days"][-60:]
                            self.state["last_update"] = _now_iso()
                            self.state["last_decision"] = "reverse_od"
                            self.state["last_reason"] = (
                                "Trend long reversed to OD short"
                            )
                            self.save()
                            result.update(action="reverse",
                                          reason=self.state["last_reason"],
                                          trade=trade,
                                          position=self.state["positions"][0],
                                          stats=self.stats())
                            return result
                    else:
                        self.state["od_days"].append(str(today))
                        self.state["od_days"] = self.state["od_days"][-60:]

            self._mark(latest, now_utc)
            self.state["last_update"] = _now_iso()
            self.state["last_decision"] = "mark"
            self.state["last_reason"] = "open position"
            self.save()
            result.update(action="mark", position=self.state["positions"][0],
                          stats=self.stats())
            return result

        # ---- flat: entries
        if self.is_halted():
            result["reason"] = "halted: new paper entries blocked"
            self.state["last_update"] = _now_iso()
            self.state["last_decision"] = "halted"
            self.state["last_reason"] = result["reason"]
            self.save()
            result["stats"] = self.stats()
            return result

        if not (time(9, 30) <= now_et.time() <= time(15, 35)):
            # No position, no entry window: stay quiet so the GitHub runner
            # does not create a state commit every five minutes.
            result["reason"] = "outside entry window"
            result["stats"] = self.stats()
            return result

        today = now_et.date()
        od_days = set(str(x) for x in self.state.get("od_days", []))

        # ---- Opening Drive: at most one decision per day
        if str(today) not in od_days:
            cands = sorted(
                t for t in od
                if signal_day(t) == today and t <= now_utc
            )
            if cands:
                sig_ts = cands[0]
                age = (now_utc - sig_ts).total_seconds()
                if age > MAX_ENTRY_DELAY_SEC:
                    self.state["od_days"].append(str(today))
                    self.state["od_days"] = self.state["od_days"][-60:]
                    result["reason"] = (
                        f"OD signal {sig_ts.tz_convert(TZ).strftime('%H:%M')} "
                        f"missed by {age / 60:.1f} min"
                    )
                    self.state["last_update"] = _now_iso()
                    self.state["last_decision"] = "skip"
                    self.state["last_reason"] = result["reason"]
                    self.save()
                    result["stats"] = self.stats()
                    return result

                d = int(od[sig_ts])
                entry = latest + d * SLIPPAGE_TICKS * TICK
                stop = entry * (1 - d * OD_STOP)
                self._open(d, entry, stop, None, "od", now_utc,
                           abs(entry - stop), signal_time=sig_ts)
                self.state["od_days"].append(str(today))
                self.state["od_days"] = self.state["od_days"][-60:]
                self.state["last_update"] = _now_iso()
                self.state["last_decision"] = "open_od"
                self.state["last_reason"] = (
                    f"OD {'long' if d == 1 else 'short'} signal "
                    f"{sig_ts.tz_convert(TZ).strftime('%H:%M')}"
                )
                self.save()
                result.update(action="open", reason=self.state["last_reason"],
                              position=self.state["positions"][0],
                              stats=self.stats())
                return result

        # ---- Trend FVG: latest unexpired signal, never on an OD day
        if str(today) not in set(str(x) for x in self.state.get("od_days", [])):
            cands = [
                t for t in trend
                if signal_day(t) == today and t <= now_utc
                and (now_utc - t).total_seconds() <= MAX_ENTRY_DELAY_SEC
            ]
            if cands:
                sig_ts = max(cands)
                _, signal_stop = trend[sig_ts]
                entry = latest + SLIPPAGE_TICKS * TICK
                risk = entry - signal_stop
                if risk > 0:
                    target = entry + RR * risk
                    self._open(1, entry, signal_stop, target, "trend",
                               now_utc, risk, signal_time=sig_ts)
                    self.state["last_update"] = _now_iso()
                    self.state["last_decision"] = "open_trend"
                    self.state["last_reason"] = (
                        "Trend bullish FVG "
                        f"{sig_ts.tz_convert(TZ).strftime('%H:%M')}"
                    )
                    self.save()
                    result.update(action="open",
                                  reason=self.state["last_reason"],
                                  position=self.state["positions"][0],
                                  stats=self.stats())
                    return result

        result["reason"] = "no entry signal"
        self.state["last_update"] = _now_iso()
        self.state["last_decision"] = "idle"
        self.state["last_reason"] = result["reason"]
        self.save()
        result["stats"] = self.stats()
        return result
