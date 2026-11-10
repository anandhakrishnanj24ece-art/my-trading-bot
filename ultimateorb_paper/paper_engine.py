"""Paper-only execution ledger for the UltimateORB 6.6/10 research setting.

Safety model
------------
* There is no broker adapter, no API key, and no order-routing code here.
* ``open_trade`` refuses to work when the halt flag is set.
* Halt does not block marking or closing an existing paper position.
* The only side effects are local JSON and CSV files under this folder.

This is a manual/CSV-fed paper ledger, not a live trading system. A signal
source must call ``open_trade`` explicitly.
"""

from __future__ import annotations

import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER_ONLY = True

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "paper",
    "strategy": "UltimateORB 6.6/10 paper proxy",
    "target_points": 6.6,
    "stop_points": 10.0,
    "commission_points": 0.95,
    "point_value_usd": 2.0,
    "contracts": 1,
    "starting_capital": 10000.0,
    "max_open_positions": 1,
    "allow_live_orders": False,
}


class TradingHalted(RuntimeError):
    """Raised when a new paper entry is attempted while halted."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


class PaperEngine:
    """Local paper ledger with a hard entry halt switch."""

    def __init__(self, base_dir: str | Path | None = None, starting_capital: float | None = None):
        if not PAPER_ONLY:
            raise RuntimeError("paper-only guard failed")
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.state_dir = self.base_dir / "state"
        self.config_path = self.base_dir / "config.json"
        self.state_path = self.state_dir / "paper_state.json"
        self.control_path = self.state_dir / "control.json"
        self.trades_csv_path = self.state_dir / "paper_trades.csv"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_create_config(starting_capital)
        self.control = self._load_or_create_control()
        self.state = self._load_or_create_state()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load_or_create_config(self, starting_capital: float | None) -> dict[str, Any]:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        else:
            config = dict(DEFAULT_CONFIG)
        if starting_capital is not None:
            config["starting_capital"] = float(starting_capital)
        config["mode"] = "paper"
        config["allow_live_orders"] = False
        self.config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return config

    def _load_or_create_control(self) -> dict[str, Any]:
        if self.control_path.exists():
            with self.control_path.open("r", encoding="utf-8") as handle:
                control = json.load(handle)
        else:
            control = {"halted": False, "reason": "", "updated_at": None}
        control.setdefault("halted", False)
        control.setdefault("reason", "")
        control.setdefault("updated_at", None)
        self._save_json(self.control_path, control)
        return control

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            capital = float(self.config["starting_capital"])
            state = {
                "capital": capital,
                "cash": capital,
                "equity": capital,
                "realized_pnl": 0.0,
                "positions": [],
                "trades": [],
                "equity_curve": [{"timestamp": _now_iso(), "equity": capital}],
                "last_update": None,
                "last_price": None,
            }
        state.setdefault("positions", [])
        state.setdefault("trades", [])
        state.setdefault("equity_curve", [])
        state.setdefault("cash", state.get("capital", self.config["starting_capital"]))
        state.setdefault("equity", state["cash"])
        state.setdefault("realized_pnl", 0.0)
        state.setdefault("last_update", None)
        state.setdefault("last_price", None)
        self._save_json(self.state_path, state)
        return state

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _save_state(self) -> None:
        self._save_json(self.state_path, self.state)

    # ------------------------------------------------------------------
    # halt / resume
    # ------------------------------------------------------------------
    def set_halted(self, halted: bool, reason: str = "") -> dict[str, Any]:
        self.control = {
            "halted": bool(halted),
            "reason": reason if halted else "",
            "updated_at": _now_iso(),
        }
        self._save_json(self.control_path, self.control)
        return self.control

    def halt(self, reason: str = "Manual dashboard halt") -> dict[str, Any]:
        return self.set_halted(True, reason)

    def resume(self) -> dict[str, Any]:
        return self.set_halted(False, "")

    def is_halted(self) -> bool:
        return bool(self.control.get("halted", False))

    # ------------------------------------------------------------------
    # entries / exits
    # ------------------------------------------------------------------
    def open_trade(
        self,
        direction: str,
        entry_price: float,
        timestamp: str | None = None,
        target_points: float | None = None,
        stop_points: float | None = None,
        contracts: int | None = None,
    ) -> dict[str, Any]:
        if self.is_halted():
            raise TradingHalted(
                "Paper entries are halted. Resume from the dashboard/CLI before opening a trade."
            )

        direction = direction.strip().lower()
        if direction not in {"long", "short"}:
            raise ValueError("direction must be 'long' or 'short'")
        entry_price = float(entry_price)
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError("entry_price must be a positive finite number")

        max_positions = int(self.config.get("max_open_positions", 1))
        if len(self.state["positions"]) >= max_positions:
            raise ValueError(f"max_open_positions={max_positions} already reached")

        target_points = float(target_points if target_points is not None else self.config["target_points"])
        stop_points = float(stop_points if stop_points is not None else self.config["stop_points"])
        if target_points <= 0 or stop_points <= 0:
            raise ValueError("target_points and stop_points must be positive")

        contracts = int(contracts if contracts is not None else self.config["contracts"])
        if contracts < 1:
            raise ValueError("contracts must be >= 1")

        if direction == "long":
            target = entry_price + target_points
            stop = entry_price - stop_points
        else:
            target = entry_price - target_points
            stop = entry_price + stop_points

        position = {
            "id": uuid.uuid4().hex[:12],
            "direction": direction,
            "entry_price": _round(entry_price),
            "target_price": _round(target),
            "stop_price": _round(stop),
            "target_points": _round(target_points),
            "stop_points": _round(stop_points),
            "contracts": contracts,
            "opened_at": timestamp or _now_iso(),
            "last_price": _round(entry_price),
            "unrealized_points": 0.0,
            "unrealized_usd": 0.0,
        }
        self.state["positions"].append(position)
        self.state["last_price"] = _round(entry_price)
        self.state["last_update"] = _now_iso()
        self._mark_equity()
        self._save_state()
        return position

    def mark(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Mark a bar and close any position whose target/stop was touched.

        If both barriers are touched in the same bar, the stop is assumed first.
        That is the conservative paper-fill convention.
        """
        high, low, close = float(high), float(low), float(close)
        if not all(math.isfinite(x) for x in (high, low, close)):
            raise ValueError("high/low/close must be finite")
        if low > high:
            raise ValueError("low cannot be greater than high")

        closed: list[dict[str, Any]] = []
        for position in list(self.state["positions"]):
            direction = position["direction"]
            stop = float(position["stop_price"])
            target = float(position["target_price"])

            stop_hit = low <= stop if direction == "long" else high >= stop
            target_hit = high >= target if direction == "long" else low <= target

            exit_price: float | None = None
            reason = ""
            if stop_hit:
                exit_price = stop
                reason = "stop"
            elif target_hit:
                exit_price = target
                reason = "target"

            if exit_price is not None:
                closed.append(self._close_position(position, exit_price, reason, timestamp))
            else:
                position["last_price"] = _round(close)
                position["unrealized_points"] = _round(self._points(position, close))
                position["unrealized_usd"] = _round(
                    self._points(position, close) * float(self.config["point_value_usd"]) * int(position["contracts"])
                )

        self.state["last_price"] = _round(close)
        self.state["last_update"] = _now_iso()
        self._mark_equity()
        self._save_state()
        return {"closed": closed, "positions": self.state["positions"]}

    def close_manual(self, price: float, reason: str = "manual", timestamp: str | None = None) -> dict[str, Any]:
        if not self.state["positions"]:
            raise ValueError("no open paper position")
        position = self.state["positions"][0]
        closed = self._close_position(position, float(price), reason, timestamp)
        self.state["last_price"] = _round(float(price))
        self.state["last_update"] = _now_iso()
        self._mark_equity()
        self._save_state()
        return closed

    def _points(self, position: dict[str, Any], price: float) -> float:
        if position["direction"] == "long":
            return float(price) - float(position["entry_price"])
        return float(position["entry_price"]) - float(price)

    def _close_position(
        self,
        position: dict[str, Any],
        exit_price: float,
        reason: str,
        timestamp: str | None,
    ) -> dict[str, Any]:
        gross_points = self._points(position, exit_price)
        commission_points = float(self.config["commission_points"]) * int(position["contracts"])
        net_points = gross_points - commission_points
        point_value = float(self.config["point_value_usd"])
        net_usd = net_points * point_value * int(position["contracts"])

        trade = {
            "id": position["id"],
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": _round(exit_price),
            "target_price": position["target_price"],
            "stop_price": position["stop_price"],
            "contracts": position["contracts"],
            "opened_at": position["opened_at"],
            "closed_at": timestamp or _now_iso(),
            "gross_points": _round(gross_points),
            "commission_points": _round(commission_points),
            "net_points": _round(net_points),
            "net_usd": _round(net_usd, 2),
            "reason": reason,
        }
        self.state["trades"].append(trade)
        self.state["positions"] = [p for p in self.state["positions"] if p["id"] != position["id"]]
        self.state["cash"] = _round(float(self.state["cash"]) + net_usd, 2)
        self.state["realized_pnl"] = _round(float(self.state["realized_pnl"]) + net_usd, 2)
        self._append_trade_csv(trade)
        return trade

    def _append_trade_csv(self, trade: dict[str, Any]) -> None:
        fields = [
            "id", "direction", "entry_price", "exit_price", "target_price", "stop_price",
            "contracts", "opened_at", "closed_at", "gross_points", "commission_points",
            "net_points", "net_usd", "reason",
        ]
        exists = self.trades_csv_path.exists()
        with self.trades_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: trade.get(key) for key in fields})

    def _mark_equity(self) -> None:
        unrealized = sum(
            self._points(p, float(p["last_price"])) * float(self.config["point_value_usd"]) * int(p["contracts"])
            for p in self.state["positions"]
        )
        self.state["equity"] = _round(float(self.state["cash"]) + unrealized, 2)
        curve = self.state["equity_curve"]
        point = {"timestamp": _now_iso(), "equity": self.state["equity"]}
        if not curve or abs(float(curve[-1]["equity"]) - self.state["equity"]) > 1e-9:
            curve.append(point)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        trades = self.state["trades"]
        wins = [t for t in trades if float(t["net_usd"]) > 0]
        losses = [t for t in trades if float(t["net_usd"]) < 0]
        gross_profit = sum(float(t["net_usd"]) for t in wins)
        gross_loss = abs(sum(float(t["net_usd"]) for t in losses))
        equity_curve = self.state["equity_curve"]
        peak = -math.inf
        max_dd = 0.0
        for point in equity_curve:
            equity = float(point["equity"])
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        return {
            "equity": self.state["equity"],
            "cash": self.state["cash"],
            "realized_pnl": self.state["realized_pnl"],
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
            "avg_win_usd": (gross_profit / len(wins)) if wins else 0.0,
            "avg_loss_usd": (-gross_loss / len(losses)) if losses else 0.0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
            "max_drawdown_pct": max_dd * 100,
            "open_positions": len(self.state["positions"]),
            "halted": self.is_halted(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "control": self.control,
            "state": self.state,
            "stats": self.stats(),
            "paper_only": True,
        }

    def export_trades(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.trades_csv_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "id", "direction", "entry_price", "exit_price", "target_price", "stop_price",
            "contracts", "opened_at", "closed_at", "gross_points", "commission_points",
            "net_points", "net_usd", "reason",
        ]
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for trade in self.state["trades"]:
                writer.writerow({key: trade.get(key) for key in fields})
        return target
