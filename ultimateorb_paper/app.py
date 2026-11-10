"""Local dashboard for the UltimateORB paper ledger.

Run:
    python -m ultimateorb_paper.app

The server binds to 127.0.0.1 only. There is no broker connection and no live
order path. The halt button writes state/control.json; the engine refuses new
entries while that flag is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .paper_engine import PaperEngine, TradingHalted

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard.html"


class DashboardHandler(BaseHTTPRequestHandler):
    engine: PaperEngine

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("invalid body")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/dashboard", "/dashboard.html"}:
            self._send_html(DASHBOARD)
        elif path == "/api/state":
            self._send_json(self.engine.snapshot())
        elif path == "/api/trades.csv":
            csv_path = self.engine.export_trades()
            body = csv_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="ultimateorb_paper_trades.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            data = self._read_json()
            if path == "/api/halt":
                result = self.engine.halt(str(data.get("reason", "Manual dashboard halt")))
                self._send_json({"ok": True, "control": result})
            elif path == "/api/resume":
                result = self.engine.resume()
                self._send_json({"ok": True, "control": result})
            elif path == "/api/open":
                position = self.engine.open_trade(
                    direction=str(data.get("direction", "")),
                    entry_price=float(data["price"]),
                    timestamp=data.get("timestamp"),
                    target_points=float(data["target_points"]) if data.get("target_points") is not None else None,
                    stop_points=float(data["stop_points"]) if data.get("stop_points") is not None else None,
                )
                self._send_json({"ok": True, "position": position})
            elif path == "/api/mark":
                result = self.engine.mark(
                    high=float(data["high"]),
                    low=float(data["low"]),
                    close=float(data["close"]),
                    timestamp=data.get("timestamp"),
                )
                self._send_json({"ok": True, **result})
            elif path == "/api/close":
                trade = self.engine.close_manual(
                    price=float(data["price"]),
                    reason=str(data.get("reason", "manual")),
                    timestamp=data.get("timestamp"),
                )
                self._send_json({"ok": True, "trade": trade})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except TradingHalted as exc:
            self._send_json({"ok": False, "error": str(exc), "halted": True}, 409)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="UltimateORB paper dashboard (local only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--base", default=None, help="folder containing config.json/state")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing to bind a non-loopback address. This dashboard has no auth.")

    engine = PaperEngine(base_dir=Path(args.base) if args.base else None)
    DashboardHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"UltimateORB paper dashboard: http://{args.host}:{args.port}")
    print("PAPER ONLY — no broker, no live orders. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
