"""Lightweight local HTTP server for both A-Share Sentiment Command Center & Options Intraday Monitor.

Routes:
  GET  /                     -> results_live/index.html (Sentiment Command Center)
  GET  /options              -> results_live/options_intraday.html (Options Intraday Monitor)
  GET  /api/status           -> Pipeline status & metadata
  GET  /api/payload          -> Latest daily sentiment JSON
  POST /api/refresh          -> Run daily_update.py --fast (~3.5s)
  GET  /api/options          -> Latest options intraday JSON
  POST /api/options/refresh  -> Run options_intraday.py (~6-7s, live 5,200 stocks + 5 option chains)

Usage:
  .venv/bin/python serve_dashboard.py --port 8765
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from daily_update import run_update, LIVE_DIR
from options_intraday import run_options_snapshot

_LOCK_DAILY = threading.Lock()
_LOCK_OPT = threading.Lock()


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, path) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path_only = self.path.split("?", 1)[0]
        if path_only in ("/", "/index.html"):
            p = LIVE_DIR / "index.html"
            if not p.exists():
                run_update(fast=True)
            self._send_html(p)
        elif path_only in ("/options", "/options.html"):
            p = LIVE_DIR / "options_intraday.html"
            if not p.exists():
                run_options_snapshot()
            self._send_html(p)
        elif path_only == "/api/status":
            p = LIVE_DIR / "live_payload.json"
            meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {}) if p.exists() else {}
            self._send_json(200, {"ok": True, "meta": meta, "options_enabled": True})
        elif path_only == "/api/payload":
            p = LIVE_DIR / "live_payload.json"
            if not p.exists():
                self._send_json(404, {"ok": False, "error": "No payload yet"})
                return
            self._send_json(200, json.loads(p.read_text(encoding="utf-8")))
        elif path_only == "/api/options":
            p = LIVE_DIR / "options_intraday.json"
            if not p.exists():
                run_options_snapshot()
            self._send_json(200, json.loads(p.read_text(encoding="utf-8")))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        path_only = self.path.split("?", 1)[0]
        if path_only == "/api/refresh":
            if not _LOCK_DAILY.acquire(blocking=False):
                self._send_json(409, {"ok": False, "error": "Daily refresh already in progress"})
                return
            try:
                payload = run_update(fast=True)
                self._send_json(200, {"ok": True, "meta": payload["meta"]})
            except Exception as e:
                self._send_json(500, {"ok": False, "error": str(e)})
            finally:
                _LOCK_DAILY.release()
        elif path_only == "/api/options/refresh":
            if not _LOCK_OPT.acquire(blocking=False):
                self._send_json(409, {"ok": False, "error": "Options refresh already in progress"})
                return
            try:
                payload = run_options_snapshot()
                self._send_json(200, {"ok": True, "meta": payload["meta"]})
            except Exception as e:
                self._send_json(500, {"ok": False, "error": str(e)})
            finally:
                _LOCK_OPT.release()
        else:
            self.send_error(404, "Not Found")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Sentiment Command Center & Options Intraday Monitor")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN/VPS)")
    ap.add_argument("--port", type=int, default=8765, help="Bind port")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard server listening on http://{args.host}:{args.port} (Main: / | Options: /options)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
