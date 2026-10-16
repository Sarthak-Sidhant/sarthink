#!/usr/bin/env python3
"""Serve the Sarthink web app on http://127.0.0.1:8080 and proxy the API to the backend.

  python3 web/serve.py                       # backend at http://127.0.0.1:8000 (python3 run.py)
  python3 web/serve.py --api http://host:8000 --port 8080

Static files come from web/; /api/* and /processed_data/* go to the backend unchanged
(streamed, so the assistant's server-sent progress events arrive live).
"""
import argparse
import http.client
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

WEB = Path(__file__).resolve().parent
PROXY_PREFIXES = ("/api/", "/processed_data/")
LOGIN_PATHS = ("/login", "/logout")       # the backend's sign-in pages (only when it has a login configured)
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "host", "content-length"}
mimetypes.add_type("text/javascript", ".js")


class Handler(BaseHTTPRequestHandler):
    api = None

    def log_message(self, fmt, *args):   # quiet: one line per API call, nothing for static files
        if self.path.startswith("/api/"):
            sys.stderr.write(f"{self.command} {self.path[:120]} {args[1] if len(args) > 1 else ''}\n")

    def _to_backend(self):
        return self.path.startswith(PROXY_PREFIXES) or urlsplit(self.path).path in LOGIN_PATHS

    def do_GET(self):
        self._proxy() if self._to_backend() else self._static()

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self._proxy() if self._to_backend() else self.send_error(405)

    def do_DELETE(self):          # people: remove an account from a person, remove a person
        self._proxy() if self.path.startswith(PROXY_PREFIXES) else self.send_error(405)

    def _static(self):
        rel = urlsplit(self.path).path.lstrip("/") or "index.html"
        f = (WEB / rel).resolve()
        if WEB not in f.parents or not f.is_file() or f.suffix == ".py":
            f = WEB / "index.html"                  # single-page app: unknown paths get the shell
        if f.suffix == ".html" and self._needs_login():
            self.send_response(303)
            self.send_header("Location", "/login?next=" + quote(self.path, safe=""))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _needs_login(self):
        """True when the backend has a login and this browser isn't signed in (its pages need it too)."""
        u = urlsplit(self.api)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
        try:
            conn.request("GET", "/api/auth/check", headers={"Cookie": self.headers.get("Cookie", "")})
            return conn.getresponse().status == 401
        except OSError:
            return False                            # backend down: the page loads and shows the API error
        finally:
            conn.close()

    def _proxy(self):
        u = urlsplit(self.api)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            r = conn.getresponse()
        except OSError as e:
            msg = f'{{"detail": "backend unreachable at {self.api}: {e}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while chunk := r.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="Sarthink backend (scripts/api/server.py)")
    a = ap.parse_args()
    Handler.api = a.api.rstrip("/")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Sarthink web · http://{a.host}:{a.port}/  (API → {Handler.api})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
