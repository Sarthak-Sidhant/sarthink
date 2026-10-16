"""PIN + password login for the hosted Sarthink (one shared login across s1d.me apps).

Credentials live in one shared file, so every app on the server uses the same PIN and password:

  S1D_AUTH_FILE (default /etc/s1d/auth.env), or the same keys as environment variables:
    S1D_AUTH_PIN_HASH=scrypt$...        hash of the PIN
    S1D_AUTH_PASSWORD_HASH=scrypt$...   hash of the password
    S1D_AUTH_SECRET=<random>            signs login cookies; keep it the same across restarts

Make the file with:  python3 scripts/api/auth.py init [path]   (asks for the PIN and password)

With no credentials configured the login is off (local use on a laptop). Set SARTHINK_REQUIRE_AUTH=1
on a server so it refuses to start without them.
"""
import base64
import getpass
import hashlib
import hmac
import html
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

COOKIE = "s1d_session"
SESSION_DAYS = 30
MAX_FAILS, LOCKOUT_S = 5, 15 * 60          # per client IP; the PIN alone is guessable, so lock out early
GLOBAL_MAX_FAILS = 20                      # across all IPs, so spoofed forwarding headers can't dodge the limit
OPEN_PATHS = {"/login", "/logout", "/api/health"}


def hash_secret(value, salt=None):
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(value.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def check_secret(value, stored):
    try:
        scheme, salt, dk = stored.split("$")
        if scheme != "scrypt":
            return False
        return hmac.compare_digest(hash_secret(value, base64.b64decode(salt)), stored)
    except (ValueError, TypeError):
        return False


def _read_env_file(path):
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def load_config():
    cfg = _read_env_file(os.environ.get("S1D_AUTH_FILE", "/etc/s1d/auth.env"))
    for k in ("S1D_AUTH_PIN_HASH", "S1D_AUTH_PASSWORD_HASH", "S1D_AUTH_SECRET"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    if not (cfg.get("S1D_AUTH_PIN_HASH") and cfg.get("S1D_AUTH_PASSWORD_HASH")):
        return None
    if not cfg.get("S1D_AUTH_SECRET"):
        cfg["S1D_AUTH_SECRET"] = secrets.token_hex(32)   # logins then last until the next restart
    return cfg


class Auth:
    def __init__(self, cfg):
        self.pin_hash, self.pw_hash = cfg["S1D_AUTH_PIN_HASH"], cfg["S1D_AUTH_PASSWORD_HASH"]
        self.key = cfg["S1D_AUTH_SECRET"].encode()
        self.fails, self.lock = {}, threading.Lock()

    def make_cookie(self):
        exp = str(int(time.time()) + SESSION_DAYS * 86400)
        return exp + "." + hmac.new(self.key, exp.encode(), hashlib.sha256).hexdigest()

    def cookie_ok(self, value):
        exp, _, sig = (value or "").partition(".")
        good = hmac.new(self.key, exp.encode(), hashlib.sha256).hexdigest()
        return exp.isdigit() and hmac.compare_digest(sig, good) and int(exp) > time.time()

    def locked_out(self, ip):
        with self.lock:
            for key, limit in ((ip, MAX_FAILS), ("*", GLOBAL_MAX_FAILS)):
                n, since = self.fails.get(key, (0, 0))
                if time.time() - since >= LOCKOUT_S:
                    self.fails.pop(key, None)
                elif n >= limit:
                    return True
            return False

    def attempt(self, ip, pin, password):
        ok = check_secret(pin, self.pin_hash) & check_secret(password, self.pw_hash)   # check both, always
        with self.lock:
            if ok:
                self.fails.pop(ip, None)
            else:
                for key in (ip, "*"):
                    n, _ = self.fails.get(key, (0, 0))
                    self.fails[key] = (n + 1, time.time())
        return ok


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sarthink · sign in</title><style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0d12;color:#e6e8ee;font:15px system-ui,sans-serif}}
form{{width:min(320px,90vw);display:grid;gap:12px}} h1{{font-size:20px;margin:0 0 8px}}
input,button{{font:inherit;padding:10px 12px;border-radius:8px;border:1px solid #2a2f3a;background:#141821;color:inherit}}
button{{background:#3b5bdb;border-color:#3b5bdb;cursor:pointer}} .err{{color:#ff8787;min-height:1.2em;margin:0}}
</style></head><body><form method="post" action="/login">
<h1>Sarthink</h1>
<input name="pin" type="password" inputmode="numeric" autocomplete="off" placeholder="PIN" required autofocus>
<input name="password" type="password" autocomplete="current-password" placeholder="Password" required>
<input type="hidden" name="next" value="{next}">
<button type="submit">Sign in</button><p class="err">{error}</p>
</form></body></html>"""


def _client_ip(request):
    # The app only listens on 127.0.0.1, so the client is whatever the proxy in front appended last
    # (earlier X-Forwarded-For entries are client-supplied).
    if request.headers.get("cf-connecting-ip"):
        return request.headers["cf-connecting-ip"].strip()
    if request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[-1].strip()
    return request.client.host if request.client else "?"


def _safe_next(n):
    return n if n.startswith("/") and not n.startswith("//") else "/"


def install(app):
    """Protect every route of a FastAPI app with the shared login. Returns False when no login is configured."""
    from fastapi import Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

    cfg = load_config()
    if cfg is None:
        if os.environ.get("SARTHINK_REQUIRE_AUTH") == "1":
            sys.exit("SARTHINK_REQUIRE_AUTH=1 but no PIN/password configured (see scripts/api/auth.py).")
        return False
    auth = Auth(cfg)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path in OPEN_PATHS or auth.cookie_ok(request.cookies.get(COOKIE)):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "login required"}, status_code=401)
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target)}", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    def login_form(next: str = "/"):
        return LOGIN_PAGE.format(next=html.escape(_safe_next(next)), error="")

    @app.post("/login")
    def login(request: Request, pin: str = Form(...), password: str = Form(...), next: str = Form("/")):
        ip = _client_ip(request)
        nxt = html.escape(_safe_next(next))
        if auth.locked_out(ip):
            return HTMLResponse(LOGIN_PAGE.format(next=nxt, error="Too many attempts. Try again in 15 minutes."),
                                status_code=429)
        if not auth.attempt(ip, pin, password):
            return HTMLResponse(LOGIN_PAGE.format(next=nxt, error="Wrong PIN or password."), status_code=401)
        r = RedirectResponse(_safe_next(next), status_code=303)
        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        r.set_cookie(COOKIE, auth.make_cookie(), max_age=SESSION_DAYS * 86400, httponly=True, secure=secure,
                     samesite="lax")
        return r

    @app.get("/logout")
    def logout():
        r = RedirectResponse("/login", status_code=303)
        r.delete_cookie(COOKIE)
        return r

    return True


def _init(path):
    pin = getpass.getpass("PIN: ")
    if getpass.getpass("PIN again: ") != pin:
        sys.exit("PINs don't match.")
    pw = getpass.getpass("Password: ")
    if getpass.getpass("Password again: ") != pw:
        sys.exit("Passwords don't match.")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"S1D_AUTH_PIN_HASH={hash_secret(pin)}\nS1D_AUTH_PASSWORD_HASH={hash_secret(pw)}\n"
                 f"S1D_AUTH_SECRET={secrets.token_hex(32)}\n")
    os.chmod(p, 0o640)
    print(f"Wrote {p} (mode 640). Give the app's user read access, e.g. chgrp s1d {p}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        _init(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("S1D_AUTH_FILE", "/etc/s1d/auth.env"))
    else:
        print(__doc__)
