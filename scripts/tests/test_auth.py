"""PIN + password login (scripts/api/auth.py). Run: python3 -m pytest scripts/tests/test_auth.py"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import auth  # noqa: E402


def make_app(tmp_path, monkeypatch, configured=True):
    f = tmp_path / "auth.env"
    if configured:
        f.write_text(f"S1D_AUTH_PIN_HASH={auth.hash_secret('4821')}\n"
                     f"S1D_AUTH_PASSWORD_HASH={auth.hash_secret('correct horse')}\nS1D_AUTH_SECRET=test\n")
    monkeypatch.setenv("S1D_AUTH_FILE", str(f))
    app = FastAPI()
    on = auth.install(app)

    @app.get("/")
    def home():
        return {"page": "viewer"}

    @app.get("/api/search")
    def search():
        return {"hits": []}

    @app.get("/api/health")
    def health():
        return {"ok": True}

    return on, TestClient(app)


def test_off_without_credentials(tmp_path, monkeypatch):
    on, c = make_app(tmp_path, monkeypatch, configured=False)
    assert not on and c.get("/api/search").status_code == 200


def test_require_auth_refuses_to_start(tmp_path, monkeypatch):
    monkeypatch.setenv("SARTHINK_REQUIRE_AUTH", "1")
    with pytest.raises(SystemExit):
        make_app(tmp_path, monkeypatch, configured=False)


def test_blocks_until_login(tmp_path, monkeypatch):
    on, c = make_app(tmp_path, monkeypatch)
    assert on
    assert c.get("/api/search").status_code == 401
    r = c.get("/?x=1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=/%3Fx%3D1"
    assert c.get("/api/health").status_code == 200
    assert "PIN" in c.get("/login").text


def test_needs_both_pin_and_password(tmp_path, monkeypatch):
    _, c = make_app(tmp_path, monkeypatch)
    assert c.post("/login", data={"pin": "4821", "password": "wrong"}).status_code == 401
    assert c.post("/login", data={"pin": "0000", "password": "correct horse"}).status_code == 401
    r = c.post("/login", data={"pin": "4821", "password": "correct horse", "next": "/"}, follow_redirects=False)
    assert r.status_code == 303 and auth.COOKIE in r.cookies
    assert c.get("/api/search").status_code == 200
    c.get("/logout")
    assert c.get("/api/search").status_code == 401


def test_open_redirect_and_forged_cookie(tmp_path, monkeypatch):
    _, c = make_app(tmp_path, monkeypatch)
    r = c.post("/login", data={"pin": "4821", "password": "correct horse", "next": "//evil.com"},
               follow_redirects=False)
    assert r.headers["location"] == "/"
    c.cookies.clear()
    c.cookies.set(auth.COOKIE, "9999999999.deadbeef")
    assert c.get("/api/search").status_code == 401


def test_lockout_after_repeated_failures(tmp_path, monkeypatch):
    _, c = make_app(tmp_path, monkeypatch)
    for _ in range(auth.MAX_FAILS):
        c.post("/login", data={"pin": "0000", "password": "x"})
    r = c.post("/login", data={"pin": "4821", "password": "correct horse"})
    assert r.status_code == 429


def test_global_lockout_ignores_spoofed_ips(tmp_path, monkeypatch):
    _, c = make_app(tmp_path, monkeypatch)
    for i in range(auth.GLOBAL_MAX_FAILS):
        c.post("/login", data={"pin": "0000", "password": "x"}, headers={"x-forwarded-for": f"10.0.0.{i}"})
    r = c.post("/login", data={"pin": "4821", "password": "correct horse"}, headers={"x-forwarded-for": "10.9.9.9"})
    assert r.status_code == 429
