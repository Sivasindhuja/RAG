from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.main as backend_main  # noqa: E402


def test_parse_cors_origins_defaults_to_local_dev_when_unset():
    origins = backend_main._parse_cors_origins(None)
    assert origins == backend_main.DEFAULT_DEV_CORS_ORIGINS
    assert "*" not in origins


def test_parse_cors_origins_defaults_to_local_dev_when_empty():
    origins = backend_main._parse_cors_origins("  ,  ,, ")
    assert origins == backend_main.DEFAULT_DEV_CORS_ORIGINS


def test_parse_cors_origins_supports_explicit_and_multiple_origins():
    origins = backend_main._parse_cors_origins("https://a.example, https://b.example ,,http://localhost:5173")
    assert origins == ["https://a.example", "https://b.example", "http://localhost:5173"]


def test_parse_cors_origins_explicit_wildcard_only_when_configured():
    origins = backend_main._parse_cors_origins("*,https://a.example")
    assert origins == ["*"]


def _cors_test_client(origins):
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return TestClient(app)


def test_cors_allows_explicit_origin():
    client = _cors_test_client(["https://allowed.example"])
    resp = client.get("/ping", headers={"Origin": "https://allowed.example"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "https://allowed.example"


def test_cors_rejects_unlisted_origin_by_omitting_allow_origin():
    client = _cors_test_client(["https://allowed.example"])
    resp = client.get("/ping", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_cors_multiple_origins_allow_listed_and_reject_unlisted():
    client = _cors_test_client(["https://a.example", "https://b.example"])
    a = client.get("/ping", headers={"Origin": "https://a.example"})
    b = client.get("/ping", headers={"Origin": "https://b.example"})
    c = client.get("/ping", headers={"Origin": "https://c.example"})
    assert a.headers.get("access-control-allow-origin") == "https://a.example"
    assert b.headers.get("access-control-allow-origin") == "https://b.example"
    assert "access-control-allow-origin" not in c.headers


def test_cors_explicit_wildcard_allows_any_origin_header():
    client = _cors_test_client(["*"])
    resp = client.get("/ping", headers={"Origin": "https://any.example"})
    assert resp.status_code == 200
    # Starlette returns "*" for wildcard allow_origins
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_local_dev_default_cors_is_usable_for_vite_origin():
    origins = backend_main._parse_cors_origins(None)
    client = _cors_test_client(origins)
    resp = client.get("/ping", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
