"""API tests using FastAPI's TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from scripts.make_sample import make_image

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "rus" in body["languages"]


def test_index_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_recognize_endpoint(tmp_path):
    img = make_image("Тест", tmp_path / "t.png", font_size=52)
    with open(img, "rb") as fh:
        resp = client.post(
            "/api/recognize",
            files={"file": ("t.png", fh, "image/png")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "тест" in body["text"].lower()
    assert body["confidence"] > 0


def test_recognize_rejects_non_image():
    resp = client.post(
        "/api/recognize",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415
