"""API tests using FastAPI's TestClient."""

from __future__ import annotations

import time

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


def _run_to_completion(img_path, params=None, timeout_s=60):
    with open(img_path, "rb") as fh:
        resp = client.post(
            "/api/recognize",
            params=params or {},
            files={"file": (img_path.name, fh, "image/png")},
        )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f"/api/progress/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


def test_recognize_endpoint(tmp_path):
    img = make_image("Тест", tmp_path / "t.png", font_size=52)
    job = _run_to_completion(img, params={"engine": "tesseract"})
    assert job["status"] == "done", job.get("error")
    assert "тест" in job["text"].lower()
    assert job["confidence"] > 0
    assert job["timing"]["total_ms"] >= 0


def test_metrics_endpoint():
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "cpu_percent" in body
    assert "ram_percent" in body


def test_recognize_rejects_non_image():
    resp = client.post(
        "/api/recognize",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415
