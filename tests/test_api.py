from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from app.main import create_app
from app.ocr import FakeOCREngine
from tests.conftest import make_png_bytes


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine"] == "fake"
    assert payload["ready"] is True


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Распознать" in response.text


def test_static_assets(client):
    css = client.get("/static/styles.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "dropzone" in css.text
    assert "/api/recognize" in js.text


def test_ui_has_copy_and_download(client):
    html = client.get("/").text
    assert "Копировать" in html
    assert "Скачать .txt" in html


def test_recognize_success(client):
    response = client.post(
        "/api/recognize",
        files={"file": ("note.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "Привет, мир" in payload["text"]
    assert payload["engine"] == "fake"
    assert payload["lines"][0]["confidence"] == 0.91
    assert payload["average_confidence"] == 0.91


def test_recognize_without_enhance(client):
    response = client.post(
        "/api/recognize?enhance=false",
        files={"file": ("note.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["text"].startswith("Привет")


def test_empty_file_rejected(client):
    response = client.post(
        "/api/recognize",
        files={"file": ("note.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "Пустой" in response.json()["detail"]


def test_unsupported_format_rejected(client):
    response = client.post(
        "/api/recognize",
        files={"file": ("note.gif", b"GIF89a-not-really", "image/gif")},
    )
    assert response.status_code == 400


def test_broken_image_rejected(client):
    response = client.post(
        "/api/recognize",
        files={"file": ("note.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
    assert "прочитать" in response.json()["detail"]


def test_no_text_found():
    empty_client = TestClient(create_app(engine=FakeOCREngine(text="")))
    response = empty_client.post(
        "/api/recognize",
        files={"file": ("note.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert "не найден" in response.json()["detail"].lower()


def test_too_large_file_rejected(client, monkeypatch):
    monkeypatch.setattr("app.preprocess.MAX_UPLOAD_BYTES", 32)
    response = client.post(
        "/api/recognize",
        files={"file": ("note.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert "большой" in response.json()["detail"]


def test_jpeg_upload(client):
    buffer = BytesIO()
    from PIL import Image

    Image.new("RGB", (32, 32), (240, 240, 240)).save(buffer, format="JPEG")
    response = client.post(
        "/api/recognize",
        files={"file": ("page.jpg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
