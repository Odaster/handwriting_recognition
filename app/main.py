"""FastAPI application exposing the Russian OCR pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.ocr import DEFAULT_LANG, recognize_bytes, tesseract_info

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",
    "image/webp",
}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

app = FastAPI(title="Russian Handwriting Recognition", version="0.1.0")


@app.get("/api/health")
def health() -> dict:
    """Report service status plus the detected Tesseract version/languages."""
    return {"status": "ok", **tesseract_info()}


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    lang: str = DEFAULT_LANG,
) -> JSONResponse:
    """Accept an uploaded image and return recognized text with confidence."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    try:
        result = recognize_bytes(data, lang=lang)
    except Exception as exc:  # noqa: BLE001 - surface any decode/OCR failure to the client
        raise HTTPException(status_code=422, detail=f"Recognition failed: {exc}") from exc

    return JSONResponse(result.to_dict())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
