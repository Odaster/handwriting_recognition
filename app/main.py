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
    engine: str = "tesseract",
    spellcheck: bool = False,
) -> JSONResponse:
    """Recognize text on an uploaded image.

    ``engine`` selects the backend:
      * ``tesseract`` — fast, best for printed/neat text (default);
      * ``trocr`` — neural handwriting model for cursive Russian (heavier/slower,
        requires ``requirements-trocr.txt``).

    ``spellcheck`` additionally returns a dictionary-corrected version of the text.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )
    if engine not in {"tesseract", "trocr"}:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    try:
        if engine == "trocr":
            from app.htr import recognize_bytes as recognize_handwriting

            result = recognize_handwriting(data)
        else:
            result = recognize_bytes(data, lang=lang)
    except RuntimeError as exc:
        # Missing optional dependencies for the handwriting engine.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface any decode/OCR failure to the client
        raise HTTPException(status_code=422, detail=f"Recognition failed: {exc}") from exc

    payload = result.to_dict()

    if spellcheck:
        try:
            from app.postprocess import correct_text

            payload["text_corrected"] = correct_text(payload["text"])
        except Exception as exc:  # noqa: BLE001 - spellcheck is best-effort
            payload["text_corrected"] = None
            payload["spellcheck_error"] = str(exc)

    return JSONResponse(payload)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
