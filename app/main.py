"""FastAPI application exposing the Russian OCR pipeline."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import jobs, metrics
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


CORRECTORS = {"none", "spell", "context"}
ENGINES = {"tesseract", "trocr", "chandra"}


def _run_job(job_id: str, data: bytes, lang: str, engine: str, corrector: str) -> None:
    """Background worker: recognize + (optionally) correct, updating the job."""
    start = time.time()
    jobs.update(job_id, status="running", phase="recognize", progress=0.0)

    def progress(fraction, phase="recognize", **extra):
        fields = {"phase": phase}
        if fraction is not None:
            fields["progress"] = max(0.0, min(0.999, float(fraction)))
        fields.update(extra)
        jobs.update(job_id, **fields)

    try:
        image = Image.open(io.BytesIO(data))
        if engine == "chandra":
            from app.chandra_engine import recognize_image as rec

            result = rec(image, progress=progress)
        elif engine == "trocr":
            from app.htr import recognize_image as rec

            result = rec(image, progress=progress)
        else:
            progress(0.2, "recognize")
            result = recognize_bytes(data, lang=lang)

        recognize_ms = int((time.time() - start) * 1000)
        payload = result.to_dict()
        jobs.update(
            job_id,
            text=payload["text"],
            words=len(payload.get("words", [])),
            confidence=payload.get("confidence"),
        )

        corrected = None
        correct_start = time.time()
        if corrector != "none":
            jobs.update(job_id, phase="correct", progress=0.99)
            try:
                if corrector == "context":
                    from app.corrector import correct_text
                else:
                    from app.postprocess import correct_text
                corrected = correct_text(payload["text"])
            except Exception as exc:  # noqa: BLE001 - correction is best-effort
                jobs.update(job_id, corrector_error=str(exc))
        correct_ms = int((time.time() - correct_start) * 1000)

        jobs.update(
            job_id,
            status="done",
            phase="done",
            progress=1.0,
            text_corrected=corrected,
            timing={
                "recognize_ms": recognize_ms,
                "correct_ms": correct_ms,
                "total_ms": int((time.time() - start) * 1000),
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        jobs.update(job_id, status="error", phase="error", error=str(exc))


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    lang: str = DEFAULT_LANG,
    engine: str = "tesseract",
    corrector: str = "none",
) -> JSONResponse:
    """Start an async recognition job and return its ``job_id``.

    Poll ``GET /api/progress/{job_id}`` for live progress, timing and the result.

    ``engine``: ``tesseract`` (printed, CPU), ``trocr`` (handwriting, CPU) or
    ``chandra`` (Chandra OCR 2, page-level VLM, needs GPU).
    ``corrector``: ``none`` | ``spell`` (dictionary) | ``context`` (SAGE).
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    if engine not in ENGINES:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
    if corrector not in CORRECTORS:
        raise HTTPException(status_code=400, detail=f"Unknown corrector: {corrector}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB)")

    job_id = jobs.create_job(engine=engine, corrector=corrector)
    threading.Thread(
        target=_run_job, args=(job_id, data, lang, engine, corrector), daemon=True
    ).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/progress/{job_id}")
def progress(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return JSONResponse(job)


@app.get("/api/metrics")
def system_metrics() -> JSONResponse:
    return JSONResponse(metrics.get_metrics())


@app.get("/")
def index() -> FileResponse:
    # Disable caching so the browser never runs a stale UI (the inline JS must
    # stay in sync with the API contract).
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
