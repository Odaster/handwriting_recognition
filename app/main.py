"""FastAPI application exposing the Russian OCR pipeline."""

from __future__ import annotations

import io
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import jobs, metrics
from app.ocr import DEFAULT_LANG, tesseract_info

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",
    "image/webp",
    "application/pdf",
}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB (PDFs can be larger)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _count_words(text: str) -> int:
    """Count word-like tokens, ignoring any HTML/layout markup."""
    plain = re.sub(r"<[^>]+>", " ", text or "")
    return len(_WORD_RE.findall(plain))

app = FastAPI(title="Russian Handwriting Recognition", version="0.1.0")


@app.get("/api/health")
def health() -> dict:
    """Report service status plus the detected Tesseract version/languages."""
    return {"status": "ok", **tesseract_info()}


CORRECTORS = {"none", "spell", "context"}
ENGINES = {"tesseract", "trocr", "chandra"}


def _recognize_one(image, engine: str, lang: str, progress):
    """Run one page image through the selected engine."""
    if engine == "chandra":
        from app.chandra_engine import recognize_image as rec

        return rec(image, progress=progress)
    if engine == "trocr":
        from app.htr import recognize_image as rec

        return rec(image, progress=progress)
    from app.ocr import recognize_image as rec

    if progress:
        progress(0.5, "recognize")
    return rec(image, lang=lang)


def _run_job(job_id: str, data: bytes, content_type: str, lang: str, engine: str, corrector: str) -> None:
    """Background worker: recognize (image or PDF) + optionally correct."""
    start = time.time()
    jobs.update(job_id, status="running", phase="recognize", progress=0.0)

    def make_progress(page_index: int, n_pages: int):
        span = 1.0 / n_pages
        base = page_index / n_pages

        def progress(fraction, phase="recognize", **extra):
            fields = dict(extra)
            if fraction is not None:
                fields["progress"] = max(0.0, min(0.999, base + float(fraction) * span))
            fields["phase"] = phase if n_pages == 1 else f"{phase} · стр. {page_index + 1}/{n_pages}"
            jobs.update(job_id, **fields)

        return progress

    try:
        if content_type == "application/pdf":
            from app.pdfutil import pdf_to_images

            jobs.update(job_id, phase="pdf")
            images = pdf_to_images(data)
            if not images:
                raise RuntimeError("В PDF не найдено страниц")
        else:
            images = [Image.open(io.BytesIO(data))]

        n_pages = len(images)
        texts: list[str] = []
        engine_name = engine
        confidence = None
        for i, image in enumerate(images):
            result = _recognize_one(image, engine, lang, make_progress(i, n_pages))
            payload = result.to_dict()
            engine_name = payload["engine"]
            texts.append(payload["text"])
            if n_pages == 1:
                confidence = payload.get("confidence")

        combined = "\n\n".join(texts) if n_pages > 1 else (texts[0] if texts else "")
        recognize_ms = int((time.time() - start) * 1000)
        jobs.update(
            job_id,
            text=combined,
            words=_count_words(combined),
            confidence=confidence,
            engine=engine_name,
            pages=n_pages,
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
                corrected = correct_text(combined)
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
        target=_run_job,
        args=(job_id, data, file.content_type, lang, engine, corrector),
        daemon=True,
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
