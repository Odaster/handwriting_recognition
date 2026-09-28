"""Tiny in-memory job store for asynchronous recognition with live progress.

Recognition (especially the neural engines) can take a while, so the API starts
a job and the client polls its progress. This is process-local and not durable —
fine for a single-process dev/demo server.
"""

from __future__ import annotations

import threading
import time
import uuid

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_MAX_JOBS = 50


def create_job(engine: str, corrector: str) -> str:
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",  # queued | running | done | error
            "phase": "queued",
            "progress": 0.0,
            "engine": engine,
            "corrector": corrector,
            "token_count": 0,
            "text": None,
            "text_corrected": None,
            "words": 0,
            "confidence": None,
            "error": None,
            "corrector_error": None,
            "timing": {},
            "created": time.time(),
            "elapsed_ms": 0,
        }
        if len(_JOBS) > _MAX_JOBS:
            for key, _ in sorted(_JOBS.items(), key=lambda kv: kv[1]["created"])[
                : len(_JOBS) - _MAX_JOBS
            ]:
                _JOBS.pop(key, None)
    return job_id


def update(job_id: str, **fields) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["elapsed_ms"] = int((time.time() - job["created"]) * 1000)


def get(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
