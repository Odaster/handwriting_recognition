"""Схемы ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecognizedLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[list[float]] | None = None


class RecognitionResponse(BaseModel):
    text: str
    lines: list[RecognizedLine]
    engine: str
    language: list[str]
    average_confidence: float | None = None


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    engine: str
    ready: bool
