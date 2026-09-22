"""HTTP-приложение: загрузка изображения и распознавание текста."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import APP_NAME, OCR_BACKEND
from app.ocr import OCREngine, RecognitionResult, create_engine
from app.preprocess import ImageValidationError, prepare_image, validate_upload
from app.schemas import HealthResponse, RecognitionResponse, RecognizedLine

STATIC_DIR = Path(__file__).parent / "static"


def get_engine(request: Request) -> OCREngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        engine = create_engine()
        request.app.state.engine = engine
    return engine


def result_to_response(result: RecognitionResult) -> RecognitionResponse:
    return RecognitionResponse(
        text=result.text,
        lines=[
            RecognizedLine(text=line.text, confidence=line.confidence, box=line.box)
            for line in result.lines
        ],
        engine=result.engine,
        language=result.languages,
        average_confidence=result.average_confidence,
    )


def create_app(engine: OCREngine | None = None) -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=__version__,
        description="Загрузите фото рукописного русского текста и получите электронную расшифровку.",
    )
    app.state.engine = engine

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="Интерфейс не найден.")
        return FileResponse(index_path)

    @app.get("/api/health", response_model=HealthResponse)
    def health(ocr: OCREngine = Depends(get_engine)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app=APP_NAME,
            version=__version__,
            engine=getattr(ocr, "name", OCR_BACKEND),
            ready=True,
        )

    @app.post("/api/recognize", response_model=RecognitionResponse)
    def recognize(
        file: UploadFile = File(..., description="Изображение с рукописным текстом"),
        enhance: bool = Query(True, description="Предварительная обработка изображения"),
        ocr: OCREngine = Depends(get_engine),
    ) -> RecognitionResponse:
        data = file.file.read()
        try:
            validate_upload(file.filename, file.content_type, data)
            image = prepare_image(data, enhance=enhance)
            result = ocr.recognize(image)
        except ImageValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"Не удалось распознать текст: {exc}",
            ) from exc

        if not result.text.strip():
            raise HTTPException(
                status_code=422,
                detail="Текст не найден. Попробуйте снимок крупнее и при лучшем освещении.",
            )
        return result_to_response(result)

    return app


app = create_app()
