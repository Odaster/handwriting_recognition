# handwriting_recognition

Сервис: скан / фото / PDF с русским текстом → UTF-8. FastAPI, UI на `/`, асинхронные джобы.

## Движки (`engine`)

| Id | Модель | Когда |
| --- | --- | --- |
| `tesseract` | Tesseract rus | печать, CPU, без torch |
| `trocr` | `kazars24/trocr-base-handwritten-ru` | курсив по словам, CPU, вход 384×384 |
| `chandra` | `datalab-to/chandra-ocr-2` (~5B, CUDA) | страница / бланк / тетрадь |

## Корректоры (`corrector`)

| Id | Что |
| --- | --- |
| `none` | только OCR |
| `spell` | словарь `pyspellchecker` |
| `context` | SAGE 1.7B (`ai-forever/sage-v1.1.0` + `FRED-T5-1.7B`), затем `cursivefix` (pymorphy3, lookalike для OOV) |

Chandra и корректор в VRAM **по очереди** (`app/models.py`).

## Обработка Chandra

PDF → RGB 300 DPI. Enhance (deskew, линии, красные пометки). Страница — 3 горизонтальные полосы (`CHANDRA_STRIPS`). HTML лейаута → текст (`app/textout.py`): теги, оборванный `data-bbox`, петли декодера, повтор хвоста полосы. SAGE не трогает даты/прочерки бланка (`« 08 »`, `_____`).

Лимит генерации: 1536 токенов/полоса, 2048 без полос (`CHANDRA_MAX_TOKENS`).

## Вход / выход

Вход: PNG, JPEG, BMP, TIFF, WebP, PDF. Скан: цвет, 300 DPI, режим «фото», без вырезания фона.

Выход: `text`, `text_corrected`. Экспорт: `GET /api/export/{job_id}?format=txt|md|docx|xlsx|pdf&which=text|corrected`.

## Установка

Linux: `bash scripts/install.sh` (Tesseract rus, `.venv`, `requirements.txt`).

TrOCR: CPU-torch + `requirements-trocr.txt`. Chandra: CUDA-torch + `requirements-chandra.txt`. Веса Hugging Face качаются при первом запросе (Chandra ~10–12 ГБ, SAGE ~7 ГБ). Chandra без CUDA не грузит веса.

Windows: `.\scripts\install.ps1`. Tesseract — отдельно (`winget install UB-Mannheim.TesseractOCR`, rus). Нет в PATH: `TESSERACT_CMD`. Для Chandra Tesseract не нужен.

Uvicorn запускать из каталога с `app\`.

## Запуск

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

UI: `/`. Chandra + `corrector=context`.

SSL/прокси pip: `--trusted-host pypi.org --trusted-host files.pythonhosted.org`. Старый HF-кэш: `python scripts/cleanup_old_hf_cache.py`.

## API

| Метод | Путь | |
| --- | --- | --- |
| GET | `/` | UI |
| GET | `/api/health` | статус, ids моделей |
| POST | `/api/recognize` | `file` + query `engine`, `corrector` → `{ job_id }` |
| GET | `/api/progress/{job_id}` | прогресс, `text`, `text_corrected` |
| GET | `/api/export/{job_id}` | файл |
| GET | `/api/metrics` | CPU / RAM / GPU |

## Переменные

| | по умолчанию | |
| --- | --- | --- |
| `HTR_MODEL` | `kazars24/trocr-base-handwritten-ru` | |
| `HTR_BEAMS` | `4` | |
| `CORRECTOR_MODEL` / `SAGE_MODEL` | `ai-forever/sage-v1.1.0` | |
| `SAGE_TOKENIZER` | `ai-forever/FRED-T5-1.7B` | |
| `SAGE_BEAMS` | `1` (1.7B) | |
| `CHANDRA_PROMPT` | `ocr` | `ocr_layout` — markdown лейаута |
| `CHANDRA_ENHANCE` | `1` | |
| `CHANDRA_STRIPS` | `3` | `1` — целая страница |
| `CHANDRA_REFINE` | `0` | второй проход Chandra |
| `CHANDRA_MAX_TOKENS` | `1536` / `2048` | |
| `TESSERACT_CMD` | | путь к `tesseract.exe` |

## Ограничения

Tesseract не читает курсив. Chandra путает похожие буквы и может подставить другое словарное слово. SAGE чинит опечатки, не смысл. `CHANDRA_REFINE=1` дописывает текст.
