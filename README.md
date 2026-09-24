# handwriting_recognition

Приложение для перевода русского рукописного/печатного текста в электронный формат.

Распознавание построено на **Tesseract OCR** (движок LSTM) с русской языковой моделью —
без `torch`/CUDA, поэтому установка лёгкая и работает офлайн.

## Стек

- **FastAPI** — REST API и веб-интерфейс
- **Tesseract OCR** (`tesseract-ocr`, `tesseract-ocr-rus`) — распознавание
- **pytesseract** — Python-обёртка над Tesseract
- **OpenCV / Pillow / NumPy** — предобработка изображений (масштабирование, бинаризация Otsu)

## Установка

```bash
bash scripts/install.sh
```

Скрипт идемпотентный: ставит системный Tesseract с русским языком (через `apt`),
создаёт виртуальное окружение `.venv` и устанавливает Python-зависимости.

> Если вы за корпоративным прокси с перехватом SSL, добавьте к pip-командам
> `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

## Запуск

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Откройте <http://localhost:8000> — загрузите изображение и получите распознанный текст.

## API

| Метод | Путь              | Описание                                            |
| ----- | ----------------- | --------------------------------------------------- |
| GET   | `/`               | Веб-интерфейс загрузки                              |
| GET   | `/api/health`     | Статус + версия Tesseract и список языков           |
| POST  | `/api/recognize`  | `multipart/form-data` с полем `file` → JSON с текстом |

Пример:

```bash
python scripts/make_sample.py --text "Привет, мир!" --out sample.png
curl -s -F "file=@sample.png" http://localhost:8000/api/recognize
```

## Тесты

```bash
.venv/bin/pytest -q
```

Тесты генерируют изображения с русским текстом и проверяют, что Tesseract
корректно распознаёт их через OCR-модуль и через HTTP-эндпоинт.
