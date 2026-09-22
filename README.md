# handwriting_recognition

Приложение на Python, которое превращает **русский рукописный текст** с фото или скана в электронный текст.

A Python web app that converts Russian handwriting from photos and scans into editable text.

## Возможности

- Веб-интерфейс: перетащите изображение или выберите файл
- Распознавание русского и английского текста через EasyOCR
- Предварительная обработка: ориентация по EXIF, шумоподавление, контраст
- Копирование результата и скачивание `.txt`
- HTTP API для программной интеграции

## Требования

- Python 3.11+
- 2–4 ГБ свободной памяти на первый запуск (скачиваются модели EasyOCR)

## Установка и запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

Откройте [http://127.0.0.1:8000](http://127.0.0.1:8000).

Первое распознавание может занять несколько минут: движок загружает модели.

Запуск без моделей (демо-текст для разработки):

```bash
OCR_BACKEND=fake python -m app
```

## Переменные окружения

| Переменная | По умолчанию | Смысл |
| --- | --- | --- |
| `OCR_BACKEND` | `easyocr` | `easyocr` или `fake` (синоним: `OCR_ENGINE`) |
| `OCR_LANGUAGES` | `ru,en` | Языки EasyOCR |
| `OCR_USE_GPU` | `0` | `1`, если есть CUDA (синоним: `OCR_GPU`) |
| `MAX_UPLOAD_BYTES` | `10485760` | Лимит размера файла (или `MAX_UPLOAD_MB`) |
| `OCR_MAX_SIDE` | `3000` | Длинная сторона изображения после уменьшения |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Адрес сервера |

## API

`GET /api/health` — статус сервиса.

`POST /api/recognize` — multipart-загрузка поля `file`.

Необязательный query-параметр `enhance=true|false`.

```bash
curl -F "file=@page.jpg" http://127.0.0.1:8000/api/recognize
```

## Тесты

Тесты используют поддельный OCR-движок и не скачивают нейросетевые модели.

```bash
pip install -r requirements-test.txt
pytest
```

## Docker

```bash
docker build -t handwriting-recognition .
docker run --rm -p 8000:8000 handwriting-recognition
```

## Советы по качеству

- Снимайте страницу сверху, без сильного наклона
- Обеспечьте ровный свет, без бликов и густых теней
- Чем крупнее буквы на кадре, тем устойчивее результат
- Рукописный курсив распознаётся хуже печатного и аккуратного письма

## Структура

```
app/            FastAPI-приложение и OCR
app/static/     Веб-интерфейс
tests/          Pytest
```

## English

Install dependencies with `pip install -r requirements.txt`, then run `python -m app` and open http://127.0.0.1:8000. Upload a photo of handwritten Russian text to get a digital transcript you can copy or download as `.txt`.

The first EasyOCR run downloads language models (hundreds of MB). Tests never do that: `pip install -r requirements-test.txt && pytest` uses a fake OCR engine. For a UI demo without models, start the server with `OCR_BACKEND=fake`.
