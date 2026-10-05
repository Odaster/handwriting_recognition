# handwriting_recognition

Приложение для перевода русской рукописи и печатного текста (сканы, фото, PDF) в чистый электронный текст.

Три движка:

- **Tesseract OCR** — быстрый, без `torch`. Годится для **печатного** текста. Курсив почти не читает.
- **TrOCR** (`kazars24/trocr-base-handwritten-ru`) — нейросеть по словам (~1.3 ГБ). Читает курсив на CPU, но слабее Chandra на целой странице.
- **Chandra OCR 2** (`datalab-to/chandra-ocr-2`) — page-level VLM ~5B (Qwen3.5-VL). Лучший вариант для тетрадей, бланков и PDF. На **RTX 12 ГБ** работает, если корректор грузится **после** выгрузки Chandra, а не вместе с ней.

TrOCR ужимает вход до 384×384: страница режется на строки и слова. Chandra читает полосы страницы целиком.

## Стек

- **FastAPI** — REST и веб-интерфейс
- **Tesseract** + **pytesseract** — печать
- **TrOCR** (`transformers`, CPU-`torch`) — рукопись без GPU
- **Chandra OCR 2** — рукопись/бланки на CUDA
- **OpenCV / Pillow / NumPy** — предобработка страницы (выравнивание, линии тетради, красные пометки)
- **PyMuPDF** — PDF → изображения, 300 DPI
- **Коррекция** (после распознавания):
  - **SAGE 1.7B** (`ai-forever/sage-v1.1.0` + токенизатор `FRED-T5-1.7B`) — seq2seq, орфография и типичные OCR-опечатки. На 12 ГБ: dense bf16/fp32, ~1.74B весов.
  - **cursivefix** — словарные замены курсивных крокозябр (`лацубу` → `чащобу`), если слово не из OpenCorpora и есть единственный дешёвый lookalike.
  - **pyspellchecker** — лёгкая словарная проверка без контекста (`corrector=spell`).
  - **T-lite 8B** — опция `CORRECTOR_MODEL=t-tech/T-lite-it-2.1` (4-bit на ≤16 ГБ). Не держит 8B в bf16 на 12 ГБ.

Chandra и корректор **не живут в VRAM одновременно** (`app/models.py`). Типичный прогон на 4070 Super 12 ГБ: Chandra → выгрузка → SAGE.

Неиспользуемый кэш старых чекпойнтов:

```bash
python scripts/cleanup_old_hf_cache.py
```

### Пайплайн Chandra + SAGE

1. Скан/фото/PDF → RGB, при необходимости enhance (deskew, линии, CLAHE).
2. Страница режется на **3 горизонтальные полосы** с перекрытием (`CHANDRA_STRIPS`), чтобы буквы были крупнее.
3. Chandra пишет HTML с лейаутом; `html_to_text` снимает теги, обрезанный `data-bbox`, петли декодера и повтор хвоста полосы.
4. `cursivefix` правит несловарные токены вроде `кинуза`.
5. SAGE 1.7B правит опечатки по предложениям, абзацы сохраняются. Даты и прочерки бланка (`« 08 »`, `_____`) SAGE не трогает.
6. В UI два поля: распознанный текст и «После коррекции».

Лимит генерации Chandra по умолчанию **1536 токенов на полосу** (2048 без полос). Раньше 512 обрывал бланки посередине HTML.

### Переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `HTR_MODEL` | `kazars24/trocr-base-handwritten-ru` | TrOCR |
| `HTR_BEAMS` | `4` | Beam search TrOCR (`1` = greedy) |
| `CORRECTOR_MODEL` | `ai-forever/sage-v1.1.0` | Контекстный корректор (`SAGE_MODEL` — синоним) |
| `SAGE_TOKENIZER` | `ai-forever/FRED-T5-1.7B` | Токенизатор SAGE 1.7B |
| `SAGE_BEAMS` | `1` для 1.7B, иначе `4` | Beam SAGE; `1` на 12 ГБ, чтобы generate не зависал |
| `CORRECTOR_QUANT` | `auto` | Для T-lite: `4bit` на GPU ≤16 ГБ; SAGE 1.7B dense |
| `CHANDRA_PROMPT` | `ocr` | `ocr` — русский рукописный промпт; `ocr_layout` — markdown с лейаутом |
| `CHANDRA_ENHANCE` | `1` | Препроцесс страницы |
| `CHANDRA_STRIPS` | `3` | Число горизонтальных полос (`1` = вся страница) |
| `CHANDRA_REFINE` | `0` | Второй проход Chandra; часто выдумывает «литературный» заголовок |
| `CHANDRA_MAX_TOKENS` | `1536` / `2048` | Потолок токенов на полосу / страницу |

## Форматы ввода

PNG/JPEG/BMP/TIFF/WebP и **PDF** (каждая страница — отдельное изображение). Для МФУ (Lexmark и аналоги) лучше **фото/цвет, 300 DPI, без вырезания фона**. ЧБ «текст» убивает карандаш и тонкий стержень.

## Вывод и экспорт

Результат — **чистый текст**: HTML Chandra → абзацы, `<br/>` → перевод строки, теги и `data-bbox` снимаются (в том числе оборванные теги, если модель упёрлась в лимит токенов). Скачать: `GET /api/export/{job_id}` или кнопки UI — **TXT, Markdown, Word, Excel, PDF**. Параметр `which=text|corrected`.

## Установка

```bash
bash scripts/install.sh
```

Идемпотентно: Tesseract с русским языком, `.venv`, лёгкие зависимости.

> За корпоративным прокси с перехватом SSL добавьте к pip `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

### TrOCR (опционально, CPU)

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -r requirements-trocr.txt
```

### Chandra OCR 2 (рекомендуется, CUDA)

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
.venv/bin/pip install -r requirements-chandra.txt
```

`datalab-to/chandra-ocr-2` (~10–12 ГБ) и SAGE 1.7B (~7 ГБ) качаются при первом прогоне. На CPU Chandra сразу возвращает ошибку, без загрузки весов. Без своего GPU: [Datalab](https://datalab.to).

### Windows

`scripts/install.sh` рассчитан на Linux. Нативно:

```powershell
.\scripts\install.ps1
.\.venv\Scripts\python -m pip install -r requirements-trocr.txt
.\.venv\Scripts\python -m pip install -r requirements-chandra.txt   # GPU
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Tesseract: `winget install --id UB-Mannheim.TesseractOCR` (язык `rus`). Если нет в PATH: `$env:TESSERACT_CMD='C:\Program Files\Tesseract-OCR\tesseract.exe'`. Для Chandra Tesseract не нужен.

Запускайте uvicorn из каталога, где лежит `app\` (не из вложенного git-клона, если их два). В логе должно быть `app loaded from ...\app`.

## Запуск

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Откройте <http://localhost:8000> (на Windows часто <http://127.0.0.1:8080>). Движок — Chandra, коррекция — контекстная (SAGE).

## API

| Метод | Путь | Описание |
| ----- | ---- | -------- |
| GET | `/` | Веб-интерфейс |
| GET | `/api/health` | Статус, Tesseract, id корректора |
| POST | `/api/recognize` | `multipart/form-data`, поле `file` → `job_id` |
| GET | `/api/progress/{job_id}` | Прогресс и готовый текст |
| GET | `/api/export/{job_id}` | Скачать txt/md/docx/xlsx/pdf |
| GET | `/api/metrics` | CPU / RAM / GPU |

Параметры `/api/recognize`:

- `engine` — `tesseract` \| `trocr` \| `chandra`
- `corrector` — `none` \| `spell` \| `context` (SAGE 1.7B)

```bash
python scripts/make_sample.py --text "Привет, мир!" --out sample.png
curl -s -F "file=@sample.png" "http://127.0.0.1:8080/api/recognize?engine=tesseract"

curl -s -F "file=@page.pdf" "http://127.0.0.1:8080/api/recognize?engine=chandra&corrector=context"
```

Ответ сразу отдаёт `job_id`; текст и `text_corrected` приходят через `/api/progress/{job_id}`.

## Ограничения

- Tesseract не читает школьный курсив — берите Chandra.
- Chandra путает похожие буквы (`ч/л`, `щ/ц`) и целые слова (`Дом`/`Ночь`). Это не словарь модели.
- SAGE чинит опечатки (`стиках` → `струйках`), не «угадывает» другое словарное слово и не должна переписывать сочинение. Крокозябры вроде `лацубу` закрывает cursivefix, не SAGE.
- Второй проход Chandra (`CHANDRA_REFINE=1`) часто литературно выдумывает заголовок — по умолчанию выключен.
- 8B в bf16 на 12 ГБ не влезает; offload в RAM — это PCIe, не «ещё 32 ГБ видеопамяти».

## Тесты

```bash
.venv/bin/pytest -q
```

CI/локально корректор в тестах — маленький `sage-fredt5-distilled-95m` (`conftest.py`), не 1.7B.

<img width="1851" height="882" alt="image" src="https://github.com/user-attachments/assets/cb27370e-0344-4f35-b07c-cd49865fcdae" />
