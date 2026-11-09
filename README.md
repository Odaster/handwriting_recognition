# handwriting_recognition

Сервис: скан, фото или PDF с русским текстом → UTF-8. FastAPI, страница на `/`, распознавание идёт фоновой задачей.

Руководство ниже ставит проект с чистой машины. Шаги идут по порядку: не перескакивайте. Каталог, из которого всё запускается, — тот, где лежат папки `app` и `scripts` и файлы `requirements*.txt`. Если после клонирования вы видите `handwriting_recognition\handwriting_recognition\app`, зайдите во внутренний каталог. Виртуальное окружение `.venv` создаётся именно там. Uvicorn тоже запускается оттуда, иначе импорт `app` не найдётся.

Нужен Python 3.11, 64-bit. Подойдут 3.10 и 3.12. Python 3.13 и новее не ставьте: колёса PyTorch для них не гарантированы, и установка оборвётся уже на torch.

Свободное место: около 8 ГБ на базовое окружение. Chandra и корректор SAGE качают веса при первом запросе (Chandra около 10–12 ГБ, SAGE около 7 ГБ). На этот вариант заложите 30 ГБ.

## Что ставить по задаче

Базовые пакеты (`requirements.txt`) поднимают сайт, PDF и движок Tesseract. Нейросетей в них нет.

Дальше одна ветка, не обе сразу. CPU-torch и CUDA-torch в одном окружении затирают друг друга.

| Задача | Дополнительно |
| --- | --- |
| Печатный текст, без видеокарты | Tesseract с языком `rus` |
| Рукопись на CPU (TrOCR) | torch с индекса CPU, затем `requirements-trocr.txt` |
| Страница / бланк / тетрадь (Chandra) и правка SAGE | NVIDIA-драйвер, torch с индекса CUDA 12.1, `requirements-chandra.txt`, затем пакеты SAGE из `requirements-trocr.txt`, и снова CUDA-torch |

Chandra без рабочей CUDA веса не грузит. На 12 ГБ видеопамяти Chandra и SAGE занимают карту по очереди, не вместе.

`scripts/install.ps1` и `scripts/install.sh` ставят только базу (venv + `requirements-dev.txt`) и Tesseract на Linux. Chandra, TrOCR и политику PowerShell они не делают. Ниже полный путь вручную. Скрипт можно запустить только после Python и, на Windows, после политики — и только вместо шагов с venv и базовым `pip`.

## Windows

Команды — в Windows PowerShell. «Терминал» в Cursor и PowerShell 7 тоже подходят. Командная строка `cmd` эти шаги не повторяет: политика и `Activate.ps1` там не действуют.

1. Откройте PowerShell. Проверьте политику:

   ```powershell
   Get-ExecutionPolicy -List
   ```

   Смотрите строку `CurrentUser`. Если там `Restricted` или `Undefined`, локальные скрипты запрещены. Без смены политики не запустится ни `.\scripts\install.ps1`, ни активация окружения `.\.venv\Scripts\Activate.ps1`. Создание `.venv` командой `py -3.11 -m venv` само по себе скрипт не запускает, но следующий шаг — активация — упрётся в эту ошибку: `running scripts is disabled on this system`.

2. Разрешите локальные скрипты только текущему пользователю. Права администратора для этой области не нужны:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

   На вопрос ответьте `Y`. `RemoteSigned` пускает локальные неподписанные скрипты (активация venv и `install.ps1`) и требует подпись у скриптов, скачанных из интернета. Проверка:

   ```powershell
   Get-ExecutionPolicy -Scope CurrentUser
   ```

   Ожидается `RemoteSigned`. Если команда отвечает, что политику перекрывает групповая политика, её меняет администратор домена. Обход через `Bypass` в это руководство не входит: тогда подхватится любой скачанный скрипт.

3. Установите Git для Windows с https://git-scm.com/download/win . В мастере оставьте настройки по умолчанию. Закройте PowerShell и откройте новое окно, затем:

   ```powershell
   git --version
   ```

4. Установите Python 3.11, 64-bit, с https://www.python.org/downloads/windows/ (не из Microsoft Store). В первом окне установщика включите **Add python.exe to PATH** и **Install pip**. Установка «для всех пользователей» удобнее, но тогда нужны права администратора. После установки снова закройте PowerShell и откройте новое окно: старое не видит обновлённый PATH.

5. Отключите заглушки Store, иначе команда `python` открывает магазин и окружение создаётся не тем интерпретатором. Параметры → Приложения → Дополнительные параметры приложений → Псевдонимы выполнения приложений → выключите `python.exe` и `python3.exe`.

   Проверка должна показать 3.11 и путь не из `WindowsApps`:

   ```powershell
   py -3.11 --version
   py -3.11 -c "import sys; print(sys.executable)"
   ```

   Если `py` не найден, установщик Python не поставил лаунчер. Переустановите 3.11 с теми же галочками. Дальше везде используется `py -3.11`, а не голый `python`.

6. Клонируйте репозиторий и зайдите в каталог с `app`:

   ```powershell
   git clone https://github.com/Odaster/handwriting_recognition.git
   cd handwriting_recognition
   ```

   Проверка: `dir app`, `dir requirements.txt` оба находятся. Если `app` нет — вы не в корне репозитория.

7. Создайте окружение этим же Python:

   ```powershell
   py -3.11 -m venv .venv
   ```

   Рядом появится папка `.venv`. Повторно команду не запускайте, если папка уже есть: она перезапишет уже поставленные пакеты только если её удалить. Удалять имеет смысл, если окружение собрали не тем Python.

8. Активируйте:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   В начале строки появится `(.venv)`. Если PowerShell снова пишет, что скрипты запрещены, шаг 2 не применился в этом окне: откройте новое окно PowerShell, повторите проверку политики и активируйте снова. Команды ниже предполагают активное окружение. Проверка, что `python` теперь из `.venv`:

   ```powershell
   python -c "import sys; print(sys.executable)"
   ```

   Путь должен заканчиваться на `.venv\Scripts\python.exe`.

9. Поставьте базу:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Это веб-сервер, разбор PDF и клиент Tesseract. Torch здесь ещё нет.

   Если pip ругается на SSL или прокси, добавьте к команде `--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org`.

10. Поставьте один движок.

    Печать без видеокарты. Сначала Tesseract:

    ```powershell
    winget install --id UB-Mannheim.TesseractOCR
    ```

    В установщике отметьте языковые данные **Russian**. Закройте и откройте PowerShell, активируйте `.venv` снова и проверьте:

    ```powershell
    tesseract --list-langs
    ```

    В списке должна быть строка `rus`. Если команды `tesseract` нет, путь не попал в PATH. Перед запуском сервера задайте его в этом же окне (путь установщика по умолчанию):

    ```powershell
    $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

    Переменная живёт до закрытия окна. Для постоянной настройки её задают в переменных среды пользователя.

    Рукопись на CPU (TrOCR), без Chandra:

    ```powershell
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    python -m pip install -r requirements-trocr.txt
    ```

    Chandra и корректор SAGE. Сначала убедитесь, что драйвер NVIDIA живой:

    ```powershell
    nvidia-smi
    ```

    Команда должна показать видеокарту. Если её нет, CUDA-torch встанет, но Chandra не запустится. Дальше именно индекс `cu121`, не обычный PyPI и не CPU-индекс:

    ```powershell
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
    python -m pip install -r requirements-chandra.txt
    python -m pip install -r requirements-trocr.txt
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
    ```

    Третий `pip` ставит пакеты SAGE (transformers, sentencepiece, accelerate, bitsandbytes). Четвёртый возвращает CUDA-сборку torch: шаг с `requirements-trocr.txt` может подменить её на другую. Проверка:

    ```powershell
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```

    Вторая часть должна быть `True`. Если `False`, в окружении не CUDA-сборка: повторите четвёртую команду и проверьте снова, сервер при этом не запускайте.

11. Запуск из того же каталога, где лежит `app`, с уже активированным `.venv`:

    ```powershell
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
    ```

    Окно не закрывайте: это и есть сервер. Остановка — Ctrl+C. Повторный запуск после нового окна PowerShell: снова `cd` в этот каталог, снова `.\.venv\Scripts\Activate.ps1`, затем та же команда uvicorn. Само окружение заново создавать не нужно.

    Адрес в браузере: http://127.0.0.1:8080 . Проверка процесса: http://127.0.0.1:8080/api/health — в JSON поле `"status": "ok"` и путь `app_path` к папке `app` этого репозитория. Если страница не открывается, смотрите, не занят ли порт 8080 другим процессом, и что uvicorn пишет в этом окне, а не в старом.

    Чтобы сервер был виден другим машинам в сети, вместо `127.0.0.1` укажите `0.0.0.0`. Windows может спросить разрешение брандмауэра — без ответа с других компьютеров порт закрыт.

12. Первый запрос с Chandra или SAGE качает веса с Hugging Face в кэш пользователя (обычно `C:\Users\<имя>\.cache\huggingface`). Пока идёт загрузка, задача в интерфейсе стоит на фазе загрузки. Повторный запрос той же модели веса не качает. Старый кэш других моделей чистится так: `python scripts/cleanup_old_hf_cache.py` из корня репозитория, при активированном `.venv`.

## Linux

Шаги рассчитаны на Debian и Ubuntu с `apt`. На другом дистрибутиве те же пакеты называются иначе; порядок не меняется: Python 3.11 с модулем `venv`, системные библиотеки для OpenCV, клон, `.venv`, один движок, uvicorn из корня репозитория.

1. Обновите список пакетов и поставьте базу. `sudo` спросит пароль.

   ```bash
   sudo apt-get update
   sudo apt-get install -y --no-install-recommends \
     git python3 python3-venv python3-pip \
     libgl1 libglib2.0-0
   ```

   `python3-venv` обязателен. Без него `python3 -m venv` падает с текстом `ensurepip is not available`, и окружение не создаётся. `libgl1` и `libglib2.0-0` нужны OpenCV, без них импорт падает уже при старте распознавания.

2. Проверьте версию:

   ```bash
   python3 --version
   ```

   Нужна 3.10, 3.11 или 3.12. Если системный Python старше, поставьте 3.11 из репозитория дистрибутива (`python3.11` и `python3.11-venv`) и дальше подставляйте `python3.11` вместо `python3`.

3. Клонируйте и зайдите в каталог с `app`:

   ```bash
   git clone https://github.com/Odaster/handwriting_recognition.git
   cd handwriting_recognition
   test -d app && test -f requirements.txt && echo "корень репозитория"
   ```

   Строка `корень репозитория` должна появиться. Если нет — `cd` не в тот каталог.

4. Создайте и включите окружение:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   В начале строки появится `(.venv)`. Проверка:

   ```bash
   python -c "import sys; print(sys.executable)"
   ```

   Путь должен содержать `handwriting_recognition/.venv/`. Новое окно терминала активацию не помнит: снова `cd` в этот каталог и снова `source .venv/bin/activate`.

5. Поставьте базу:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   При ошибке SSL или прокси добавьте `--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org`.

6. Поставьте один движок.

   Печать без видеокарты:

   ```bash
   sudo apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus
   tesseract --list-langs
   ```

   В списке должна быть `rus`.

   Рукопись на CPU (TrOCR):

   ```bash
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   python -m pip install -r requirements-trocr.txt
   ```

   Chandra и SAGE. Сначала драйвер:

   ```bash
   nvidia-smi
   ```

   Затем CUDA-torch и пакеты. Последняя команда возвращает CUDA-сборку, если предыдущая её подменила:

   ```bash
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   python -m pip install -r requirements-chandra.txt
   python -m pip install -r requirements-trocr.txt
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   Ожидается `True`. При `False` сервер не запускайте: повторите установку torch с индекса `cu121`.

7. Запуск из корня репозитория с активным `.venv`:

   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
   ```

   Страница: http://127.0.0.1:8080 . Проверка: http://127.0.0.1:8080/api/health , в ответе `"status": "ok"`. Остановка — Ctrl+C. Для доступа из сети замените адрес на `0.0.0.0` и откройте порт в файрволе дистрибутива.

8. Первый запрос Chandra или SAGE скачивает веса в `~/.cache/huggingface`. Это один раз на модель. Старый кэш: `python scripts/cleanup_old_hf_cache.py` из корня, с активным `.venv`.

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
| `TESSERACT_CMD` | | путь к `tesseract.exe`, если его нет в PATH |

## Ограничения

Tesseract не читает курсив. Chandra путает похожие буквы и может подставить другое словарное слово. SAGE чинит опечатки, не смысл. `CHANDRA_REFINE=1` дописывает текст.
