# handwriting_recognition

Сервис переводит скан, фото или PDF с русским текстом в обычный текст. Сервер на FastAPI, страница открывается на `/`.

Перед установкой:

- Python 3.11, 64 бит. 3.10 и 3.12 тоже подойдут. 3.13 и новее не ставьте.
- На базовую установку хватит примерно 8 ГБ. Chandra и SAGE при первом запуске скачают ещё около 10–12 ГБ и 7 ГБ.
- Команды выполняются в каталоге, где лежат папка `app` и файл `requirements.txt`. Сервер запускается оттуда же.
- Движок один на окружение: Tesseract, TrOCR или Chandra. CPU-сборку torch и CUDA-сборку вместе не ставьте, вторая затрёт первую.

## Установка на Windows

Дальше PowerShell. В cmd эти команды не сработают.

1. Разрешите запуск локальных скриптов. Без этого не откроется окружение: `Activate.ps1` ответит `running scripts is disabled on this system`. Администратор не нужен.

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

   На вопрос ответьте `Y`. Потом:

   ```powershell
   Get-ExecutionPolicy -Scope CurrentUser
   ```

   Должно быть `RemoteSigned`.

2. Поставьте Git: https://git-scm.com/download/win . В мастере оставьте как есть. Закройте PowerShell, откройте новое окно.

   ```powershell
   git --version
   ```

3. Поставьте Python 3.11 64-bit с https://www.python.org/downloads/windows/ . Из Microsoft Store не ставьте. В первом окне установщика отметьте `Add python.exe to PATH` и `Install pip`. Снова закройте PowerShell и откройте новое, старое окно новый PATH не видит.

4. Выключите подмену Python магазином: Параметры, Приложения, Дополнительные параметры приложений, Псевдонимы выполнения приложений, снять галки с `python.exe` и `python3.exe`.

   ```powershell
   py -3.11 --version
   ```

   В ответе версия 3.11.

5. Склонируйте репозиторий и зайдите в него.

   ```powershell
   git clone https://github.com/Odaster/handwriting_recognition.git
   cd handwriting_recognition
   ```

   В этом каталоге должны быть папка `app` и файл `requirements.txt`. Если `app` лежит глубже, в `handwriting_recognition\handwriting_recognition`, зайдите туда.

6. Создайте окружение.

   ```powershell
   py -3.11 -m venv .venv
   ```

   Появится папка `.venv`.

7. Включите его.

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Слева в строке будет `(.venv)`.

   ```powershell
   python -c "import sys; print(sys.executable)"
   ```

   Путь должен заканчиваться на `.venv\Scripts\python.exe`.

8. Поставьте пакеты сайта.

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Если pip ругается на сертификат или прокси, допишите к команде `--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org`.

9. Поставьте один движок.

   Печать, без видеокарты:

   ```powershell
   winget install --id UB-Mannheim.TesseractOCR
   ```

   В установщике отметьте язык Russian. Новое окно PowerShell, снова включите `.venv`, затем:

   ```powershell
   tesseract --list-langs
   ```

   В списке должна быть строка `rus`. Если команда `tesseract` не находится:

   ```powershell
   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

   Эта переменная живёт, пока открыто окно.

   Рукопись на процессоре (TrOCR):

   ```powershell
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   python -m pip install -r requirements-trocr.txt
   ```

   Страница, бланк, тетрадь (Chandra и правка SAGE). Сначала драйвер:

   ```powershell
   nvidia-smi
   ```

   В таблице должна быть ваша видеокарта. Затем пакеты. Последняя команда ставит CUDA-torch ещё раз: файл `requirements-trocr.txt` может подменить его другой сборкой.

   ```powershell
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   python -m pip install -r requirements-chandra.txt
   python -m pip install -r requirements-trocr.txt
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   В конце строки должно быть `True`. На 12 ГБ памяти карты Chandra и SAGE работают по очереди, не одновременно.

10. Запустите сервер из того же каталога, `.venv` уже включён.

    ```powershell
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
    ```

    Страница: http://127.0.0.1:8080 . Проверка: http://127.0.0.1:8080/api/health , в ответе `"status": "ok"`. Окно не закрывайте. Остановка: Ctrl+C. После нового входа в систему снова зайдите в каталог и снова выполните `.\.venv\Scripts\Activate.ps1`.

    Если сервер нужен другим компьютерам, вместо `127.0.0.1` напишите `0.0.0.0`. Windows спросит, пускать ли программу через брандмауэр.

11. Первый запрос с Chandra или SAGE скачает веса в `C:\Users\<имя>\.cache\huggingface`. Пока качается, в интерфейсе висит загрузка. Старые чужие модели из кэша:

    ```powershell
    python scripts/cleanup_old_hf_cache.py
    ```

Скрипт `scripts\install.ps1` ставит только базовые пакеты. Политику PowerShell и Chandra он не трогает.

## Установка на Linux

Ниже команды для Debian и Ubuntu.

1. Поставьте системные пакеты.

   ```bash
   sudo apt-get update
   sudo apt-get install -y --no-install-recommends git python3 python3-venv python3-pip libgl1 libglib2.0-0
   ```

   `python3-venv` нужен обязательно: без него `python3 -m venv` пишет `ensurepip is not available`. `libgl1` и `libglib2.0-0` нужны для картинок, без них распознавание падает на импорте.

2. Проверьте Python.

   ```bash
   python3 --version
   ```

   Нужна 3.10, 3.11 или 3.12. Если версия старше, поставьте `python3.11` и `python3.11-venv` и в командах ниже пишите `python3.11`.

3. Склонируйте и зайдите в каталог.

   ```bash
   git clone https://github.com/Odaster/handwriting_recognition.git
   cd handwriting_recognition
   ```

   Здесь должны лежать `app` и `requirements.txt`.

4. Создайте окружение и включите его.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Слева в строке будет `(.venv)`. Новое окно терминала это не помнит: зайдите в каталог и выполните `source` ещё раз.

5. Поставьте пакеты сайта.

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   При ошибке сертификата те же `--trusted-host`, что в пункте 8 для Windows.

6. Поставьте один движок.

   Печать:

   ```bash
   sudo apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus
   tesseract --list-langs
   ```

   В списке должна быть `rus`.

   Рукопись на процессоре:

   ```bash
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   python -m pip install -r requirements-trocr.txt
   ```

   Chandra и SAGE:

   ```bash
   nvidia-smi
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   python -m pip install -r requirements-chandra.txt
   python -m pip install -r requirements-trocr.txt
   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   `nvidia-smi` показывает карту. В проверке torch последнее слово `True`.

7. Запустите сервер из этого каталога.

   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
   ```

   Страница: http://127.0.0.1:8080 . Проверка: http://127.0.0.1:8080/api/health . Для других машин хост `0.0.0.0`, порт откройте в файрволе.

8. Веса Chandra и SAGE попадут в `~/.cache/huggingface`. Старый кэш:

   ```bash
   python scripts/cleanup_old_hf_cache.py
   ```

Скрипт `scripts/install.sh` ставит только базовые пакеты и Tesseract. Chandra он не ставит.

## Движки

| Имя | Для чего | Модель |
| --- | --- | --- |
| `tesseract` | печать, без torch | Tesseract, язык `rus` |
| `trocr` | курсив по словам, CPU, кадр 384×384 | `kazars24/trocr-base-handwritten-ru` |
| `chandra` | страница, бланк, тетрадь, нужна CUDA | `datalab-to/chandra-ocr-2`, около 5B |

## Корректоры

| Имя | Что делает |
| --- | --- |
| `none` | ничего |
| `spell` | словарь `pyspellchecker` |
| `context` | SAGE 1.7B: `ai-forever/sage-v1.1.0` и токенайзер `ai-forever/FRED-T5-1.7B`, затем правки pymorphy3 |

Chandra и SAGE на видеокарте идут по очереди.

Вход: PNG, JPEG, BMP, TIFF, WebP, PDF. Скан лучше цветной, 300 DPI, режим «фото».

Файл результата: `GET /api/export/{job_id}?format=txt|md|docx|xlsx|pdf&which=text|corrected`.

## API

| Метод | Путь | |
| --- | --- | --- |
| GET | `/` | страница |
| GET | `/api/health` | статус и имена моделей |
| POST | `/api/recognize` | файл, в query `engine` и `corrector`, в ответе `job_id` |
| GET | `/api/progress/{job_id}` | ход работы и текст |
| GET | `/api/export/{job_id}` | файл |
| GET | `/api/metrics` | CPU, память, видеокарта |

## Переменные

| Имя | По умолчанию |
| --- | --- |
| `HTR_MODEL` | `kazars24/trocr-base-handwritten-ru` |
| `HTR_BEAMS` | `4` |
| `CORRECTOR_MODEL`, `SAGE_MODEL` | `ai-forever/sage-v1.1.0` |
| `SAGE_TOKENIZER` | `ai-forever/FRED-T5-1.7B` |
| `SAGE_BEAMS` | `1` |
| `CHANDRA_PROMPT` | `ocr`. Другое значение: `ocr_layout` |
| `CHANDRA_ENHANCE` | `1` |
| `CHANDRA_STRIPS` | `3`. Значение `1` — вся страница сразу |
| `CHANDRA_REFINE` | `0` |
| `CHANDRA_MAX_TOKENS` | `1536` на полосу, `2048` если полос нет |
| `TESSERACT_CMD` | путь к `tesseract.exe`, если его нет в PATH |

Tesseract курсив не читает. Chandra путает похожие буквы и иногда подставляет другое слово. SAGE правит опечатки, смысл не меняет. `CHANDRA_REFINE=1` дописывает текст, лучше не включать.
