# Беседа по проекту (чат)

Файл для нашей переписки в удобном markdown-формате: сюда вы скидываете код и
ошибки, я отвечаю правками и командами. Ветка: `cursor/linux-install-debug-dd79`.

---

## Как вести беседу

Чтобы я мог быстро помочь, при проблеме присылайте:

1. **Команду**, которую запускали.
2. **Полный текст ошибки** (весь вывод, не только последнюю строку).
3. **Окружение** (если менялось): `python3 --version`, дистрибутив, venv/системный Python.

Код и вывод оформляйте в блоках с тройными кавычками — так markdown сохраняет форматирование:

````markdown
```bash
команда, которую запускал
```

```text
текст ошибки из терминала
```
````

Свои сообщения можно писать обычным текстом между блоками кода.

---

## Статус проекта

- **Стек:** FastAPI + Tesseract OCR (`tesseract-ocr-rus`) + OpenCV/Pillow. Без `torch`/CUDA.
- **Запуск:** `bash scripts/install.sh` → `.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- **Тесты:** `.venv/bin/pytest -q` (7 тестов).
- Подробности — в [`README.md`](README.md).

## Журнал

- ✅ Диагностирована ошибка установки: `easyocr` тянул `torch`+CUDA (гигабайты), сеть рвала закачку + перехват SSL.
- ✅ Выбран путь B — Tesseract (лёгкий, офлайн). Собрано приложение: OCR-модуль, API, веб-интерфейс, тесты.
- ✅ Проверено end-to-end: распознавание русского текста с уверенностью ~95%.
- ✅ `scripts/install.sh` доработан под чистую Ubuntu 24.04 (добавлен `python3-venv`/`python3-pip`).
- ✅ Разобрана ошибка `not a git repository` — нужно клонировать репозиторий заново (инструкция была в `text_chat`).
- ✅ Результат распознавания добавлен в конец `README.md` (скриншот).

---

## Установка полной версии (рукописный движок TrOCR + корректор SAGE)

Ubuntu 24.04, всё выполняем в папке репозитория `~/handwriting_recognition`.

```bash
# 0) Обновить код из ветки
cd ~/handwriting_recognition
git pull origin cursor/linux-install-debug-dd79

# 1) Базовая установка (Tesseract + venv + лёгкие зависимости). Идемпотентно.
bash scripts/install.sh

# 2) Тяжёлые зависимости для рукописного движка (CPU-torch, без CUDA)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -r requirements-trocr.txt

# 3) Запуск
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Откройте `http://localhost:8000`. В UI: **Движок = «Рукописный (TrOCR)»**,
**Коррекция = «Контекстная (SAGE)»**. Загрузите фото рукописи → «Распознать».

### Что скачается при первом запуске (с HuggingFace, по HTTPS)

- TrOCR `kazars24/trocr-base-handwritten-ru` — ~1.3 ГБ
- Корректор `ai-forever/sage-fredt5-large` — ~3.3 ГБ
- CPU-`torch` — ~190 МБ

### Скорость vs качество (env-переменные)

По умолчанию — максимум качества (beam search + большой корректор), одна страница
на CPU считается единицы–десятки минут. Чтобы ускорить:

```bash
HTR_BEAMS=1 \
SAGE_MODEL=ai-forever/sage-fredt5-distilled-95m \
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

(`sage-fredt5-distilled-95m` — всего ~350 МБ вместо 3.3 ГБ.)

### Если корпоративный прокси с перехватом SSL

- Для pip: добавляйте `--trusted-host pypi.org --trusted-host files.pythonhosted.org
  --trusted-host download.pytorch.org`.
- Для загрузки моделей (HuggingFace тоже по HTTPS): укажите ваш корпоративный CA:
  ```bash
  export REQUESTS_CA_BUNDLE=/путь/к/corp-ca.crt
  export SSL_CERT_FILE=/путь/к/corp-ca.crt
  ```
  CA-сертификат можно попросить у админа. Без него загрузка моделей упрётся в тот же
  `self-signed certificate in certificate chain`.

### Быстрая проверка из терминала

```bash
curl -s -F "file=@page.jpg" \
  "http://localhost:8000/api/recognize?engine=trocr&corrector=context" | python3 -m json.tool
```

---

## Новые сообщения

<!-- Пишите ниже. Новые сообщения удобно добавлять сверху этого раздела. -->
vmuser@vmuser-VMware-Virtual-Platform:~/handwriting_recognition$ .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
INFO:     Started server process [8076]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:56366 - "GET / HTTP/1.1" 200 OK
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 480/480 [00:00<00:00, 1510.28it/s]
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[transformers] Both `max_new_tokens` (=32) and `max_length`(=64) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 789/789 [00:00<00:00, 628kB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20.3k/20.3k [00:00<00:00, 14.5MB/s]
vocab.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.81M/1.81M [00:00<00:00, 6.45MB/s]
merges.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.27M/1.27M [00:00<00:00, 5.60MB/s]
added_tokens.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.72k/2.72k [00:00<00:00, 5.21MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 689/689 [00:00<00:00, 1.94MB/s]
model.safetensors: downloading bytes: ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|  171MB, 2.04MB/s  
model.safetensors: reconstructing file: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|  383MB /  383MB, 14.4MB/s  
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 1962.20it/s]
[transformers] The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 184/184 [00:00<00:00, 814kB/s]
[transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer GPT2Tokenizer. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
INFO:     127.0.0.1:38580 - "POST /api/recognize?engine=trocr&corrector=context HTTP/1.1" 200 OK

Распознанный текст
Ноль в легу
Деревня была где-то за лесом. исли изти в ней по
большой дороге , нупинь отмахать не один десяток х
километров, если пойти лесними тропинками путь
урежется врвое. Полстие корни обхватили извилистуль тропу.
лес ищлит, устнаивает. В стымом воздухе кружатся О
жухлие листья Тропинка, петли среди деревьев, поднимости
на пригорхи, слускается в логибинки забирали в лагуобу
осинника внбегает на зарастающий ельником полены, и
кажется, что она так и не выведет тебл никуда.
Но вот вместе с листьми начинает кругипться
снежинки Их становита больше л. б 1 8 п
хогроводе не видно уше ничего: н. больше, и в снежном
Осенный день нах свена тмет- падыющих листьев ни трожи.
учаснет. На лес наваливных см-тит туекши огнём и
визно: не знаемь, куда иджи. Сумерни, и дороги совим не
Ищтко и странию в темнож о л. п
дольми рискованю: осенью севе Марина совсем одна Идти
Марина забирается на дерево и ремса страши вынами.
ночь в лесу. о 1 пр переждать злегныю
Монрый снег напоми в ч
обмороженные ноги. Наюска пальто. Нолодно, и ныт
неоглиданно э п х о ед в промогла ел
сли зманими ащитрай

После коррекции
Ноль в лигу.
Деревня была где-то за лесом. Если идти в ней по...
Большой дороге, нупинь отмахать не один десяток х.
километров, если пойти лесными тропинками - путь.
Урежется втрое. Полстые корни обхватили извилистую тропу.
Лес ищет, устраивает. В стомом воздухе кружатся О.
Жухлые листья Тропинка, петли среди деревьев, поднимаемости.
На пригорхи, случается, в логибинки забирали в лагуобу.
Особинника набегает на зарастающий ельником полены и...
Кажется, что она так и не выведет тебя никуда.
Но вот вместе с листьями начинает крупиться.
Снежинки. Их становится больше: л. б. 1,8 п.
Хогровода не видно уже ничего: н. больше и в снежном.
Осенний день на свёнах тёт, падыющих листьев ни трожжи.
Участник. На лес наваливных см-тит туекши огнём и...
Визано: не знаем, куда идти. Сумерни и дороги совсем не.
Ищетко и странию в темнож, о л. п.
Дольми рискованю: осенью Севе Марина совсем одна Идти.
Марина забирается на дерево, и ремса страши вынами.
Ночь в лесу. О 1 пр. Переждать злегную.
Мёрзный снег, напоми в ч.
Обмороженные ноги. Наюска пальто. Нолодно и нут.
Неожиданно ЭП Х о ЕД в промогла ЕЛ.
Если зманими защитрай.
Движок: trocr Уверенность: — Слов: 147
