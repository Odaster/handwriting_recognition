FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md ./

EXPOSE 8000

ENV OCR_BACKEND=easyocr
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "-m", "app"]
