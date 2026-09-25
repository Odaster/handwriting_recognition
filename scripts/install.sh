#!/usr/bin/env bash
# Idempotent setup for the Russian OCR app (path B: Tesseract, no torch/CUDA).
set -euo pipefail

cd "$(dirname "$0")/.."

# 1) System packages: OCR engine, Russian data, and venv support.
#    On a fresh Ubuntu 24.04 `python3 -m venv` fails unless python3-venv is
#    present ("ensurepip is not available"), so we ensure it here too.
need_apt=0
command -v tesseract >/dev/null 2>&1 || need_apt=1
tesseract --list-langs 2>/dev/null | grep -qx 'rus' || need_apt=1
python3 -c "import ensurepip" >/dev/null 2>&1 || need_apt=1

if [ "$need_apt" -eq 1 ]; then
  echo ">> Installing system packages (tesseract, Russian data, python venv)..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-rus \
    python3-venv python3-pip \
    libgl1 libglib2.0-0
fi

# 2) Python virtual environment + dependencies.
if [ ! -d .venv ]; then
  echo ">> Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi

echo ">> Installing Python dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt

echo ""
echo "Done."
echo "  $(tesseract --version | head -1)"
echo "  Languages: $(tesseract --list-langs 2>/dev/null | tail -n +2 | tr '\n' ' ')"
echo ""
echo "Run the app:  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000"
