# Windows (PowerShell) setup for the Russian OCR app.
# Run from the repo root:  .\scripts\install.ps1
# NOTE: scripts/install.sh is Linux-only (apt/sudo) and will NOT run on native
# Windows. Use this script instead, or use WSL2 (see README).

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# 1) Virtual environment + base Python dependencies.
if (-not (Test-Path .venv)) {
    Write-Host ">> Creating virtual environment (.venv)..."
    python -m venv .venv
}
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt

Write-Host ""
Write-Host "Base dependencies installed."
Write-Host ""
Write-Host "Tesseract (only needed for engine=tesseract) is a separate Windows install:"
Write-Host "  winget install --id UB-Mannheim.TesseractOCR"
Write-Host "  In the installer, tick the Russian language data (rus)."
Write-Host "  If tesseract.exe is not on PATH, set:  `$env:TESSERACT_CMD='C:\Program Files\Tesseract-OCR\tesseract.exe'"
Write-Host ""
Write-Host "Optional engines:"
Write-Host "  TrOCR (CPU handwriting):  .\.venv\Scripts\python -m pip install -r requirements-trocr.txt"
Write-Host "  Chandra OCR 2 (needs GPU): .\.venv\Scripts\python -m pip install -r requirements-chandra.txt"
Write-Host ""
Write-Host "Run the app:"
Write-Host "  .\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
