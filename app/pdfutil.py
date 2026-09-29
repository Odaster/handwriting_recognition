"""PDF -> page images, using PyMuPDF (no external Poppler/Ghostscript needed)."""

from __future__ import annotations

from PIL import Image

_MISSING = (
    "Распознавание PDF требует PyMuPDF. Установите: pip install pymupdf "
    "(входит в requirements.txt)."
)


def pdf_to_images(data: bytes, dpi: int = 200, max_pages: int = 50) -> list[Image.Image]:
    """Render each PDF page to a PIL image at the given DPI."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # older PyMuPDF exposes the module as `fitz`
        except ImportError as exc:
            raise RuntimeError(_MISSING) from exc

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            if len(images) >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return images
