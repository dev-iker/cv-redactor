import os
import json
import logging
from collections import Counter

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cv-redactor")

API_KEY = os.getenv("REDACT_API_KEY")
RECT_PADDING = float(os.getenv("REDACT_PADDING", "2"))
# Images covering >= this fraction of the page are treated as full-page
# backgrounds and left untouched (so we don't wipe a CV's whole design).
BG_COVERAGE_SKIP = float(os.getenv("REDACT_BG_COVERAGE_SKIP", "0.85"))

app = FastAPI(title="CV Redactor", version="2.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _dominant_color_around(page, pm, rect, margin=8, step=8):
    """Most common color in a thin ring just outside the image rect, so the
    removed photo can be filled with the surrounding background color."""
    W, H = page.rect.width, page.rect.height
    pts = []
    for x in (rect.x1 + margin, rect.x0 - margin):      # right / left edges
        if 0 <= x < W:
            y = rect.y0
            while y < rect.y1:
                if 0 <= y < H:
                    pts.append((x, y))
                y += step
    for y in (rect.y0 - margin, rect.y1 + margin):      # top / bottom edges
        if 0 <= y < H:
            x = rect.x0
            while x < rect.x1:
                if 0 <= x < W:
                    pts.append((x, y))
                x += step
    cols = []
    for (px, py) in pts:
        ix = int(min(max(px, 0), pm.width - 1))
        iy = int(min(max(py, 0), pm.height - 1))
        cols.append(pm.pixel(ix, iy))
    if not cols:
        return (255, 255, 255)
    return Counter(cols).most_common(1)[0][0]


def _remove_images(page):
    """Remove raster images (photos) and fill their area with the surrounding
    background color. Skips near-full-page background images."""
    imgs = page.get_images(full=True)
    if not imgs:
        return 0
    page_area = page.rect.get_area()
    pm = page.get_pixmap(dpi=72)  # 1 pt ~ 1 px, used only for color sampling
    fills = []
    removed = 0
    for img in imgs:
        xref = img[0]
        for r in page.get_image_rects(xref):
            visible = r & page.rect
            if visible.is_empty or page_area <= 0:
                continue
            if (visible.get_area() / page_area) >= BG_COVERAGE_SKIP:
                continue  # likely a full-page background -> leave it
            color = _dominant_color_around(page, pm, r)
            fills.append((r, tuple(c / 255 for c in color)))
            page.add_redact_annot(r, fill=False)
            removed += 1
    if removed:
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_REMOVE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )
        for (r, col) in fills:
            page.draw_rect(r, color=col, fill=col, width=0)
    return removed


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    terms: str = Form(...),
    remove_images: str = Form("true"),
    x_api_key: str | None = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    try:
        term_list = json.loads(terms)
        if not isinstance(term_list, list):
            raise ValueError
        term_list = [str(t) for t in term_list if str(t).strip()]
    except Exception:
        raise HTTPException(
            status_code=400, detail="`terms` must be a JSON array of strings"
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="not a valid PDF")

    do_images = str(remove_images).lower() in ("1", "true", "yes", "on")

    total_text = 0
    total_imgs = 0
    for page in doc:
        # 1) remove photo(s), blending with the surrounding background
        if do_images:
            total_imgs += _remove_images(page)
        # 2) remove PII text (no box; only the glyphs are deleted)
        page_hits = 0
        for term in term_list:
            for rect in page.search_for(term):
                padded = fitz.Rect(
                    rect.x0 - RECT_PADDING, rect.y0 - RECT_PADDING,
                    rect.x1 + RECT_PADDING, rect.y1 + RECT_PADDING,
                )
                page.add_redact_annot(padded, fill=False)
                page_hits += 1
        if page_hits:
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
        total_text += page_hits

    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    log.info(
        "removed %d image(s); redacted %d text occurrence(s) across %d terms",
        total_imgs, total_text, len(term_list),
    )
    return Response(
        content=out,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="cv_ciego.pdf"'},
    )
