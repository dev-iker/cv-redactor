import os
import io
import json
import logging
from collections import Counter

import fitz  # PyMuPDF
from PIL import Image, ImageChops
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cv-redactor")

BASE = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.getenv("REDACT_API_KEY")
RECT_PADDING = float(os.getenv("REDACT_PADDING", "2"))
# Images covering >= this fraction of the page are treated as full-page
# backgrounds and left untouched (so we don't wipe a CV's whole design).
BG_COVERAGE_SKIP = float(os.getenv("REDACT_BG_COVERAGE_SKIP", "0.85"))

# --- Branding (Behum) ---
ICON_PATH = os.getenv("BEHUM_ICON", os.path.join(BASE, "behum_icon.png"))
LOGO_PATH = os.getenv("BEHUM_LOGO", os.path.join(BASE, "behum_logo.png"))
WATERMARK_OPACITY = float(os.getenv("WATERMARK_OPACITY", "0.22"))
WATERMARK_WIDTH = float(os.getenv("WATERMARK_WIDTH", "0.60"))   # fraction of page width
LOGO_MAX_WIDTH = float(os.getenv("LOGO_MAX_WIDTH", "0.90"))     # fraction of page width

app = FastAPI(title="CV Redactor", version="3.0.0")


def _load_keyed_png(path, opacity=1.0):
    """Load a logo, turn its (black) background transparent by deriving alpha
    from brightness, and optionally scale the alpha to make it faint."""
    im = Image.open(path).convert("RGB")
    r, g, b = im.split()
    alpha = ImageChops.lighter(ImageChops.lighter(r, g), b)  # max channel
    if opacity < 1.0:
        alpha = alpha.point(lambda v: int(v * opacity))
    im = im.convert("RGBA")
    im.putalpha(alpha)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue(), im.size


# Pre-build branding assets once at startup (graceful if files are missing).
try:
    _WM_PNG, _WM_SIZE = _load_keyed_png(ICON_PATH, WATERMARK_OPACITY)
except Exception as e:
    _WM_PNG, _WM_SIZE = None, None
    log.warning("watermark icon not loaded (%s): %s", ICON_PATH, e)
try:
    _LOGO_PNG, _LOGO_SIZE = _load_keyed_png(LOGO_PATH, 1.0)
except Exception as e:
    _LOGO_PNG, _LOGO_SIZE = None, None
    log.warning("logo not loaded (%s): %s", LOGO_PATH, e)


@app.get("/health")
def health():
    return {"status": "ok", "branding": bool(_WM_PNG and _LOGO_PNG)}


def _sample_color(page, pm, rect, margin=8, step=8):
    """Most common color in a ring just outside the rect (left/right edges)."""
    W, H = page.rect.width, page.rect.height
    cols = []
    for x in (rect.x1 + margin, rect.x0 - margin):
        if 0 <= x < W:
            y = rect.y0
            while y < rect.y1:
                if 0 <= y < H:
                    cols.append(pm.pixel(int(min(max(x, 0), pm.width - 1)),
                                         int(min(max(y, 0), pm.height - 1))))
                y += step
    return Counter(cols).most_common(1)[0][0] if cols else (255, 255, 255)


def _remove_images(page):
    """Remove raster images (photos) WITHOUT deleting nearby text, then fill
    the photo area with the surrounding background color. Skips full-page
    backgrounds and never fills over text below the image."""
    imgs = page.get_images(full=True)
    if not imgs:
        return 0
    page_area = page.rect.get_area()
    if page_area <= 0:
        return 0
    pm = page.get_pixmap(dpi=72)  # original colors for sampling
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    removed = 0
    for img in imgs:
        xref = img[0]
        rects = page.get_image_rects(xref)
        is_bg = False
        usable = []
        for r in rects:
            vis = r & page.rect
            if vis.is_empty:
                continue
            if vis.get_area() / page_area >= BG_COVERAGE_SKIP:
                is_bg = True
                break
            usable.append(r)
        if is_bg or not usable:
            continue
        page.delete_image(xref)  # removes image, keeps text
        for r in usable:
            inter = [b for b in blocks if fitz.Rect(b[:4]).intersects(r)]
            fill_bottom = (min(b[1] for b in inter) - 2) if inter else r.y1
            color = tuple(c / 255 for c in _sample_color(page, pm, r))
            fr = fitz.Rect(r.x0, r.y0, r.x1, max(r.y0, fill_bottom))
            if fr.get_area() > 0:
                page.draw_rect(fr, color=color, fill=color, width=0)
        removed += 1
    return removed


def _add_branding(page):
    """Centered faint 'B' watermark + full logo placed in the top clear zone."""
    W, H = page.rect.width, page.rect.height

    if _WM_PNG:
        iw, ih = _WM_SIZE
        ww = WATERMARK_WIDTH * W
        wh = ww * ih / iw
        page.insert_image(
            fitz.Rect((W - ww) / 2, (H - wh) / 2, (W + ww) / 2, (H + wh) / 2),
            stream=_WM_PNG, overlay=True, keep_proportion=True,
        )

    if _LOGO_PNG:
        lw, lh = _LOGO_SIZE
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        margin = 16
        band_x0 = 0.45 * W
        right_text = [b for b in blocks if b[2] > band_x0]
        top_y = min((b[1] for b in right_text), default=0.30 * H)
        clear_h = max(top_y - margin - 6, 0)
        logo_w = LOGO_MAX_WIDTH * W
        logo_h = logo_w * lh / lw
        if logo_h > clear_h and clear_h > 20:   # fit within the clear zone
            logo_h = clear_h
            logo_w = logo_h * lw / lh
        x1 = W - margin
        x0 = x1 - logo_w
        y0 = margin
        y1 = y0 + logo_h
        page.insert_image(
            fitz.Rect(x0, y0, x1, y1),
            stream=_LOGO_PNG, overlay=True, keep_proportion=True,
        )


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    terms: str = Form(...),
    remove_images: str = Form("true"),
    add_branding: str = Form("true"),
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
    do_brand = str(add_branding).lower() in ("1", "true", "yes", "on")

    total_text = 0
    total_imgs = 0
    for page in doc:
        # 1) remove photo(s) without harming text
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
        # 3) Behum branding (watermark + logo)
        if do_brand:
            _add_branding(page)

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