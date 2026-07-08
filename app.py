import os
import io
import json
import logging
import re
from collections import Counter

import fitz  # PyMuPDF
from PIL import Image, ImageChops
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cv-redactor")

BASE = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.getenv("REDACT_API_KEY")
# Vertical inset (fraction of each match's height) applied to redaction boxes so
# they stay within the target line and never bleed into the line above/below.
REDACT_VINSET = float(os.getenv("REDACT_VINSET", "0.25"))
# A whole line is removed (e.g. "email | phone | linkedin") only when, after
# removing the PII values, nothing but these filler chars remains. This clears
# leftover separators on contact lines without touching legit pipes elsewhere.
_FILLER = re.compile(r"[\s|·•—–/,.:;\-]+")
# Images covering >= this fraction of the page are treated as full-page
# backgrounds (typically a scanned/photographed CV with an OCR text layer on
# top) and are NOT deleted like a normal photo would be — see has_bg_image /
# _cover_pii_on_image below for how their PII is handled instead.
BG_COVERAGE_SKIP = float(os.getenv("REDACT_BG_COVERAGE_SKIP", "0.85"))
# Thin horizontal strokes inside an already-redacted region are leftover
# underlines (e.g. a redacted hyperlink). They are painted over with the
# surrounding background color. Restricted to redacted zones so color panels
# and legit rules/borders stay untouched.
UNDERLINE_MIN_WIDTH = float(os.getenv("UNDERLINE_MIN_WIDTH", "15"))
UNDERLINE_MAX_HEIGHT = float(os.getenv("UNDERLINE_MAX_HEIGHT", "3.5"))
UNDERLINE_PROBE_PAD = float(os.getenv("UNDERLINE_PROBE_PAD", "3"))

# --- Branding (Behum) ---
ICON_PATH = os.getenv("BEHUM_ICON", os.path.join(BASE, "behum_icon.png"))
LOGO_PATH = os.getenv("BEHUM_LOGO", os.path.join(BASE, "behum_logo.png"))
WATERMARK_OPACITY = float(os.getenv("WATERMARK_OPACITY", "0.22"))
WATERMARK_WIDTH = float(os.getenv("WATERMARK_WIDTH", "0.60"))   # fraction of page width
LOGO_MAX_WIDTH = float(os.getenv("LOGO_MAX_WIDTH", "0.62"))     # fraction of page width

app = FastAPI(title="CV Redactor", version="3.1.0")


def _load_keyed_png(path, opacity=1.0):
    """Turn the (black) background transparent and recolor the artwork to its
    own brand color, so anti-aliased edges keep no dark halo."""
    im = Image.open(path).convert("RGB")
    r, g, b = im.split()
    alpha = ImageChops.lighter(ImageChops.lighter(r, g), b)  # brightness -> alpha
    colors = im.getcolors(maxcolors=16777216) or []
    bright = [(cnt, col) for cnt, col in colors if max(col) > 150]
    brand = max(bright, key=lambda t: t[0])[1] if bright else (245, 190, 0)
    if opacity < 1.0:
        alpha = alpha.point(lambda v: int(v * opacity))
    solid = Image.new("RGBA", im.size, brand + (255,))
    solid.putalpha(alpha)
    buf = io.BytesIO()
    solid.save(buf, "PNG")
    return buf.getvalue(), solid.size


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


def _remove_images(page, apply_removal=True):
    """Remove raster images (photos) WITHOUT deleting nearby text, then fill
    the photo area with the surrounding background color. Skips full-page
    backgrounds and never fills over text below the image.

    Returns (removed_count, has_bg_image): has_bg_image is True when this
    page contains an image covering >= BG_COVERAGE_SKIP of the page — i.e.
    the CV is effectively a full-page photo/scan with an OCR text layer on
    top (see /redact for how that case is handled: PII text is painted over
    via _cover_pii_on_image instead of deleting the image).

    apply_removal=False runs pure detection (has_bg_image) without touching
    the page, so callers can still raise the Nivel 0 warning even when the
    caller disabled photo removal via the remove_images form field — the
    RGPD safety net must not depend on that optional toggle.
    """
    imgs = page.get_images(full=True)
    if not imgs:
        return 0, False
    page_area = page.rect.get_area()
    if page_area <= 0:
        return 0, False
    pm = page.get_pixmap(dpi=72)  # original colors for sampling
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    removed = 0
    has_bg_image = False
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
        if is_bg:
            has_bg_image = True
            continue
        if not usable or not apply_removal:
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
    return removed, has_bg_image


def _remove_underlines(page, red_rects, pm):
    """Paint over thin horizontal strokes (link/text underlines) that fall in an
    already-redacted region, using the surrounding background color. Restricted
    to redacted zones, so color panels and legit rules stay untouched."""
    if not red_rects:
        return 0
    removed = 0
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < UNDERLINE_MIN_WIDTH or r.height > UNDERLINE_MAX_HEIGHT:
            continue
        in_red = any(
            fitz.Rect(rr.x0, rr.y0, rr.x1, rr.y1 + UNDERLINE_PROBE_PAD).intersects(r)
            for rr in red_rects
        )
        if not in_red:
            continue
        color = tuple(c / 255 for c in _sample_color(page, pm, r))
        page.draw_rect(
            fitz.Rect(r.x0, r.y0 - 0.6, r.x1, r.y1 + 0.6),
            color=color, fill=color, width=0,
        )
        removed += 1
    return removed


def _cover_pii_on_image(page, red_rects, pm):
    """For 'image-based' CVs (a full-page photo/scan with an invisible OCR text
    layer on top), apply_redactions() only deletes the invisible glyphs — the
    PII pixels baked into the image itself are still visible underneath. Since
    the OCR layer's bounding boxes line up with the visible text (search_for
    already located it), paint over each PII rect with the sampled surrounding
    color — same technique already used for photos and underlines.

    The OCR bbox is a close but imperfect fit around the rendered glyphs (font
    metric estimation, not pixel-perfect), so a tight box can leave a sliver of
    the last character visible at the right edge. A small asymmetric pad
    compensates: more padding on the right (where the mismatch was observed on
    a real CV) than on the left (where padding would otherwise eat into the
    preceding word's punctuation, e.g. "Nombre:"). Vertical pad stays small,
    capped well under half the tightest line gap seen in real CVs (~2.6pt), so
    it never bleeds into the line above/below.

    This does NOT remove a face/photo embedded in the image — that requires
    face detection (a separate, not-yet-implemented step, "Nivel 2"). Callers
    should treat pages where this ran as still needing manual photo review.
    """
    covered = 0
    for r in red_rects:
        h = r.y1 - r.y0
        pad_left = 1.5
        pad_right = max(6.0, 0.7 * h)
        pad_y = min(1.0, 0.15 * h)
        rr = fitz.Rect(r.x0 - pad_left, r.y0 - pad_y, r.x1 + pad_right, r.y1 + pad_y)
        color = tuple(c / 255 for c in _sample_color(page, pm, rr))
        page.draw_rect(rr, color=color, fill=color, width=0)
        covered += 1
    return covered


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
        margin = 8
        gap = 4
        band_x0 = 0.45 * W
        right_text = [b for b in blocks if b[2] > band_x0]
        top_y = min((b[1] for b in right_text), default=0.30 * H)
        clear_h = max(top_y - margin - gap, 0)
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


def _redaction_rects(page, term_list):
    """Rects to redact: each PII match, plus the full bbox of any line that is
    entirely PII + separators (so leftover '|' etc. on contact lines go too)."""
    rects = []
    for term in term_list:
        rects.extend(page.search_for(term))
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"])
            if not any(t in text for t in term_list):
                continue
            residual = text
            for t in term_list:
                residual = residual.replace(t, "")
            if _FILLER.sub("", residual) == "":
                rects.append(fitz.Rect(line["bbox"]))
    return rects


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
    total_lines = 0
    total_covered = 0
    image_based_pages = []
    for page_idx, page in enumerate(doc):
        # 1) remove photo(s) without harming text; detect "image-based" CVs
        # (full-page background image + OCR text layer) regardless of the
        # remove_images flag, so the RGPD warning below always fires.
        removed, has_bg_image = _remove_images(page, apply_removal=do_images)
        if do_images:
            total_imgs += removed
        if has_bg_image:
            image_based_pages.append(page_idx + 1)

        # 2) remove PII text (no box; only the glyphs are deleted)
        red_rects = _redaction_rects(page, term_list)
        for r in red_rects:
            inset = REDACT_VINSET * (r.y1 - r.y0)
            page.add_redact_annot(
                fitz.Rect(r.x0, r.y0 + inset, r.x1, r.y1 - inset), fill=False
            )
        if red_rects:
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
            # 2b) remove leftover underlines inside the redacted zones
            pm_bg = page.get_pixmap(dpi=72)
            total_lines += _remove_underlines(page, red_rects, pm_bg)
            # 2c) image-based CV: the PII is also baked into the background
            # image itself (apply_redactions only touched the invisible OCR
            # text). Paint over it too, so the visible page doesn't leak PII.
            if has_bg_image:
                total_covered += _cover_pii_on_image(page, red_rects, pm_bg)
        total_text += len(red_rects)

        # 3) Behum branding (watermark + logo)
        if do_brand:
            _add_branding(page)

    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    log.info(
        "removed %d image(s); redacted %d text occurrence(s); cleared %d underline(s); "
        "covered %d PII region(s) over background image across %d terms",
        total_imgs, total_text, total_lines, total_covered, len(term_list),
    )

    # Nivel 0 (RGPD safety net): an "image-based" CV (full-page photo/scan +
    # invisible OCR layer) still has its embedded photo visible after this —
    # face removal (Nivel 2) is not implemented yet. Flag it clearly instead
    # of silently shipping a CV that only *looks* anonymized: both in the
    # response headers (for the calling workflow to branch on) and in the
    # filename (so a human glancing at the file also sees it).
    image_based = bool(image_based_pages)
    if image_based:
        log.warning(
            "image-based CV detected on page(s) %s - scanned/photographed CV "
            "(full-page image + OCR text layer). PII text was covered, but any "
            "face/photo embedded in that image is NOT removed by this service "
            "yet. Flag for manual photo review before sending to a client.",
            image_based_pages,
        )
        filename = "cv_ciego_REVISAR_FOTO.pdf"
    else:
        filename = "cv_ciego.pdf"

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Cv-Redactor-Image-Based": "true" if image_based else "false",
        "X-Cv-Redactor-Image-Pages": ",".join(str(p) for p in image_based_pages),
        "X-Cv-Redactor-Needs-Manual-Photo-Review": "true" if image_based else "false",
    }
    return Response(
        content=out,
        media_type="application/pdf",
        headers=headers,
    )
