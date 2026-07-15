import os
import io
import json
import logging
import re
from collections import Counter

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageChops
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

try:
    import cv2
except Exception:  # pragma: no cover - opencv missing entirely
    cv2 = None

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

# --- Nivel 2: face detection on image-based CVs (OpenCV YuNet, local/offline) ---
# Runs entirely inside this container — no candidate photo is ever sent to a
# third-party service. See _cover_faces_on_image for why.
FACE_MODEL_PATH = os.getenv(
    "FACE_MODEL_PATH", os.path.join(BASE, "face_detection_yunet_2023mar.onnx")
)
FACE_SCORE_THRESHOLD = float(os.getenv("FACE_SCORE_THRESHOLD", "0.6"))
FACE_NMS_THRESHOLD = float(os.getenv("FACE_NMS_THRESHOLD", "0.3"))
# CV photos are almost always in the header/top area of the page. Searching
# that crop first is faster and more reliable than the full page (a small
# face lost in a huge page image is much harder for the model to find). Falls
# back to the full image if nothing turns up in the crop.
FACE_TOP_CROP_FRACTION = float(os.getenv("FACE_TOP_CROP_FRACTION", "0.40"))
# A detected face is just the anchor point. What actually gets covered is the
# WHOLE photo block it sits in (background, clothing, date-stamp and all) -
# covering only a face-sized box left the surrounding photo clearly visible,
# which still reads as "there was a photo here". _find_photo_bbox finds that
# block by isolating pixels that differ from the page's own background color
# (sampled from the image's own border, see _page_background_color) and
# taking the connected blob that contains the face. FACE_BG_DIFF_THRESHOLD is
# the per-pixel color-distance cutoff; FACE_PHOTO_SEARCH_FRACTION is how far
# down the page to look (a bit more than FACE_TOP_CROP_FRACTION, in case the
# photo is taller than the face-detection crop); FACE_MIN_PHOTO_AREA_FRACTION
# discards tiny stray components (JPEG noise, a stray mark) as not a photo.
FACE_BG_DIFF_THRESHOLD = float(os.getenv("FACE_BG_DIFF_THRESHOLD", "30"))
FACE_PHOTO_SEARCH_FRACTION = float(os.getenv("FACE_PHOTO_SEARCH_FRACTION", "0.50"))
FACE_MORPH_KERNEL = int(os.getenv("FACE_MORPH_KERNEL", "15"))
FACE_MIN_PHOTO_AREA_FRACTION = float(os.getenv("FACE_MIN_PHOTO_AREA_FRACTION", "0.005"))
FACE_PHOTO_PAD_PX = float(os.getenv("FACE_PHOTO_PAD_PX", "3"))
# Fallback only: if the photo block can't be isolated (unusual/non-uniform
# background), cover a generous margin around just the face instead of
# leaving it fully exposed. Multiple of the face box's own width/height.
FACE_MARGIN_X = float(os.getenv("FACE_MARGIN_X", "1.15"))
FACE_MARGIN_TOP = float(os.getenv("FACE_MARGIN_TOP", "1.2"))
FACE_MARGIN_BOTTOM = float(os.getenv("FACE_MARGIN_BOTTOM", "1.3"))
FACE_COVER_COLOR = (0.25, 0.28, 0.32)  # neutral slate; only used by that fallback
# Until this is proven reliable on enough real CVs, keep the Nivel 0 "revisar
# a mano" warning ON even when a face was found and covered. Flip to true
# once you trust it — then the warning only stays on for pages where NO face
# was found at all (i.e. a possible miss).
TRUST_FACE_DETECTION = os.getenv("TRUST_FACE_DETECTION", "false").lower() in (
    "1", "true", "yes", "on",
)

app = FastAPI(title="CV Redactor", version="3.3.0")


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


# Lazily-created singleton: the model is loaded once per process, not per
# request. If the model file is missing or unreadable, face-covering is
# disabled but the rest of the service (Nivel 0 + Nivel 1) keeps working —
# same graceful-degradation pattern as the branding assets above.
_face_detector = None
_face_detector_load_attempted = False


def _get_face_detector():
    global _face_detector, _face_detector_load_attempted
    if _face_detector_load_attempted:
        return _face_detector
    _face_detector_load_attempted = True
    if cv2 is None:
        log.warning("opencv not installed - face covering (Nivel 2) disabled")
        return None
    if not os.path.exists(FACE_MODEL_PATH):
        log.warning(
            "face model not found at %s - face covering (Nivel 2) disabled",
            FACE_MODEL_PATH,
        )
        return None
    try:
        _face_detector = cv2.FaceDetectorYN_create(
            FACE_MODEL_PATH, "", (320, 320),
            score_threshold=FACE_SCORE_THRESHOLD,
            nms_threshold=FACE_NMS_THRESHOLD,
        )
    except Exception as e:
        log.warning("could not load face model (%s): %s", FACE_MODEL_PATH, e)
        _face_detector = None
    return _face_detector


@app.get("/health")
def health():
    return {
        "status": "ok",
        "branding": bool(_WM_PNG and _LOGO_PNG),
        "face_detection": _get_face_detector() is not None,
    }


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

    Returns (removed_count, bg_images): bg_images is a list of (xref, rect)
    for every image covering >= BG_COVERAGE_SKIP of the page — i.e. the CV is
    effectively a full-page photo/scan with an OCR text layer on top (see
    /redact for how that case is handled: PII text is painted over via
    _cover_pii_on_image, and any face in it via _cover_faces_on_image, instead
    of deleting the image). Truthiness of bg_images works as the old
    has_bg_image boolean did.

    apply_removal=False runs pure detection (bg_images) without touching the
    page, so callers can still raise the Nivel 0 warning even when the caller
    disabled photo removal via the remove_images form field — the RGPD safety
    net must not depend on that optional toggle.
    """
    imgs = page.get_images(full=True)
    if not imgs:
        return 0, []
    page_area = page.rect.get_area()
    if page_area <= 0:
        return 0, []
    pm = page.get_pixmap(dpi=72)  # original colors for sampling
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    removed = 0
    bg_images = []
    for img in imgs:
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception as exc:  # XObject registered as image but not a real one
            log.warning("skipping xref %s (not a usable image): %s", xref, exc)
            continue
        is_bg = False
        bg_rect = None
        usable = []
        for r in rects:
            vis = r & page.rect
            if vis.is_empty:
                continue
            if vis.get_area() / page_area >= BG_COVERAGE_SKIP:
                is_bg = True
                bg_rect = vis
                break
            usable.append(r)
        if is_bg:
            bg_images.append((xref, bg_rect))
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
    return removed, bg_images


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

    This does NOT remove a face/photo embedded in the image — that's handled
    separately by _cover_faces_on_image ("Nivel 2").
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


def _detect_faces_px(bgr_image):
    """Run the YuNet face detector on a BGR uint8 numpy image. Returns a list
    of (x, y, w, h) pixel boxes (top-left corner + size), one per detected
    face, already filtered by FACE_SCORE_THRESHOLD/FACE_NMS_THRESHOLD."""
    detector = _get_face_detector()
    if detector is None:
        return []
    h, w = bgr_image.shape[:2]
    if h <= 0 or w <= 0:
        return []
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr_image)
    if faces is None:
        return []
    return [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces]


def _page_background_color(bgr_image):
    """Median color of a thin border strip around the image's own edges. For
    an image-based CV the 'page' IS this image, so its true edges (top row,
    bottom row, left/right columns) are reliably outside any photo and give
    the real background color (typically white paper) to blend a cover into."""
    edge = 5
    border = np.concatenate([
        bgr_image[:edge, :, :].reshape(-1, 3),
        bgr_image[-edge:, :, :].reshape(-1, 3),
        bgr_image[:, :edge, :].reshape(-1, 3),
        bgr_image[:, -edge:, :].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def _find_photo_bbox(bgr_region, anchor_x, anchor_y):
    """Find the full rectangular photo block that contains pixel (anchor_x,
    anchor_y) - the center of a detected face - by isolating everything that
    differs from the region's own background color and picking the connected
    blob covering that point. A photo is a large solid blob; text lines are
    thin, well-separated blobs, so FACE_MIN_PHOTO_AREA_FRACTION cleanly tells
    them apart (confirmed on a real CV: the photo blob was ~5% of the search
    area, the largest text-line blob under 1%).

    Returns ((x, y, w, h) in bgr_region's pixel coords, bg_color as BGR
    tuple), or (None, bg_color) if no blob covering the anchor point is found.
    """
    img_h, img_w = bgr_region.shape[:2]
    bg_color = _page_background_color(bgr_region)
    diff = np.linalg.norm(
        bgr_region.astype(np.int16) - bg_color.reshape(1, 1, 3), axis=2
    )
    mask = (diff > FACE_BG_DIFF_THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((FACE_MORPH_KERNEL, FACE_MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    min_area = FACE_MIN_PHOTO_AREA_FRACTION * img_w * img_h
    ax, ay = int(anchor_x), int(anchor_y)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if x <= ax < x + w and y <= ay < y + h:
            return (x, y, w, h), tuple(bg_color)
    return None, tuple(bg_color)


def _cover_faces_on_image(page, xref, img_rect):
    """Nivel 2: find and cover any face baked into a full-page background
    image (see bg_images from _remove_images). Runs 100% locally in this
    container via OpenCV/YuNet - the candidate's photo is never sent to a
    third-party service, which would otherwise make that provider a new data
    processor of the candidate's personal data.

    Covers the WHOLE photo block (via _find_photo_bbox), not just a box
    around the face - otherwise the surrounding photo (background, clothing,
    a camera date-stamp, ...) stays visible and it still obviously reads as
    "there was a photo here". The cover is filled with the image's own
    background color so it blends in rather than leaving an obvious redacted
    box. Falls back to a padded box around just the face (see FACE_MARGIN_*)
    if the photo block can't be cleanly isolated.

    Returns the number of faces covered on this page.
    """
    detector = _get_face_detector()
    if detector is None:
        return 0
    doc = page.parent
    try:
        base = doc.extract_image(xref)
        img_bytes = base["image"]
    except Exception as e:
        log.warning("could not extract background image (xref %s): %s", xref, e)
        return 0
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return 0
    img_h, img_w = bgr.shape[:2]
    if img_h <= 0 or img_w <= 0:
        return 0

    crop_h = max(1, int(img_h * FACE_TOP_CROP_FRACTION))
    boxes = _detect_faces_px(bgr[:crop_h, :, :])
    if not boxes:
        boxes = _detect_faces_px(bgr)  # fallback: search the whole image
    if not boxes:
        return 0

    # A bit taller than the face-detection crop, in case the photo itself
    # extends lower than the face within it (shoulders, a lanyard, etc.).
    search_h = max(crop_h, min(img_h, int(img_h * FACE_PHOTO_SEARCH_FRACTION)))
    search_region = bgr[:search_h, :, :]

    sx = img_rect.width / img_w
    sy = img_rect.height / img_h
    covered = 0
    for (x, y, w, h) in boxes:
        cx, cy = x + w / 2, y + h / 2
        photo_box, bg_color = _find_photo_bbox(search_region, cx, cy)
        if photo_box is not None:
            px, py, pw, ph = photo_box
            pad = FACE_PHOTO_PAD_PX
            fx0 = img_rect.x0 + (px - pad) * sx
            fy0 = img_rect.y0 + (py - pad) * sy
            fx1 = img_rect.x0 + (px + pw + pad) * sx
            fy1 = img_rect.y0 + (py + ph + pad) * sy
            b, g, r = bg_color
            fill = (r / 255, g / 255, b / 255)
        else:
            # Couldn't isolate the photo block cleanly (unusual background) -
            # cover a generous margin around the face itself so we never
            # leave skin visible, even if the rest of the photo remains.
            fx0 = img_rect.x0 + x * sx
            fy0 = img_rect.y0 + y * sy
            fx1 = fx0 + w * sx
            fy1 = fy0 + h * sy
            fw, fh = fx1 - fx0, fy1 - fy0
            ccx, ccy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
            fx0 = ccx - fw * FACE_MARGIN_X
            fx1 = ccx + fw * FACE_MARGIN_X
            fy0 = ccy - fh * FACE_MARGIN_TOP
            fy1 = ccy + fh * FACE_MARGIN_BOTTOM
            fill = FACE_COVER_COLOR
        cover = fitz.Rect(fx0, fy0, fx1, fy1)
        cover &= page.rect  # never draw outside the page
        if cover.get_area() > 0:
            page.draw_rect(cover, color=fill, fill=fill, width=0)
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
    cover_faces: str = Form("true"),
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
    do_faces = str(cover_faces).lower() in ("1", "true", "yes", "on")

    total_text = 0
    total_imgs = 0
    total_lines = 0
    total_covered = 0
    total_faces = 0
    image_based_pages = []
    pages_without_face = []  # image-based pages where 0 faces were found/covered
    for page_idx, page in enumerate(doc):
        # 1) remove photo(s) without harming text; detect "image-based" CVs
        # (full-page background image + OCR text layer) regardless of the
        # remove_images flag, so the RGPD warning below always fires.
        removed, bg_images = _remove_images(page, apply_removal=do_images)
        if do_images:
            total_imgs += removed
        has_bg_image = bool(bg_images)
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

        # 2d) Nivel 2: cover any face baked into the background image itself.
        if has_bg_image and do_faces:
            faces_this_page = sum(
                _cover_faces_on_image(page, xref, rect) for xref, rect in bg_images
            )
            total_faces += faces_this_page
            if faces_this_page == 0:
                pages_without_face.append(page_idx + 1)
        elif has_bg_image:
            pages_without_face.append(page_idx + 1)

        # 3) Behum branding (watermark + logo)
        if do_brand:
            _add_branding(page)

    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    log.info(
        "removed %d image(s); redacted %d text occurrence(s); cleared %d underline(s); "
        "covered %d PII region(s) and %d face(s) over background image(s) across %d terms",
        total_imgs, total_text, total_lines, total_covered, total_faces, len(term_list),
    )

    # Nivel 0 (RGPD safety net): an "image-based" CV (full-page photo/scan +
    # invisible OCR layer) needs extra scrutiny. Nivel 2 now covers faces it
    # finds automatically, but until that's proven reliable on enough real
    # CVs (TRUST_FACE_DETECTION=false by default) we keep flagging every
    # image-based CV for manual review regardless of what Nivel 2 did. Once
    # trusted, the flag only stays on for pages where NO face was found at
    # all (a possible miss) - never silently, always visible in both the
    # response headers (for the calling workflow) and the filename (for a
    # human glancing at the file).
    image_based = bool(image_based_pages)
    if TRUST_FACE_DETECTION:
        needs_review_pages = pages_without_face
    else:
        needs_review_pages = image_based_pages
    needs_review = bool(needs_review_pages)

    if image_based:
        log.warning(
            "image-based CV detected on page(s) %s - scanned/photographed CV "
            "(full-page image + OCR text layer). PII text covered; %d face(s) "
            "covered automatically; page(s) %s had no face covered. "
            "needs_manual_review=%s (TRUST_FACE_DETECTION=%s)",
            image_based_pages, total_faces, pages_without_face or "none",
            needs_review, TRUST_FACE_DETECTION,
        )
        filename = "cv_ciego_REVISAR_FOTO.pdf" if needs_review else "cv_ciego.pdf"
    else:
        filename = "cv_ciego.pdf"

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Cv-Redactor-Image-Based": "true" if image_based else "false",
        "X-Cv-Redactor-Image-Pages": ",".join(str(p) for p in image_based_pages),
        "X-Cv-Redactor-Faces-Covered": str(total_faces),
        "X-Cv-Redactor-Pages-Without-Face": ",".join(str(p) for p in pages_without_face),
        "X-Cv-Redactor-Needs-Manual-Photo-Review": "true" if needs_review else "false",
    }
    return Response(
        content=out,
        media_type="application/pdf",
        headers=headers,
    )
