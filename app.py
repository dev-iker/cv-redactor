import hashlib
import os
import io
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections import Counter, OrderedDict
from difflib import SequenceMatcher

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageChops
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response, JSONResponse
from starlette.concurrency import run_in_threadpool

try:
    import cv2
except Exception:  # pragma: no cover - opencv missing entirely
    cv2 = None

try:
    import pytesseract
except Exception:  # pragma: no cover - tesseract wrapper missing
    pytesseract = None

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
# Los metadatos del PDF son una fuga de PII silenciosa: el CV de InfoJobs que
# disparó todo esto llevaba title="C.V.-JOSE-LUIS ACTUALIZADO", visible en la
# barra de título de cualquier visor aunque la página esté impecable.
SCRUB_METADATA = os.getenv("SCRUB_METADATA", "true").lower() in (
    "1", "true", "yes", "on")

# --- Branding (Behum) ---
ICON_PATH = os.getenv("BEHUM_ICON", os.path.join(BASE, "behum_icon.png"))
LOGO_PATH = os.getenv("BEHUM_LOGO", os.path.join(BASE, "behum_logo.png"))
WATERMARK_OPACITY = float(os.getenv("WATERMARK_OPACITY", "0.22"))
WATERMARK_WIDTH = float(os.getenv("WATERMARK_WIDTH", "0.60"))   # fraction of page width
# Ancho preferente del logo, en fracción del ancho de página. Antes esto era
# LOGO_MAX_WIDTH=0.62 y en la práctica el logo SIEMPRE salía a ese tamaño: la
# única rama que lo encogía pedía `clear_h > 20`, o sea que solo actuaba si
# había hueco, y cuando no lo había lo dibujaba a tamaño completo encima del
# contenido. En un CV de InfoJobs el hueco es exactamente 0 pt, porque su
# propia marca "CV inscrito desde InfoJobs" está a 10 pt del borde.
LOGO_WIDTH = float(os.getenv("LOGO_WIDTH", "0.30"))
# Si al tamaño preferente el logo pisa texto, se va probando esta escalera
# antes de cambiar de esquina o de renunciar.
LOGO_SCALE_LADDER = tuple(float(s) for s in os.getenv(
    "LOGO_SCALE_LADDER", "1.0,0.8,0.65,0.5").split(","))
# Esquinas candidatas, en orden de preferencia.
LOGO_CORNERS = tuple(c.strip() for c in os.getenv(
    "LOGO_CORNERS", "top-right,top-left,bottom-right,bottom-left").split(","))
LOGO_MARGIN = float(os.getenv("LOGO_MARGIN", "8"))
# Holgura que se exige alrededor del logo para no dejarlo pegado al texto.
LOGO_CLEARANCE = float(os.getenv("LOGO_CLEARANCE", "4"))

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
# ===========================================================================
# Nivel 3 — OCR para CVs raster puros (InfoJobs y similares)
#
# Un CV descargado de InfoJobs viene rasterizado por ImageMagick: una imagen
# a página completa y CERO caracteres extraíbles. Sin capa de texto,
# page.search_for() devuelve 0 resultados, red_rects queda vacío y todo el
# Nivel 1 (apply_redactions + _cover_pii_on_image) nunca llega a ejecutarse:
# el microservicio no ve ni un carácter del CV. Eso también dejaba sin texto
# al nodo "Extraer texto CV" de la Automatización 1, así que el resumen de
# Claude se hacía solo con las respuestas de la entrevista.
#
# La única salida es generar nosotros el texto Y SUS COORDENADAS con OCR
# local (Tesseract dentro del contenedor: ningún CV sale a un tercero, que
# convertiría al proveedor en nuevo encargado del tratamiento).
# ===========================================================================
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes", "on")
OCR_LANG = os.getenv("OCR_LANG", "spa+eng")
# Un raster de InfoJobs viene a 150 dpi. Renderizar por encima de la
# resolución nativa no añade información y multiplica el tiempo de OCR.
OCR_DPI_MIN = int(os.getenv("OCR_DPI_MIN", "150"))
OCR_DPI_MAX = int(os.getenv("OCR_DPI_MAX", "260"))
# Por debajo de estos caracteres extraíbles se considera que la página NO
# tiene capa de texto usable y hay que pasar por OCR.
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "120"))
# Si la página SÍ trae capa de texto (escaneo con OCR incrustado), por
# defecto se deja el camino de siempre, que ya funciona. Ponerlo a true suma
# el OCR propio como red extra a costa de unos segundos por página.
OCR_ON_TEXT_LAYER_PAGES = os.getenv("OCR_ON_TEXT_LAYER_PAGES", "false").lower() in (
    "1", "true", "yes", "on")
# Pasadas de OCR: cada una preprocesa la página de forma distinta y se hace la
# UNIÓN de lo que encuentra cada una. Una sola pasada NO es fiable: en un CV
# real el apellido del candidato iba sobre una banda de color y Tesseract lo
# ignora por completo en la imagen en crudo (binarización global); solo
# aparece con "bgnorm". Ese fallo, con una sola pasada, dejaba el apellido
# visible en el CV ciego.
OCR_PASSES = [p.strip() for p in os.getenv(
    "OCR_PASSES", "bgnorm,gray_otsu,darkinv").split(",") if p.strip()]
# Pasadas usadas por /extract-text (ahí manda la calidad del texto, no la
# recall de cajas, y una sola pasada es 3x más rápida).
OCR_TEXT_PASSES = [p.strip() for p in os.getenv(
    "OCR_TEXT_PASSES", "bgnorm").split(",") if p.strip()]
# n8n llama a /extract-text y después a /redact con EL MISMO PDF, así que las
# mismas páginas se pasaban por OCR dos veces (5 pasadas en total sobre un CV
# de 7 páginas ≈ 50 s). Se cachean las líneas por (hash del PDF, página,
# pasada) para que la segunda llamada reutilice lo que ya leyó la primera.
# Se guardan solo las líneas (unos KB), no las imágenes.
OCR_CACHE_DOCS = int(os.getenv("OCR_CACHE_DOCS", "8"))
OCR_CACHE_TTL = float(os.getenv("OCR_CACHE_TTL", "900"))  # segundos
# Techo de páginas a pasar por OCR en una petición. Las que sobren se marcan
# para revisión manual en vez de tumbar el worker con un PDF de 40 páginas.
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "20"))
# Umbral de similitud para dar por bueno un match difuso. El OCR comete
# errores sistemáticos ("@"->"O", "i"->"¡"), así que el match exacto no sirve:
# el email "j_gonzalez37@hotmail.com" se lee "¡_gonzalez37Ohotmail.com".
OCR_FUZZY_THRESHOLD = float(os.getenv("OCR_FUZZY_THRESHOLD", "0.80"))
OCR_TERM_MIN_FUZZY = int(os.getenv("OCR_TERM_MIN_FUZZY", "5"))
OCR_TERM_MIN_EXACT = int(os.getenv("OCR_TERM_MIN_EXACT", "3"))
OCR_MAX_WINDOW = int(os.getenv("OCR_MAX_WINDOW", "8"))
# Campos de sesgo (nacionalidad, fecha nac., edad, estado civil, sexo).
# DESACTIVADO por defecto para mantener paridad con el pipeline de texto, que
# solo tapa los términos que manda n8n. A true si Behum decide que el CV
# ciego también debe ocultarlos. Se informan siempre en la cabecera
# X-Cv-Redactor-Bias-Labels-Found para poder decidir con datos.
REDACT_BIAS_FIELDS = os.getenv("REDACT_BIAS_FIELDS", "false").lower() in (
    "1", "true", "yes", "on")
# El pintado sobre la imagen se hace en PÍXELES y se sustituye el stream de la
# imagen en el PDF (destructivo). El draw_rect de siempre solo dibuja un
# rectángulo vectorial ENCIMA: los píxeles originales siguen dentro del PDF y
# se recuperan borrando la capa de dibujo. A false vuelve al comportamiento
# antiguo (y marca el CV para revisión manual).
OCR_DESTRUCTIVE = os.getenv("OCR_DESTRUCTIVE", "true").lower() in (
    "1", "true", "yes", "on")
OCR_JPEG_QUALITY = int(os.getenv("OCR_JPEG_QUALITY", "88"))

app = FastAPI(title="CV Redactor", version="4.1.0")


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


_tesseract_ok = None


def _has_tesseract():
    """Comprueba una sola vez que el binario de Tesseract responde. Si no está,
    el Nivel 3 se desactiva pero el resto del servicio sigue funcionando (mismo
    patrón de degradación que el branding y el modelo de caras)."""
    global _tesseract_ok
    if _tesseract_ok is not None:
        return _tesseract_ok
    if pytesseract is None:
        _tesseract_ok = False
        return False
    try:
        pytesseract.get_tesseract_version()
        _tesseract_ok = True
    except Exception as e:
        log.warning("tesseract no disponible - Nivel 3 (OCR) desactivado: %s", e)
        _tesseract_ok = False
    return _tesseract_ok


@app.get("/health")
def health():
    langs = []
    if _has_tesseract():
        try:
            langs = pytesseract.get_languages(config="")
        except Exception:
            langs = []
    return {
        "status": "ok",
        "version": app.version,
        "branding": bool(_WM_PNG and _LOGO_PNG),
        "face_detection": _get_face_detector() is not None,
        "ocr": _has_tesseract(),
        "ocr_langs": langs,
        "ocr_destructive": OCR_DESTRUCTIVE,
    }


# --------------------------------------------------------------------------
# Normalización de texto
# --------------------------------------------------------------------------
_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(s):
    """Minúsculas, sin acentos, solo [a-z0-9]. Es la forma canónica que se usa
    para comparar términos con lo que ha leído el OCR: absorbe acentos mal
    leídos, espacios de más, separadores y puntuación."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _ALNUM.sub("", s.lower())


def _ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------
# Etiquetas de PII (es/en). El valor a tapar es lo que va DESPUÉS de la
# etiqueta; la etiqueta se deja visible para que el CV siga leyéndose.
# --------------------------------------------------------------------------
CONTACT_LABELS = [
    "nombre", "nombres", "nombre y apellidos", "nombre completo", "apellidos",
    "apellido", "name", "full name", "surname", "candidato", "candidata",
    "dni", "nif", "nie", "documento de identidad", "pasaporte", "passport",
    # OJO: "calle"/"avenida"... NO van aquí. Si estuvieran, se tratarían como
    # etiqueta y se dejaría el tipo de vía visible ("Calle ███████"). Se
    # gestionan en STREET_WORDS, que tapa la línea entera.
    "direccion", "direccion postal", "domicilio", "address",
    "localidad", "poblacion", "municipio", "ciudad", "city", "provincia",
    "codigo postal", "cp", "c p", "zip", "zip code",
    "telefono", "telefonos", "telf", "tlf", "tel", "movil", "moviles",
    "celular", "phone", "mobile", "telephone", "contacto", "contact",
    "email", "e mail", "correo", "correo electronico", "mail",
    "linkedin", "skype", "web", "website", "instagram", "twitter", "github",
]
BIAS_LABELS = [
    "fecha de nacimiento", "fecha nacimiento", "fecha naciemiento",
    "nacimiento", "f nacimiento", "fec nacimiento", "date of birth",
    "birth date", "birthdate", "dob",
    "edad", "age", "anos", "nacionalidad", "nationality", "estado civil",
    "marital status", "sexo", "genero", "gender", "lugar de nacimiento",
]
# Etiquetas cuyo valor puede continuar en las líneas siguientes (una dirección
# postal ocupa 2-3 líneas alineadas en la misma columna).
MULTILINE_LABELS = {
    "direccion", "direccion postal", "domicilio", "address",
    "localidad", "poblacion", "municipio", "ciudad", "city",
}
_CONTACT_NORM = {_norm(l): l for l in CONTACT_LABELS}
_BIAS_NORM = {_norm(l): l for l in BIAS_LABELS}
_MULTILINE_NORM = {_norm(l) for l in MULTILINE_LABELS}
_ALL_LABELS_NORM = dict(_CONTACT_NORM)
_ALL_LABELS_NORM.update(_BIAS_NORM)

# Tipos de vía: una línea que empieza por uno de estos es una dirección postal
# aunque no lleve etiqueta delante (típico en plantillas modernas de CV).
STREET_WORDS = {
    "calle", "c", "cl", "avenida", "avda", "av", "plaza", "pza", "pl",
    "paseo", "po", "camino", "cami", "carrer", "carretera", "ctra", "ronda",
    "travesia", "urbanizacion", "urb", "poligono", "pol", "barrio", "via",
    "rua", "street", "st", "road", "rd", "avenue", "ave",
}

# --------------------------------------------------------------------------
# Regex de PII (red de seguridad, independiente de los términos de n8n)
# --------------------------------------------------------------------------
_MAIL_DOMAINS = (
    "hotmail|gmail|yahoo|outlook|icloud|live|msn|protonmail|proton|aol|"
    "terra|telefonica|movistar|orange|vodafone|ya|wanadoo|mixmail|correo"
)
PII_PATTERNS = [
    # email bien leído
    ("email", re.compile(r"[A-Za-z0-9._%+\-]{2,}@[A-Za-z0-9.\-]{2,}\.[A-Za-z]{2,}")),
    # email con la "@" mal leída por el OCR (@ -> O/G/©/0) contra un dominio conocido
    ("email", re.compile(
        r"[A-Za-z0-9._%+\-]{2,}[@G0Oo©ª]?(?:" + _MAIL_DOMAINS + r")[A-Za-z0-9.\-]*\.[A-Za-z]{2,}",
        re.I)),
    # cualquier token con arroba
    ("email", re.compile(r"\S*@\S*")),
    # teléfono ES (fijo/móvil), con o sin prefijo, separado por espacios/./-
    ("phone", re.compile(
        r"(?<!\d)(?:\+\s?\d{1,3}[\s.\-]?)?[6789]\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}(?!\d)")),
    ("phone", re.compile(
        r"(?<!\d)(?:\+\s?\d{1,3}[\s.\-]?)?[6789]\d{2}[\s.\-]?\d{3}[\s.\-]?\d{3}(?!\d)")),
    # internacional genérico: +NN seguido de 7-14 dígitos
    ("phone", re.compile(r"\+\s?\d{1,3}(?:[\s.\-]?\d){7,14}(?!\d)")),
    # DNI / NIE
    ("dni", re.compile(r"(?<![A-Za-z0-9])\d{8}\s?[\-–]?\s?[A-HJ-NP-TV-Z](?![A-Za-z0-9])", re.I)),
    ("dni", re.compile(r"(?<![A-Za-z0-9])[XYZ]\s?[\-–]?\s?\d{7}\s?[\-–]?\s?[A-HJ-NP-TV-Z](?![A-Za-z0-9])", re.I)),
    # IBAN español
    ("iban", re.compile(r"(?<![A-Za-z0-9])ES\s?\d{2}(?:[\s\-]?\d{4}){5}(?![A-Za-z0-9])", re.I)),
    # perfiles personales
    ("url", re.compile(
        r"(?:https?://)?(?:www\.)?(?:linkedin\.com|instagram\.com|facebook\.com|"
        r"twitter\.com|x\.com|github\.com|t\.me|tiktok\.com)/[\w\-./%?=&+]+", re.I)),
    # código postal español suelto
    ("cp", re.compile(r"(?<!\d)(?:0[1-9]|[1-4]\d|5[0-2])\d{3}(?!\d)")),
]

# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


def _otsu(gray):
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _flatten_background(gray, ksize=31):
    """Divide la imagen por su propio fondo (cierre morfológico) -> cualquier
    fondo uniforme (banda de color, panel, sombra de escáner) se vuelve blanco
    y el texto oscuro se conserva. Es lo que rescata el texto sobre bandas de
    color, que la binarización global de Tesseract se come."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    return cv2.divide(gray, bg, scale=255)


def _ocr_variants(rgb, passes=None):
    """Devuelve [(nombre, imagen_uint8)] con las pasadas de OCR configuradas."""
    if cv2 is None:
        return [("raw", rgb)]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = []
    for name in (passes or OCR_PASSES):
        if name == "raw":
            out.append((name, rgb))
        elif name == "gray_otsu":
            out.append((name, _otsu(gray)))
        elif name == "bgnorm":
            out.append((name, _flatten_background(gray)))
        elif name == "darkinv":
            # texto claro sobre fondo oscuro (barra lateral de plantillas):
            # al invertir, el texto pasa a oscuro y el fondo a claro.
            out.append((name, _flatten_background(255 - gray)))
    return out or [("bgnorm", _flatten_background(gray))]


def _ocr_lines(image_u8, lang=None, psm=3):
    """OCR de una imagen ya preprocesada. Devuelve líneas:
    [{"words": [{"text","x0","y0","x1","y1","conf"}], "x0","y0","x1","y1","text"}]
    en coordenadas de píxel de la imagen recibida."""
    if pytesseract is None:
        return []
    data = pytesseract.image_to_data(
        Image.fromarray(image_u8), lang=lang or OCR_LANG,
        config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT,
    )
    groups = {}
    n = len(data["text"])
    for i in range(n):
        t = data["text"][i]
        if not t or not t.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append({
            "text": t.strip(),
            "x0": float(data["left"][i]),
            "y0": float(data["top"][i]),
            "x1": float(data["left"][i] + data["width"][i]),
            "y1": float(data["top"][i] + data["height"][i]),
            "conf": float(data["conf"][i]),
        })
    lines = []
    for words in groups.values():
        words.sort(key=lambda w: w["x0"])
        lines.append({
            "words": words,
            "x0": min(w["x0"] for w in words),
            "y0": min(w["y0"] for w in words),
            "x1": max(w["x1"] for w in words),
            "y1": max(w["y1"] for w in words),
            "text": " ".join(w["text"] for w in words),
        })
    lines.sort(key=lambda l: (round(l["y0"] / 6), l["x0"]))
    _annotate_vertical_pads(lines)
    return lines


def _annotate_vertical_pads(lines, max_pad=6.0, min_pad=1.0, ratio=0.12):
    """Calcula, por línea, cuánto se puede engordar la caja hacia arriba y
    hacia abajo. Hace falta porque las dos situaciones opuestas conviven en el
    mismo CV: en un titular de 56 px hay que sobrepasar la caja del OCR varios
    píxeles para no dejar los remates de las letras a la vista, mientras que en
    un bloque de datos con líneas a 1 px de distancia cualquier margen se come
    los descendentes de la línea de arriba ('Española' justo encima de la
    línea de dirección). El límite es la mitad del hueco real con la línea
    vecina que se solape horizontalmente."""
    for idx, line in enumerate(lines):
        h = max(1.0, line["y1"] - line["y0"])
        want = min(max(ratio * h, min_pad), max_pad)
        gap_up, gap_down = 1e9, 1e9
        for other in lines:
            if other is line:
                continue
            if other["x1"] <= line["x0"] or other["x0"] >= line["x1"]:
                continue  # no se solapan en horizontal: no molesta
            if other["y1"] <= line["y0"]:
                gap_up = min(gap_up, line["y0"] - other["y1"])
            elif other["y0"] >= line["y1"]:
                gap_down = min(gap_down, other["y0"] - line["y1"])
        pu = max(0.0, min(want, gap_up / 2.0))
        pd = max(0.0, min(want, gap_down / 2.0))
        for w in line["words"]:
            w["pad_up"] = pu
            w["pad_down"] = pd


# --------------------------------------------------------------------------
# Detección de PII sobre las líneas del OCR
# --------------------------------------------------------------------------
def _line_offsets(line):
    """Texto de la línea + mapa posición_de_caracter -> índice de palabra, para
    poder traducir un match de regex a las cajas de las palabras afectadas."""
    parts = []
    owner = []
    for idx, w in enumerate(line["words"]):
        if parts:
            parts.append(" ")
            owner.append(-1)
        for _ in w["text"]:
            owner.append(idx)
        parts.append(w["text"])
    return "".join(parts), owner


def _words_in_span(line, owner, start, end):
    idxs = {owner[i] for i in range(start, min(end, len(owner))) if owner[i] >= 0}
    return [line["words"][i] for i in sorted(idxs)]


def _flat_words(lines):
    """Todas las palabras en orden de lectura, con su índice de línea. Permite
    buscar términos que cruzan de una línea a la siguiente (un email partido
    en 'joselopez_bdn@hotmail.c' + 'om', o un nombre en dos líneas)."""
    flat = []
    for li, line in enumerate(lines):
        for w in line["words"]:
            flat.append((li, w))
    return flat


def _window_match(norms, i, tn, fuzzy_ok):
    """Busca la ventana MÍNIMA de palabras consecutivas que empieza en o
    después de `i` y equivale a `tn`. Devuelve (i_ajustado, j) o None.

    Dos detalles que importan: (1) se corta la ventana en cuanto pasa de
    ~1.4x la longitud del término, si no un simple 'el término está contenido
    en la ventana' se tragaba la etiqueta y la línea anterior completa
    ('DATOS DE PERSONALES: Nombre y apellidos: José Luis González Piñero');
    (2) una vez encontrada, se recorta por la izquierda mientras siga
    cumpliendo, para quedarse con las palabras justas."""
    limit = int(1.4 * len(tn)) + 6
    acc = ""
    end = None
    for j in range(i, min(i + OCR_MAX_WINDOW, len(norms))):
        acc += norms[j]
        if len(acc) > limit:
            break
        if fuzzy_ok:
            ok = (tn in acc) or _ratio(acc, tn) >= OCR_FUZZY_THRESHOLD
        else:
            ok = acc == tn
        if ok:
            end = j
            break
    if end is None:
        return None
    start = i
    while start < end:
        acc2 = "".join(norms[start + 1:end + 1])
        if fuzzy_ok:
            ok = (tn in acc2) or _ratio(acc2, tn) >= OCR_FUZZY_THRESHOLD
        else:
            ok = acc2 == tn
        if not ok:
            break
        start += 1
    return start, end


def _match_terms(lines, term_list):
    """Localiza cada término de n8n en las palabras del OCR.

    Dos estrategias que se complementan:
      1. Ventana de palabras consecutivas para el término completo (máxima
         precisión).
      2. Token a token (nombre, apellido1, apellido2...). Imprescindible
         porque el OCR NO garantiza que las palabras de un nombre queden
         contiguas en el orden de lectura: en un CV real el nombre iba en un
         bloque y el apellido en otro, con la columna de contacto en medio,
         así que la ventana del término completo nunca llegaba a formarse.
         Tokens de 4-5 caracteres solo por igualdad exacta (un 'Luis' difuso
         taparía medio CV); de 6 en adelante, difuso o contenido.

    Devuelve (cajas, no_encontrados, parciales).
    """
    flat = _flat_words(lines)
    norms = [_norm(w["text"]) for _, w in flat]
    boxes = []
    unmatched = []
    partial = []
    for term in term_list:
        tn = _norm(term)
        if len(tn) < OCR_TERM_MIN_EXACT:
            continue
        fuzzy_ok = len(tn) >= OCR_TERM_MIN_FUZZY
        full_hit = False
        i = 0
        while i < len(flat):
            if not norms[i]:
                i += 1
                continue
            m = _window_match(norms, i, tn, fuzzy_ok)
            if m:
                s, e = m
                for k in range(s, e + 1):
                    if norms[k]:
                        boxes.append(flat[k][1])
                full_hit = True
                i = e + 1
            else:
                i += 1

        # --- estrategia 2: token a token ---
        tokens = [_norm(t) for t in re.split(r"[\s,;/|]+", str(term)) if _norm(t)]
        tokens = [t for t in tokens if len(t) >= 4]
        missing = []
        for tok in tokens:
            hit = False
            for idx, wn in enumerate(norms):
                if not wn:
                    continue
                if len(tok) >= 6:
                    ok = (tok in wn) or _ratio(wn, tok) >= 0.88
                else:
                    ok = wn == tok
                if ok:
                    boxes.append(flat[idx][1])
                    hit = True
            if not hit:
                missing.append(tok)
        if not full_hit and missing and len(missing) == len(tokens):
            unmatched.append(term)
        elif missing:
            partial.append(f"{term} (sin localizar: {', '.join(missing)})")
    return boxes, unmatched, partial


def _wraps_into_next(line, nxt):
    """¿La línea acaba cortada a media palabra y sigue en la de abajo?

    Solo entonces tiene sentido concatenar las dos líneas sin espacio para
    buscar PII partida ('joselopez_bdn@hotmail.c' + 'om' en una columna
    estrecha). Concatenar siempre es peligroso: en una plantilla a dos
    columnas, unir 'PERFIL PROFESIONAL' con la línea de contacto de al lado
    producía 'PROFESIONALlucia@correo.es', que el regex de email daba por
    válido y tapaba el titular del CV.
    """
    if not line["words"] or not nxt["words"]:
        return False
    last = line["words"][-1]["text"]
    first = nxt["words"][0]["text"]
    if "@" in last:                       # email cortado después de la arroba
        return True
    if last[-1:].isdigit() and first.replace(" ", "").isdigit():
        return True                       # teléfono cortado entre dos grupos
    return False


def _match_patterns(lines):
    """Barrido regex sobre cada línea y sobre cada par de líneas consecutivas
    unidas sin espacio (así se capturan emails/teléfonos partidos por el salto
    de línea, que es habitual en las columnas estrechas de las plantillas)."""
    boxes = []
    kinds = set()

    def scan(text, resolve):
        for kind, rx in PII_PATTERNS:
            for m in rx.finditer(text):
                if not m.group(0).strip():
                    continue
                ws = resolve(m.start(), m.end())
                if ws:
                    boxes.extend(ws)
                    kinds.add(kind)

    for li, line in enumerate(lines):
        text, owner = _line_offsets(line)
        scan(text, lambda s, e, l=line, o=owner: _words_in_span(l, o, s, e))

        # unión con la línea siguiente si está justo debajo, alineada y la
        # línea de arriba parece cortada a media palabra
        if li + 1 < len(lines):
            nxt = lines[li + 1]
            h = max(1.0, line["y1"] - line["y0"])
            aligned = abs(nxt["x0"] - line["x0"]) < 4 * h
            close = 0 <= (nxt["y0"] - line["y1"]) < 1.2 * h
            if aligned and close and _wraps_into_next(line, nxt):
                t2, o2 = _line_offsets(nxt)
                joined = text + t2
                cut = len(text)

                def resolve(s, e, l=line, o=owner, n=nxt, o2=o2, cut=cut):
                    ws = []
                    if s < cut:
                        ws += _words_in_span(l, o, s, min(e, cut))
                    if e > cut:
                        ws += _words_in_span(n, o2, max(0, s - cut), e - cut)
                    return ws
                scan(joined, resolve)
    return boxes, kinds


def _label_at_line_start(line):
    """Si la línea empieza por una etiqueta de PII, devuelve
    (etiqueta_normalizada, índice de la primera palabra del valor). El ':'
    puede venir pegado o como palabra suelta, y la etiqueta puede llevar
    faltas del OCR o del propio candidato ('Fecha naciemiento'), así que se
    compara de forma difusa."""
    words = line["words"]
    if len(words) < 2:
        return None, None
    # 1) hay ':' en las primeras palabras -> la etiqueta es lo de antes
    for k in range(min(5, len(words))):
        t = words[k]["text"]
        if ":" in t:
            label = _norm(" ".join(w["text"] for w in words[:k + 1]))
            if not label:
                return None, None
            start = k + 1 if t.rstrip().endswith(":") else k
            if t.rstrip().endswith(":") and start >= len(words):
                return None, None
            best = None
            for ln in _ALL_LABELS_NORM:
                r = _ratio(label, ln)
                if r >= 0.82 and (best is None or r > best[1]):
                    best = (ln, r)
            if best:
                return best[0], start
            return None, None
    # 2) sin ':' -> ¿coinciden las 1-3 primeras palabras con una etiqueta?
    for k in range(min(3, len(words) - 1), 0, -1):
        label = _norm(" ".join(w["text"] for w in words[:k]))
        if len(label) < 4:
            continue
        for ln in _ALL_LABELS_NORM:
            if _ratio(label, ln) >= 0.9:
                return ln, k
    return None, None


def _is_street_line(line):
    """¿La línea es una dirección postal sin etiqueta delante? (típico en
    plantillas modernas: 'Calle menendez pidal ,' / '08490, Tordera').

    Las abreviaturas cortas piden más pruebas que el tipo de vía completo: en
    un CV real la línea de una lista, 'c)  En el mantenimiento y reparación
    de llenadoras...', se tapaba entera porque 'c' está en STREET_WORDS. Para
    una abreviatura se exige la barra o el punto de 'C/' o 'Av.' y además un
    número en la línea; para 'calle', 'avenida'... basta la palabra (una
    dirección puede no llevar número: 'Calle menendez pidal ,').
    """
    if not line["words"]:
        return False
    raw = line["words"][0]["text"]
    w0 = _norm(raw)
    if w0 not in STREET_WORDS:
        return False
    if "(" in raw or ")" in raw:
        return False
    if len(w0) <= 2:
        if not ("/" in raw or "." in raw):
            return False
        if not re.search(r"\d", line["text"]):
            return False
    return True


def _match_labels(lines):
    """Reglas por etiqueta: tapa el VALOR que sigue a 'Nombre:', 'Telf:',
    'Email:', 'Direccion:'... y las líneas de continuación de una dirección
    (alineadas en la misma columna y pegadas verticalmente).
    Devuelve (cajas, etiquetas_de_sesgo_detectadas)."""
    boxes = []
    bias_seen = []
    i = 0
    while i < len(lines):
        line = lines[i]
        label, start = _label_at_line_start(line)
        is_street = label is None and _is_street_line(line)

        if label in _BIAS_NORM:
            bias_seen.append(label)
            if not REDACT_BIAS_FIELDS:
                i += 1
                continue

        if label is None and not is_street:
            i += 1
            continue

        value_words = line["words"][start:] if label is not None else line["words"]
        if not value_words:
            i += 1
            continue
        boxes.extend(value_words)

        multiline = is_street or (label in _MULTILINE_NORM)
        if multiline:
            col_x0 = value_words[0]["x0"]
            h = max(1.0, line["y1"] - line["y0"])
            prev = line
            taken = 0
            j = i + 1
            while j < len(lines) and taken < 3:
                nxt = lines[j]
                nlabel, _ = _label_at_line_start(nxt)
                gap = nxt["y0"] - prev["y1"]
                # La continuación puede alinearse con la columna del valor
                # ("Direccion :   Santiago..." / "   Hospitalet...") o con el
                # inicio de la propia línea ("Calle menendez pidal," /
                # "08490, Tordera"). Vale cualquiera de las dos.
                tol = max(12.0, 0.9 * h)
                same_col = (abs(nxt["x0"] - col_x0) <= tol
                            or abs(nxt["x0"] - prev["x0"]) <= tol)
                if (nlabel is not None or not same_col or not (-0.3 * h <= gap <= 1.6 * h)
                        or len(nxt["words"]) > 8):
                    break
                boxes.extend(nxt["words"])
                prev = nxt
                taken += 1
                j += 1
            i = j
        else:
            i += 1
    return boxes, bias_seen


def _pii_boxes(lines, term_list):
    """Todas las cajas de PII de una pasada de OCR, ya fusionadas por línea.
    Devuelve (rects, info) con rects = [(x0,y0,x1,y1)] en píxeles."""
    b1, unmatched, partial = _match_terms(lines, term_list)
    b2, kinds = _match_patterns(lines)
    b3, bias_seen = _match_labels(lines)
    words = b1 + b2 + b3
    rects = _merge_word_boxes(words)
    return rects, {
        "unmatched_terms": unmatched,
        "partial_terms": partial,
        "pattern_kinds": sorted(kinds),
        "bias_labels": sorted(set(bias_seen)),
        "n_term_words": len(b1),
        "n_pattern_words": len(b2),
        "n_label_words": len(b3),
    }


def _merge_word_boxes(words, pad_ratio_x=0.28):
    """Convierte cajas de palabra en rectángulos con un pequeño margen y funde
    las que están en la misma línea y a menos de un espacio de distancia, para
    que el resultado sea un bloque limpio y no una fila de parches."""
    if not words:
        return []
    boxes = []
    for w in words:
        h = max(1.0, w["y1"] - w["y0"])
        # El margen horizontal es generoso (la caja del OCR queda justa y deja
        # ver medio carácter al final); el vertical lo decide _annotate_
        # vertical_pads según el hueco real con las líneas vecinas.
        pu = w.get("pad_up", 1.0)
        pd = w.get("pad_down", 1.0)
        boxes.append([
            w["x0"] - pad_ratio_x * h, w["y0"] - pu,
            w["x1"] + pad_ratio_x * h, w["y1"] + pd,
        ])
    boxes.sort(key=lambda b: (b[1], b[0]))
    merged = []
    for b in boxes:
        placed = False
        for m in merged:
            h = min(m[3] - m[1], b[3] - b[1])
            v_overlap = min(m[3], b[3]) - max(m[1], b[1])
            h_gap = max(m[0], b[0]) - min(m[2], b[2])
            if v_overlap > 0.45 * h and h_gap < 1.2 * h:
                m[0] = min(m[0], b[0]); m[1] = min(m[1], b[1])
                m[2] = max(m[2], b[2]); m[3] = max(m[3], b[3])
                placed = True
                break
        if not placed:
            merged.append(list(b))
    return [tuple(m) for m in merged]


def _merge_rects(rects):
    """Funde rectángulos solapados venidos de pasadas de OCR distintas."""
    out = []
    for r in sorted(rects, key=lambda r: (r[1], r[0])):
        merged = False
        for m in out:
            if not (r[2] <= m[0] or r[0] >= m[2] or r[3] <= m[1] or r[1] >= m[3]):
                m[0] = min(m[0], r[0]); m[1] = min(m[1], r[1])
                m[2] = max(m[2], r[2]); m[3] = max(m[3], r[3])
                merged = True
                break
        if not merged:
            out.append(list(r))
    return [tuple(m) for m in out]


# --------------------------------------------------------------------------
# Pintado destructivo en píxeles
# --------------------------------------------------------------------------
def _paint_boxes(rgb, rects, probe=6):
    """Tapa cada rectángulo pintando, FILA A FILA, el color mediano de los
    píxeles inmediatamente a izquierda y derecha de esa misma fila. Así el
    parche se funde con el fondo real (papel blanco, banda de color, panel
    oscuro) en vez de dejar un rectángulo de un color plano.

    Modifica `rgb` in place. Devuelve el nº de rectángulos pintados.
    """
    H, W = rgb.shape[:2]
    painted = 0
    for (x0, y0, x1, y1) in rects:
        ix0 = max(0, int(np.floor(x0))); iy0 = max(0, int(np.floor(y0)))
        ix1 = min(W, int(np.ceil(x1))); iy1 = min(H, int(np.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        left = rgb[iy0:iy1, max(0, ix0 - probe):ix0, :]
        right = rgb[iy0:iy1, ix1:min(W, ix1 + probe), :]
        samples = [s for s in (left, right) if s.size]
        if samples:
            ring = np.concatenate(samples, axis=1)          # (h, n, 3)
            fill = np.median(ring, axis=1).astype(np.uint8)  # color por fila
            rgb[iy0:iy1, ix0:ix1, :] = fill[:, None, :]
        else:
            rgb[iy0:iy1, ix0:ix1, :] = 255
        painted += 1
    return painted


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
    effectively a full-page photo/scan, with or without an OCR text layer on
    top (see /redact for how that case is handled). Truthiness of bg_images
    works as the old has_bg_image boolean did.

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

    OJO: esto pinta un rectángulo vectorial ENCIMA de la imagen; los píxeles
    originales siguen dentro del PDF. La ruta de Nivel 3 (_redact_raster_page)
    sustituye directamente el stream de la imagen, que sí es irreversible.

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


def _page_text_boxes(page):
    """Cajas de los bloques de texto reales de la página, en puntos. Es contra
    esto contra lo que se comprueba el solape del logo."""
    return [fitz.Rect(b[:4]) for b in page.get_text("blocks")
            if b[6] == 0 and b[4].strip()]


def _logo_rect(W, H, corner, w, h):
    m = LOGO_MARGIN
    if corner == "top-right":
        return fitz.Rect(W - m - w, m, W - m, m + h)
    if corner == "top-left":
        return fitz.Rect(m, m, m + w, m + h)
    if corner == "bottom-right":
        return fitz.Rect(W - m - w, H - m - h, W - m, H - m)
    if corner == "bottom-left":
        return fitz.Rect(m, H - m - h, m + w, H - m)
    return None


def _logo_fits(rect, boxes):
    probe = fitz.Rect(rect.x0 - LOGO_CLEARANCE, rect.y0 - LOGO_CLEARANCE,
                      rect.x1 + LOGO_CLEARANCE, rect.y1 + LOGO_CLEARANCE)
    return not any(probe.intersects(b) for b in boxes)


def _add_watermark(page):
    if not _WM_PNG:
        return
    W, H = page.rect.width, page.rect.height
    iw, ih = _WM_SIZE
    ww = WATERMARK_WIDTH * W
    wh = ww * ih / iw
    page.insert_image(
        fitz.Rect((W - ww) / 2, (H - wh) / 2, (W + ww) / 2, (H + wh) / 2),
        stream=_WM_PNG, overlay=True, keep_proportion=True,
    )


def _apply_branding(doc, boxes_by_page):
    """Marca de agua en todas las páginas + logo del mismo tamaño y en la misma
    esquina en todas ellas, garantizando que no pisa texto.

    Se decide en dos fases a propósito. Primero se busca la combinación
    (escala, esquina) que cabe en el MAYOR número de páginas del documento, y
    solo después se dibuja. Así el logo sale igual en todo el CV: uno que
    cambia de tamaño o de esquina de página en página se ve peor que uno
    constante, y además hace el resultado imposible de predecir.

    boxes_by_page: {índice de página: [Rect]} con las cajas de texto. En una
    página raster no hay bloques de texto que consultar (es una imagen), así
    que la ruta de Nivel 3 las saca de las líneas del OCR y las pasa aquí.
    Sin eso el logo se dibujaba a ciegas encima del contenido.
    """
    for page in doc:
        _add_watermark(page)
    if not _LOGO_PNG:
        return None

    lw, lh = _LOGO_SIZE
    n_pages = doc.page_count
    boxes = {i: (boxes_by_page.get(i) if boxes_by_page else None)
             for i in range(n_pages)}
    for i in range(n_pages):
        if boxes[i] is None:
            boxes[i] = _page_text_boxes(doc[i])

    best = None  # (nº de páginas donde cabe, escala, esquina)
    for scale in LOGO_SCALE_LADDER:
        for corner in LOGO_CORNERS:
            fits = 0
            for i in range(n_pages):
                page = doc[i]
                w = LOGO_WIDTH * page.rect.width * scale
                r = _logo_rect(page.rect.width, page.rect.height, corner, w,
                               w * lh / lw)
                if r is not None and _logo_fits(r, boxes[i]):
                    fits += 1
            if best is None or fits > best[0]:
                best = (fits, scale, corner)
            if fits == n_pages:
                break
        if best and best[0] == n_pages:
            break

    fits, scale, corner = best
    placed = 0
    for i in range(n_pages):
        page = doc[i]
        W, H = page.rect.width, page.rect.height
        w = LOGO_WIDTH * W * scale
        r = _logo_rect(W, H, corner, w, w * lh / lw)
        if r is None or not _logo_fits(r, boxes[i]):
            # Esta página concreta no admite la colocación elegida: se prueban
            # las demás esquinas al mismo tamaño antes de renunciar.
            r = None
            for alt in LOGO_CORNERS:
                cand = _logo_rect(W, H, alt, w, w * lh / lw)
                if cand is not None and _logo_fits(cand, boxes[i]):
                    r = cand
                    break
        if r is None:
            log.info("logo omitido en la página %d: no hay hueco libre", i + 1)
            continue
        page.insert_image(r, stream=_LOGO_PNG, overlay=True, keep_proportion=True)
        placed += 1
    log.info("branding: logo al %.0f%% en '%s', colocado en %d/%d página(s)",
             scale * 100, corner, placed, n_pages)
    return {"scale": scale, "corner": corner, "placed": placed}


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
# --- Caché de OCR ---------------------------------------------------------
# Clave: (sha256 del PDF, índice de página, nombre de pasada) -> líneas.
# Con esto /redact reutiliza lo que ya leyó /extract-text sobre el mismo
# fichero. Además de ahorrar la mitad del OCR, garantiza que los términos que
# detecta la IA y las cajas que se tapan salen del MISMO texto.
_ocr_cache = OrderedDict()
_ocr_cache_lock = threading.Lock()


def _cache_get(doc_key, page_idx, pass_name):
    if not doc_key:
        return None
    with _ocr_cache_lock:
        entry = _ocr_cache.get(doc_key)
        if entry is None:
            return None
        if time.time() - entry["t"] > OCR_CACHE_TTL:
            _ocr_cache.pop(doc_key, None)
            return None
        _ocr_cache.move_to_end(doc_key)
        return entry["pages"].get((page_idx, pass_name))


def _cache_put(doc_key, page_idx, pass_name, lines):
    if not doc_key:
        return
    with _ocr_cache_lock:
        entry = _ocr_cache.get(doc_key)
        if entry is None:
            entry = {"t": time.time(), "pages": {}}
            _ocr_cache[doc_key] = entry
        entry["pages"][(page_idx, pass_name)] = lines
        entry["t"] = time.time()
        _ocr_cache.move_to_end(doc_key)
        while len(_ocr_cache) > OCR_CACHE_DOCS:
            _ocr_cache.popitem(last=False)


def _ocr_lines_cached(image_u8, doc_key, page_idx, pass_name):
    hit = _cache_get(doc_key, page_idx, pass_name)
    if hit is not None:
        log.info("OCR cache HIT (página %d, pasada %s)", page_idx + 1, pass_name)
        return hit
    lines = _ocr_lines(image_u8)
    _cache_put(doc_key, page_idx, pass_name, lines)
    return lines


def _doc_key(pdf_bytes):
    return hashlib.sha256(pdf_bytes).hexdigest() if pdf_bytes else None


def _native_dpi(page, bg_images):
    """dpi al que conviene renderizar: el nativo de la imagen de fondo,
    acotado a [OCR_DPI_MIN, OCR_DPI_MAX]."""
    doc = page.parent
    best = OCR_DPI_MIN
    for xref, rect in bg_images:
        try:
            info = doc.extract_image(xref)
            w = info.get("width") or 0
        except Exception:
            continue
        if w and page.rect.width > 0:
            best = max(best, int(round(w / page.rect.width * 72)))
    return max(OCR_DPI_MIN, min(OCR_DPI_MAX, best))


def _render_rgb(page, dpi):
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width, 3).copy()


def _photo_rects_on_array(rgb):
    """Nivel 2 sobre la página ya renderizada. Antes se trabajaba sobre la
    imagen extraída del PDF, que obliga a decodificarla aparte y falla si el
    códec no lo soporta cv2; sobre el render funciona siempre y además pilla
    fotos compuestas por varios objetos. Devuelve rects en píxeles."""
    if cv2 is None or _get_face_detector() is None:
        return []
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr.shape[:2]
    crop_h = max(1, int(img_h * FACE_TOP_CROP_FRACTION))
    boxes = _detect_faces_px(bgr[:crop_h, :, :]) or _detect_faces_px(bgr)
    if not boxes:
        return []
    search_h = max(crop_h, min(img_h, int(img_h * FACE_PHOTO_SEARCH_FRACTION)))
    region = bgr[:search_h, :, :]
    out = []
    for (x, y, w, h) in boxes:
        photo_box, _bg = _find_photo_bbox(region, x + w / 2, y + h / 2)
        if photo_box is not None:
            px, py, pw, ph = photo_box
            pad = FACE_PHOTO_PAD_PX
            out.append((px - pad, py - pad, px + pw + pad, py + ph + pad))
        else:
            # No se pudo aislar el bloque de la foto: se tapa un margen amplio
            # alrededor de la cara para no dejar piel a la vista.
            cx, cy = x + w / 2, y + h / 2
            out.append((cx - w * FACE_MARGIN_X, cy - h * FACE_MARGIN_TOP,
                        cx + w * FACE_MARGIN_X, cy + h * FACE_MARGIN_BOTTOM))
    return out


def _replace_page_images(page, bg_xrefs, rgb, prefer_jpeg=True):
    """Sustituye el stream de las imágenes de fondo por el render ya pintado.
    Esto es lo que hace la anonimización IRREVERSIBLE: draw_rect solo dibuja
    encima y los píxeles originales se quedan dentro del PDF.

    Si hay varias imágenes a página completa apiladas se sustituyen TODAS por
    el mismo render (ocupan el mismo rect, así que el resultado visual es el
    mismo y no queda ninguna con los píxeles originales).
    """
    buf = io.BytesIO()
    img = Image.fromarray(rgb)
    if prefer_jpeg:
        img.save(buf, format="JPEG", quality=OCR_JPEG_QUALITY, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    ok = 0
    for xref in bg_xrefs:
        try:
            page.replace_image(xref, stream=data)
            ok += 1
        except Exception as e:
            log.error("replace_image falló en xref %s: %s", xref, e)
    return ok == len(bg_xrefs)


def _strip_identifying_links(page):
    """Quita los enlaces de la página. Un mailto:/tel:/linkedin sigue siendo
    dato personal aunque el texto visible esté tapado, y apply_redactions no
    los toca. Devuelve cuántos se han quitado."""
    n = 0
    try:
        links = page.get_links()
    except Exception:
        return 0
    for _ in links:
        try:
            page.delete_link(page.get_links()[0])
            n += 1
        except Exception:
            break
    return n


def _redact_raster_page(page, bg_images, term_list, do_faces, page_idx=0,
                        doc_key=None):
    """Anonimiza una página que es una imagen (sin capa de texto usable).

    Devuelve un dict con el resultado para poder construir cabeceras y
    decidir si hace falta revisión manual.
    """
    res = {
        "ocr": False, "rects": 0, "faces": 0, "destructive": False,
        "unmatched": [], "partial": [], "bias": [], "kinds": [],
        "text": "", "error": None, "text_boxes": [],
    }
    if pytesseract is None:
        res["error"] = "pytesseract/tesseract no disponible"
        return res

    rot = page.rotation
    if rot:
        # El render y la imagen almacenada tienen que coincidir en orientación
        # para poder sustituir el stream; se trabaja sin rotación y se
        # restaura al final.
        page.set_rotation(0)
    try:
        dpi = _native_dpi(page, bg_images)
        rgb = _render_rgb(page, dpi)
        H, W = rgb.shape[:2]

        all_rects = []
        unmatched_sets = []
        partial_sets = []
        bias = set()
        kinds = set()
        best_text = (-1, "")
        for name, variant in _ocr_variants(rgb):
            try:
                lines = _ocr_lines_cached(variant, doc_key, page_idx, name)
            except Exception as e:
                log.warning("pasada OCR '%s' falló: %s", name, e)
                continue
            rects, info = _pii_boxes(lines, term_list)
            all_rects.extend(rects)
            unmatched_sets.append(set(info["unmatched_terms"]))
            partial_sets.append(set(info["partial_terms"]))
            bias.update(info["bias_labels"])
            kinds.update(info["pattern_kinds"])
            score = sum(1 for l in lines for w in l["words"] if w["conf"] >= 60)
            if score > best_text[0]:
                best_text = (score, "\n".join(l["text"] for l in lines))
                # Cajas de las líneas en PUNTOS: es lo único que sabe dónde hay
                # contenido en una página que es solo píxeles, y es contra esto
                # contra lo que _apply_branding comprueba el solape del logo.
                sx_pt = page.rect.width / W
                sy_pt = page.rect.height / H
                res["text_boxes"] = [
                    fitz.Rect(l["x0"] * sx_pt, l["y0"] * sy_pt,
                              l["x1"] * sx_pt, l["y1"] * sy_pt)
                    for l in lines
                ]
            log.info("OCR pasada %s: %d lineas, %d rects (%s)",
                     name, len(lines), len(rects), info)
        if not unmatched_sets:
            res["error"] = "ninguna pasada de OCR produjo resultado"
            return res

        res["ocr"] = True
        res["text"] = best_text[1]
        # Un término solo cuenta como no localizado si NINGUNA pasada lo vio.
        res["unmatched"] = sorted(set.intersection(*unmatched_sets))
        res["partial"] = sorted(set.intersection(*partial_sets)) if partial_sets else []
        res["bias"] = sorted(bias)
        res["kinds"] = sorted(kinds)

        rects = _merge_rects(all_rects)
        face_rects = _photo_rects_on_array(rgb) if do_faces else []
        res["faces"] = len(face_rects)

        painted = _paint_boxes(rgb, rects + face_rects)
        res["rects"] = painted

        bg_xrefs = [x for x, _r in bg_images]
        if OCR_DESTRUCTIVE:
            res["destructive"] = _replace_page_images(page, bg_xrefs, rgb)
        if not res["destructive"]:
            # Plan B: pintar rectángulos vectoriales encima (recuperable, así
            # que la página se marca para revisión manual).
            sx = page.rect.width / W
            sy = page.rect.height / H
            for (x0, y0, x1, y1) in rects + face_rects:
                r = fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                page.draw_rect(r, color=(0.15, 0.15, 0.15),
                               fill=(0.15, 0.15, 0.15), width=0)

        # Si además hubiera capa de texto debajo, se borra en las mismas zonas.
        if len(page.get_text().strip()) > 0 and rects:
            sx = page.rect.width / W
            sy = page.rect.height / H
            for (x0, y0, x1, y1) in rects:
                page.add_redact_annot(
                    fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy), fill=False)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                  graphics=fitz.PDF_REDACT_LINE_ART_NONE)
        _strip_identifying_links(page)
        return res
    except Exception as e:
        log.exception("fallo anonimizando página raster: %s", e)
        res["error"] = str(e)
        return res
    finally:
        if rot:
            page.set_rotation(rot)


def _ocr_page_text(page, bg_images, page_idx=0, doc_key=None):
    """Texto OCR de una página raster, para /extract-text. Usa las pasadas de
    OCR_TEXT_PASSES y se queda con la de más palabras fiables."""
    if pytesseract is None:
        return ""
    dpi = _native_dpi(page, bg_images) if bg_images else OCR_DPI_MIN
    rgb = _render_rgb(page, dpi)
    best = (-1, "")
    for name, variant in _ocr_variants(rgb, passes=OCR_TEXT_PASSES):
        try:
            lines = _ocr_lines_cached(variant, doc_key, page_idx, name)
        except Exception as e:
            log.warning("pasada OCR '%s' falló: %s", name, e)
            continue
        score = sum(1 for l in lines for w in l["words"] if w["conf"] >= 60)
        if score > best[0]:
            best = (score, "\n".join(l["text"] for l in lines))
    return best[1]


def _flag(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def _parse_terms(terms):
    try:
        term_list = json.loads(terms)
        if not isinstance(term_list, list):
            raise ValueError
        return [str(t) for t in term_list if str(t).strip()]
    except Exception:
        raise HTTPException(
            status_code=400, detail="`terms` must be a JSON array of strings"
        )


def _open_pdf(pdf_bytes):
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        return fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="not a valid PDF")


# ---------------------------------------------------------------------------
# El trabajo pesado va en funciones SÍNCRONAS y los endpoints las lanzan con
# run_in_threadpool. Antes estaba todo dentro del `async def`, y como el OCR y
# PyMuPDF son CPU y bloquean, un CV de 7 páginas tenía el bucle de eventos
# ocupado ~30 s: el segundo candidato que llegase esperaba el turno completo y
# durante ese rato ni /health respondía. Con esto uvicorn atiende en paralelo
# aunque siga con --workers 1.
# ---------------------------------------------------------------------------
def _do_redact(pdf_bytes, term_list, do_images, do_brand, do_faces, do_ocr):
    doc = _open_pdf(pdf_bytes)
    doc_key = _doc_key(pdf_bytes)

    total_text = 0
    total_imgs = 0
    total_lines = 0
    total_covered = 0
    total_faces = 0
    ocr_redactions = 0
    image_based_pages = []
    pages_without_face = []   # image-based pages where 0 faces were found/covered
    ocr_pages = []            # páginas resueltas por la ruta de Nivel 3 (OCR)
    ocr_failed_pages = []     # raster que no se pudo anonimizar por OCR
    non_destructive_pages = []
    # Un término solo cuenta como "no localizado" si no aparece en NINGUNA
    # página. Acumularlo por página marcaría toda la PII como perdida en
    # cuanto el CV tenga una segunda página sin datos de contacto.
    seen_terms = set()
    partial_terms = set()
    bias_found = set()
    pattern_kinds = set()
    boxes_by_page = {}        # para colocar el logo sin pisar contenido

    for page_idx, page in enumerate(doc):
        # 1) remove photo(s) without harming text; detect "image-based" CVs
        # regardless of the remove_images flag, so the RGPD warning below
        # always fires.
        removed, bg_images = _remove_images(page, apply_removal=do_images)
        if do_images:
            total_imgs += removed
        has_bg_image = bool(bg_images)
        if has_bg_image:
            image_based_pages.append(page_idx + 1)

        text_chars = len(page.get_text().strip())
        # Nivel 3 solo entra donde el camino de siempre está roto: página que
        # es una imagen y NO tiene capa de texto usable. Si trae capa OCR
        # incrustada, se respeta el flujo existente (que ya funciona) salvo que
        # se fuerce con OCR_ON_TEXT_LAYER_PAGES.
        raster_mode = (
            has_bg_image and do_ocr
            and (text_chars < OCR_MIN_TEXT_CHARS or OCR_ON_TEXT_LAYER_PAGES)
        )
        if raster_mode and len(ocr_pages) >= OCR_MAX_PAGES:
            log.warning("OCR_MAX_PAGES (%d) alcanzado: la página %d se marca "
                        "para revisión manual", OCR_MAX_PAGES, page_idx + 1)
            ocr_failed_pages.append(page_idx + 1)
            raster_mode = False

        if raster_mode:
            res = _redact_raster_page(page, bg_images, term_list, do_faces,
                                      page_idx=page_idx, doc_key=doc_key)
            if res["ocr"]:
                ocr_pages.append(page_idx + 1)
                ocr_redactions += res["rects"]
                total_faces += res["faces"]
                seen_terms.update(set(term_list) - set(res["unmatched"]))
                partial_terms.update(res["partial"])
                bias_found.update(res["bias"])
                pattern_kinds.update(res["kinds"])
                boxes_by_page[page_idx] = res["text_boxes"]
                if not res["destructive"]:
                    non_destructive_pages.append(page_idx + 1)
                if res["faces"] == 0:
                    pages_without_face.append(page_idx + 1)
            else:
                log.error("OCR no pudo procesar la página %d: %s",
                          page_idx + 1, res["error"])
                ocr_failed_pages.append(page_idx + 1)
                pages_without_face.append(page_idx + 1)
            continue

        # --- ruta de siempre: PDF con capa de texto real ---
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
            # 2c) image-based CV con capa OCR: la PII también está grabada en
            # la imagen (apply_redactions solo borró los glifos invisibles).
            if has_bg_image:
                total_covered += _cover_pii_on_image(page, red_rects, pm_bg)
            # 2d) un mailto:/tel: sigue siendo dato personal aunque el texto
            # visible esté tapado, y apply_redactions no toca los enlaces.
            _strip_identifying_links(page)
        total_text += len(red_rects)

        # 2e) Nivel 2: cover any face baked into the background image itself.
        if has_bg_image and do_faces:
            faces_this_page = sum(
                _cover_faces_on_image(page, xref, rect) for xref, rect in bg_images
            )
            total_faces += faces_this_page
            if faces_this_page == 0:
                pages_without_face.append(page_idx + 1)
        elif has_bg_image:
            pages_without_face.append(page_idx + 1)

    # 3) Branding, en una segunda pasada sobre el documento ya anonimizado:
    # así se puede elegir un tamaño y una esquina que valgan para TODAS las
    # páginas, en vez de decidir a ciegas página a página.
    brand_info = None
    if do_brand:
        brand_info = _apply_branding(doc, boxes_by_page)

    unmatched_terms = (set(term_list) - seen_terms) if ocr_pages else set()
    # Los "parciales" de un término que no se localizó en ninguna página ya
    # están cubiertos por unmatched_terms; no se reportan dos veces.
    partial_terms = {p for p in partial_terms
                     if p.split(" (sin localizar:")[0] not in unmatched_terms}

    if SCRUB_METADATA:
        # title/author/subject del PDF llevan PII con frecuencia
        # ("C.V.-JOSE-LUIS ACTUALIZADO" en un CV real de InfoJobs).
        try:
            doc.set_metadata({})
            doc.del_xml_metadata()
        except Exception as e:
            log.warning("no se pudieron limpiar los metadatos: %s", e)

    n_pages = doc.page_count
    out = doc.tobytes(garbage=4, deflate=True, clean=True)
    doc.close()
    log.info(
        "removed %d image(s); redacted %d text occurrence(s); cleared %d underline(s); "
        "covered %d PII region(s) and %d face(s); OCR: %d region(s) on page(s) %s "
        "across %d terms",
        total_imgs, total_text, total_lines, total_covered, total_faces,
        ocr_redactions, ocr_pages or "none", len(term_list),
    )

    # Nivel 0 (red de seguridad RGPD). Hay dos motivos independientes para
    # pedir revisión manual y se suman:
    #   - la foto: hasta que TRUST_FACE_DETECTION=true se marca toda página
    #     basada en imagen (comportamiento de siempre, sin cambios).
    #   - el OCR: se marca SIEMPRE que algo no haya salido perfecto (un término
    #     que no se localizó, una página que el OCR no pudo procesar, o un
    #     pintado no destructivo). Nunca falla en silencio.
    image_based = bool(image_based_pages)
    if TRUST_FACE_DETECTION:
        photo_review = set(pages_without_face)
    else:
        photo_review = set(image_based_pages)
    ocr_review = set(ocr_failed_pages) | set(non_destructive_pages)
    if unmatched_terms or partial_terms:
        ocr_review |= set(ocr_pages)
    needs_review_pages = sorted(photo_review | ocr_review)
    needs_review = bool(needs_review_pages)

    if image_based:
        log.warning(
            "image-based CV en página(s) %s. OCR aplicado en %s; %d cara(s) "
            "cubiertas; sin cara en %s; términos sin localizar %s; parciales %s; "
            "no destructivo en %s; OCR fallido en %s. needs_manual_review=%s "
            "(TRUST_FACE_DETECTION=%s)",
            image_based_pages, ocr_pages or "none", total_faces,
            pages_without_face or "none", sorted(unmatched_terms) or "none",
            sorted(partial_terms) or "none", non_destructive_pages or "none",
            ocr_failed_pages or "none", needs_review, TRUST_FACE_DETECTION,
        )
        filename = "cv_ciego_REVISAR.pdf" if needs_review else "cv_ciego.pdf"
    else:
        filename = "cv_ciego.pdf"

    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Cv-Redactor-Image-Based": "true" if image_based else "false",
        "X-Cv-Redactor-Image-Pages": ",".join(str(p) for p in image_based_pages),
        "X-Cv-Redactor-Faces-Covered": str(total_faces),
        "X-Cv-Redactor-Pages-Without-Face": ",".join(str(p) for p in pages_without_face),
        "X-Cv-Redactor-Needs-Manual-Photo-Review": "true" if needs_review else "false",
        # --- Nivel 3 ---
        "X-Cv-Redactor-Ocr-Pages": ",".join(str(p) for p in ocr_pages),
        "X-Cv-Redactor-Ocr-Redactions": str(ocr_redactions),
        "X-Cv-Redactor-Ocr-Failed-Pages": ",".join(str(p) for p in ocr_failed_pages),
        "X-Cv-Redactor-Destructive": "false" if non_destructive_pages else "true",
        "X-Cv-Redactor-Unmatched-Terms": " | ".join(sorted(unmatched_terms))[:900],
        "X-Cv-Redactor-Partial-Terms": " | ".join(sorted(partial_terms))[:900],
        "X-Cv-Redactor-Bias-Labels-Found": ",".join(sorted(bias_found))[:400],
        "X-Cv-Redactor-Pii-Kinds": ",".join(sorted(pattern_kinds)),
        "X-Cv-Redactor-Review-Pages": ",".join(str(p) for p in needs_review_pages),
        "X-Cv-Redactor-Logo": (
            "{}@{:.2f} en {}/{} pag".format(
                brand_info["corner"], brand_info["scale"],
                brand_info["placed"], n_pages)
            if brand_info else "none"),
    }
    return out, headers


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    terms: str = Form(...),
    remove_images: str = Form("true"),
    add_branding: str = Form("true"),
    cover_faces: str = Form("true"),
    ocr: str = Form("true"),
    x_api_key: str | None = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    term_list = _parse_terms(terms)
    pdf_bytes = await file.read()
    out, headers = await run_in_threadpool(
        _do_redact, pdf_bytes, term_list,
        _flag(remove_images), _flag(add_branding), _flag(cover_faces),
        _flag(ocr) and OCR_ENABLED and _has_tesseract(),
    )
    return Response(content=out, media_type="application/pdf", headers=headers)


# ---------------------------------------------------------------------------
# /extract-text — texto del CV para la Automatización 1 (resumen con IA).
# En un CV raster el nodo "Extraer texto CV" recibía cadena vacía y el resumen
# se hacía solo con las respuestas de la entrevista. Este endpoint devuelve la
# capa de texto si existe y, si no, el OCR de la página.
# ---------------------------------------------------------------------------
def _do_extract_text(pdf_bytes, forced):
    doc = _open_pdf(pdf_bytes)
    doc_key = _doc_key(pdf_bytes)
    pages = []
    sources = set()
    for page_idx, page in enumerate(doc):
        native = page.get_text().strip()
        _removed, bg_images = _remove_images(page, apply_removal=False)
        use_ocr = forced or len(native) < OCR_MIN_TEXT_CHARS
        text, source = native, "text-layer"
        if use_ocr and OCR_ENABLED and _has_tesseract():
            if page_idx < OCR_MAX_PAGES:
                ocr_text = _ocr_page_text(page, bg_images, page_idx=page_idx,
                                          doc_key=doc_key)
                if len(ocr_text.strip()) > len(native):
                    text, source = ocr_text, "ocr"
            else:
                source = "skipped-page-limit"
        elif use_ocr:
            source = "no-text-layer-and-no-ocr"
        sources.add(source)
        pages.append({
            "page": page_idx + 1,
            "source": source,
            "image_based": bool(bg_images),
            "chars": len(text),
            "text": text,
        })
    doc.close()

    full = "\n\n".join(p["text"] for p in pages if p["text"])
    payload = {
        "text": full,
        "chars": len(full),
        "pages": pages,
        "source": "mixed" if len(sources) > 1 else (sources.pop() if sources else "empty"),
        "image_based": any(p["image_based"] for p in pages),
    }
    log.info("extract-text: %d página(s), %d caracteres, fuentes=%s",
             len(pages), len(full), payload["source"])
    return payload


@app.post("/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    force_ocr: str = Form("false"),
    x_api_key: str | None = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    pdf_bytes = await file.read()
    payload = await run_in_threadpool(_do_extract_text, pdf_bytes, _flag(force_ocr))
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# /convert-to-pdf — convierte CVs en Word (.doc/.docx/.odt/.rtf) a PDF usando
# LibreOffice headless, para que puedan seguir el mismo flujo que un CV que
# ya llega en PDF. Se usa desde n8n justo después de descargar el CV original,
# antes de "Extraer texto CV" y antes de "/redact".
# ---------------------------------------------------------------------------
ALLOWED_CONVERT_EXTENSIONS = {".doc", ".docx", ".odt", ".rtf"}


def _do_convert_to_pdf(file_bytes, original_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, original_name)
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        try:
            subprocess.run(
                [
                    "soffice", "--headless", "--norestore",
                    "--convert-to", "pdf", "--outdir", tmpdir, input_path,
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log.error(
                "LibreOffice conversion failed for %s: %s",
                original_name, e.stderr.decode(errors="ignore"),
            )
            raise HTTPException(
                status_code=500, detail="fallo al convertir el documento a PDF"
            )
        except subprocess.TimeoutExpired:
            log.error("LibreOffice conversion timeout for %s", original_name)
            raise HTTPException(status_code=504, detail="timeout convirtiendo a PDF")

        pdf_name = os.path.splitext(original_name)[0] + ".pdf"
        pdf_path = os.path.join(tmpdir, pdf_name)
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=500, detail="LibreOffice no generó el PDF esperado"
            )

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
    return pdf_bytes, pdf_name


@app.post("/convert-to-pdf")
async def convert_to_pdf(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    original_name = file.filename or "document"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_CONVERT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"extensión no soportada para conversión: {ext or '(sin extensión)'}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    pdf_bytes, pdf_name = await run_in_threadpool(
        _do_convert_to_pdf, file_bytes, original_name)

    log.info(
        "converted %s (%d bytes) to PDF (%d bytes)",
        original_name, len(file_bytes), len(pdf_bytes),
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{pdf_name}"'},
    )
