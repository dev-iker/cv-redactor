import os
import json
import logging

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cv-redactor")

# If REDACT_API_KEY is set, callers must send the same value in the X-API-Key
# header. If it is unset, auth is disabled (only acceptable for local testing).
API_KEY = os.getenv("REDACT_API_KEY")

# Extra margin (in points) added around each match so descenders/edges are
# fully covered. Tunable without touching code.
RECT_PADDING = float(os.getenv("REDACT_PADDING", "2"))

app = FastAPI(title="CV Redactor", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    terms: str = Form(...),
    x_api_key: str | None = Header(default=None),
):
    # --- auth ---
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    # --- validate terms ---
    try:
        term_list = json.loads(terms)
        if not isinstance(term_list, list):
            raise ValueError
        term_list = [str(t) for t in term_list if str(t).strip()]
    except Exception:
        raise HTTPException(
            status_code=400, detail="`terms` must be a JSON array of strings"
        )

    # --- read pdf ---
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="not a valid PDF")

    # --- redact ---
    total_hits = 0
    for page in doc:
        page_hits = 0
        for term in term_list:
            for rect in page.search_for(term):
                padded = fitz.Rect(
                    rect.x0 - RECT_PADDING,
                    rect.y0 - RECT_PADDING,
                    rect.x1 + RECT_PADDING,
                    rect.y1 + RECT_PADDING,
                )
                # fill=False -> no box is drawn; only the text is removed.
                page.add_redact_annot(padded, fill=False)
                page_hits += 1
        if page_hits:
            # Remove ONLY the text glyphs, leaving the page background
            # (white space, colored banners, images) untouched -> the
            # redacted area blends in instead of showing a black bar.
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
        total_hits += page_hits

    # garbage=4 purges the now-unreferenced content so the original data
    # cannot be recovered from leftover objects in the file.
    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    # IMPORTANT: log counts only, never the PII values themselves.
    log.info("redacted %d occurrences across %d terms", total_hits, len(term_list))

    return Response(
        content=out,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="cv_ciego.pdf"'},
    )
