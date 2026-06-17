FROM python:3.12-slim

WORKDIR /app

# PyMuPDF and Pillow ship prebuilt wheels, so no compiler/system libs needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + bundled Behum branding assets
COPY app.py behum_icon.png behum_logo.png ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
