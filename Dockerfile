FROM python:3.12-slim

WORKDIR /app

# LibreOffice headless, usado por /convert-to-pdf para convertir CVs
# en Word (.doc/.docx/.odt/.rtf) a PDF antes de procesarlos.
# libreoffice-writer basta (no hace falta el paquete completo libreoffice,
# que incluye Calc/Impress y pesa bastante más).
#
# Tesseract (Nivel 3) para los CVs que llegan rasterizados —InfoJobs los
# sirve como una imagen a página completa, sin capa de texto— tanto para
# anonimizarlos como para darle texto al resumen de IA vía /extract-text.
# El OCR corre dentro de este contenedor: ningún CV sale a un tercero, que
# se convertiría en un nuevo encargado del tratamiento de datos personales.
# -spa y -eng son los que usa OCR_LANG por defecto; -cat va instalado para
# poder poner OCR_LANG="spa+cat+eng" si aparecen CVs en catalán (cada idioma
# extra suma tiempo de OCR, así que no está activado por defecto).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    tesseract-ocr-cat \
    && rm -rf /var/lib/apt/lists/*

# PyMuPDF, Pillow and opencv-python-headless ship prebuilt wheels, so no
# compiler/system libs should be needed. If the build ever fails on a missing
# shared library (e.g. libglib2.0-0) for opencv, add it here with apt-get.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Nivel 2 (face detection) model - fetched at build time from the official
# OpenCV model zoo instead of committing a binary to this repo. ~230KB.
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 'face_detection_yunet_2023mar.onnx')"

# App + bundled Behum branding assets
COPY app.py behum_icon.png behum_logo.png ./

# Cada worker de uvicorn puede lanzar un OCR a la vez; Tesseract ya usa
# OpenMP por dentro y dejarle todos los hilos con varias páginas en paralelo
# provoca thrashing en un VPS pequeño.
ENV OMP_THREAD_LIMIT=2

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
