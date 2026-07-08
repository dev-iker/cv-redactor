FROM python:3.12-slim

WORKDIR /app

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

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
