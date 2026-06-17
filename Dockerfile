FROM python:3.12-slim

WORKDIR /app

# PyMuPDF ships prebuilt manylinux wheels, so no compiler/system libs needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# Single worker is plenty for this load; bump --workers if you process
# many CVs in parallel.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
