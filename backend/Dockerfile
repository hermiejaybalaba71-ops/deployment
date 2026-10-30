# FastAPI ML service (RapidOCR + scikit-learn Random Forest) - Render.
#
# Docker, NOT Render's native Python runtime: rapidocr-onnxruntime hard-requires
# opencv-python (not the -headless variant), and that wheel needs libGL + libglib
# at import time; scikit-learn needs libgomp (OpenMP). Render's native image has
# none of them, so `import cv2` would raise "libGL.so.1: cannot open shared
# object file" and the health check would report ocr_ready: false.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first: editing Python code must not re-download ~1 GB of
# wheels (onnxruntime + opencv + scikit-learn).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Render assigns $PORT at runtime; 10000 is the local fallback.
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
