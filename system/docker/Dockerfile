# CrimeNet API image.
#
# The application source is *not* copied in. `docker-compose.yml` bind-mounts
# `CriminalNetwork/system` at `/app`, so the container always runs the working
# tree you are editing and the pipeline gets every file it expects
# (`run.py`, `src/`, `loader/`, `db_schema/`, `.env`, case state, uploads).
# This image therefore only supplies the interpreter and the Python
# dependencies from requirements.txt.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /tmp

# OCR binary for image overlays (CCTV timestamps, camera ids — doc 11 §6.4).
# Installed as root before the USER switch; hin+eng matches the pipeline's
# OCR language list, and the code falls back to eng if hin is unavailable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    # insightface pulls opencv-python (Qt build) whose libxcb dependency does
    # not exist in slim; replace it with the headless wheel so `import cv2`
    # works in the container (face recognition stages 1.5/2/2.5).
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless==5.0.0.93 \
    && rm /tmp/requirements.txt

# Overridden by the `user:` entry in docker-compose.yml so that files the
# pipeline writes into the bind mount stay owned by your host user.
USER 1000:1000

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
