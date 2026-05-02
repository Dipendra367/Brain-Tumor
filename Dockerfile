# ── Base image ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# ── System dependencies for OpenCV + TensorFlow ────────────
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app code ──────────────────────────────────────────
COPY app/ ./app/
COPY models/best_model.keras ./models/best_model.keras
COPY models/production_model_info.txt ./models/production_model_info.txt
COPY firebase-credentials.json ./firebase-credentials.json

# ── Expose FastAPI port ────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────
HEALTHCHECK CMD curl --fail http://localhost:8000/api/health || exit 1

# ── Start FastAPI ──────────────────────────────────────────
WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]