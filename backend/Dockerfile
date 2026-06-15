# ── Stage 1: builder ──────────────────────────────────────────────
FROM python:3.10-alpine3.20 AS builder

WORKDIR /app

# Alpine: install gcc + musl-dev needed to compile some Python packages
RUN apk add --no-cache gcc musl-dev libffi-dev

# Install Python dependencies into an isolated prefix
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────
FROM python:3.10-alpine3.20 AS runtime

WORKDIR /app

# Tesseract OCR + runtime libs (all from Alpine's apk — no Debian mirrors)
RUN apk add --no-cache \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    libgcc \
    libstdc++ \
    zlib \
    libjpeg \
    libpng \
    freetype

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source (whole project root so backend/ is a package)
COPY . /app

# Create persistent data directories
RUN mkdir -p /app/data/resumes /app/data/user_contexts

# Non-root user for security
RUN addgroup -S nexus && adduser -S -G nexus -u 1001 nexus && \
    chown -R nexus:nexus /app
USER nexus

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
