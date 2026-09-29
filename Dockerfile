# SkyGuard AI - single container: FastAPI backend + built dashboard + edge C library (zig cc).
FROM node:22-alpine AS ui
WORKDIR /ui
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config config
COPY ml ml
COPY injection injection
COPY backend backend
COPY edge edge
COPY scripts scripts
COPY data/raw data/raw
COPY data/demo data/demo
COPY artifacts/final_model artifacts/final_model
COPY --from=ui /ui/dist dashboard/dist
# station registry from the preserved raw data (offline) + edge C library for this platform
RUN python scripts/build_dataset.py && python edge/build_edge.py \
 && useradd --create-home skyguard && mkdir -p /app/data/db && chown -R skyguard /app/data /app/artifacts
USER skyguard
ENV SKYGUARD_DB=/app/data/db/skyguard.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
