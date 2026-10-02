# One image for the web dashboard: the Angular app (built in stage 1) is
# served by the FastAPI back end together with the API.
#   docker build -t eu-innovation-panel .
#   docker run -p 8000:8000 eu-innovation-panel      → http://localhost:8000
#   add -e ANTHROPIC_API_KEY=… to enable the research assistant

# ── stage 1: build the Angular front end ─────────────────────────────────
FROM node:24-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ── stage 2: Python API + built front end ────────────────────────────────
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt requirements-api.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-llm.txt
COPY . .
COPY --from=web /web/dist/web/browser ./web/dist/web/browser
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
