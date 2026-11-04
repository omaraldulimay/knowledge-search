FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed HF_HOME=/opt/hf

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# dependencies first (better layer caching), then the code
COPY pyproject.toml README.md ./
COPY ks ./ks
# editable install: the code stays in /app, so data/ and eval/ are found relative to it
RUN pip install -e ".[local,anthropic]"

# download the embedding model at build time, so the container starts offline and fast
RUN python -c "from ks.embed import FastEmbedEmbedder; FastEmbedEmbedder()" \
    && chmod -R a+rX /opt/fastembed

COPY data ./data
COPY eval ./eval
RUN mkdir -p /app/state && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "ks.api:app", "--host", "0.0.0.0", "--port", "8000"]
