FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY tools ./tools
COPY core ./core
COPY tests ./tests
COPY config ./config
COPY pyproject.toml VERSION alembic.ini ./
COPY core/infrastructure/database/migrations ./core/infrastructure/database/migrations

EXPOSE 8080
CMD ["python", "-m", "core.runtime.main"]
