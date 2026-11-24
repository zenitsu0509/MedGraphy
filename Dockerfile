FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
# Retrieval uses CPU inference. On Linux ARM, the default PyPI torch wheel can
# pull several hundred MB of CUDA libraries that this container cannot use.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 && \
    python -m pip install -r requirements.txt

COPY . .
RUN chmod +x /app/docker-entrypoint.sh && mkdir -p /app/runtime

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
