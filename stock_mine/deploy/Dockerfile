FROM python:3.12-slim@sha256:9e01bf1ae5db7649a236da7be1e94ffbbbdd7a93f867dd0d8d5720d9e1f89fab

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    STOCK_MINE_DB_PATH=/app/data/stock_mine.db \
    TZ=Asia/Shanghai \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY deploy/requirements.lock /app/deploy/requirements.lock
COPY deploy/wheels /wheels
RUN pip install --no-index --find-links=/wheels --no-cache-dir -r /app/deploy/requirements.lock \
    && groupadd --gid 1000 stockmine \
    && useradd --uid 1000 --gid 1000 --no-create-home stockmine \
    && mkdir -p /app/data \
    && chown stockmine:stockmine /app/data \
    && mkdir -p /usr/share/zoneinfo/Asia \
    && cp /usr/local/lib/python3.12/site-packages/tzdata/zoneinfo/Asia/Shanghai /usr/share/zoneinfo/Asia/Shanghai

COPY backend /app/backend
COPY scripts /app/scripts
USER stockmine
EXPOSE 8000

# Scan and backtest job registries are in-process; never scale this to multiple workers.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"]
