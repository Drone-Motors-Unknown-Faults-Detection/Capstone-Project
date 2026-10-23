# syntax=docker/dockerfile:1
# 冷啟動 PHM 即時展示（web.server）映像。build context 是 repo 根目錄：
#   docker compose -f docker/app.compose.yaml build
# 依賴照 uv.lock 安裝（--locked，不重新解析）；不含 test extras 與 legacy TensorFlow。
# data/ 不進映像，執行時以唯讀 volume 掛入（AGENT.md 鐵則 5）。

FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

FROM python:3.10.21-slim-bookworm AS builder
# hdbscan 0.8.44 沒有 aarch64 wheel（GB10 / DGX Spark），要從 sdist 編譯
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --python /usr/local/bin/python3.10

FROM python:3.10.21-slim-bookworm
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# core.runtime_environment 以 /app/pyproject.toml 認程式根、雜湊 /app/uv.lock；少了它們，存下的結果 lock_status 是 UNKNOWN
COPY pyproject.toml uv.lock ./
COPY src ./src
# AI 問答的 get_experiment／lookup_knowledge 讀實驗手冊（web/assistant.py EXPERIMENTS_DIR）
COPY docs/experiments ./docs/experiments
# 逐步導覽的 /guide/docs/*（web/guide.py routes）與實驗頁 8-2 封存卡（web/experiments.py _load_exp8_results）
COPY docs/navigation ./docs/navigation
COPY reports/exp8_health_index_results ./reports/exp8_health_index_results
# setup_run() 寫相對路徑 logs/、output/，來源綁定寫 .lineage_private/（#311）；compose 會把 repo 的同名目錄掛進來
RUN mkdir -p logs output data .lineage_private && chmod 0777 logs output .lineage_private
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8600
ENTRYPOINT ["python", "-m", "web.server"]
CMD ["--data-root", "/app/data", "--bind-address", "0.0.0.0", "--port", "8600"]
