FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY . .

# 使用 uv 安裝 Python 依賴
RUN uv sync
ENV PATH="/app/.venv/bin:$PATH"

# 暴露端口
EXPOSE 8000

# 啟動命令
CMD [ "uvicorn", "app.main:app", "--host=0.0.0.0", "--port", "8000" ]
