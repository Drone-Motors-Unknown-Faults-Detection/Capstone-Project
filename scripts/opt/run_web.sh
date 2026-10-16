#!/bin/bash
# 啟動冷啟動 PHM 即時展示伺服器（參數原樣轉給 web/server.py，如 --port 8600 --motor T1 --rpm 8000rpm）
set -e
cd "$(dirname "$0")/../.."
exec venv/bin/python -m web.server "$@"