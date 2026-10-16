#!/bin/bash
# 跑 ruff 風格檢查（AGENT.md 鐵則 12：提交前要零錯誤）；額外參數原樣轉給 ruff，如 --fix
set -e
cd "$(dirname "$0")/../.."
exec venv/bin/python -m ruff check . "$@"
