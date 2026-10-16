#!/bin/bash
# 跑全套 pytest（AGENT.md 鐵則 10：提交前要通過）；額外參數原樣轉給 pytest，如 -k test_foo
set -e
cd "$(dirname "$0")"
exec venv/bin/python -m pytest src/tests "$@"
