#!/bin/bash
# 啟動 AI 問答用的本地模型（OpenAI 相容 API，預設 http://127.0.0.1:8000/v1），供 web/assistant.py 連線。
# LangGraph agent 靠 tool calling 讀機台狀態，所以兩種引擎都要開工具呼叫：
#   vLLM      --enable-auto-tool-choice --tool-call-parser hermes（Qwen 系列用 hermes 格式）
#   llama.cpp --jinja（用模型內建 chat template 解析工具呼叫）
#
# 用法：
#   ./scripts/opt/run_llm.sh                              # vLLM + Qwen/Qwen3-30B-A3B-Instruct-2507
#   ./scripts/opt/run_llm.sh --engine llamacpp            # llama-server + 同模型 GGUF Q4_K_M
#   ./scripts/opt/run_llm.sh --engine vllm --model Qwen/Qwen3-8B --port 8000
# 引擎本身（vllm 或 llama-server）需先自行安裝並在 PATH 上；本腳本不安裝任何東西。
# 換了 --port 時，web 端要對應 ./scripts/opt/run_web.sh --llm-url http://127.0.0.1:<port>/v1
set -e

ENGINE=vllm
MODEL=""
PORT=8000
CTX=32768
while [ $# -gt 0 ]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        *) echo "未知參數：$1" >&2; exit 2 ;;
    esac
done

case "$ENGINE" in
    vllm)
        command -v vllm >/dev/null 2>&1 || { echo "找不到 vllm；請先安裝 vLLM（GB10/aarch64 建議用 NVIDIA vLLM 容器）" >&2; exit 2; }
        MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
        exec vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len "$CTX" \
            --enable-auto-tool-choice --tool-call-parser hermes
        ;;
    llamacpp)
        command -v llama-server >/dev/null 2>&1 || { echo "找不到 llama-server；請先編譯 llama.cpp（-DGGML_CUDA=ON）" >&2; exit 2; }
        MODEL="${MODEL:-unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M}"
        exec llama-server -hf "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -ngl 999 --jinja
        ;;
    *)
        echo "--engine 只接受 vllm 或 llamacpp" >&2; exit 2 ;;
esac
