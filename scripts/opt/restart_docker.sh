#!/bin/bash
# 移除舊的 docker compose 容器後重新建置並啟動（docker/ 下的 compose 檔）。
#
# 用法：
#   ./scripts/opt/restart_docker.sh            # 模型（vLLM）+ 展示一起重啟
#   ./scripts/opt/restart_docker.sh app        # 只重啟展示伺服器（模型不動，不必重新載入權重）
#   ./scripts/opt/restart_docker.sh vllm       # 只重啟模型
#   ./scripts/opt/restart_docker.sh all --no-build   # 不重新建置映像
#
# down 只移除容器與網路；all 另加 --remove-orphans 清掉改名後殘留的舊服務。app／vllm 不加——
# 三個 compose 檔共用專案名 lineage，加了會把另一個服務當孤兒刪掉。
# 不加 -v、不刪映像，HF 模型快取（主機 ~/.cache/huggingface）與 data/ logs/ output/ 都保留。
# docker/.env 不存在時由 .env.example 複製，並把 APP_UID/APP_GID 設成目前使用者。
set -e
cd "$(dirname "$0")/../.."

TARGET="${1:-all}"
ORPHANS=""
BUILD="--build"
[ "${2:-}" = "--no-build" ] && BUILD=""

case "$TARGET" in
    all) FILE=compose.yaml; ORPHANS="--remove-orphans" ;;
    app) FILE=app.compose.yaml ;;
    vllm) FILE=vllm.compose.yaml; BUILD="" ;;
    *) echo "用法：$0 [all|app|vllm] [--no-build]" >&2; exit 2 ;;
esac

command -v docker >/dev/null 2>&1 || { echo "找不到 docker" >&2; exit 2; }
docker info >/dev/null 2>&1 || {
    echo "無法連線 Docker daemon；確認服務已啟動，且使用者在 docker 群組（sudo usermod -aG docker \$USER 後重新登入）" >&2
    exit 2
}

if [ "$TARGET" != "vllm" ] && [ ! -d data ]; then
    echo "找不到 $(pwd)/data；原始資料需放在專案根目錄 data/ 才能啟動展示" >&2
    exit 2
fi

# vLLM 要 Docker 能把 GPU 交給容器：主機需有 NVIDIA Container Toolkit。
# 沒有時 compose 會報 could not select device driver "nvidia"；在拆舊容器前先擋下。
NVIDIA_HOOK="${NVIDIA_HOOK:-nvidia-container-runtime-hook}"
if [ "$TARGET" != "app" ] && ! command -v "$NVIDIA_HOOK" >/dev/null 2>&1 \
    && ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    cat >&2 <<'EOF'
Docker 無法使用 NVIDIA GPU（找不到 NVIDIA Container Toolkit）。在這台主機執行：
  sudo apt-get install -y nvidia-container-toolkit   # 套件來源見 https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  docker run --rm --gpus all ubuntu nvidia-smi       # 看得到 GPU 即完成
只想先開展示（不含 AI 模型）：./scripts/opt/restart_docker.sh app
EOF
    exit 2
fi

# vLLM 記憶體比例依 GPU 種類：nvidia-smi 回報 [N/A] 是統一記憶體（GB10 / DGX Spark），
# 與系統共用 128 GB → 0.5；獨立顯示卡（L40S、A100、RTX…）顯存專用 → 0.9。
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
case "$GPU_MEM" in
    "") MEM_UTIL="" ;;
    *[!0-9]*) MEM_UTIL=0.5 ;;
    *) MEM_UTIL=0.9 ;;
esac

if [ ! -f docker/.env ]; then
    sed -e "s/^APP_UID=.*/APP_UID=$(id -u)/" -e "s/^APP_GID=.*/APP_GID=$(id -g)/" \
        docker/.env.example > docker/.env
    if [ -n "$MEM_UTIL" ]; then
        sed -i "s/^VLLM_GPU_MEMORY_UTILIZATION=.*/VLLM_GPU_MEMORY_UTILIZATION=$MEM_UTIL/" docker/.env
    fi
    echo "已由 .env.example 建立 docker/.env（APP_UID=$(id -u)、APP_GID=$(id -g)、VLLM_GPU_MEMORY_UTILIZATION=${MEM_UTIL:-預設}）"
elif [ -n "$MEM_UTIL" ]; then
    CURRENT=$(sed -n 's/^VLLM_GPU_MEMORY_UTILIZATION=//p' docker/.env | tail -1)
    if [ -n "$CURRENT" ] && [ "$CURRENT" != "$MEM_UTIL" ]; then
        echo "注意：docker/.env 的 VLLM_GPU_MEMORY_UTILIZATION=$CURRENT，這台 GPU（memory.total=$GPU_MEM）建議 $MEM_UTIL" >&2
    fi
fi

cd docker
echo "== 移除舊容器：docker compose -f $FILE down $ORPHANS"
docker compose -f "$FILE" down $ORPHANS
echo "== 重新啟動：docker compose -f $FILE up -d $BUILD"
docker compose -f "$FILE" up -d $BUILD
docker compose -f "$FILE" ps
echo "展示：http://localhost:$(sed -n 's/^WEB_PORT=//p' .env | tail -1 | grep . || echo 8600)"
echo "看模型載入進度：docker compose -f $FILE logs -f vllm"
