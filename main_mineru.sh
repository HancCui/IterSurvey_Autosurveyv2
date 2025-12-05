#!/bin/bash

# mineru 服务启动脚本
DEFAULT_PORT=8000
DEFAULT_HOST="127.0.0.1"

PORT=${1:-$DEFAULT_PORT}
HOST=${2:-$DEFAULT_HOST}
conda activate steven312
# 日志目录 - 使用脚本所在目录的相对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p $LOG_DIR

# 检查端口是否被占用
if ss -tuln | grep -q ":$PORT "; then
    echo "端口 $PORT 已被占用，正在终止占用进程..."
    PID=$(ss -tulpn | grep ":$PORT " | awk '{print $7}' | cut -d',' -f2 | cut -d'=' -f2)
    if [ ! -z "$PID" ]; then
        kill -9 $PID 2>/dev/null
        sleep 2
    fi
fi

echo "启动 mineru 服务 (端口: $PORT)..."

# 启动 mineru 服务
nohup env MINERU_VIRTUAL_VRAM_SIZE=2 MINERU_MODEL_SOURCE=local mineru-api --host $HOST --port $PORT -mem-fraction-static 0.5 --enable-torch-compile --local > $LOG_DIR/mineru.log 2>&1 &

MINERU_PID=$!
echo $MINERU_PID > $LOG_DIR/mineru.pid

# 等待服务启动
echo "等待服务启动..."
for i in {1..20}; do
    if curl -s "http://$HOST:$PORT" > /dev/null 2>&1; then
        echo "✓ mineru 服务启动成功！"
        echo "  服务地址: http://$HOST:$PORT"
        echo "  进程 PID: $MINERU_PID"
        echo "  日志文件: $LOG_DIR/mineru.log"
        exit 0
    fi
    sleep 1
done

echo "服务启动失败，请检查日志: $LOG_DIR/mineru.log"
exit 1
