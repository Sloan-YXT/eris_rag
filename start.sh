#!/bin/bash
# 启动服务器，自动重启 + 日志追加（不覆盖）
cd "$(dirname "$0")"

while true; do
    echo "========== $(date) SERVER START ==========" >> server.log
    python -m src.server >> server.log 2>&1
    EXIT_CODE=$?
    echo "========== $(date) SERVER EXITED (code=$EXIT_CODE) ==========" >> server.log

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Server exited normally, not restarting."
        break
    fi

    echo "Server crashed, restarting in 5 seconds..."
    sleep 5
done
