#!/bin/bash
# macOS / Linux：直接雙擊這個檔案就會開始安裝
cd "$(dirname "$0")" || exit 1
clear
if command -v python3 >/dev/null 2>&1; then
  python3 install.py
else
  echo "找不到 Python。請先到 https://www.python.org/downloads/ 安裝，再重新雙擊這個檔案。"
  read -r -p "按 Enter 關閉…"
fi
