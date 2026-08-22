@echo off
REM Windows：直接雙擊這個檔案就會開始安裝
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 install.py
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python install.py
    ) else (
        echo 找不到 Python。請先到 https://www.python.org/downloads/ 安裝，
        echo 安裝時記得勾選 "Add Python to PATH"，然後重新雙擊這個檔案。
        pause
    )
)
