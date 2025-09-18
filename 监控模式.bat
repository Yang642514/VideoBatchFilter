@echo off
chcp 65001 >nul
title 视频批量筛选工具 - 监控模式

echo ========================================
echo 视频批量筛选工具 - 监控模式
echo ========================================
echo.
echo 此模式将持续监控data文件夹中的新文件
echo 当检测到新的CSV或Excel文件时，会自动处理
echo 按 Ctrl+C 可以停止监控
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到Python，请先安装Python
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查是否存在主程序
if not exist "video_filter.py" (
    echo ❌ 未找到video_filter.py文件
    echo 请确保此批处理文件与video_filter.py在同一目录下
    pause
    exit /b 1
)

REM 检查data文件夹
if not exist "data" (
    echo 📁 创建data文件夹...
    mkdir data
    echo ✅ data文件夹已创建
)

echo 🔍 启动监控模式...
echo.
python video_filter.py --watch

echo.
echo 监控已停止
pause