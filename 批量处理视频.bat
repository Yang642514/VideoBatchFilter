@echo off
chcp 65001 >nul
title 视频批量筛选工具

echo ========================================
echo 视频批量筛选工具 - 批量处理模式
echo ========================================
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
    echo.
    echo 请将待处理的CSV或Excel文件放入data文件夹，然后重新运行此脚本
    pause
    exit /b 0
)

REM 运行批处理脚本
echo 🚀 启动批量处理...
echo.
python batch_process.py

echo.
echo 处理完成！
pause