@echo off
echo 激活VideoBatchFilter虚拟环境...
call C:\Users\30305\Anaconda3\Scripts\activate.bat VideoBatchFilter
echo 环境已激活！
echo 当前Python版本：
python --version
echo.
echo 可用命令：
echo   python video_filter.py --help    - 查看主程序帮助
echo   python video_filter.py --batch   - 批量处理模式
echo   pip install -r requirements.txt  - 安装依赖
echo.
cmd /k