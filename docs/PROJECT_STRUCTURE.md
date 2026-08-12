# 项目结构

```text
VideoBatchFilter/
├── data/
│   └── example_links.csv
├── docs/
│   ├── INSTALLATION_GUIDE.md
│   └── PROJECT_STRUCTURE.md
├── scripts/
│   ├── activate_env.bat
│   ├── 批量处理视频.bat
│   └── 监控模式.bat
├── utils/
│   ├── video_info_extractor.py
│   └── video_processor.py
├── batch_process.py
├── config.json
├── requirements-minimal.txt
├── requirements.txt
└── video_filter.py
```

## 核心文件

- `video_filter.py`：CLI、在线链接处理、本地目录筛选和监控模式。
- `utils/video_processor.py`：结果结构、规则判断和 CSV 写回。
- `utils/video_info_extractor.py`：缺少 `yt-dlp` 时的有限备用提取器。
- `config.json`：列名、筛选阈值、可选内容标签和 Cookie 路径。
- `batch_process.py`：适合 Windows 双击使用的批处理入口。

## 数据目录

`data/example_links.csv` 仅用于展示输入格式，不含真实平台数据。`.gitignore` 会忽略其他 CSV 和 Excel 文件，避免误提交实际业务数据。
