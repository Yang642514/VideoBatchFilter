# 项目结构说明

## 目录结构

```
VideoBatchFilter/
├── data/                           # 数据文件夹
│   ├── example_links.csv          # 示例视频链接文件
│   └── vedio_links_cleaned.csv    # 清理后的视频链接文件
├── docs/                          # 文档文件夹
│   ├── DEPENDENCY_STATUS_UPDATED.md  # 依赖状态报告
│   ├── INSTALLATION_GUIDE.md      # 安装指南
│   └── PROJECT_STRUCTURE.md       # 项目结构说明（本文件）
├── scripts/                       # 脚本文件夹
│   ├── activate_env.bat          # 激活虚拟环境脚本
│   ├── 批量处理视频.bat          # 批量处理脚本
│   └── 监控模式.bat              # 监控模式脚本
├── utils/                         # 工具模块文件夹
│   ├── __init__.py               # Python包初始化文件
│   └── video_info_extractor.py   # 轻量级视频信息提取器
├── .gitignore                     # Git忽略文件
├── README.md                      # 项目说明文档
├── batch_process.py               # 批量处理脚本
├── config.json                    # 配置文件
├── requirements-minimal.txt       # 最小依赖列表
├── requirements.txt               # 完整依赖列表
└── video_filter.py               # 主程序
```

## 文件夹说明

### data/
存放输入和输出的数据文件：
- `example_links.csv`: 示例视频链接文件，用于测试
- `vedio_links_cleaned.csv`: 处理后的视频链接文件

### docs/
存放项目文档：
- `DEPENDENCY_STATUS_UPDATED.md`: 依赖状态详细报告
- `INSTALLATION_GUIDE.md`: 详细的安装和配置指南
- `PROJECT_STRUCTURE.md`: 项目结构说明文档

### scripts/
存放批处理脚本和工具脚本：
- `activate_env.bat`: 激活conda虚拟环境的便捷脚本
- `批量处理视频.bat`: 一键批量处理视频链接
- `监控模式.bat`: 启动文件夹监控模式

### utils/
存放工具模块和辅助功能：
- `__init__.py`: Python包初始化文件
- `video_info_extractor.py`: 轻量级视频信息提取器

## 核心文件说明

### video_filter.py
主程序文件，提供以下功能：
- 批量处理模式 (`--batch`)
- 监控模式 (`--watch`)
- 单文件处理
- 本地视频处理

### batch_process.py
批量处理脚本，用于处理多个文件。

### config.json
配置文件，包含：
- 筛选条件设置
- 支持的视频格式
- 日志配置
- 其他运行参数

### requirements.txt 和 requirements-minimal.txt
- `requirements.txt`: 完整的依赖列表，包含所有可选功能
- `requirements-minimal.txt`: 最小依赖列表，仅包含核心功能所需的包

## 使用说明

1. **环境准备**: 使用 `scripts/activate_env.bat` 激活虚拟环境
2. **批量处理**: 使用 `scripts/批量处理视频.bat` 进行一键批量处理
3. **监控模式**: 使用 `scripts/监控模式.bat` 启动文件夹监控
4. **手动运行**: 直接运行 `python video_filter.py` 进行单文件处理

## 开发说明

- 所有脚本文件位于 `scripts/` 文件夹中，便于管理
- 工具模块位于 `utils/` 文件夹中，支持模块化导入
- 文档统一存放在 `docs/` 文件夹中
- 数据文件统一存放在 `data/` 文件夹中

这种结构使项目更加清晰、易于维护和扩展。