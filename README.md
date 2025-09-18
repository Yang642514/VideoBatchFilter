# 视频批量筛选工具

这是一个用于批量筛选视频文件的Python工具，可以根据分辨率、时长、文件大小等条件筛选视频。支持在线视频链接检测和本地视频文件筛选。

## 功能特点

- 🎥 支持多种视频格式（mp4, avi, mkv, mov, flv, wmv）
- 📏 可根据分辨率筛选视频
- ⏱️ 可根据视频时长筛选视频
- 📦 可根据文件大小筛选视频
- ⚙️ 支持自定义配置文件
- 📊 支持CSV/Excel文件批量处理
- 🔄 支持批量处理模式（一键处理多个文件）
- 👁️ 支持监控模式（自动处理新文件）
- 🌐 支持在线视频链接检测
- 🛡️ 智能重试机制和错误处理

## 安装

### 方法一：使用conda虚拟环境（推荐）

1. 克隆仓库：
```bash
git clone https://github.com/Yang642514/VideoBatchFilter.git
cd VideoBatchFilter
```

2. 创建并激活虚拟环境：
```bash
# 创建虚拟环境
conda create -n VideoBatchFilter python=3.9 -y

# 激活环境（Windows）
activate VideoBatchFilter
# 或使用脚本
scripts\activate_env.bat
```

3. 安装依赖：
```bash
# 安装最小依赖（推荐）
pip install -r requirements-minimal.txt

# 或安装完整依赖
pip install -r requirements.txt
```

### 方法二：直接安装

```bash
pip install -r requirements-minimal.txt
```

> 💡 **提示**: 推荐使用虚拟环境以避免依赖冲突。项目已配置专用的conda环境。

## 使用说明

### 🚀 快速开始（推荐）

**最简单的使用方式：**

1. **将待处理文件放入data文件夹**
   - 将包含视频链接的CSV或Excel文件放入 `data` 文件夹
   - 确保文件中包含 `link` 列（或在config.json中自定义列名）

2. **双击运行批处理脚本**
   - Windows用户：双击 `批量处理视频.bat`
   - 或者双击 `batch_process.py`

3. **等待处理完成**
   - 程序会自动处理data文件夹中的所有文件
   - 结果会直接写回原文件

### 📁 文件夹结构

```
VideoBatchFilter/
├── data/                    # 数据文件夹（放置待处理文件）
│   ├── example_links.csv    # 示例文件
│   └── vedio_links_cleaned.csv # 处理结果文件
├── docs/                    # 文档文件夹
│   ├── DEPENDENCY_STATUS_UPDATED.md # 依赖状态报告
│   └── INSTALLATION_GUIDE.md        # 安装指南
├── scripts/                 # 脚本文件夹
│   ├── activate_env.bat     # 环境激活脚本
│   ├── 批量处理视频.bat      # 一键批量处理（Windows）
│   └── 监控模式.bat         # 监控模式（Windows）
├── utils/                   # 工具模块文件夹
│   ├── __init__.py          # 包初始化文件
│   └── video_info_extractor.py # 轻量级视频信息提取器
├── batch_process.py         # 批量处理脚本
├── video_filter.py          # 主程序
├── config.json             # 配置文件
├── requirements.txt         # 完整依赖列表
└── requirements-minimal.txt # 最小依赖列表
```

### 🔧 使用模式

#### 1. 批量处理模式（推荐）
```bash
# 处理data文件夹中的所有文件
python video_filter.py --batch

# 指定其他文件夹
python video_filter.py --batch --data-dir your_folder
```

#### 2. 监控模式
```bash
# 持续监控data文件夹，自动处理新文件
python video_filter.py --watch

# 或双击运行：监控模式.bat
```

#### 3. 单文件处理模式
```bash
# 处理单个Excel文件
python video_filter.py --excel your_file.xlsx

# 处理单个CSV文件
python video_filter.py --excel your_file.csv
```

#### 4. 本地视频文件模式
```bash
# 处理本地视频文件
python video_filter.py -i input_folder -o output_folder
```

### 📋 文件格式要求

**CSV/Excel文件必须包含以下列：**
- `link`: 视频链接列（必需）
- 其他列可自定义

**示例文件内容：**
```csv
link,title,description
https://example.com/video1.mp4,视频1,描述1
https://example.com/video2.mp4,视频2,描述2
```

**处理后会自动添加结果列：**
- `status`: 筛选状态（pass/reject/error）
- `reason`: 筛选原因
- `duration`: 视频时长
- `title`: 视频标题
- `description`: 视频描述

## 配置文件

配置文件示例（config.json）：

```json
{
    "filters": {
        "min_resolution": [1280, 720],
        "min_duration": 10,
        "max_duration": 3600,
        "min_size": 5,
        "max_size": 2000
    },
    "video_extensions": [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"],
    "logging": {
        "level": "INFO",
        "file": "video_filter.log"
    }
}
```

## 依赖

### 核心依赖
- Python 3.7+
- yt-dlp (视频信息获取)
- pandas (数据处理)
- requests (网络请求)
- tqdm (进度条)

### 可选依赖
- openpyxl (Excel文件支持)
- opencv-python (本地视频处理)
- numpy (数值计算)

### 安装依赖
```bash
# 安装最小依赖
pip install -r requirements-minimal.txt

# 安装完整依赖
pip install -r requirements.txt
```

## 常见问题

### Q: 如何自定义链接列名？
A: 在config.json中设置：
```json
{
    "link_column": "your_column_name"
}
```

### Q: 如何修改筛选条件？
A: 编辑config.json文件中的filters部分：
```json
{
    "filters": {
        "min_duration": 30,
        "max_duration": 1800,
        "min_resolution": [720, 480]
    }
}
```

### Q: 程序运行出错怎么办？
A: 
1. 检查网络连接
2. 确保文件格式正确
3. 查看生成的日志文件
4. 检查Python和依赖是否正确安装

## 更新日志

### v1.1.0 (2025-01-18)
**重要修复：**
- 🐛 修复了pandas读取CSV时数据类型推断导致的float转换错误
- 🔧 修复了配置文件中时长筛选参数的读取逻辑
- ✅ 解决了批量处理模式下"could not convert string to float"错误
- 📊 改进了CSV文件的数据类型处理，所有列现在都以字符串形式读取

**技术改进：**
- 在pandas读取CSV时添加了`dtype=str`参数，避免自动类型推断
- 修正了配置文件中`filters.min_duration`和`filters.max_duration`的读取路径
- 增强了错误处理和调试信息输出

**影响：**
- 批量处理功能现在可以正常处理包含空值的CSV文件
- 提高了程序的稳定性和容错能力
- 改善了用户体验，减少了运行时错误

### v1.0.0 (2025-01-17)
- 🎉 初始版本发布
- 支持批量处理、监控模式、在线视频检测等核心功能

## 许可证

MIT