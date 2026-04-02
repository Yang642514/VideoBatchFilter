# 视频批量筛选工具

> **一句话说清：** 把成百上千个视频链接扔进去，自动过滤掉画质差、时长不合规、大小不达标的视频，结果直接写回原文件。

---

## 🔍 解决什么问题

在视频搬运、分发、聚合等场景中，你手里往往有一大堆视频链接，需要人工逐一检查每个视频的：
- 画质是否达到要求（比如至少 720p）？
- 时长是否在范围内（比如 10 秒 ~ 1 小时）？
- 文件大小是否合理？

**VideoBatchFilter** 把这件事自动化了——只要你有一条视频链接，工具就能自动获取视频信息并按规则筛选。

---

## 📋 输入 / 输出示例

**处理前（你的 CSV 文件）：**

| link | title | notes |
|------|-------|-------|
| https://example.com/v1.mp4 | 视频1 | 来自A站 |
| https://example.com/v2.mp4 | 视频2 | 来自B站 |
| https://example.com/v3.mp4 | 视频3 | 来自C站 |

**处理后（同一文件多了几列）：**

| link | title | notes | status | reason | duration | resolution |
|------|-------|-------|--------|--------|----------|------------|
| https://example.com/v1.mp4 | 视频1 | 来自A站 | ✅ pass | — | 120s | 1920×1080 |
| https://example.com/v2.mp4 | 视频2 | 来自B站 | ❌ reject | 时长不足(5s<10s) | 5s | 1280×720 |
| https://example.com/v3.mp4 | 视频3 | 来自C站 | ✅ pass | — | 300s | 1920×1080 |

---

## ✨ 功能特点

- 📺 支持多平台在线视频（YouTube、B站、抖音等，依赖 yt-dlp）
- 📁 支持本地视频文件（mp4 / avi / mkv / mov / flv / wmv）
- 📏 按分辨率筛选（宽高均可自定义）
- ⏱️ 按时长筛选（最小 / 最大时长）
- 📦 按文件大小筛选
- 📊 支持批量处理 CSV / Excel 文件
- 👁️ 支持监控模式（新文件自动处理）
- 🛡️ 智能重试 + 详细日志

---

## 🚀 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/Yang642514/VideoBatchFilter.git
cd VideoBatchFilter
pip install -r requirements-minimal.txt
```

### 2. 准备文件

将包含视频链接的 CSV 或 Excel 文件放入 `data/` 文件夹，文件需包含 `link` 列。

### 3. 运行

**方式一：双击（Windows）**
```
双击运行 scripts\批量处理视频.bat
```

**方式二：命令行**
```bash
python batch_process.py
```

### 4. 查看结果

处理完成后，打开原文件，最后几列就是筛选结果。

---

## 📁 项目结构

```
VideoBatchFilter/
├── batch_process.py          # 一键批量处理入口
├── video_filter.py            # 核心筛选逻辑
├── config.json               # 筛选规则配置
├── data/                     # 放待处理文件
├── scripts/
│   ├── 批量处理视频.bat        # Windows 一键处理
│   └── 监控模式.bat           # Windows 监控文件夹
└── utils/
    └── video_info_extractor.py  # 视频信息提取器
```

---

## ⚙️ 筛选规则配置

编辑 `config.json` 即可自定义筛选条件：

```json
{
    "filters": {
        "min_resolution": [1280, 720],   // 最小分辨率
        "min_duration": 10,               // 最短时长（秒）
        "max_duration": 3600,             // 最长时长（秒）
        "min_size": 5,                   // 最小文件大小（MB）
        "max_size": 2000                  // 最大文件大小（MB）
    }
}
```

---

## 📖 进阶用法

```bash
# 批量处理 data 文件夹下所有文件
python video_filter.py --batch

# 单文件处理
python video_filter.py --excel my_links.csv

# 监控模式（自动处理新文件）
python video_filter.py --watch

# 本地视频文件筛选
python video_filter.py -i ./input -o ./output
```

---

## ❓ 常见问题

**Q: 链接列名不是 `link` 怎么办？**
在 `config.json` 中添加：`"link_column": "你的列名"`

**Q: 想保留某些链接不受筛选影响？**
在 CSV 中加一列 `force_pass`，值设为 `true`，该链接会跳过筛选直接通过。

**Q: 程序中途失败了怎么办？**
工具支持断点续传，已处理的链接不会重复抓取，重新运行即可。

---

## 📦 依赖

| 依赖 | 用途 |
|------|------|
| yt-dlp | 获取在线视频信息 |
| pandas | CSV/Excel 数据处理 |
| requests | 网络请求 |
| tqdm | 进度条 |

完整依赖见 `requirements.txt`，最小依赖见 `requirements-minimal.txt`。

---

## 许可证

MIT
