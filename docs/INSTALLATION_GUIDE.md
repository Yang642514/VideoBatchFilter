# 依赖安装指南

## 推荐环境

- Python 3.10+
- 独立虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## 在线链接与 CSV/Excel

```bash
pip install -r requirements-minimal.txt
```

主要依赖：

- `yt-dlp`：获取在线视频元信息
- `pandas`、`openpyxl`：读写 Excel
- `requests`：下载缩略图
- `tqdm`：进度显示

平台页面和访问限制会变化。如果 `yt-dlp` 无法读取某个链接，请先升级它：

```bash
python -m pip install --upgrade yt-dlp
```

部分受限内容可能需要本地 `cookies.txt`。Cookie 是账号凭据，不要提交或分享。

## 本地视频文件

本地目录筛选需要 OpenCV，安装完整依赖：

```bash
pip install -r requirements.txt
```

## 可选内容标签

`torch`、`transformers`、`Pillow` 用于基于缩略图的 CLIP 内容标签。首次运行可能下载模型，CPU 环境也可以运行但速度较慢。

内容标签是模型估计，不是人工审核结论。

## 检查安装

```bash
python video_filter.py --help
```

程序启动时会输出可用依赖。缺少可选依赖时，对应功能会跳过。
