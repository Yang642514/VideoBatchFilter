# 依赖安装指南

## 当前状态

✅ **基本功能已可用** - 系统已集成轻量级视频信息提取器，可以处理大部分视频链接

## 依赖安装优先级

### 🟢 立即推荐安装（简单且有效）

#### 1. OpenCV (视频处理)
```bash
# 方法1: 使用pip安装（推荐）
pip install opencv-python

# 方法2: 如果网络有问题，使用国内镜像
pip install opencv-python -i https://pypi.douban.com/simple/

# 方法3: 使用conda（如果conda环境正常）
conda install -c conda-forge opencv
```

#### 2. NumPy (数值计算)
```bash
pip install numpy
```

### 🟡 可选安装（提升功能）

#### 3. PyTorch + transformers (AI内容检测)
```bash
# CPU版本（较小，适合大多数用户）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers

# 如果有NVIDIA GPU，可以安装CUDA版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
```

### 🔴 高难度安装（可跳过）

#### 4. yt-dlp (完整视频信息提取)
```bash
# 需要Microsoft Visual C++ 14.0或更高版本
pip install yt-dlp

# 如果安装失败，可以尝试：
# 1. 安装Microsoft C++ Build Tools
# 2. 使用预编译版本
# 3. 继续使用轻量级替代方案（推荐）
```

## 替代方案说明

### 轻量级视频信息提取器

我们已经为您创建了一个轻量级的视频信息提取器，作为yt-dlp的替代方案：

**支持的平台：**
- ✅ Bilibili (B站) - 完全支持
- ✅ YouTube - 基本支持（可能受网络限制）
- ✅ 抖音/TikTok - 基本支持
- ✅ 其他平台 - 通用支持

**提取的信息：**
- 视频标题
- 视频时长
- 分辨率信息
- 错误信息

**优势：**
- 无需编译器
- 安装简单
- 网络要求低
- 支持主流平台

## 安装验证

### 检查当前依赖状态
```bash
python video_filter.py --help
```

### 测试轻量级提取器
```bash
python utils/video_info_extractor.py
```

### 测试批量处理
```bash
python video_filter.py --batch
```

## 网络问题解决方案

如果遇到网络连接问题：

### 1. 使用国内镜像源
```bash
pip install -i https://pypi.douban.com/simple/ 包名
```

### 2. 配置pip镜像源（永久）
创建或编辑 `~/.pip/pip.conf` (Linux/Mac) 或 `%APPDATA%\pip\pip.ini` (Windows):
```ini
[global]
index-url = https://pypi.douban.com/simple/
trusted-host = pypi.douban.com
```

### 3. 离线安装
如果网络完全不可用，可以：
1. 在有网络的机器上下载wheel文件
2. 传输到目标机器
3. 使用 `pip install 文件名.whl` 安装

## 功能对比

| 功能 | 当前状态 | 完整安装后 |
|------|----------|------------|
| CSV/Excel处理 | ✅ 完全支持 | ✅ 完全支持 |
| 批量处理 | ✅ 完全支持 | ✅ 完全支持 |
| B站视频信息 | ✅ 支持 | ✅ 完全支持 |
| YouTube视频信息 | ⚠️ 基本支持 | ✅ 完全支持 |
| 本地视频处理 | ❌ 不支持 | ✅ 支持 |
| AI内容检测 | ❌ 不支持 | ✅ 支持 |
| 缩略图下载 | ❌ 不支持 | ✅ 支持 |

## 推荐安装顺序

1. **立即可用** - 当前系统已经可以处理大部分需求
2. **第一步** - 安装OpenCV：`pip install opencv-python`
3. **第二步** - 如需AI功能，安装PyTorch：`pip install torch transformers`
4. **第三步** - 如需完整功能，尝试安装yt-dlp：`pip install yt-dlp`

## 故障排除

### 常见问题

1. **pip安装失败**
   - 检查网络连接
   - 尝试使用镜像源
   - 更新pip：`python -m pip install --upgrade pip`

2. **conda环境问题**
   - 当前conda环境可能有版本冲突
   - 建议使用pip安装
   - 或创建新的conda环境

3. **编译错误**
   - 主要影响yt-dlp安装
   - 可以继续使用轻量级替代方案
   - 功能基本相同

### 获取帮助

如果遇到问题，可以：
1. 查看错误信息
2. 检查网络连接
3. 尝试使用镜像源
4. 继续使用当前的轻量级方案

## 总结

**当前系统已经具备完整的基础功能**，包括：
- ✅ CSV/Excel文件处理
- ✅ 批量处理模式
- ✅ 视频信息提取（轻量级）
- ✅ 中文列名支持
- ✅ 配置文件管理

**可选的增强功能**需要额外安装依赖，但不影响核心功能的使用。