#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频批量筛选工具
用于根据指定条件批量筛选视频文件 或 读取Excel中的视频链接进行筛选
"""

import os
import sys
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 依赖检查和可选导入
try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV(cv2) 未安装，本地视频处理功能将不可用")

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("NumPy 未安装，部分功能可能受限")

try:
    from tqdm import tqdm
except ImportError:
    logger.error("tqdm 未安装，这是必需的依赖")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests 未安装，缩略图下载功能将不可用")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.warning("pandas 未安装，Excel处理将降级为CSV模式")

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None
    logger.warning("yt-dlp 未安装，将使用轻量级替代方案")

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    torch = None
    CLIPProcessor = None
    CLIPModel = None
    logger.warning("PyTorch/transformers 未安装，内容检测功能将不可用")

try:
    from PIL import Image
    from io import BytesIO
except ImportError:
    Image = None
    BytesIO = None
    logger.warning("Pillow 未安装，图像处理功能将不可用")

# 导入核心处理模块
try:
    from utils.video_processor import VideoProcessor, FileProcessor, VideoInfo, ProcessingResult
    from utils.video_info_extractor import LightVideoExtractor
except ImportError as e:
    logger.error(f"核心模块导入失败: {e}")
    sys.exit(1)


class DependencyChecker:
    """依赖检查器"""

    def __init__(self):
        self.dependencies = {
            'yt_dlp': YoutubeDL is not None,
            'tqdm': 'tqdm' in sys.modules,
            'requests': requests is not None,
            'pandas': pd is not None,
            'opencv': cv2 is not None,
            'torch': torch is not None,
            'pil': Image is not None,
            'light_extractor': False
        }

        # 检查轻量级提取器
        try:
            self.light_extractor = LightVideoExtractor()
            self.dependencies['light_extractor'] = True
            logger.info("✓ 轻量级视频信息提取器已加载")
        except Exception as e:
            self.light_extractor = None
            logger.warning(f"轻量级视频信息提取器加载失败: {e}")

    def print_status(self):
        """打印依赖状态"""
        logger.info("依赖检查完成:")
        for name, available in self.dependencies.items():
            status = "✓" if available else "✗"
            logger.info(f"- {name}: {status}")
        logger.info("")

    def is_available(self, name: str) -> bool:
        """检查特定依赖是否可用"""
        return self.dependencies.get(name, False)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.processor = VideoProcessor(self.config)

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"成功加载配置文件: {config_path}")
                return config
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                sys.exit(1)
        else:
            # 默认配置
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "columns": {
                "link": "vediolink",
                "title": "video_title",
                "duration": "duration",
                "resolution": "max_resolution",
                "resolution_ok": "resolution_ok",
                "violence_blood": "has_violence_blood",
                "smoking": "has_smoking",
                "overall_pass": "overall_pass",
                "notes": "notes",
                "error": "error_message"
            },
            "filters": {
                "min_resolution": [1920, 1080],
                "min_duration": 10,
                "max_duration": 3600,
                "min_size": 5,
                "max_size": 2000
            },
            "video_extensions": [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"],
            "processing": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "timeout": 30
            }
        }


class ContentDetector:
    """基于CLIP的内容检测器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = self._check_dependencies()

        if self.enabled:
            self._initialize_model()
        else:
            logger.warning("未检测到CLIP相关依赖，内容识别将被跳过")
            self._setup_fallback_config()

    def _check_dependencies(self) -> bool:
        """检查依赖"""
        return all([
            torch is not None,
            CLIPProcessor is not None,
            CLIPModel is not None,
            Image is not None,
            requests is not None
        ])

    def _initialize_model(self):
        """初始化模型"""
        try:
            prefer_device = self.config.get('device', {}).get('prefer', 'cuda')
            self.device = 'cuda' if (prefer_device == 'cuda' and torch.cuda.is_available()) else 'cpu'

            model_id = self.config.get('detector', {}).get('clip_model_id', 'openai/clip-vit-base-patch32')
            logger.info(f"加载CLIP模型: {model_id} on {self.device}")

            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model.to(self.device)

            # 标签与阈值配置
            detector_config = self.config.get('detector', {})
            self.enabled_labels = detector_config.get('enabled_labels', ['blood', 'gore', 'violence', 'smoking'])
            self.thresholds = detector_config.get('clip_thresholds', {
                'blood': 0.28,
                'gore': 0.30,
                'violence': 0.30,
                'smoking': 0.32
            })
            self.max_thumbs = detector_config.get('thumbnail_samples', 5)

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self.enabled = False
            self._setup_fallback_config()

    def _setup_fallback_config(self):
        """设置降级配置"""
        self.model = None
        self.processor = None
        self.device = 'cpu'
        self.enabled_labels = ['blood', 'gore', 'violence', 'smoking']
        self.thresholds = {
            'blood': 0.28,
            'gore': 0.30,
            'violence': 0.30,
            'smoking': 0.32
        }
        self.max_thumbs = 5

    def _compose_prompts(self) -> Dict[str, List[str]]:
        """为每个标签构造提示语"""
        prompts = {
            'blood': ["a photo with blood", "bloody scene", "blood stains"],
            'gore': ["gory scene", "graphic violence", "gore"],
            'violence': ["violent scene", "people fighting", "assault"],
            'smoking': ["person smoking", "cigarette", "cigar"]
        }
        return {label: prompts.get(label, [label]) for label in self.enabled_labels}

    def _clip_scores(self, images, texts) -> List[List[float]]:
        """计算CLIP图文相似度"""
        if not self.enabled:
            return []

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1).detach().cpu().tolist()
        return probs

    def detect_from_thumbnails(self, thumb_urls: List[str]) -> Dict[str, bool]:
        """对多张缩略图进行检测"""
        if not self.enabled:
            return {label: False for label in self.enabled_labels}

        if not thumb_urls:
            return {label: False for label in self.enabled_labels}

        # 读取图片
        images = []
        for url in thumb_urls[:self.max_thumbs]:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                images.append(img)
            except Exception as e:
                logger.debug(f"下载缩略图失败: {url}, {e}")

        if not images:
            return {label: False for label in self.enabled_labels}

        # 计算得分
        label_texts = self._compose_prompts()
        results = {}
        for label, texts in label_texts.items():
            probs = self._clip_scores(images, texts)
            try:
                max_prob = max((max(row) for row in probs), default=0.0)
            except Exception:
                max_prob = 0.0

            thr = float(self.thresholds.get(label, 0.3))
            results[label] = bool(max_prob >= thr)

        return results


def format_error_message(error_text: str) -> str:
    """格式化错误信息"""
    if not error_text:
        return error_text

    # 清理ANSI代码
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', error_text)
    color_codes = re.compile(r'\[[\d;]*m')
    cleaned = color_codes.sub('', cleaned)

    # 常见错误类型的简化映射
    error_patterns = {
        r'.*Unsupported URL.*': '不支持的链接格式',
        r'.*Video unavailable.*': '视频不可用',
        r'.*Private video.*': '私有视频，无法访问',
        r'.*This video is not available.*': '视频不可用',
        r'.*HTTP Error 404.*': '视频不存在(404错误)',
        r'.*HTTP Error 403.*': '访问被拒绝(403错误)',
        r'.*Connection.*timeout.*': '网络连接超时',
        r'.*Unable to download.*': '下载失败',
        r'.*Sign in to confirm.*': '需要登录验证',
        r'.*This video has been removed.*': '视频已被删除',
        r'.*This video is private.*': '私有视频',
        r'.*This video is unavailable.*': '视频不可用',
        r'.*Age-restricted.*': '年龄限制视频',
        r'.*Premieres.*': '首映视频，暂不可用',
        r'.*Live stream.*': '直播流，暂不支持',
        r'.*yt-dlp.*not.*install.*': 'yt-dlp未安装，无法提取视频信息',
    }

    # 尝试匹配并简化错误信息
    for pattern, simplified in error_patterns.items():
        if re.search(pattern, cleaned, re.IGNORECASE):
            return simplified

    # 如果没有匹配到特定模式，返回清理后的原始错误（截取前100个字符）
    if len(cleaned) > 100:
        return cleaned[:97] + "..."

    return cleaned


def extract_video_info(url: str, cookies_path: Optional[str] = None,
                      max_retries: int = 3, retry_delay: float = 1.0) -> VideoInfo:
    """提取视频元信息"""

    # 如果yt-dlp不可用，使用轻量级提取器
    if YoutubeDL is None:
        return _extract_with_light_extractor(url)

    return _extract_with_yt_dlp(url, cookies_path, max_retries, retry_delay)


def _extract_with_light_extractor(url: str) -> VideoInfo:
    """使用轻量级提取器"""
    checker = DependencyChecker()
    if not checker.light_extractor:
        return VideoInfo(
            url=url,
            error='yt-dlp 和轻量级提取器都不可用'
        )

    try:
        info = checker.light_extractor.extract_video_info(url)

        # 转换时长格式
        duration_seconds = None
        duration_str = info.get('duration', '')
        if duration_str and ':' in duration_str:
            try:
                parts = duration_str.split(':')
                if len(parts) == 2:
                    duration_seconds = int(parts[0]) * 60 + int(parts[1])
            except (ValueError, TypeError):
                pass

        # 解析分辨率
        width, height = 0, 0
        resolution = info.get('resolution', '')
        if resolution and 'x' in resolution:
            try:
                w, h = resolution.split('x')
                width, height = int(w), int(h)
            except (ValueError, TypeError):
                pass
        elif '1080' in resolution:
            width, height = 1920, 1080
        elif '720' in resolution:
            width, height = 1280, 720

        return VideoInfo(
            url=url,
            title=info.get('title', ''),
            duration=duration_seconds,
            width=width,
            height=height,
            max_resolution=f"{width}x{height}" if width and height else resolution,
            error=info.get('error', '')
        )

    except Exception as e:
        return VideoInfo(
            url=url,
            error=f'轻量级提取器失败: {str(e)}'
        )


def _extract_with_yt_dlp(url: str, cookies_path: Optional[str], max_retries: int, retry_delay: float) -> VideoInfo:
    """使用yt-dlp提取"""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'noplaylist': True,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'socket_timeout': 30,
        'retries': 2,
    }

    if cookies_path and Path(cookies_path).exists():
        ydl_opts['cookiefile'] = cookies_path

    last_error = None
    for attempt in range(max_retries):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            # 解析最大分辨率
            max_h, max_w = 0, 0
            formats = info.get('formats') or []
            for f in formats:
                h = f.get('height') or 0
                w = f.get('width') or 0
                if h and h > max_h:
                    max_h, max_w = h, w or max_w

            thumbnails = info.get('thumbnails') or []
            thumb_urls = [t.get('url') for t in sorted(thumbnails, key=lambda x: x.get('height', 0), reverse=True) if t.get('url')]

            return VideoInfo(
                url=url,
                title=info.get('title', ''),
                duration=info.get('duration'),
                width=max_w,
                height=max_h,
                max_resolution=f"{max_w}x{max_h}" if max_h else '',
                thumb_urls=thumb_urls
            )

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"提取视频信息失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"提取视频信息最终失败: {str(e)}")

    return VideoInfo(
        url=url,
        error=format_error_message(str(last_error))
    )


class VideoFilter:
    """本地视频筛选器"""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.processor = config_manager.processor

    def filter_videos(self, input_dir: str, output_dir: Optional[str] = None) -> List[Path]:
        """筛选本地视频文件"""
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            logger.error(f"输入目录不存在: {input_dir}")
            return []

        # 检查OpenCV
        if cv2 is None:
            logger.error("OpenCV(cv2) 未安装，无法处理本地视频文件")
            return []

        # 获取所有视频文件
        video_extensions = self.config.get('video_extensions', ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'])
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(input_path.glob(f"**/*{ext}")))

        logger.info(f"找到 {len(video_files)} 个视频文件")

        # 筛选视频
        filtered_videos = []
        filters = self.config.get('filters', {})

        for video_path in tqdm(video_files, desc="筛选视频"):
            if self._check_video(video_path, filters):
                filtered_videos.append(video_path)

                # 如果指定了输出目录，则复制文件
                if output_dir:
                    self._copy_video(video_path, output_dir)

        logger.info(f"筛选出 {len(filtered_videos)} 个符合条件的视频")
        return filtered_videos

    def _check_video(self, video_path: Path, filters: Dict[str, Any]) -> bool:
        """检查单个视频文件"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"无法打开视频: {video_path}")
                return False

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            # 检查分辨率
            min_width, min_height = filters.get('min_resolution', [1920, 1080])
            if width < min_width or height < min_height:
                cap.release()
                return False

            # 检查时长
            min_duration = filters.get('min_duration', 0)
            max_duration = filters.get('max_duration', 3600)
            if duration < min_duration or duration > max_duration:
                cap.release()
                return False

            # 检查文件大小
            min_size = filters.get('min_size', 0)
            max_size = filters.get('max_size', float('inf'))
            if min_size > 0 or max_size < float('inf'):
                file_size = video_path.stat().st_size / (1024 * 1024)  # MB
                if file_size < min_size or file_size > max_size:
                    cap.release()
                    return False

            cap.release()
            return True

        except Exception as e:
            logger.error(f"检查视频时出错 {video_path}: {e}")
            return False

    def _copy_video(self, video_path: Path, output_dir: str):
        """复制视频文件"""
        try:
            output_path = Path(output_dir) / video_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(video_path, output_path)
            logger.debug(f"已复制: {video_path} -> {output_path}")
        except Exception as e:
            logger.error(f"复制视频时出错 {video_path}: {e}")


class DataFileProcessor:
    """数据文件处理器（Excel/CSV）"""

    def __init__(self, config_manager: ConfigManager, dependency_checker: DependencyChecker):
        self.config = config_manager.config
        self.processor = config_manager.processor
        self.dependency_checker = dependency_checker
        self.detector = ContentDetector(self.config) if self.dependency_checker.is_available('torch') else None

    def process_file(self, file_path: str, link_column: str) -> bool:
        """处理单个数据文件"""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error(f"文件不存在: {file_path}")
                return False

            # 根据文件类型选择处理方法
            if file_path_obj.suffix.lower() == '.csv':
                return self._process_csv(file_path, link_column)
            else:
                return self._process_excel(file_path, link_column)

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def _process_csv(self, csv_path: str, link_column: str) -> bool:
        """处理CSV文件"""
        logger.info(f"处理CSV文件: {csv_path}")

        # 读取CSV
        rows, fieldnames = FileProcessor.read_csv(csv_path, link_column)
        if not rows:
            logger.warning(f"CSV文件为空: {csv_path}")
            return True

        # 处理数据
        processed_rows = self.processor.process_data_rows(
            rows, link_column,
            lambda url: extract_video_info(
                url,
                self.config.get('auth', {}).get('cookies_path'),
                self.config.get('processing', {}).get('max_retries', 3),
                self.config.get('processing', {}).get('retry_delay', 1.0)
            ),
            self.detector
        )

        # 获取输出列名
        output_columns = FileProcessor.get_output_columns(fieldnames, self.processor)

        # 写回文件
        FileProcessor.write_csv(csv_path, processed_rows, output_columns)
        logger.info(f"CSV处理完成: {csv_path}")
        return True

    def _process_excel(self, excel_path: str, link_column: str) -> bool:
        """处理Excel文件"""
        if pd is None:
            logger.error("pandas未安装，无法处理Excel文件")
            return False

        logger.info(f"处理Excel文件: {excel_path}")

        try:
            # 读取Excel
            df = pd.read_excel(excel_path, dtype=str)
            if link_column not in df.columns:
                logger.error(f"Excel中未找到链接列: {link_column}")
                return False

            # 转换为字典列表
            rows = df.to_dict('records')

            # 处理数据
            processed_rows = self.processor.process_data_rows(
                rows, link_column,
                lambda url: extract_video_info(
                    url,
                    self.config.get('auth', {}).get('cookies_path'),
                    self.config.get('processing', {}).get('max_retries', 3),
                    self.config.get('processing', {}).get('retry_delay', 1.0)
                ),
                self.detector
            )

            # 转换回DataFrame
            result_df = pd.DataFrame(processed_rows)

            # 写回Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                result_df.to_excel(writer, index=False)

            logger.info(f"Excel处理完成: {excel_path}")
            return True

        except Exception as e:
            logger.error(f"处理Excel文件失败: {e}")
            return False


class BatchProcessor:
    """批量文件处理器"""

    def __init__(self, config_manager: ConfigManager, dependency_checker: DependencyChecker):
        self.config = config_manager.config
        self.file_processor = DataFileProcessor(config_manager, dependency_checker)

    def process_batch(self, data_dir: str, link_column: str) -> Dict[str, int]:
        """批量处理文件夹中的所有数据文件"""
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"数据文件夹不存在: {data_path}")
            return {'total': 0, 'success': 0, 'failed': 0}

        # 查找所有数据文件
        csv_files = list(data_path.glob('*.csv'))
        excel_files = list(data_path.glob('*.xlsx')) + list(data_path.glob('*.xls'))
        all_files = csv_files + excel_files

        if not all_files:
            logger.info(f"在 {data_path} 中未找到任何CSV或Excel文件")
            return {'total': 0, 'success': 0, 'failed': 0}

        logger.info(f"找到 {len(all_files)} 个文件待处理:")
        for file_path in all_files:
            logger.info(f"  - {file_path.name}")

        # 处理统计
        stats = {'total': len(all_files), 'success': 0, 'failed': 0}

        # 处理每个文件
        for file_path in all_files:
            logger.info(f"\n开始处理文件: {file_path.name}")
            try:
                success = self.file_processor.process_file(str(file_path), link_column)
                if success:
                    stats['success'] += 1
                    logger.info(f"✓ 文件处理完成: {file_path.name}")
                else:
                    stats['failed'] += 1
                    logger.error(f"✗ 文件处理失败: {file_path.name}")
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"✗ 文件处理失败: {file_path.name}, 错误: {str(e)}")

        logger.info(f"\n批量处理完成，共处理 {stats['total']} 个文件，成功 {stats['success']} 个，失败 {stats['failed']} 个")
        return stats


class FolderWatcher:
    """文件夹监控器"""

    def __init__(self, config_manager: ConfigManager, dependency_checker: DependencyChecker):
        self.config = config_manager.config
        self.file_processor = DataFileProcessor(config_manager, dependency_checker)
        self.processed_files: set[str] = set()

    def watch_folder(self, data_dir: str, link_column: str):
        """监控文件夹，自动处理新文件"""
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"数据文件夹不存在: {data_path}")
            return

        logger.info(f"开始监控文件夹: {data_path}")
        logger.info("请将CSV或Excel文件放入该文件夹，程序将自动处理")
        logger.info("按 Ctrl+C 停止监控")

        try:
            # 先处理现有文件
            self._process_existing_files(data_path, link_column)

            # 开始监控
            while True:
                self._check_for_new_files(data_path, link_column)
                time.sleep(2)  # 每2秒检查一次

        except KeyboardInterrupt:
            logger.info("\n监控已停止")

    def _process_existing_files(self, data_path: Path, link_column: str):
        """处理现有文件"""
        csv_files = list(data_path.glob('*.csv'))
        excel_files = list(data_path.glob('*.xlsx')) + list(data_path.glob('*.xls'))
        existing_files = csv_files + excel_files

        if existing_files:
            logger.info(f"发现 {len(existing_files)} 个现有文件，开始处理...")
            for file_path in existing_files:
                self._process_file_safe(file_path, link_column)

    def _check_for_new_files(self, data_path: Path, link_column: str):
        """检查新文件"""
        csv_files = list(data_path.glob('*.csv'))
        excel_files = list(data_path.glob('*.xlsx')) + list(data_path.glob('*.xls'))
        current_files = {str(f) for f in csv_files + excel_files}

        # 找到新文件
        new_files = current_files - self.processed_files

        for file_path_str in new_files:
            file_path = Path(file_path_str)
            logger.info(f"\n检测到新文件: {file_path.name}")
            self._process_file_safe(file_path, link_column)

    def _process_file_safe(self, file_path: Path, link_column: str):
        """安全地处理单个文件"""
        try:
            success = self.file_processor.process_file(str(file_path), link_column)
            if success:
                logger.info(f"✓ 文件处理完成: {file_path.name}")
                self.processed_files.add(str(file_path))
            else:
                logger.error(f"✗ 文件处理失败: {file_path.name}")
        except Exception as e:
            logger.error(f"✗ 文件处理失败: {file_path.name}, 错误: {str(e)}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频批量筛选工具')
    parser.add_argument('-i', '--input', help='输入视频目录（本地模式）')
    parser.add_argument('-o', '--output', help='输出目录（可选，仅本地模式）')
    parser.add_argument('-c', '--config', help='配置文件路径')

    # Excel/CSV模式
    parser.add_argument('--excel', help='Excel/CSV文件路径（链接批量模式）')
    parser.add_argument('--link-column', default='vediolink', help='链接列名，默认vediolink')

    # 批量处理模式
    parser.add_argument('--batch', action='store_true', help='批量处理模式，自动处理data文件夹中的所有CSV/Excel文件')
    parser.add_argument('--data-dir', default='data', help='数据文件夹路径，默认为data')

    # 监控模式
    parser.add_argument('--watch', action='store_true', help='监控模式，持续监控data文件夹中的新文件')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查依赖
    dep_checker = DependencyChecker()
    dep_checker.print_status()

    # 加载配置
    config_manager = ConfigManager(args.config)

    # 批量处理模式优先
    if args.batch:
        logger.info(f"批量处理模式：处理文件夹 {args.data_dir}")
        batch_processor = BatchProcessor(config_manager, dep_checker)
        stats = batch_processor.process_batch(args.data_dir, args.link_column)
        return

    # 监控模式
    if args.watch:
        logger.info(f"监控模式：监控文件夹 {args.data_dir}")
        watcher = FolderWatcher(config_manager, dep_checker)
        watcher.watch_folder(args.data_dir, args.link_column)
        return

    # 单文件模式
    if args.excel:
        logger.info(f"单文件模式：处理文件 {args.excel}")
        file_processor = DataFileProcessor(config_manager, dep_checker)
        success = file_processor.process_file(args.excel, args.link_column)
        if success:
            logger.info(f"✓ 文件处理完成: {args.excel}")
        else:
            logger.error(f"✗ 文件处理失败: {args.excel}")
        return

    # 本地目录模式
    if not args.input:
        logger.error("请指定输入文件或目录，或使用 --batch 进行批量处理")
        logger.info("使用示例:")
        logger.info("  批量处理: python video_filter.py --batch")
        logger.info("  监控模式: python video_filter.py --watch")
        logger.info("  单文件模式: python video_filter.py --excel file.xlsx")
        logger.info("  本地模式: python video_filter.py -i input_dir -o output_dir")
        return

    # 处理本地视频文件
    video_filter = VideoFilter(config_manager)
    filtered_videos = video_filter.filter_videos(args.input, args.output)

    if filtered_videos:
        logger.info(f"筛选完成，共找到 {len(filtered_videos)} 个符合条件的视频")
    else:
        logger.info("没有找到符合条件的视频")


if __name__ == "__main__":
    main()