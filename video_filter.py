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

# 依赖检查和可选导入
try:
    import cv2
except ImportError:
    cv2 = None
    print("警告: OpenCV(cv2) 未安装，本地视频处理功能将不可用")

try:
    import numpy as np
except ImportError:
    np = None
    print("警告: NumPy 未安装，部分功能可能受限")

try:
    from tqdm import tqdm
except ImportError:
    print("错误: tqdm 未安装，这是必需的依赖")
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None
    print("警告: requests 未安装，缩略图下载功能将不可用")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("警告: pandas 未安装，Excel处理将降级为CSV模式")

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None
    print("警告: yt-dlp 未安装，将使用轻量级替代方案")

# 导入轻量级视频信息提取器
try:
    from utils.video_info_extractor import LightVideoExtractor
    light_extractor = LightVideoExtractor()
    print("✓ 轻量级视频信息提取器已加载")
except ImportError:
    light_extractor = None
    print("警告: 轻量级视频信息提取器加载失败")

try:
    from PIL import Image
    from io import BytesIO
except ImportError:
    Image = None
    BytesIO = None
    print("警告: Pillow 未安装，图像处理功能将不可用")

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    torch = None
    CLIPProcessor = None
    CLIPModel = None
    print("警告: PyTorch/transformers 未安装，内容检测功能将不可用")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_ansi_codes(text):
    """
    清理文本中的ANSI颜色代码
    
    Args:
        text (str): 包含ANSI代码的文本
        
    Returns:
        str: 清理后的文本
    """
    if not text:
        return text
    
    # ANSI颜色代码的正则表达式
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    # 清理ANSI代码
    cleaned = ansi_escape.sub('', text)
    
    # 清理形如 [0;31m 的颜色代码
    color_codes = re.compile(r'\[[\d;]*m')
    cleaned = color_codes.sub('', cleaned)
    
    return cleaned


def format_error_message(error_text):
    """
    格式化错误信息，使其更清晰易读
    
    Args:
        error_text (str): 原始错误信息
        
    Returns:
        str: 格式化后的错误信息
    """
    if not error_text:
        return error_text
    
    # 先清理ANSI代码
    cleaned = clean_ansi_codes(error_text)
    
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
    }
    
    # 尝试匹配并简化错误信息
    for pattern, simplified in error_patterns.items():
        if re.search(pattern, cleaned, re.IGNORECASE):
            return simplified
    
    # 如果没有匹配到特定模式，返回清理后的原始错误（截取前100个字符）
    if len(cleaned) > 100:
        return cleaned[:97] + "..."
    
    return cleaned

print(f"依赖检查完成:")
print(f"- yt-dlp: {'✓' if YoutubeDL else '✗'}")
print(f"- tqdm: {'✓' if 'tqdm' in sys.modules else '✗'}")
print(f"- requests: {'✓' if requests else '✗'}")
print(f"- pandas: {'✓' if pd else '✗'}")
print(f"- opencv: {'✓' if cv2 else '✗'}")
print(f"- torch: {'✓' if torch else '✗'}")
print(f"- PIL: {'✓' if Image else '✗'}")
print()

class VideoFilter:
    """视频筛选器主类（本地目录模式）"""
    
    def __init__(self, config_path=None):
        """
        初始化视频筛选器
        
        Args:
            config_path: 配置文件路径，默认为None
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
        else:
            self.config = {}
    
    def load_config(self, config_path):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    def filter_videos(self, input_dir, output_dir=None, filters=None):
        """
        根据条件筛选视频（本地目录模式）
        """
        if filters is None:
            filters = self.config.get('filters', {})
        
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            logger.error(f"输入目录不存在: {input_dir}")
            return []
        
        # 获取所有视频文件
        video_extensions = self.config.get('video_extensions', ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'])
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(input_path.glob(f"**/*{ext}")))
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        # 筛选视频
        filtered_videos = []
        for video_path in tqdm(video_files, desc="筛选视频"):
            if self._check_video(video_path, filters):
                filtered_videos.append(video_path)
                
                # 如果指定了输出目录，则复制文件
                if output_dir:
                    self._copy_video(video_path, output_dir)
        
        logger.info(f"筛选出 {len(filtered_videos)} 个符合条件的视频")
        return filtered_videos
    
    def _check_video(self, video_path, filters):
        """检查视频是否符合筛选条件（本地文件）"""
        # 新增：OpenCV可用性检查
        if cv2 is None:
            logger.error("OpenCV(cv2) 未安装，无法检查本地视频，请先安装 opencv-python 依赖或仅使用链接模式。")
            return False
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
            if 'min_resolution' in filters:
                min_width, min_height = filters['min_resolution']
                if width < min_width or height < min_height:
                    cap.release()
                    return False
            
            # 检查时长
            if 'min_duration' in filters and duration < filters['min_duration']:
                cap.release()
                return False
            if 'max_duration' in filters and duration > filters['max_duration']:
                cap.release()
                return False
            
            # 检查文件大小
            if 'min_size' in filters or 'max_size' in filters:
                file_size = video_path.stat().st_size / (1024 * 1024)  # MB
                if 'min_size' in filters and file_size < filters['min_size']:
                    cap.release()
                    return False
                if 'max_size' in filters and file_size > filters['max_size']:
                    cap.release()
                    return False
            
            cap.release()
            return True
            
        except Exception as e:
            logger.error(f"检查视频时出错 {video_path}: {e}")
            return False
    
    def _copy_video(self, video_path, output_dir):
        """复制视频到输出目录（本地文件）"""
        try:
            output_path = Path(output_dir) / video_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(video_path, output_path)
            logger.debug(f"已复制: {video_path} -> {output_path}")
        except Exception as e:
            logger.error(f"复制视频时出错 {video_path}: {e}")
    
    def check_video(self, video_url):
        """
        检查在线视频链接
        
        Args:
            video_url: 视频链接
            
        Returns:
            dict: 包含status, reason, duration, title, description等信息
        """
        logger.debug(f"开始检查视频: {video_url}")
        
        if not YoutubeDL:
            return {
                'status': 'error',
                'reason': 'yt-dlp未安装，无法处理在线视频',
                'duration': '',
                'title': '',
                'description': ''
            }
        
        try:
            # 配置yt-dlp选项
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,  # 只获取信息，不下载
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                # 获取视频信息
                info = ydl.extract_info(video_url, download=False)
                
                # 提取基本信息
                title = info.get('title', '')
                description = info.get('description', '')
                duration = info.get('duration', 0)
                
                # 格式化时长
                if duration:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    if hours > 0:
                        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration_str = f"{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = ''
                
                # 应用筛选条件
                filters = self.config.get('filters', {})
                
                # 检查时长筛选
                min_duration = filters.get('min_duration', 0)
                max_duration = filters.get('max_duration', float('inf'))
                
                if min_duration > 0 or max_duration < float('inf'):
                    
                    if duration < min_duration:
                        return {
                            'status': 'filtered',
                            'reason': f'视频时长过短: {duration_str}',
                            'duration': duration_str,
                            'title': title,
                            'description': description
                        }
                    
                    if duration > max_duration:
                        return {
                            'status': 'filtered',
                            'reason': f'视频时长过长: {duration_str}',
                            'duration': duration_str,
                            'title': title,
                            'description': description
                        }
                
                # 检查关键词筛选
                if 'keywords' in filters:
                    exclude_keywords = filters['keywords'].get('exclude', [])
                    include_keywords = filters['keywords'].get('include', [])
                    
                    # 检查排除关键词
                    for keyword in exclude_keywords:
                        if keyword.lower() in title.lower() or keyword.lower() in description.lower():
                            return {
                                'status': 'filtered',
                                'reason': f'包含排除关键词: {keyword}',
                                'duration': duration_str,
                                'title': title,
                                'description': description
                            }
                    
                    # 检查包含关键词
                    if include_keywords:
                        found_keyword = False
                        for keyword in include_keywords:
                            if keyword.lower() in title.lower() or keyword.lower() in description.lower():
                                found_keyword = True
                                break
                        
                        if not found_keyword:
                            return {
                                'status': 'filtered',
                                'reason': '未包含必需关键词',
                                'duration': duration_str,
                                'title': title,
                                'description': description
                            }
                
                # 通过所有筛选条件
                return {
                    'status': 'passed',
                    'reason': '通过筛选',
                    'duration': duration_str,
                    'title': title,
                    'description': description
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'reason': f'获取视频信息失败: {str(e)}',
                'duration': '',
                'title': '',
                'description': ''
            }

# ========================= 新增：Excel/CSV 链接处理与内容识别 =========================

# 实用函数：安全读取配置

def get_cfg(config: dict, path: list, default=None):
    """安全读取配置，支持默认值"""
    cur = config
    try:
        for p in path:
            cur = cur[p]
        return cur
    except Exception:
        return default

# 新增：备注拼接函数

def append_note(orig_note: str, msg: str) -> str:
    """将提示信息追加到备注字段，避免重复添加相同信息"""
    if not orig_note or str(orig_note).strip() == "" or str(orig_note).lower() == "nan":
        return msg
    
    # 检查是否已经存在相同的消息，避免重复添加
    orig_note_str = str(orig_note)
    if msg in orig_note_str:
        return orig_note_str
    
    return f"{orig_note_str}; {msg}"

class ContentDetector:
    """基于CLIP的零样本内容识别（缩略图MVP，支持依赖缺失时降级）"""
    def __init__(self, config: dict):
        self.config = config or {}
        self.enabled = (torch is not None and CLIPProcessor is not None and CLIPModel is not None and Image is not None and requests is not None)
        if not self.enabled:
            logger.warning("未检测到CLIP相关依赖(torch/transformers/Pillow/requests)，内容识别将被跳过。")
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
            return
        prefer_device = get_cfg(self.config, ['device', 'prefer'], 'cuda')
        self.device = 'cuda' if (prefer_device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        model_id = get_cfg(self.config, ['detector', 'clipModelId'], 'openai/clip-vit-base-patch32')
        logger.info(f"加载CLIP模型: {model_id} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.to(self.device)
        
        # 标签与阈值
        self.enabled_labels = get_cfg(self.config, ['detector', 'enabledLabels'], [
            'blood', 'gore', 'violence', 'smoking'
        ])
        self.thresholds = get_cfg(self.config, ['detector', 'clipThresholds'], {
            'blood': 0.28,
            'gore': 0.30,
            'violence': 0.30,
            'smoking': 0.32
        })
        self.max_thumbs = int(get_cfg(self.config, ['detector', 'thumbnailSamples'], 5))

    def _compose_prompts(self):
        """为每个标签构造提示语"""
        prompts = {
            'blood': ["a photo with blood", "bloody scene", "blood stains"],
            'gore': ["gory scene", "graphic violence", "gore"],
            'violence': ["violent scene", "people fighting", "assault"],
            'smoking': ["person smoking", "cigarette", "cigar"]
        }
        label_texts = {label: prompts.get(label, [label]) for label in self.enabled_labels}
        return label_texts

    def _clip_scores(self, images, texts):
        """计算CLIP图文相似度，返回Python list（避免强依赖numpy）"""
        if not self.enabled:
            return []
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # [N_img, N_txt]
            probs = torch.softmax(logits_per_image, dim=-1).detach().cpu().tolist()  # list[list[float]]
        return probs

    def detect_from_thumbnails(self, thumb_urls: list):
        """对多张缩略图进行检测，返回标签->bool；当未启用时全部返回False"""
        if not self.enabled:
            return {label: False for label in ['blood', 'gore', 'violence', 'smoking']}
        if not thumb_urls:
            return {label: False for label in self.enabled_labels}
        # 读取图片
        images = []
        for url in thumb_urls[: self.max_thumbs]:
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
            probs = self._clip_scores(images, texts)  # list[list[float]]
            try:
                max_prob = max((max(row) for row in probs), default=0.0)
            except Exception:
                max_prob = 0.0
            thr = float(self.thresholds.get(label, 0.3))
            results[label] = bool(max_prob >= thr)
        return results


def extract_video_info(url: str, cookies_path: str = None, max_retries: int = 3, retry_delay: float = 1.0):
    """提取视频元信息，优先使用yt-dlp，不可用时使用轻量级提取器
    
    Args:
        url: 视频链接
        cookies_path: cookies文件路径
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
    
    返回: {title, duration, max_resolution, width, height, thumbnails(list), error}
    """
    # 如果yt-dlp不可用，使用轻量级提取器
    if YoutubeDL is None:
        if light_extractor is not None:
            try:
                info = light_extractor.extract_video_info(url)
                # 转换格式以匹配原有接口
                duration_str = info.get('duration', '')
                duration_seconds = None
                if duration_str and ':' in duration_str:
                    try:
                        parts = duration_str.split(':')
                        if len(parts) == 2:
                            duration_seconds = int(parts[0]) * 60 + int(parts[1])
                    except:
                        pass
                
                # 解析分辨率
                resolution = info.get('resolution', '')
                width, height = 0, 0
                if resolution and 'x' in resolution:
                    try:
                        w, h = resolution.split('x')
                        width, height = int(w), int(h)
                    except:
                        pass
                elif '1080' in resolution:
                    width, height = 1920, 1080
                elif '720' in resolution:
                    width, height = 1280, 720
                
                return {
                    'title': info.get('title'),
                    'duration': duration_seconds,
                    'max_resolution': f"{width}x{height}" if width and height else resolution,
                    'width': width,
                    'height': height,
                    'thumb_urls': [],  # 轻量级提取器暂不支持缩略图
                    'error': info.get('error') or None
                }
            except Exception as e:
                return {
                    'title': None,
                    'duration': None,
                    'max_resolution': None,
                    'width': 0,
                    'height': 0,
                    'thumb_urls': [],
                    'error': f'轻量级提取器失败: {str(e)}'
                }
        else:
            return {
                'title': None,
                'duration': None,
                'max_resolution': None,
                'width': 0,
                'height': 0,
                'thumb_urls': [],
                'error': 'yt-dlp 和轻量级提取器都不可用'
            }
    
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'noplaylist': True,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'socket_timeout': 30,  # 30秒超时
        'retries': 2,  # yt-dlp内部重试
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
            
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'max_resolution': f"{max_w}x{max_h}" if max_h else None,
                'width': max_w,
                'height': max_h,
                'thumb_urls': thumb_urls,
                'error': None
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"提取视频信息失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay * (attempt + 1))  # 递增延迟
            else:
                logger.error(f"提取视频信息最终失败: {str(e)}")
    
    return {
        'title': None,
        'duration': None,
        'max_resolution': None,
        'width': 0,
        'height': 0,
        'thumb_urls': [],
        'error': str(last_error)
    }


# 新增：CSV 处理函数（在未安装pandas时也可运行）

def process_csv_links(csv_path: str, link_column: str, config_path: str = None):
    import csv
    # 加载配置
    config = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"读取配置失败，将使用默认配置: {e}")
            config = {}
    min_width, min_height = tuple(get_cfg(config, ['filters', 'minResolution'], [1920, 1080]))
    cookies_path = get_cfg(config, ['auth', 'cookiesPath'], None)

    # 读取CSV
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if link_column not in fieldnames:
            raise ValueError(f"CSV中未找到链接列: {link_column}")
        rows = list(reader)

    # 预备输出列 - 根据用户需求重新设计
    out_cols = [
        '清晰度是否1080', '是否有吸烟', '是否有血腥暴力', '是否通过全部筛选', 
        '视频标题', '视频时长', '最大分辨率', '备注', '错误信息'
    ]
    fieldnames_extended = fieldnames + [c for c in out_cols if c not in fieldnames]

    # 初始化内容检测器
    detector = ContentDetector(config)

    for idx, row in enumerate(tqdm(rows, desc='处理链接')):
        url = str(row.get(link_column, '')).strip()
        # 初始化输出字段
        for c in out_cols:
            row.setdefault(c, None)
        if not url.startswith(('http://', 'https://')):
            row['错误信息'] = '无效链接'
            row['是否通过全部筛选'] = False
            continue
        info = extract_video_info(url, cookies_path)
        row['视频标题'] = info.get('title')
        row['视频时长'] = info.get('duration')
        row['最大分辨率'] = info.get('max_resolution')
        if info.get('error'):
            row['错误信息'] = format_error_message(info['error'])
            row['是否通过全部筛选'] = False
            continue
        # 分辨率判定 - 检查是否为1080p
        try:
            width = int(info.get('width') or 0) if info.get('width') not in [None, ''] else 0
            height = int(info.get('height') or 0) if info.get('height') not in [None, ''] else 0
            res_ok = (width >= min_width and height >= min_height)
        except (ValueError, TypeError):
            res_ok = False
        row['清晰度是否1080'] = bool(res_ok)
        if not res_ok:
            row['是否通过全部筛选'] = False
            row['备注'] = append_note(row.get('备注'), '分辨率不达标')
            continue
        # 内容识别（可降级）
        if not detector.enabled:
            row['是否有血腥暴力'] = False
            row['是否有吸烟'] = False
            row['是否通过全部筛选'] = bool(res_ok)
            row['备注'] = append_note(row.get('备注'), '内容检测未启用')
        else:
            detect_res = detector.detect_from_thumbnails(info.get('thumb_urls') or [])
            has_violence_blood = bool(detect_res.get('violence', False) or detect_res.get('blood', False) or detect_res.get('gore', False))
            has_smoking = bool(detect_res.get('smoking', False))
            row['是否有血腥暴力'] = has_violence_blood
            row['是否有吸烟'] = has_smoking
            row['是否通过全部筛选'] = bool(res_ok and (not has_violence_blood) and (not has_smoking))

    # 写回CSV（覆盖）- 使用临时文件避免权限问题
    import shutil
    temp_path = csv_path + '.tmp'
    try:
        with open(temp_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_extended)
            writer.writeheader()
            writer.writerows(rows)
        # 替换原文件
        shutil.move(temp_path, csv_path)
        logger.info(f"处理完成，结果已写回：{csv_path}")
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def process_excel_links(excel_path: str, link_column: str = 'vediolink', config_path: str = None):
    """读取Excel/CSV链接，按规则筛选并写回原文件"""
    # CSV 分支：不依赖 pandas
    lower = str(excel_path).lower()
    if lower.endswith('.csv'):
        return process_csv_links(excel_path, link_column, config_path)

    # Excel 分支：需要 pandas 和 openpyxl
    if pd is None:
        raise RuntimeError('未安装 pandas，无法处理 Excel。可将文件另存为 CSV 或先安装 pandas/openpyxl。')

    # 加载配置
    config = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"读取配置失败，将使用默认配置: {e}")
            config = {}
    min_width, min_height = tuple(get_cfg(config, ['filters', 'minResolution'], [1920, 1080]))
    cookies_path = get_cfg(config, ['auth', 'cookiesPath'], None)

    # 读取Excel
    df = pd.read_excel(excel_path)
    if link_column not in df.columns:
        raise ValueError(f"Excel中未找到链接列: {link_column}")

    # 预备输出列
    out_cols = [
        'title', 'duration', 'max_available_resolution', 'resolution_ok',
        'has_violence_blood', 'has_smoking', 'overall_pass', 'notes', 'error_message'
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = None

    # 初始化内容检测器
    detector = ContentDetector(config)

    links = df[link_column].astype(str).tolist()
    for idx, url in enumerate(tqdm(links, desc='处理链接')):
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            df.at[idx, 'error_message'] = '无效链接'
            df.at[idx, 'overall_pass'] = False
            continue
        info = extract_video_info(url, cookies_path)
        df.at[idx, '视频标题'] = info['title']
        df.at[idx, '视频时长'] = info['duration']
        df.at[idx, '最大分辨率'] = info['max_resolution']
        if info['error']:
            df.at[idx, '错误信息'] = format_error_message(info['error'])
            df.at[idx, '是否通过全部筛选'] = False
            continue
        # 分辨率判定 - 检查是否为1080p
        res_ok = (int(info.get('width') or 0) >= min_width and int(info.get('height') or 0) >= min_height)
        df.at[idx, '清晰度是否1080'] = bool(res_ok)
        if not res_ok:
            df.at[idx, '是否通过全部筛选'] = False
            df.at[idx, '备注'] = append_note(df.at[idx, '备注'], '分辨率不达标')
            continue
        # 内容识别（可降级）
        if not detector.enabled:
            df.at[idx, '是否有血腥暴力'] = False
            df.at[idx, '是否有吸烟'] = False
            df.at[idx, '是否通过全部筛选'] = bool(res_ok)
            df.at[idx, '备注'] = append_note(df.at[idx, '备注'], '内容检测未启用')
        else:
            detect_res = detector.detect_from_thumbnails(info['thumb_urls'])
            has_violence_blood = bool(detect_res.get('violence', False) or detect_res.get('blood', False) or detect_res.get('gore', False))
            has_smoking = bool(detect_res.get('smoking', False))
            df.at[idx, '是否有血腥暴力'] = has_violence_blood
            df.at[idx, '是否有吸烟'] = has_smoking
            df.at[idx, '是否通过全部筛选'] = bool(res_ok and (not has_violence_blood) and (not has_smoking))

    # 写回原文件（覆盖Excel）
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)
    logger.info(f"处理完成，结果已写回：{excel_path}")


def process_batch_files(data_dir: str = 'data', link_column: str = 'vediolink', config_path: str = None):
    """
    批量处理指定文件夹中的所有CSV/Excel文件
    
    Args:
        data_dir: 数据文件夹路径
        link_column: 链接列名
        config_path: 配置文件路径
    """
    import glob
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据文件夹不存在: {data_path}")
        return
    
    # 查找所有CSV和Excel文件
    csv_files = list(data_path.glob('*.csv'))
    excel_files = list(data_path.glob('*.xlsx')) + list(data_path.glob('*.xls'))
    
    all_files = csv_files + excel_files
    
    if not all_files:
        logger.info(f"在 {data_path} 中未找到任何CSV或Excel文件")
        return
    
    logger.info(f"找到 {len(all_files)} 个文件待处理:")
    for file_path in all_files:
        logger.info(f"  - {file_path.name}")
    
    # 处理每个文件
    for file_path in all_files:
        logger.info(f"\n开始处理文件: {file_path.name}")
        try:
            if file_path.suffix.lower() == '.csv':
                # 处理CSV文件
                process_csv_links(str(file_path), link_column, config_path)
            else:
                # 处理Excel文件
                process_excel_links(str(file_path), link_column, config_path)
            logger.info(f"✓ 文件处理完成: {file_path.name}")
        except Exception as e:
            logger.error(f"✗ 文件处理失败: {file_path.name}, 错误: {str(e)}")
    
    logger.info(f"\n批量处理完成，共处理 {len(all_files)} 个文件")


def watch_data_folder(data_dir: str = 'data', link_column: str = 'vediolink', config_path: str = None):
    """
    监控数据文件夹，自动处理新添加的文件
    
    Args:
        data_dir: 数据文件夹路径
        link_column: 链接列名
        config_path: 配置文件路径
    """
    import time
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据文件夹不存在: {data_path}")
        return
    
    logger.info(f"开始监控文件夹: {data_path}")
    logger.info("请将CSV或Excel文件放入该文件夹，程序将自动处理")
    logger.info("按 Ctrl+C 停止监控")
    
    processed_files = set()
    
    try:
        while True:
            # 查找所有CSV和Excel文件
            csv_files = list(data_path.glob('*.csv'))
            excel_files = list(data_path.glob('*.xlsx')) + list(data_path.glob('*.xls'))
            current_files = set(csv_files + excel_files)
            
            # 找到新文件
            new_files = current_files - processed_files
            
            for file_path in new_files:
                logger.info(f"\n检测到新文件: {file_path.name}")
                try:
                    if file_path.suffix.lower() == '.csv':
                        process_csv_links(str(file_path), link_column, config_path)
                    else:
                        process_excel_links(str(file_path), link_column, config_path)
                    logger.info(f"✓ 文件处理完成: {file_path.name}")
                    processed_files.add(file_path)
                except Exception as e:
                    logger.error(f"✗ 文件处理失败: {file_path.name}, 错误: {str(e)}")
            
            time.sleep(2)  # 每2秒检查一次
            
    except KeyboardInterrupt:
        logger.info("\n监控已停止")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频批量筛选工具')
    parser.add_argument('-i', '--input', help='输入视频目录（本地模式）')
    parser.add_argument('-o', '--output', help='输出目录（可选，仅本地模式）')
    parser.add_argument('-c', '--config', help='配置文件路径')
    # Excel模式
    parser.add_argument('--excel', help='Excel文件路径（链接批量模式）')
    parser.add_argument('--link-column', default='vediolink', help='Excel中链接列名，默认vediolink')
    # 新增：批量处理模式
    parser.add_argument('--batch', action='store_true', help='批量处理模式，自动处理data文件夹中的所有CSV/Excel文件')
    parser.add_argument('--data-dir', default='data', help='数据文件夹路径，默认为data')
    parser.add_argument('--watch', action='store_true', help='监控模式，持续监控data文件夹中的新文件')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 批量处理模式优先
    if args.batch:
        logger.info(f"批量处理模式：处理文件夹 {args.data_dir}")
        process_batch_files(args.data_dir, args.link_column, args.config)
        return
    
    # 监控模式
    if args.watch:
        logger.info(f"监控模式：监控文件夹 {args.data_dir}")
        watch_data_folder(args.data_dir, args.link_column, args.config)
        return
    
    # Excel模式
    if args.excel:
        logger.info(f"Excel模式：处理文件 {args.excel}")
        process_excel_links(args.excel, args.link_column, args.config)
        return
    
    # 本地目录模式
    if not args.input:
        logger.error("请指定输入文件或目录，或使用 --batch 进行批量处理")
        logger.info("使用示例:")
        logger.info("  批量处理: python video_filter.py --batch")
        logger.info("  监控模式: python video_filter.py --watch")
        logger.info("  Excel模式: python video_filter.py --excel file.xlsx")
        logger.info("  本地模式: python video_filter.py -i input_dir")
        return
    
    filter_instance = VideoFilter(args.config)
    filter_instance.filter_videos(args.input, args.output)
    
    # 输出由filter_videos内部日志给出


if __name__ == "__main__":
    main()