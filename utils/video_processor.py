#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理核心模块
统一处理Excel和CSV文件的核心逻辑
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """视频信息数据结构"""
    url: str = ""
    title: str = ""
    duration: Optional[int] = None  # 秒
    width: int = 0
    height: int = 0
    max_resolution: str = ""
    thumb_urls: List[str] = None
    error: str = ""

    def __post_init__(self):
        if self.thumb_urls is None:
            self.thumb_urls = []


@dataclass
class ProcessingResult:
    """处理结果数据结构"""
    url: str = ""
    title: str = ""
    duration: str = ""
    max_resolution: str = ""
    resolution_ok: bool = False
    has_violence_blood: bool = False
    has_smoking: bool = False
    overall_pass: bool = False
    notes: str = ""
    error_message: str = ""


class VideoProcessor:
    """统一的视频处理核心类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.columns_config = config.get('columns', {})
        self.filters_config = config.get('filters', {})

    def get_column_name(self, column_type: str) -> str:
        """获取配置的列名"""
        return self.columns_config.get(column_type, column_type)

    def check_resolution(self, width: int, height: int) -> bool:
        """检查分辨率是否符合要求"""
        min_width, min_height = self.filters_config.get('min_resolution', [1920, 1080])
        return width >= min_width and height >= min_height

    def check_duration(self, duration_seconds: Optional[int]) -> tuple[bool, str]:
        """检查时长是否符合要求"""
        if duration_seconds is None:
            return True, ""

        min_duration = self.filters_config.get('min_duration', 0)
        max_duration = self.filters_config.get('max_duration', 3600)

        if duration_seconds < min_duration:
            return False, f"视频时长过短: {self.format_duration(duration_seconds)}"

        if duration_seconds > max_duration:
            return False, f"视频时长过长: {self.format_duration(duration_seconds)}"

        return True, ""

    def format_duration(self, seconds: Optional[int]) -> str:
        """格式化时长显示"""
        if seconds is None or seconds <= 0:
            return ""

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def process_video_info(self, video_info: VideoInfo, detector=None) -> ProcessingResult:
        """处理单个视频信息"""
        result = ProcessingResult(
            url=video_info.url,
            title=video_info.title or "",
            duration=self.format_duration(video_info.duration),
            max_resolution=video_info.max_resolution or ""
        )

        # 如果有错误，直接返回
        if video_info.error:
            result.error_message = video_info.error
            result.overall_pass = False
            result.notes = "视频信息提取失败"
            return result

        # 检查分辨率
        result.resolution_ok = self.check_resolution(video_info.width, video_info.height)
        if not result.resolution_ok:
            result.overall_pass = False
            result.notes = "分辨率不达标"
            return result

        # 检查时长
        duration_ok, duration_msg = self.check_duration(video_info.duration)
        if not duration_ok:
            result.overall_pass = False
            result.notes = duration_msg
            return result

        # 内容检测
        if detector and detector.enabled:
            detect_res = detector.detect_from_thumbnails(video_info.thumb_urls)
            result.has_violence_blood = bool(
                detect_res.get('violence', False) or
                detect_res.get('blood', False) or
                detect_res.get('gore', False)
            )
            result.has_smoking = bool(detect_res.get('smoking', False))

            # 综合判断
            result.overall_pass = (
                result.resolution_ok and
                duration_ok and
                not result.has_violence_blood and
                not result.has_smoking
            )

            if not result.overall_pass:
                if result.has_violence_blood:
                    result.notes = "检测到血腥暴力内容"
                elif result.has_smoking:
                    result.notes = "检测到吸烟内容"
        else:
            # 没有内容检测器，仅根据分辨率和时长判断
            result.overall_pass = result.resolution_ok and duration_ok
            if detector and not detector.enabled:
                result.notes = "内容检测未启用"

        return result

    def process_data_rows(self, rows: List[Dict[str, Any]], link_column: str,
                         extract_func, detector=None) -> List[Dict[str, Any]]:
        """统一处理数据行"""
        processed_rows = []

        for row in rows:
            # 创建新行，避免修改原始数据
            new_row = row.copy()

            # 获取链接
            url = str(new_row.get(link_column, '')).strip()
            if not url.startswith(('http://', 'https://')):
                result = ProcessingResult(
                    url=url,
                    error_message="无效链接",
                    overall_pass=False,
                    notes="链接格式不正确"
                )
            else:
                # 提取视频信息
                video_info = extract_func(url)
                result = self.process_video_info(video_info, detector)

            # 更新行数据
            self._update_row_with_result(new_row, result)
            processed_rows.append(new_row)

        return processed_rows

    def _update_row_with_result(self, row: Dict[str, Any], result: ProcessingResult):
        """将处理结果更新到行数据"""
        # 使用配置的列名
        row[self.get_column_name('title')] = result.title
        row[self.get_column_name('duration')] = result.duration
        row[self.get_column_name('resolution')] = result.max_resolution
        row[self.get_column_name('resolution_ok')] = result.resolution_ok
        row[self.get_column_name('violence_blood')] = result.has_violence_blood
        row[self.get_column_name('smoking')] = result.has_smoking
        row[self.get_column_name('overall_pass')] = result.overall_pass
        row[self.get_column_name('notes')] = result.notes
        row[self.get_column_name('error')] = result.error_message

    def append_note(self, original_note: str, new_note: str) -> str:
        """安全地追加备注信息"""
        if not original_note or str(original_note).strip() == "" or str(original_note).lower() == "nan":
            return new_note

        # 避免重复添加
        original_str = str(original_note)
        if new_note in original_str:
            return original_str

        return f"{original_str}; {new_note}"


class FileProcessor:
    """文件处理工具类"""

    @staticmethod
    def read_csv(file_path: str, link_column: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """读取CSV文件，自动检测编码"""
        # 尝试多种编码
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    if link_column not in fieldnames:
                        raise ValueError(f"CSV中未找到链接列: {link_column}")
                    rows = list(reader)
                logger.info(f"成功使用 {encoding} 编码读取文件: {file_path}")
                return rows, fieldnames
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                if "CSV中未找到链接列" in str(e):
                    raise e
                continue
        
        raise ValueError(f"无法读取CSV文件 {file_path}，尝试了所有编码: {encodings}")

    @staticmethod
    def write_csv(file_path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
        """写入CSV文件（使用临时文件）"""
        import shutil
        temp_path = file_path + '.tmp'
        try:
            with open(temp_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            shutil.move(temp_path, file_path)
        except Exception as e:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise e

    @staticmethod
    def get_output_columns(base_columns: List[str], processor: VideoProcessor) -> List[str]:
        """获取输出列名列表"""
        output_columns = [
            processor.get_column_name('title'),
            processor.get_column_name('duration'),
            processor.get_column_name('resolution'),
            processor.get_column_name('resolution_ok'),
            processor.get_column_name('violence_blood'),
            processor.get_column_name('smoking'),
            processor.get_column_name('overall_pass'),
            processor.get_column_name('notes'),
            processor.get_column_name('error')
        ]

        # 确保所有列都在结果中
        all_columns = base_columns.copy()
        for col in output_columns:
            if col not in all_columns:
                all_columns.append(col)

        return all_columns