#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级视频信息提取器
作为yt-dlp的替代方案，使用requests和正则表达式提取基本视频信息
"""

import re
import requests
from urllib.parse import urlparse, parse_qs
import json
import time
from typing import Dict, Optional, Tuple


class LightVideoExtractor:
    """轻量级视频信息提取器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def extract_video_info(self, url: str) -> Dict[str, str]:
        """
        提取视频基本信息
        
        Args:
            url: 视频链接
            
        Returns:
            包含视频信息的字典
        """
        try:
            # 识别视频平台
            platform = self._identify_platform(url)
            
            if platform == 'youtube':
                return self._extract_youtube_info(url)
            elif platform == 'bilibili':
                return self._extract_bilibili_info(url)
            elif platform == 'douyin':
                return self._extract_douyin_info(url)
            else:
                return self._extract_generic_info(url)
                
        except Exception as e:
            return {
                'title': '',
                'duration': '',
                'resolution': '',
                'error': f'提取失败: {str(e)}'
            }
    
    def _identify_platform(self, url: str) -> str:
        """识别视频平台"""
        domain = urlparse(url).netloc.lower()
        
        if 'youtube.com' in domain or 'youtu.be' in domain:
            return 'youtube'
        elif 'bilibili.com' in domain:
            return 'bilibili'
        elif 'douyin.com' in domain or 'tiktok.com' in domain:
            return 'douyin'
        else:
            return 'generic'
    
    def _extract_youtube_info(self, url: str) -> Dict[str, str]:
        """提取YouTube视频信息"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # 提取标题
            title_match = re.search(r'"title":"([^"]+)"', html)
            title = title_match.group(1) if title_match else ''
            
            # 提取时长
            duration_match = re.search(r'"lengthSeconds":"(\d+)"', html)
            if duration_match:
                seconds = int(duration_match.group(1))
                duration = f"{seconds//60}:{seconds%60:02d}"
            else:
                duration = ''
            
            # 提取分辨率信息
            quality_match = re.search(r'"qualityLabel":"([^"]+)"', html)
            resolution = quality_match.group(1) if quality_match else ''
            
            return {
                'title': title,
                'duration': duration,
                'resolution': resolution,
                'error': ''
            }
            
        except Exception as e:
            return {
                'title': '',
                'duration': '',
                'resolution': '',
                'error': f'YouTube提取失败: {str(e)}'
            }
    
    def _extract_bilibili_info(self, url: str) -> Dict[str, str]:
        """提取B站视频信息"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # 提取标题
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html)
            title = title_match.group(1).replace('_哔哩哔哩_bilibili', '') if title_match else ''
            
            # 提取视频信息
            info_match = re.search(r'window\.__playinfo__\s*=\s*({.+?});', html)
            if info_match:
                try:
                    play_info = json.loads(info_match.group(1))
                    duration = play_info.get('data', {}).get('dash', {}).get('duration', 0)
                    if duration:
                        duration = f"{duration//60}:{duration%60:02d}"
                    else:
                        duration = ''
                except:
                    duration = ''
            else:
                duration = ''
            
            return {
                'title': title,
                'duration': duration,
                'resolution': '1080p',  # B站默认支持1080p
                'error': ''
            }
            
        except Exception as e:
            return {
                'title': '',
                'duration': '',
                'resolution': '',
                'error': f'B站提取失败: {str(e)}'
            }
    
    def _extract_douyin_info(self, url: str) -> Dict[str, str]:
        """提取抖音视频信息"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # 提取标题
            title_match = re.search(r'"desc":"([^"]+)"', html)
            title = title_match.group(1) if title_match else ''
            
            return {
                'title': title,
                'duration': '',  # 抖音通常是短视频
                'resolution': '720p',  # 抖音默认分辨率
                'error': ''
            }
            
        except Exception as e:
            return {
                'title': '',
                'duration': '',
                'resolution': '',
                'error': f'抖音提取失败: {str(e)}'
            }
    
    def _extract_generic_info(self, url: str) -> Dict[str, str]:
        """提取通用视频信息"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # 提取页面标题作为视频标题
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html)
            title = title_match.group(1) if title_match else ''
            
            return {
                'title': title,
                'duration': '',
                'resolution': '',
                'error': ''
            }
            
        except Exception as e:
            return {
                'title': '',
                'duration': '',
                'resolution': '',
                'error': f'通用提取失败: {str(e)}'
            }


def test_extractor():
    """测试视频信息提取器"""
    extractor = LightVideoExtractor()
    
    # 测试链接
    test_urls = [
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # YouTube测试
        'https://www.bilibili.com/video/BV1xx411c7mu',  # B站测试
    ]
    
    for url in test_urls:
        print(f"\n测试链接: {url}")
        info = extractor.extract_video_info(url)
        print(f"标题: {info['title']}")
        print(f"时长: {info['duration']}")
        print(f"分辨率: {info['resolution']}")
        if info['error']:
            print(f"错误: {info['error']}")


if __name__ == '__main__':
    test_extractor()