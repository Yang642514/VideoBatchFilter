#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频批量筛选工具 - 简化批处理脚本
用户只需双击运行此脚本，即可自动处理data文件夹中的所有CSV/Excel文件

使用方法：
1. 将待处理的CSV或Excel文件放入data文件夹
2. 双击运行此脚本
3. 等待处理完成

注意：
- 确保文件中包含'link'列（或在config.json中指定其他列名）
- 处理结果会直接写回原文件
"""

import os
import sys
import time
from pathlib import Path

def main():
    """主函数 - 简化的批处理入口"""
    print("=" * 60)
    print("视频批量筛选工具 - 批处理模式")
    print("=" * 60)
    
    # 检查data文件夹是否存在
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data文件夹不存在，正在创建...")
        data_dir.mkdir()
        print("✅ data文件夹已创建")
        print("\n请将待处理的CSV或Excel文件放入data文件夹，然后重新运行此脚本")
        input("按回车键退出...")
        return
    
    # 检查是否有文件
    csv_files = list(data_dir.glob('*.csv'))
    excel_files = list(data_dir.glob('*.xlsx')) + list(data_dir.glob('*.xls'))
    all_files = csv_files + excel_files
    
    if not all_files:
        print("❌ data文件夹中没有找到CSV或Excel文件")
        print("\n请将待处理的文件放入data文件夹：")
        print("  - 支持的格式：.csv, .xlsx, .xls")
        print("  - 确保文件中包含'link'列")
        input("按回车键退出...")
        return
    
    print(f"📁 找到 {len(all_files)} 个文件待处理：")
    for i, file_path in enumerate(all_files, 1):
        print(f"  {i}. {file_path.name}")
    
    print("\n🚀 开始批量处理...")
    print("=" * 60)
    
    # 导入并运行主程序
    try:
        from video_filter import process_batch_files
        process_batch_files(data_dir='data')
        
        print("\n" + "=" * 60)
        print("✅ 批量处理完成！")
        print("📄 处理结果已写回原文件")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ 导入错误：{e}")
        print("请确保video_filter.py文件在同一目录下")
    except Exception as e:
        print(f"❌ 处理过程中出现错误：{e}")
        print("请检查文件格式和内容是否正确")
    
    print("\n处理完成，窗口将在10秒后自动关闭...")
    for i in range(10, 0, -1):
        print(f"倒计时：{i}秒", end='\r')
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n\n程序异常：{e}")
        input("按回车键退出...")