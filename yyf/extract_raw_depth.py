#!/usr/bin/env python3
"""
extract_raw_depth.py
=====================
从 RealSense bag 文件提取 16 位原始深度帧。

使用 rs-convert -r 输出 .raw 文件（640x480, 16位 Z16），然后转换为 PNG。

使用方式：
    # 提取全部深度帧
    python3 extract_raw_depth.py \
        --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
        --output /projects/calib_data/0324_calibration_depth/depth_raw

    # 只提取部分帧（快速测试）
    python3 extract_raw_depth.py \
        --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
        --output /projects/calib_data/0324_calibration_depth/depth_raw \
        --max-frames 100
"""

import os
import sys
import json
import argparse
import time
import glob
import re
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import cv2


def extract_raw_depth(
    bag_path: str,
    output_dir: str,
    max_frames: int = -1,
) -> Tuple[int, int]:
    """
    使用 rs-convert -r 提取 16 位原始深度帧。

    Args:
        bag_path: bag 文件路径
        output_dir: 输出目录
        max_frames: 最大帧数（-1 = 全部）

    Returns:
        (total_frames, saved_frames)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    print(f"Bag 文件: {bag_path}")
    print(f"输出目录: {output_dir}")

    # 使用 rs-convert -r 提取原始深度
    temp_dir = tempfile.mkdtemp(prefix="rs_depth_")
    try:
        print(f"\n步骤 1: 使用 rs-convert -r 提取原始深度到临时目录...")
        cmd = ['rs-convert', '-i', bag_path, '-d', '-r', temp_dir + '/']
        print(f"  执行: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"ERROR: rs-convert 失败: {result.stderr}")
            return 0, 0

        # 查找 .raw 文件
        raw_files = glob.glob(os.path.join(temp_dir, "*.raw"))
        print(f"  找到 {len(raw_files)} 个 .raw 文件")

        # 同时查找 metadata
        meta_files = glob.glob(os.path.join(temp_dir, "*_Depth_metadata_*.txt"))

        # 建立 counter -> 文件映射
        counter_map = {}  # {counter: raw_file_path}
        for mf in meta_files:
            with open(mf, 'r') as f:
                content = f.read()
                fc_match = re.search(r"Frame Counter:\s*(\d+)", content)
                ts_match = re.search(r"Frame Timestamp:\s*(\d+)", content)
                if fc_match:
                    counter = int(fc_match.group(1))
                    ts = ts_match.group(1) if ts_match else "0"
                    # 找对应的 .raw 文件
                    ts_in_name = os.path.basename(mf).split('_')[-1].replace('.txt', '')
                    raw_pattern = f"*_{ts_in_name}.raw"
                    matches = glob.glob(os.path.join(temp_dir, raw_pattern))
                    if matches:
                        counter_map[counter] = {
                            'raw': matches[0],
                            'meta': mf,
                            'ts': ts,
                        }

        print(f"  建立了 {len(counter_map)} 个 Counter 映射")

        # 转换为 PNG 并保存
        print(f"\n步骤 2: 转换 .raw 为 16 位 PNG...")
        saved = 0
        skipped = 0

        for counter, info in sorted(counter_map.items()):
            if max_frames > 0 and saved >= max_frames:
                break

            raw_path = info['raw']
            ts = info['ts']

            # 读取 .raw 文件 (640x480, 16-bit)
            depth_raw = np.fromfile(raw_path, dtype=np.uint16)
            if len(depth_raw) != 640 * 480:
                skipped += 1
                continue

            depth_img = depth_raw.reshape(480, 640)

            # 验证深度数据
            valid_mask = depth_img > 0
            if np.sum(valid_mask) < 100:
                skipped += 1
                continue

            # 保存为 16 位 PNG
            filename = f"depth_{counter:06d}.png"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, depth_img)

            # 复制 metadata
            meta_out = os.path.join(output_dir, "metadata", f"depth_{counter:06d}_metadata.txt")
            shutil.copy(info['meta'], meta_out)

            saved += 1
            if saved <= 3 or saved % 100 == 0:
                min_d = depth_img[valid_mask].min()
                max_d = depth_img[valid_mask].max()
                print(f"    [{saved}] counter={counter}: {min_d}-{max_d}mm")

        # 保存 counter_map
        simple_map = {k: v['ts'] for k, v in counter_map.items()}
        map_path = os.path.join(output_dir, "counter_map.json")
        with open(map_path, 'w') as f:
            json.dump(simple_map, f, indent=2)

        total = len(raw_files)
        print(f"\n完成:")
        print(f"  原始 .raw 文件: {total}")
        print(f"  有效帧: {saved}")
        print(f"  跳过: {skipped}")
        print(f"  输出: {output_dir}")

        return total, saved

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
#  命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 bag 提取 16 位原始深度")
    parser.add_argument("--bag", "-b", required=True, help="bag 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--max-frames", "-m", type=int, default=-1, help="最大帧数(-1=全部)")

    args = parser.parse_args()

    if not os.path.exists(args.bag):
        print(f"ERROR: bag 文件不存在: {args.bag}")
        sys.exit(1)

    extract_raw_depth(args.bag, args.output, args.max_frames)
