#!/usr/bin/env python3
"""
extract_raw_depth_rs2.py
========================
使用 pyrealsense2 直接从 bag 文件提取原始 Z16 深度帧。

Usage:
    python3 extract_raw_depth_rs2.py \
        --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
        --output /projects/calib_data/0324_calibration_depth/depth_raw_rs2

    # 只提取部分帧（快速测试）
    python3 extract_raw_depth_rs2.py \
        --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
        --output /projects/calib_data/0324_calibration_depth/depth_raw_rs2 \
        --max-frames 100

    # 指定分辨率过滤（默认 848x480）
    python3 extract_raw_depth_rs2.py \
        --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
        --output /projects/calib_data/0324_calibration_depth/depth_raw_rs2 \
        --width 1280 --height 720
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import cv2

# 延迟导入 pyrealsense2，便于脚本在没有 RS SDK 的环境下也能被 import 检查语法
try:
    import pyrealsense2 as rs
    _HAS_RS2 = True
except ImportError:
    _HAS_RS2 = False
    print("ERROR: pyrealsense2 未安装，请运行: pip install pyrealsense2")


def probe_bag(bag_path: str) -> Dict:
    """探测 bag 文件中的流信息（分辨率、格式等）。"""
    print(f"探测 bag 文件: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    # 启用所有流，不限制分辨率
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    streams_info = {}

    print("\n检测到的流:")
    for p in profile.get_streams():
        stream_type = str(p.stream_type())
        fps = p.fps()
        fmt = p.format()

        # 老版本 pyrealsense2 需要 as_video_stream_profile() 才能获取分辨率
        if p.stream_type() == rs.stream.depth:
            try:
                vsp = p.as_video_stream_profile()
                w, h = vsp.width(), vsp.height()
            except Exception:
                w, h = "?", "?"
            streams_info['depth'] = {
                'width': w,
                'height': h,
                'fps': fps,
                'format': str(fmt),
            }
            print(f"  深度流: {w}x{h} @ {fps}Hz, 格式={fmt}")

        elif p.stream_type() == rs.stream.color:
            try:
                vsp = p.as_video_stream_profile()
                w, h = vsp.width(), vsp.height()
            except Exception:
                w, h = "?", "?"
            streams_info['color'] = {
                'width': w,
                'height': h,
                'fps': fps,
                'format': str(fmt),
            }
            print(f"  彩色流: {w}x{h} @ {fps}Hz, 格式={fmt}")

    pipeline.stop()
    return streams_info


def extract_raw_depth(
    bag_path: str,
    output_dir: str,
    max_frames: int = -1,
    target_width: int = None,
    target_height: int = None,
) -> Tuple[int, int]:
    """
    使用 pyrealsense2 从 bag 文件提取原始 Z16 深度帧。

    Args:
        bag_path: bag 文件路径
        output_dir: 输出目录
        max_frames: 最大帧数（-1 = 全部）
        target_width: 目标宽度（None=自动选择第一个深度流）
        target_height: 目标高度

    Returns:
        (total_frames, saved_frames)
    """
    if not _HAS_RS2:
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    print(f"Bag 文件: {bag_path}")
    print(f"输出目录: {output_dir}")

    # ---------- 建立 pipeline ----------
    pipeline = rs.pipeline()
    config = rs.config()

    # 关键：enable_device_from_file 而非 enable_device
    config.enable_device_from_file(bag_path, repeat_playback=False)

    # 只启用深度流，指定 Z16 格式（fps 设为 0 由 SDK 自动匹配 bag 里的实际帧率）
    if target_width and target_height:
        config.enable_stream(
            rs.stream.depth, target_width, target_height,
            rs.format.z16, 0
        )
        print(f"\n指定深度流: {target_width}x{target_height}, 格式=Z16")
    else:
        # 让 SDK 自动选择
        config.enable_stream(rs.stream.depth)
        print("\n自动选择深度流")

    # 启动 playback
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()

    # 强制非实时播放
    playback.set_real_time(False)

    # 获取深度流的实际分辨率（自动选择时）
    if not (target_width and target_height):
        for p in profile.get_streams():
            if p.stream_type() == rs.stream.depth:
                vsp = p.as_video_stream_profile()
                target_width = vsp.width()
                target_height = vsp.height()
                break

    print(f"深度分辨率: {target_width}x{target_height}, 格式: Z16 (uint16, mm)")

    # ---------- 逐帧提取 ----------
    print(f"\n开始提取深度帧 (max={max_frames if max_frames > 0 else '全部'})...")

    saved = 0
    skipped = 0
    total = 0
    frame_times = []

    try:
        while True:
            # 等待下一帧
            frames = pipeline.wait_for_frames(10000)  # 10秒超时
            if not frames:
                print("  [WARN] 等待帧超时")
                break

            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue

            total += 1

            # 检查分辨率
            vsp = depth_frame.profile.as_video_stream_profile()
            w, h = vsp.width(), vsp.height()
            if w != target_width or h != target_height:
                skipped += 1
                continue

            # 检查 max_frames
            if max_frames > 0 and saved >= max_frames:
                break

            # 获取原始 Z16 数据
            # get_data() 返回的是 numpy array，已经就是 uint16 Z16 格式
            depth_data = np.asarray(depth_frame.get_data())

            # 验证数据
            valid_mask = depth_data > 0
            valid_count = np.sum(valid_mask)

            if valid_count < 100:
                skipped += 1
                continue

            # 获取时间戳
            ts_us = depth_frame.get_timestamp()  # 微秒
            frame_times.append({
                'frame_idx': saved,
                'timestamp_us': ts_us,
                'rs_timestamp_domain': str(depth_frame.get_frame_timestamp_domain()),
            })

            # 保存为 16 位 PNG（保持原始 mm 单位）
            # cv2.imwrite 保存 PNG 时会保持 uint16 不变
            filename = f"depth_{saved:06d}.png"
            out_path = os.path.join(output_dir, filename)

            # 关键：用 cv2.imwrite 保存 uint16 PNG
            success = cv2.imwrite(out_path, depth_data.astype(np.uint16))
            if not success:
                print(f"  [ERROR] 保存失败: {out_path}")

            # 保存 metadata
            meta = {
                'frame_idx': saved,
                'timestamp_us': ts_us,
                'width': int(w),
                'height': int(h),
                'format': 'Z16',
                'units': 'mm',
                'depth_min': int(depth_data[valid_mask].min()),
                'depth_max': int(depth_data[valid_mask].max()),
                'valid_pixels': int(valid_count),
            }
            meta_path = os.path.join(
                output_dir, "metadata", f"depth_{saved:06d}_meta.json"
            )
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            saved += 1
            if saved <= 5 or saved % 100 == 0:
                print(f"  [{saved:4d}] ts={ts_us/1e6:.3f}s, "
                      f"depth=[{meta['depth_min']}-{meta['depth_max']}]mm, "
                      f"valid={valid_count}")

    except Exception as e:
        print(f"ERROR: 提取过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()

    # 保存时间戳索引
    index_path = os.path.join(output_dir, "frame_index.json")
    with open(index_path, 'w') as f:
        json.dump(frame_times, f, indent=2)

    print(f"\n完成:")
    print(f"  总帧数: {total}")
    print(f"  有效帧: {saved}")
    print(f"  跳过:   {skipped}")
    print(f"  输出:   {output_dir}")
    print(f"  索引:   {index_path}")

    return total, saved


def verify_z16(output_dir: str) -> bool:
    """验证输出目录中的深度文件是否为真正的 Z16 格式。"""
    depth_files = sorted(glob.glob(os.path.join(output_dir, "depth_*.png")))
    if not depth_files:
        return False

    sample_file = depth_files[0]
    img = cv2.imread(sample_file, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[ERROR] 无法读取 {sample_file}")
        return False

    dtype = str(img.dtype)
    shape = img.shape
    min_val = img[img > 0].min() if img.max() > 0 else 0
    max_val = img.max()

    print(f"\nZ16 验证:")
    print(f"  数据类型: {dtype} (应为 uint16)")
    print(f"  形状: {shape}")
    print(f"  深度范围: {min_val}-{max_val} (单位: mm)")
    print(f"  是否为 Z16: {'✓ 是' if dtype == 'uint16' else '✗ 否'}")

    # Z16 格式的典型特征：
    # 1. dtype 必须是 uint16
    # 2. 深度值在 0-65535 mm 范围内
    # 3. 典型工作距离内有效值在 200-10000 mm
    if dtype == 'uint16' and max_val > 0 and max_val <= 65535:
        return True
    return False


# 为了 verify 函数需要的 glob
import glob


# ============================================================
#  命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 pyrealsense2 从 bag 提取原始 Z16 深度"
    )
    parser.add_argument("--bag", "-b", required=True, help="bag 文件路径")
    parser.add_argument(
        "--output", "-o", default=None,
        help="输出目录 (--probe 模式下可省略)"
    )
    parser.add_argument(
        "--max-frames", "-m", type=int, default=-1,
        help="最大帧数(-1=全部)"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="目标宽度 (如 848, 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="目标高度 (如 480, 720)"
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="只探测 bag 文件中的流信息，不提取"
    )
    parser.add_argument(
        "--verify", type=str,
        help="验证已有输出目录中的 Z16 格式 (传入目录路径)"
    )

    args = parser.parse_args()

    # 验证模式
    if args.verify:
        verify_z16(args.verify)
        sys.exit(0)

    # 探测模式
    if args.probe:
        if not _HAS_RS2:
            print("ERROR: pyrealsense2 未安装")
            sys.exit(1)
        probe_bag(args.bag)
        sys.exit(0)

    if not args.output:
        print("ERROR: --output/-o 是必填参数")
        sys.exit(1)

    if not os.path.exists(args.bag):
        print(f"ERROR: bag 文件不存在: {args.bag}")
        sys.exit(1)

    if args.width or args.height:
        if not (args.width and args.height):
            print("ERROR: --width 和 --height 必须同时指定")
            sys.exit(1)
        print(f"指定分辨率: {args.width}x{args.height}")

    extract_raw_depth(
        args.bag,
        args.output,
        args.max_frames,
        args.width,
        args.height,
    )

    # 提取完成后自动验证
    print("\n" + "="*60)
    print("验证提取结果:")
    print("="*60)
    verify_z16(args.output)
