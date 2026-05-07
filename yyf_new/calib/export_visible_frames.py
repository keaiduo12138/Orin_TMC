#!/usr/bin/env python3
"""
导出天眸 + RealSense 可见光帧（不依赖对齐索引）

用途：
1. 从天眸目录导出可见光帧到 output/tianmou
2. 从 bag 导出 RealSense Color 帧到 output/color
3. 支持按帧下标区间选定导出（闭区间，从 0 起算）
"""

import os
import sys
import glob
import time
import shutil
import argparse
import subprocess
import tempfile
import re
from typing import Optional, Tuple, List

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
    HAS_PYREALSENSE2 = True
except ImportError:
    HAS_PYREALSENSE2 = False

try:
    from tianmoucv.data import TianmoucDataReader
    HAS_TIANMOUCV = True
except ImportError:
    HAS_TIANMOUCV = False


def parse_inclusive_ranges(s: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """
    解析多个闭区间 [start, end]，下标从 0 开始。
    支持分号分隔多个区间，例如 "200:400;500:600" 或 "20-24,30-35,58-65"。
    每段内部格式同 parse_inclusive_range：支持 ":"、"-"、"," 作为分隔符。
    """
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(";") if p.strip()]
    if not parts:
        return None
    ranges = []
    for part in parts:
        m = re.match(r"^\s*(\d+)\s*[:,-]\s*(\d+)\s*$", part)
        if not m:
            raise ValueError(
                f"无效区间片段: {part!r}，请使用 START:END 格式，例如 200:400"
            )
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        ranges.append((a, b))
    return ranges


def parse_inclusive_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    解析闭区间 [start, end]，下标从 0 开始。
    支持 "200:400"、"200-400"、"200,400"（两端均包含）。
    """
    if s is None or not str(s).strip():
        return None
    t = str(s).strip()
    # 分隔符支持 ":"、"-"、","；连字符放在字符类末尾避免被解释为范围
    m = re.match(r"^\s*(\d+)\s*[:,-]\s*(\d+)\s*$", t)
    if not m:
        raise ValueError(f"无效区间格式: {s!r}，请使用 START:END 或 START-END，例如 200:400")
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return (a, b)


def to_np(x):
    """将 tensor/array 统一转为 numpy"""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def normalize_to_bgr(img: np.ndarray) -> np.ndarray:
    """将输入图像转换为 uint8 BGR"""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)


def _ranges_summary(ranges: List[Tuple[int, int]]) -> str:
    """human-readable 多区间摘要"""
    return ", ".join(f"[{s},{e}]" for s, e in ranges)


def export_tianmou_visible(
    tianmou_dir: str,
    output_dir: str,
    max_frames: int,
    range_pairs: Optional[List[Tuple[int, int]]],
) -> int:
    """导出天眸可见光帧（F1）。文件名含原始帧下标 tianmou_XXXXXX.png"""
    if not HAS_TIANMOUCV:
        raise RuntimeError("未安装 tianmoucv，无法读取天眸数据")

    os.makedirs(output_dir, exist_ok=True)
    reader = TianmoucDataReader(tianmou_dir, print_info=False, N=1, camera_idx=0)
    n_total = len(reader)

    if range_pairs is not None:
        all_indices: List[int] = []
        for start, end in range_pairs:
            if start < 0 or end >= n_total:
                raise ValueError(
                    f"天眸区间 [{start}, {end}] 越界，有效下标为 [0, {n_total - 1}]"
                )
            all_indices.extend(range(start, end + 1))
        print(f"天眸总帧数: {n_total}，选定 {len(range_pairs)} 个区间 {_ranges_summary(range_pairs)}，共 {len(all_indices)} 帧")
    elif max_frames > 0:
        all_indices = list(range(min(max_frames, n_total)))
        print(f"天眸总帧数: {n_total}，按 --max-frames 导出前 {len(all_indices)} 帧")
    else:
        all_indices = list(range(n_total))
        print(f"天眸总帧数: {n_total}，导出全部 {len(all_indices)} 帧")

    exported = 0
    for k, i in enumerate(all_indices):
        sample = reader[i]
        img = to_np(sample["F1"])
        img_bgr = normalize_to_bgr(img)
        out_path = os.path.join(output_dir, f"tianmou_{i:06d}.png")
        cv2.imwrite(out_path, img_bgr)
        exported += 1
        if (k + 1) % 100 == 0 or k == len(all_indices) - 1:
            print(f"  天眸导出进度: {k+1}/{len(all_indices)}")

    return exported


def _collect_sorted_color_images(directory: str):
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)


def export_realsense_color_precise_by_index(
    bag_path: str,
    output_dir: str,
    range_pairs: List[Tuple[int, int]],
) -> int:
    """
    使用 pyrealsense2 直接按 Color 帧索引区间导出（不全量落盘）。
    支持多个闭区间 [start, end]，从 0 起算。
    """
    if not HAS_PYREALSENSE2:
        raise RuntimeError("未安装 pyrealsense2，无法启用精准抽帧")

    os.makedirs(output_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, False)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)

    # 合并所有目标下标（排序、去重）
    target_indices_set = set()
    for start, end in range_pairs:
        if start < 0 or end < start:
            raise ValueError(f"非法 Color 区间: [{start}, {end}]")
        target_indices_set.update(range(start, end + 1))
    target_indices = sorted(target_indices_set)
    target_count = len(target_indices)

    try:
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        color_idx = -1
        exported = 0
        ti = 0  # target_indices pointer

        while ti < target_count:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError:
                break

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_idx += 1
            # 跳过不在目标区间的帧
            while ti < target_count and target_indices[ti] < color_idx:
                ti += 1
            if ti >= target_count:
                break
            if target_indices[ti] != color_idx:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            out_path = os.path.join(output_dir, f"color_{color_idx:06d}.png")
            cv2.imwrite(out_path, color_img)
            exported += 1
            ti += 1

            if exported % 50 == 0 or exported == target_count:
                print(f"  Color 精准导出进度: {exported}/{target_count}")

        if color_idx < target_indices[-1]:
            raise ValueError(
                f"Color 目标区间 [{target_indices[0]}, {target_indices[-1]}] 越界，"
                f"bag 中可读 Color 最大下标约为 {max(color_idx, -1)}"
            )

        return exported
    finally:
        pipeline.stop()


def export_realsense_color(
    bag_path: str,
    output_dir: str,
    range_pairs: Optional[List[Tuple[int, int]]],
) -> int:
    """
    通过 rs-convert 导出 RealSense Color。
    指定 range 时在临时目录全量解压，再仅复制区间内帧到 output_dir，
    文件名为 color_XXXXXX.png（XX 为 bag 解压排序后的原始下标）。
    """
    os.makedirs(output_dir, exist_ok=True)

    if range_pairs is not None:
        try:
            print("正在使用 pyrealsense2 按区间精准导出 Color 帧...")
            return export_realsense_color_precise_by_index(bag_path, output_dir, range_pairs)
        except Exception as e:
            print(f"pyrealsense2 精准抽帧失败，回退 rs-convert 全量+筛选。原因: {e}")

        # 合并所有目标下标（排序、去重）
        target_indices_set = set()
        for start, end in range_pairs:
            target_indices_set.update(range(start, end + 1))
        target_indices = sorted(target_indices_set)

        work_dir = tempfile.mkdtemp(prefix="rs_color_extract_")
        try:
            prefix = os.path.join(work_dir, "frame_")
            print("正在调用 rs-convert 导出 Color 帧（全量，随后按区间筛选）...")
            cmd = ["rs-convert", "-i", bag_path, "-c", "-p", prefix]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"rs-convert 导出 Color 失败:\n{result.stderr}")

            all_files = _collect_sorted_color_images(work_dir)
            n = len(all_files)
            for idx in target_indices:
                if idx >= n:
                    raise ValueError(
                        f"Color 目标区间 [{target_indices[0]}, {target_indices[-1]}] 越界，"
                        f"当前 bag 解压后共 {n} 张"
                    )
                src = all_files[idx]
                dst = os.path.join(output_dir, f"color_{idx:06d}.png")
                shutil.copy2(src, dst)
            print(f"Color 解压共 {n} 张，已复制 {len(target_indices)} 张（{_ranges_summary(range_pairs)}）到输出目录")
            return len(target_indices)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    prefix = os.path.join(output_dir, "frame_")
    print("正在调用 rs-convert 导出 Color 帧...")
    cmd = ["rs-convert", "-i", bag_path, "-c", "-p", prefix]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rs-convert 导出 Color 失败:\n{result.stderr}")

    image_files = _collect_sorted_color_images(output_dir)
    return len(image_files)


def export_visible_frames(
    tianmou_dir: str,
    bag_path: str,
    output_dir: str,
    max_frames: int,
    tianmou_ranges: Optional[List[Tuple[int, int]]],
    color_ranges: Optional[List[Tuple[int, int]]],
) -> bool:
    print(f"\n{'='*64}")
    print("导出天眸 + RealSense 可见光帧")
    print(f"{'='*64}")
    print(f"天眸目录: {tianmou_dir}")
    print(f"Bag 文件: {bag_path}")
    print(f"输出目录: {output_dir}")
    if tianmou_ranges:
        print(f"天眸选定: {_ranges_summary(tianmou_ranges)}（0 起算）")
    if color_ranges:
        print(f"Color 选定: {_ranges_summary(color_ranges)}（0 起算）")
    print(f"{'='*64}")

    if not os.path.isdir(tianmou_dir):
        print(f"错误: 天眸目录不存在: {tianmou_dir}")
        return False
    if not os.path.isfile(bag_path):
        print(f"错误: bag 文件不存在: {bag_path}")
        return False

    t0 = time.time()
    tianmou_out = os.path.join(output_dir, "tianmou")
    color_out = os.path.join(output_dir, "color")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tianmou_out, exist_ok=True)
    os.makedirs(color_out, exist_ok=True)

    try:
        tianmou_count = export_tianmou_visible(
            tianmou_dir, tianmou_out, max_frames=max_frames, range_pairs=tianmou_ranges
        )
        color_count = export_realsense_color(bag_path, color_out, range_pairs=color_ranges)
    except Exception as e:
        print(f"导出失败: {e}")
        return False

    elapsed = time.time() - t0
    print(f"\n{'='*64}")
    print("导出完成")
    print(f"天眸导出帧数: {tianmou_count}")
    print(f"Color 导出帧数: {color_count}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"输出目录: {output_dir}")
    print(f"  - 天眸: {tianmou_out}（文件名 tianmou_XXXXXX.png 为原始帧下标）")
    print(f"  - Color: {color_out}")
    print(f"{'='*64}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="导出天眸 + RealSense 可见光帧",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
帧下标均为从 0 开始的闭区间 [start, end]。
多个区间用分号分隔，例如 "20:24;30:35;58:65"。
每段格式支持 START:END、START-END、START,END。

示例：天眸第 20～24 帧 + 30～35 帧、Color 第 100～150 帧：
  python3 yyf/calib/export_visible_frames.py -t ... -b ... -o ... \\
    --tianmou-range "20:24;30:35" --color-range 100:150
        """,
    )
    parser.add_argument("--tianmou-dir", "-t", required=True, help="天眸数据目录")
    parser.add_argument("--bag", "-b", required=True, help="RealSense bag 文件")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=-1,
        help="未指定 --tianmou-range 时：仅导出天眸前 N 帧；-1 表示全部",
    )
    parser.add_argument(
        "--tianmou-range",
        type=str,
        default=None,
        metavar="RANGE",
        help="天眸帧闭区间（0 起算），支持多段如 \"20:24;30:35\"；指定后忽略 -m",
    )
    parser.add_argument(
        "--color-range",
        type=str,
        default=None,
        metavar="RANGE",
        help="Color 闭区间（0 起算），支持多段如 \"20:24;30:35\"",
    )

    args = parser.parse_args()

    try:
        tm_r = parse_inclusive_ranges(args.tianmou_range)
        c_r = parse_inclusive_ranges(args.color_range)
    except ValueError as e:
        print(f"参数错误: {e}")
        return 1

    ok = export_visible_frames(
        tianmou_dir=args.tianmou_dir,
        bag_path=args.bag,
        output_dir=args.output,
        max_frames=args.max_frames,
        tianmou_ranges=tm_r,
        color_ranges=c_r,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
