#!/usr/bin/env python3
"""
天眸 + RealSense 标定数据集导出工具

功能：
1. 解析 bag 文件，建立 Frame Counter -> (depth 帧编号, color 帧编号) 映射
2. 读取 post_process.py 输出的 JSON 索引，对齐 Tianmou / Depth / Color 三通道
3. 支持 -n 偏移将 Color 投影到 Depth（Color 滞后 Depth n 帧）
4. 输出三组对齐图片 + metadata.json，供标定脚本直接使用

输入（与总控.sh 对齐）：
  --json      post_process.py 输出的 JSON 索引文件
  --bag       RealSense bag 文件路径
  --tianmou   天眸数据目录（由 JSON 中的 tianmou_dir 决定，可省略）
  --output    输出目录
  --n         Color 滞后 Depth 的帧数偏移（可选，默认 0）

输出结构：
  output_dir/
    tianmou/   # 天眸 RGB 图像 (640x320)
    depth/     # RealSense 深度图
    color/     # RealSense Color 图像 (可选/投影后)
    metadata.json  # 每帧的 Frame Counter、时间戳、对齐信息
"""

import os
import re
import sys
import json
import glob
import shutil
import tempfile
import subprocess
import argparse
import time
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

# 尝试导入 tianmoucv
try:
    from tianmoucv.data import TianmoucDataReader
    HAS_TIANMOUCV = True
except ImportError:
    HAS_TIANMOUCV = False
    print("Warning: tianmoucv 未安装，天眸帧读取可能失败")


# ========== 进度追踪 ==========

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.stage_start_time = None
        self.current_stage = None
        self.total_files = 0

    def start_stage(self, stage_name: str, total_files: int = 0):
        if self.current_stage:
            self._finish_stage()
        self.total_files = total_files
        print(f"\n[{time.time()-self.start_time:.1f}s] {'='*50}")
        print(f"[{time.time()-self.start_time:.1f}s] 阶段: {stage_name}")
        if total_files:
            print(f"[{time.time()-self.start_time:.1f}s] 总计: {total_files} 个文件")
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self._stage_progress_printed = False

    def update_progress(self, current: int, total: int, prefix: str = "处理"):
        if total == 0:
            return
        self._stage_progress_printed = True
        percent = current / total * 100
        elapsed = time.time() - (self.stage_start_time or time.time())
        eta = (elapsed / current * (total - current)) if current > 0 else 0
        print(f"\r[{time.time()-self.start_time:.1f}s] {prefix}: [{current}/{total}] {percent:.1f}% | ETA: {eta:.1f}s", end="", flush=True)

    def _finish_stage(self):
        if self.current_stage and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            if self._stage_progress_printed:
                print()  # 换行，避免与 \r 混在一起
            print(f"[{time.time()-self.start_time:.1f}s] ✓ {self.current_stage} 完成 (耗时: {elapsed:.1f}s)")

    def finish(self, total=None):
        if self.current_stage:
            self._finish_stage()
        if total is not None:
            print(f"\n{'='*50}")
            print(f" 导出完成！共 {total} 帧，耗时: {time.time() - self.start_time:.1f}s")
            print(f"{'='*50}")
        else:
            print(f"\n{'='*50}")
            print(f" 导出完成！耗时: {time.time() - self.start_time:.1f}s")
            print(f"{'='*50}")


# ========== RealSense Metadata 解析 ==========

def parse_metadata_file(filepath: str) -> dict:
    """解析单个 metadata txt 文件"""
    res = {"counter": None, "timestamp": None, "frame_idx": None}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            fc_match = re.search(r"Frame Counter:\s*(\d+)", content)
            if fc_match:
                res["counter"] = int(fc_match.group(1))

            ft_match = re.search(r"Frame Timestamp:\s*(\d+)", content)
            if ft_match:
                res["timestamp"] = int(ft_match.group(1))
            else:
                ts_match = re.search(r"Timestamp:\s*(\d+)", content)
                if ts_match:
                    res["timestamp"] = int(ts_match.group(1))
    except Exception as e:
        print(f"  解析文件失败 {filepath}: {e}")
    return res


def build_counter_mapping(temp_dir: str) -> Tuple[dict, dict]:
    """
    解析 temp_dir 中所有 metadata，建立两个映射：
      depth_counter_map: {counter -> depth_idx_str}
      color_counter_map: {counter -> color_idx_str}
    """
    depth_map = {}
    color_map = {}

    for subdir in [os.path.join(temp_dir, "depth"), os.path.join(temp_dir, "color")]:
        if not os.path.isdir(subdir):
            continue
        for filename in os.listdir(subdir):
            if not (filename.endswith(".txt") and "_metadata_" in filename):
                continue
            filepath = os.path.join(subdir, filename)
            m = parse_metadata_file(filepath)
            if m["counter"] is None:
                continue

            digits = re.findall(r'\d+', filename)
            if not digits:
                continue
            idx_str = digits[-1]

            if "Depth" in filename:
                depth_map[m["counter"]] = idx_str
            elif "Color" in filename:
                color_map[m["counter"]] = idx_str

    return depth_map, color_map


def find_image_file(temp_dir: str, stream_type: str, idx_str: str) -> Optional[str]:
    """在 temp_dir/{depth,color}/ 子目录中查找指定格式的图像文件"""
    # 先在对应子目录中查找
    subdir = os.path.join(temp_dir, stream_type.lower())
    if os.path.isdir(subdir):
        for ext in [".png", ".jpg", ".jpeg", ".raw"]:
            guess = os.path.join(subdir, f"frame_{stream_type}_{idx_str}{ext}")
            if os.path.exists(guess):
                return guess
        # 兜底：遍历子目录
        for f in os.listdir(subdir):
            if stream_type in f and idx_str in f and not f.endswith('.txt'):
                return os.path.join(subdir, f)
    # 兜底：直接在 temp_dir 中搜索（兼容旧格式）
    for ext in [".png", ".jpg", ".jpeg", ".raw"]:
        guess = os.path.join(temp_dir, f"frame_{stream_type}_{idx_str}{ext}")
        if os.path.exists(guess):
            return guess
    for f in os.listdir(temp_dir):
        if stream_type in f and idx_str in f and not f.endswith('.txt'):
            return os.path.join(temp_dir, f)
    return None


# ========== Tianmou 帧读取 ==========

def to_np(x):
    """tensor -> numpy"""
    import torch
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return x


def get_tianmou_rgb(reader, frame_idx: int) -> Optional[Any]:
    """读取天眸 RGB 帧，返回 BGR 格式的 numpy 数组"""
    if frame_idx >= len(reader):
        return None
    try:
        sample = reader[frame_idx]
        img = to_np(sample["F1"])

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)

        return img
    except Exception as e:
        print(f"  读取天眸帧 {frame_idx} 失败: {e}")
        return None


def get_tianmou_timestamp_us(reader, frame_idx: int) -> int:
    """获取天眸帧时间戳（微秒）"""
    if frame_idx >= len(reader):
        return 0
    try:
        sample = reader[frame_idx]
        meta = sample.get("meta", {})
        Cts = meta.get("C_timestamp", None)
        if Cts is None or len(Cts) < 2:
            return 0
        ts_start_10us = int(Cts[0])
        return ts_start_10us * 10
    except:
        return 0


# ========== 核心导出逻辑 ==========

def export_calibration_set(json_path: str, bag_path: str, output_dir: str, shift_n: int = 0,
                             tianmou_dir: Optional[str] = None):
    progress = ProgressTracker()

    # ---- 0. 读取 JSON ----
    progress.start_stage("读取索引文件")
    if not os.path.exists(json_path):
        print(f"  ❌ JSON 文件不存在: {json_path}")
        return

    with open(json_path, 'r') as f:
        index_data = json.load(f)

    # JSON 里存的是 "frames"，不是 "matched_frames"
    matched_frames = index_data.get("frames", [])
    if not matched_frames:
        print("  ❌ matched_frames 为空")
        return

    # 天眸目录（优先使用传入的，否则从 JSON 中取）
    tm_dir = tianmou_dir or index_data.get("tianmou_dir", "")
    print(f"  天眸目录: {tm_dir}")
    print(f"  RealSense bag: {bag_path}")
    print(f"  索引帧数: {len(matched_frames)}")
    print(f"  Color 偏移 n: {shift_n}")

    # ---- 1. 从 bag 提取图像（Color 和 Depth 分开提取，避免文件覆盖）----
    progress.start_stage("从 bag 提取图像（可能需要几十秒）")
    temp_dir = tempfile.mkdtemp(prefix="calib_extract_")
    color_temp_dir = os.path.join(temp_dir, "color")
    depth_temp_dir = os.path.join(temp_dir, "depth")
    os.makedirs(color_temp_dir, exist_ok=True)
    os.makedirs(depth_temp_dir, exist_ok=True)

    print(f"\n  → 提取 Color 流...")
    subprocess.run(['rs-convert', '-i', bag_path, '-c', '-p', os.path.join(color_temp_dir, 'frame') + '_'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  ✓ Color 提取完成")

    print(f"  → 提取 Depth 流...")
    subprocess.run(['rs-convert', '-i', bag_path, '-d', '-p', os.path.join(depth_temp_dir, 'frame') + '_'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  ✓ Depth 提取完成")
    progress._finish_stage()

    depth_files = glob.glob(os.path.join(depth_temp_dir, "*_Depth_metadata_*.txt"))
    color_files = glob.glob(os.path.join(color_temp_dir, "*_Color_metadata_*.txt"))
    print(f"  提取到 Depth metadata: {len(depth_files)}")
    print(f"  提取到 Color metadata: {len(color_files)}")

    # ---- 2. 建立 Frame Counter 映射表（带进度条）----
    all_meta_files = depth_files + color_files
    progress.start_stage("建立 Frame Counter 映射表", total_files=len(all_meta_files))

    def build_idx_to_counter_with_progress(metadata_files, stream_label, prog):
        idx_counter_list = []
        for k, f in enumerate(metadata_files):
            prog.update_progress(k + 1, len(metadata_files), f"解析 {stream_label} metadata")
            m = parse_metadata_file(f)
            digits = re.findall(r'\d+', os.path.basename(f))
            if not digits:
                continue
            idx_str = digits[-1]
            counter = m.get("counter")
            ts = m.get("timestamp", 0)
            idx_counter_list.append((int(idx_str), counter, ts))
        idx_counter_list.sort(key=lambda x: x[0])
        return idx_counter_list

    progress.start_stage("解析 Depth Counter", total_files=len(depth_files))
    depth_pairs = build_idx_to_counter_with_progress(depth_files, "Depth", progress)
    progress._finish_stage()
    print(f"  Depth Counter 数量: {len(depth_pairs)}")

    progress.start_stage("解析 Color Counter", total_files=len(color_files))
    color_pairs = build_idx_to_counter_with_progress(color_files, "Color", progress)
    progress._finish_stage()
    print(f"  Color Counter 数量: {len(color_pairs)}")

    def safe_min(lst, key):
        vals = [key(x) for x in lst if key(x) is not None]
        return min(vals) if vals else None

    def safe_max(lst, key):
        vals = [key(x) for x in lst if key(x) is not None]
        return max(vals) if vals else None

    print(f"  Depth 帧数: {len(depth_pairs)}, Counter 范围: "
          f"[{safe_min(depth_pairs, lambda d: d[1])}, "
          f"{safe_max(depth_pairs, lambda d: d[1])}]")
    print(f"  Color 帧数: {len(color_pairs)}, Counter 范围: "
          f"[{safe_min(color_pairs, lambda d: d[1])}, "
          f"{safe_max(color_pairs, lambda d: d[1])}]")

    # ---- 3. 初始化天眸读取器 ----
    progress.start_stage("初始化天眸数据读取器")
    reader = None
    if HAS_TIANMOUCV and tm_dir and os.path.exists(tm_dir):
        try:
            reader = TianmoucDataReader(tm_dir, print_info=False, N=1, camera_idx=0, strict=True)
            print(f"  天眸总帧数: {len(reader)}")
        except Exception as e:
            print(f"  天眸读取器初始化失败: {e}")
    else:
        print(f"  ⚠ 天眸目录不存在或 tianmoucv 未安装: {tm_dir}")

    # ---- 4. 导出三组图片 ----
    # 核心原则：只有 Depth 和 Color 同时有对应帧时才输出
    progress.start_stage("导出对齐图片")
    tianmou_out = os.path.join(output_dir, "tianmou")
    depth_out = os.path.join(output_dir, "depth")
    color_out = os.path.join(output_dir, "color")
    os.makedirs(tianmou_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    os.makedirs(color_out, exist_ok=True)

    metadata_records = []
    total = len(matched_frames)
    success_tm = success_depth = success_color = 0
    skipped_frames = []
    exported_count = 0

    for i, frame in enumerate(matched_frames):
        tm_idx = frame.get("tianmou_idx")
        rs_idx = frame.get("realsense_idx")
        is_skipped = frame.get("skip", False)
        skip_type = frame.get("skip_type")
        phase = frame.get("phase", "unknown")

        # --- Depth 帧（作为主索引）---
        if rs_idx is None or rs_idx >= len(depth_pairs):
            continue
        depth_counter, depth_ts = depth_pairs[rs_idx][1], depth_pairs[rs_idx][2]
        if depth_counter is None:
            continue

        # --- Color 帧（精确按 Frame Counter 匹配，shift_n 偏移）---
        target_color_counter = depth_counter + shift_n
        color_idx_str = None
        for j in range(len(color_pairs)):
            if color_pairs[j][1] == target_color_counter:
                color_idx_str = str(color_pairs[j][0])
                break

        if color_idx_str is None:
            continue  # Color 没有对应帧，跳过这帧（不输出任何东西）

        # --- 帧命名（只在真正输出的帧上计数）---
        out_name = f"{exported_count:05d}.png"

        # --- 天眸帧 ---
        tm_img = None
        tm_ts = 0
        if reader and tm_idx is not None:
            tm_img = get_tianmou_rgb(reader, tm_idx)
            tm_ts = get_tianmou_timestamp_us(reader, tm_idx)
        if tm_img is not None:
            cv2.imwrite(os.path.join(tianmou_out, out_name), tm_img)
            success_tm += 1
        else:
            placeholder = np.zeros((320, 640, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tianmou_out, out_name), placeholder)

        # --- Depth 帧 ---
        depth_idx_str = str(depth_pairs[rs_idx][0])
        depth_src = find_image_file(temp_dir, "Depth", depth_idx_str)
        if depth_src:
            shutil.copy(depth_src, os.path.join(depth_out, out_name))
            success_depth += 1

        # --- Color 帧 ---
        color_src = find_image_file(temp_dir, "Color", color_idx_str)
        if color_src:
            shutil.copy(color_src, os.path.join(color_out, out_name))
            success_color += 1

        # --- 记录 metadata ---
        record = {
            "frame_id": exported_count,
            "frame_name": out_name,
            "tianmou_idx": tm_idx,
            "tianmou_ts_us": tm_ts,
            "tianmou_ts_str": datetime.datetime.fromtimestamp(tm_ts / 1e6).strftime("%H:%M:%S.%f")[:-3] if tm_ts else "",
            "realsense_idx": rs_idx,
            "depth_counter": int(depth_counter),
            "depth_ts_us": int(depth_ts) if depth_ts is not None else None,
            "color_counter": int(target_color_counter),
            "color_projected_n": shift_n,
            "skip": is_skipped,
            "skip_type": skip_type,
            "phase": phase,
        }
        metadata_records.append(record)

        if is_skipped:
            skipped_frames.append(record)

        exported_count += 1

        if exported_count % 20 == 0:
            print(f"\r[{time.time()-progress.start_time:.1f}s] 已导出 {exported_count} 帧...", end="", flush=True)

    total_exported = len(metadata_records)

    # ---- 5. 导出 metadata.json ----
    metadata_json = {
        "source_json": os.path.abspath(json_path),
        "source_bag": os.path.abspath(bag_path),
        "tianmou_dir": tm_dir,
        "shift_n": shift_n,
        "total_frames": len(metadata_records),
        "success_tianmou": success_tm,
        "success_depth": success_depth,
        "success_color": success_color,
        "skipped_frames_count": len(skipped_frames),
        "skipped_frames": skipped_frames,
        "frames": metadata_records,
        "depth_counter_range": {
            "min": safe_min(depth_pairs, lambda d: d[1]),
            "max": safe_max(depth_pairs, lambda d: d[1]),
        },
        "color_counter_range": {
            "min": safe_min(color_pairs, lambda d: d[1]),
            "max": safe_max(color_pairs, lambda d: d[1]),
        }
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_json, f, indent=2, ensure_ascii=False)

    print(f"\n  输出目录: {output_dir}")
    print(f"  天眸帧: {success_tm}/{total_exported}")
    print(f"  深度帧: {success_depth}/{total_exported}")
    print(f"  色彩帧: {success_color}/{total_exported}")
    print(f"  跳过帧: {len(skipped_frames)}")
    print(f"  metadata.json 已保存")

    progress.finish(total_exported)

    # ---- 清理临时文件 ----
    shutil.rmtree(temp_dir, ignore_errors=True)
    if reader is not None:
        del reader


# ========== 主入口 ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="天眸 + RealSense 标定数据集导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法（Color 无偏移）
  python export_calibration_set.py \\
      --json /projects/calib_data/20260319/index.json \\
      --bag /projects/cxr_data/2026-3-19/21-33-16.bag \\
      --output /projects/calib_data/20260319

  # Color 投影偏移（Color 滞后 Depth 8 帧）
  python export_calibration_set.py \\
      --json /projects/calib_data/20260319/index.json \\
      --bag /projects/cxr_data/2026-3-19/21-33-16.bag \\
      --output /projects/calib_data/20260319_n8 \\
      --n 8

  # 指定天眸目录（覆盖 JSON 中的路径）
  python export_calibration_set.py \\
      --json /projects/calib_data/20260319/index.json \\
      --bag /projects/cxr_data/2026-3-19/21-33-16.bag \\
      --tianmou /projects/cxr_data/20260319_2133 \\
      --output /projects/calib_data/20260319 \\
      --n 8
        """
    )
    parser.add_argument("--json", required=True, help="post_process.py 输出的 JSON 索引文件")
    parser.add_argument("--bag", required=True, help="RealSense bag 文件路径")
    parser.add_argument("--tianmou", default=None, help="天眸数据目录（可省略，从 JSON 中读取）")
    parser.add_argument("--output", "-e", required=True, help="输出目录")
    parser.add_argument("--n", type=int, default=0,
                        help="Color 滞后 Depth 的帧数偏移（Color_Counter = Depth_Counter + n）")

    args = parser.parse_args()
    export_calibration_set(
        json_path=args.json,
        bag_path=args.bag,
        output_dir=args.output,
        shift_n=args.n,
        tianmou_dir=args.tianmou,
    )
