#!/usr/bin/env python3
"""
天眸、Depth、Color 三路对齐帧导出脚本（适配新版 post_process.py）

功能：
1. 根据索引文件(output_triple_aligned.json)读取三路对齐的帧
2. 导出天眸RGB图像（带时间戳）
3. 导出RealSense Color图像（根据 Frame Counter 匹配）
4. 导出RealSense Depth图像（根据 Frame Counter 匹配）
5. 将对齐的帧合成为对比图像

适配新格式：
- 帧索引: depth_idx (对齐后的 Depth 帧索引)
- 天眸索引: tianmou_idx
- Color Frame Counter: color_counter
- Depth Frame Counter: depth_counter
- 是否有效: is_valid, has_color
"""

import os
import sys
import json
import glob
import re
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

# 尝试导入tianmoucv
try:
    from tianmoucv.data import TianmoucDataReader
    HAS_TIANMOUCV = True
except ImportError:
    HAS_TIANMOUCV = False
    print("Warning: tianmoucv 未安装，天眸帧读取可能失败")


def to_np(x):
    """将tensor转换为numpy数组"""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_tianmou_rgb(reader, frame_idx: int) -> Optional[np.ndarray]:
    """读取天眸RGB帧"""
    if frame_idx >= len(reader):
        return None

    sample = reader[frame_idx]
    img = to_np(sample["F1"])

    # 归一化到uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # 转换为BGR格式
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2BGR)

    return img_bgr


def get_tianmou_timestamp(reader, frame_idx: int) -> Tuple[int, str]:
    """获取天眸帧时间戳"""
    import datetime

    if frame_idx >= len(reader):
        return 0, ""

    sample = reader[frame_idx]
    meta = sample.get("meta", {})
    Cts = meta.get("C_timestamp", None)

    if Cts is None or len(Cts) < 2:
        ts_us = 0
    else:
        ts_start_10us = int(Cts[0])
        ts_us = ts_start_10us * 10

    ts_dt = datetime.datetime.fromtimestamp(ts_us / 1_000_000)
    ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3]

    return ts_us, ts_str


def parse_metadata_file(filepath: str) -> dict:
    """解析单个 metadata 文件，提取 Frame Counter 和 Frame Timestamp"""
    res = {"counter": None, "timestamp": None}
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
        print(f"解析文件失败 {filepath}: {e}")
    return res


def extract_realsense_raw_depth(bag_path: str, output_dir: str) -> Tuple[str, Dict[int, str]]:
    """
    使用 rs-convert -r 提取 RealSense 原始 16 位深度帧

    Args:
        bag_path: bag文件路径
        output_dir: 输出目录

    Returns:
        (output_dir, counter_map) - output_dir 和 {frame_counter: png_file_path}
    """
    import tempfile
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="rs_depth_")
    try:
        print(f"  使用 rs-convert -d -r 提取原始深度到临时目录...")
        cmd = ['rs-convert', '-i', bag_path, '-d', '-r', temp_dir + '/']
        print(f"    执行: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    Warning: rs-convert -d -r 失败: {result.stderr}")
            return output_dir, {}

        # 查找 .raw 文件和 metadata
        raw_files = glob.glob(os.path.join(temp_dir, "*.raw"))
        meta_files = glob.glob(os.path.join(temp_dir, "*_Depth_metadata_*.txt"))
        print(f"    找到 {len(raw_files)} 个 .raw 文件, {len(meta_files)} 个 metadata")

        # 建立 counter -> (raw_path, meta_path) 映射
        raw_counter_map = {}  # {counter: {'raw': path, 'meta': path}}
        for mf in meta_files:
            m = parse_metadata_file(mf)
            if m["counter"] is not None:
                # 从 metadata 文件名提取时间戳来匹配 .raw 文件
                ts_match = re.search(r'metadata_(\d+\.\d+)', os.path.basename(mf))
                if ts_match:
                    ts = ts_match.group(1)
                    raw_pattern = f"*{ts}.raw"
                    matches = glob.glob(os.path.join(temp_dir, raw_pattern))
                    if matches:
                        raw_counter_map[m["counter"]] = {
                            'raw': matches[0],
                            'meta': mf,
                            'ts': m.get("timestamp", ts),
                        }

        print(f"    建立了 {len(raw_counter_map)} 个 Counter 映射")

        # 转换为 16 位 PNG
        print(f"    转换 .raw 为 16 位 PNG...")
        counter_map = {}  # {counter: png_path}
        saved = 0

        for counter, info in sorted(raw_counter_map.items()):
            raw_path = info['raw']

            # 读取 .raw 文件 (640x480, 16-bit Z16)
            depth_raw = np.fromfile(raw_path, dtype=np.uint16)
            if len(depth_raw) != 640 * 480:
                print(f"      Warning: .raw 文件大小不对 counter={counter}")
                continue

            depth_img = depth_raw.reshape(480, 640)

            # 验证深度数据
            valid_mask = depth_img > 0
            if np.sum(valid_mask) < 100:
                print(f"      Warning: 深度数据无效 counter={counter}")
                continue

            # 保存为 16 位 PNG
            filename = f"depth_{counter:06d}.png"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, depth_img)
            counter_map[counter] = out_path
            saved += 1

            if saved <= 3 or saved % 100 == 0:
                min_d = depth_img[valid_mask].min()
                max_d = depth_img[valid_mask].max()
                print(f"      [{saved}] counter={counter}: {min_d}-{max_d}mm")

        print(f"    原始深度: {len(raw_files)} 帧 -> 有效 PNG: {saved} 帧")
        return output_dir, counter_map

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_realsense_images(bag_path: str, output_dir: str, metadata_dir: str = None,
                             use_raw_depth: bool = False) -> Tuple[str, str, Dict[int, str]]:
    """
    使用rs-convert提取RealSense Color和Depth图像和metadata

    Args:
        bag_path: bag文件路径
        output_dir: 输出目录
        metadata_dir: metadata输出目录（可选，默认使用output_dir/metadata）
        use_raw_depth: 是否使用原始16位深度（rs-convert -r）

    Returns:
        (图像目录, metadata目录, depth_counter_map)
        - depth_counter_map: {frame_counter: png_path}（仅当 use_raw_depth=True 时有效）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 清空目录
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)

    # 确定 metadata 目录
    if metadata_dir is None:
        metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    depth_counter_map = {}

    print(f"正在提取RealSense图像...")

    # 提取Color图像和metadata
    color_dir = os.path.join(output_dir, "color")
    os.makedirs(color_dir, exist_ok=True)
    color_meta_dir = os.path.join(metadata_dir, "color")
    os.makedirs(color_meta_dir, exist_ok=True)

    cmd_color = ['rs-convert', '-i', bag_path, '-c', '-p', color_dir + '/frame_']
    print(f"  执行: rs-convert -i {bag_path} -c -p {color_dir}/frame_")
    try:
        result = subprocess.run(cmd_color, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    Warning: rs-convert -c 失败: {result.stderr}")
    except Exception as e:
        print(f"    Warning: rs-convert -c 异常: {e}")

    # 提取Depth图像
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    depth_meta_dir = os.path.join(metadata_dir, "depth")
    os.makedirs(depth_meta_dir, exist_ok=True)

    if use_raw_depth:
        # 使用 rs-convert -r 提取原始 16 位深度
        extract_realsense_raw_depth(bag_path, depth_dir)
        # raw 模式下 counter_map 在函数内部已建立，这里不再重复提取
    else:
        # 使用普通 -d 提取处理后的深度
        cmd_depth = ['rs-convert', '-i', bag_path, '-d', '-p', depth_dir + '/frame_']
        print(f"  执行: rs-convert -i {bag_path} -d -p {depth_dir}/frame_")
        try:
            result = subprocess.run(cmd_depth, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"    Warning: rs-convert -d 失败: {result.stderr}")
        except Exception as e:
            print(f"    Warning: rs-convert -d 异常: {e}")

    return output_dir, metadata_dir, depth_counter_map


def find_realsense_images_by_counter(image_dir: str, prefix: str, metadata_dir: str = None, known_counters: List[int] = None, is_raw_depth: bool = False) -> Dict[int, str]:
    """
    查找RealSense图像，建立 Frame Counter 到文件路径的映射

    Args:
        image_dir: 图像目录
        prefix: "Color" 或 "Depth"
        metadata_dir: metadata目录（可选）
        known_counters: 已知的 Frame Counter 列表（可选，用于按顺序匹配）
        is_raw_depth: 是否为原始深度模式（文件名是 depth_000001.png 格式）

    Returns:
        {frame_counter: filepath}
    """
    counter_map = {}

    # 查找所有图片文件
    patterns = ['*.jpg', '*.png', '*.jpeg']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(image_dir, pattern)))

    # 按文件名排序（假设帧号按顺序）
    files = sorted(files)
    print(f"  找到 {len(files)} 张 {prefix} 图像")

    # Raw depth 模式：文件名是 depth_000001.png 格式
    if is_raw_depth:
        print(f"  使用 Raw Depth 模式解析文件名...")
        for f in files:
            fname = os.path.basename(f)
            # depth_000001.png -> counter = 1
            frame_match = re.search(r'depth_(\d+)\.png', fname)
            if frame_match:
                frame_num = int(frame_match.group(1))
                counter_map[frame_num] = f
        print(f"    建立了 {len(counter_map)} 个映射")
        if counter_map:
            sample_counters = sorted(counter_map.keys())[:5]
            print(f"    示例: Counter {sample_counters} -> {os.path.basename(counter_map[sample_counters[0]])}")
        return counter_map

    # 非 raw 模式：查找 metadata 文件
    search_dirs = [image_dir]
    if metadata_dir and os.path.exists(metadata_dir):
        search_dirs.append(metadata_dir)

    # 也尝试查找默认 metadata 目录
    default_meta_dir = "/projects/cxr_data/metadata_temp"
    if os.path.exists(default_meta_dir):
        search_dirs.append(default_meta_dir)

    meta_files = []
    for search_dir in search_dirs:
        pattern = f"*{prefix}*metadata*.txt"
        meta_files.extend(sorted(glob.glob(os.path.join(search_dir, pattern))))

    print(f"  找到 {len(meta_files)} 个 metadata 文件")

    # 调试：打印前3个 metadata 文件名
    if meta_files:
        print(f"  调试: 前3个metadata文件: {[os.path.basename(mf) for mf in meta_files[:3]]}")
        for mf in meta_files[:3]:
            with open(mf, 'r') as f:
                content = f.read()
                counter_match = re.search(r"Frame Counter:\s*(\d+)", content)
                fc_val = counter_match.group(1) if counter_match else "N/A"
            print(f"    metadata={os.path.basename(mf)}, Frame Counter={fc_val}")
    if files:
        print(f"  调试: 前3个图像文件: {[os.path.basename(f) for f in files[:3]]}")

    # 如果找到 metadata 文件，建立 Counter -> Image 映射
    if meta_files:
        for mf in meta_files:
            with open(mf, 'r') as f:
                content = f.read()
                counter_match = re.search(r"Frame Counter:\s*(\d+)", content)
                if counter_match:
                    counter = int(counter_match.group(1))

                    # 方法1：从 metadata 文件名提取时间戳，然后匹配图像
                    # metadata: frame__Color_metadata_1774259671696.23559570312500.txt
                    # 图像: frame__Color_1774259671696.23559570312500.png
                    # 通过时间戳匹配
                    ts_match = re.search(r'metadata_(\d+\.\d+)', os.path.basename(mf))
                    if ts_match:
                        ts = ts_match.group(1)
                        # 在图像目录中查找时间戳匹配的文件
                        for img_file in files:
                            if ts in img_file:
                                counter_map[counter] = img_file
                                break
                    else:
                        # 方法2：备用 - 从 metadata 文件名提取帧号
                        frame_match = re.search(r'frame[_\-]?(\d+)', os.path.basename(mf))
                        if frame_match:
                            frame_num = frame_match.group(1)
                            for ext in ['jpg', 'png', 'jpeg']:
                                for name_pattern in [
                                    f"frame_{frame_num}.{ext}",
                                    f"*{frame_num}*.{ext}",
                                ]:
                                    img_path = os.path.join(image_dir, name_pattern)
                                    matches = glob.glob(img_path)
                                    if matches:
                                        counter_map[counter] = matches[0]
                                        break
    elif known_counters and files:
        # 如果有已知的 Counter 列表，直接按顺序匹配
        # 假设 rs-convert 的图像顺序与 Counter 顺序一致
        print(f"  使用已知 Counter 列表按顺序匹配...")
        for i, counter in enumerate(known_counters):
            if i < len(files):
                counter_map[counter] = files[i]
    else:
        # 如果没有 metadata 和已知 Counter，直接使用文件名中的帧号作为 Counter
        print(f"  没有 metadata，使用文件名中的帧号作为 Counter")
        for f in files:
            fname = os.path.basename(f)
            frame_match = re.search(r'frame[_\-]?(\d+)', fname)
            if frame_match:
                frame_num = int(frame_match.group(1))
                counter_map[frame_num] = f

    print(f"  建立了 {len(counter_map)} 个映射")

    # 打印前5个映射用于调试
    if counter_map:
        sample_counters = sorted(counter_map.keys())[:5]
        print(f"  示例: Counter {sample_counters} -> {os.path.basename(counter_map[sample_counters[0]])}")

    return counter_map


def get_realsense_image_by_counter(counter: int, counter_map: Dict[int, str], is_raw_depth: bool = False) -> Optional[np.ndarray]:
    """根据 Frame Counter 读取 RealSense 图像"""
    img_path = counter_map.get(counter)
    if img_path and os.path.exists(img_path):
        if is_raw_depth:
            # 16 位深度图像
            return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return cv2.imread(img_path)
    return None


def create_triple_comparison_image(
    tianmou_img: np.ndarray,
    color_img: Optional[np.ndarray],
    depth_img: Optional[np.ndarray],
    frame_info: Dict[str, Any],
    ts_str: str,
    is_raw_depth: bool = False
) -> np.ndarray:
    """
    创建三路对比图像

    Args:
        tianmou_img: 天眸图像 (BGR)
        color_img: RealSense Color 图像 (BGR)
        depth_img: RealSense Depth 图像 (灰度 或 16位)
        frame_info: 帧信息
        ts_str: 时间戳字符串
        is_raw_depth: 是否为原始 16 位深度

    Returns:
        合成图像
    """
    # 确保所有图像大小一致
    h_tm, w_tm = tianmou_img.shape[:2]

    # 缩放 RealSense 图像到天眸大小
    def resize_to_match(img, target_h, target_w):
        if img is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if img.shape[:2] != (target_h, target_w):
            return cv2.resize(img, (target_w, target_h))
        return img

    tianmou_resized = resize_to_match(tianmou_img, h_tm, w_tm)
    color_resized = resize_to_match(color_img, h_tm, w_tm)

    # 处理深度图像
    if depth_img is not None and is_raw_depth and depth_img.dtype == np.uint16:
        # 16 位原始深度 -> 伪彩色显示 (COLORMAP_JET)
        depth_vis = depth_img.astype(np.float32)
        # 动态范围：计算实际深度分布
        valid_depth = depth_vis[depth_vis > 0]
        if len(valid_depth) > 0:
            depth_min = np.percentile(valid_depth, 1)
            depth_max = np.percentile(valid_depth, 99)
            if depth_max <= depth_min:
                depth_max = depth_min + 1000
            depth_vis = np.clip(depth_vis, depth_min, depth_max)
            depth_vis = (depth_vis - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_vis = np.zeros_like(depth_vis)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.resize(depth_vis, (w_tm, h_tm))
        # 伪彩色JET映射：蓝(近) -> 红(远)
        depth_resized = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    elif depth_img is not None:
        depth_vis = cv2.resize(depth_img, (w_tm, h_tm))
        if depth_vis.ndim == 2:
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_resized = depth_vis
    else:
        depth_resized = np.zeros((h_tm, w_tm, 3), dtype=np.uint8)

    # 水平拼接三路图像
    comparison = np.hstack([tianmou_resized, color_resized, depth_resized])

    # 添加信息栏
    info_height = 80
    info_img = np.zeros((info_height, comparison.shape[1], 3), dtype=np.uint8)

    # 信息文本
    depth_idx = frame_info.get("depth_idx", 0)
    tianmou_idx = frame_info.get("tianmou_idx", 0)
    depth_counter = frame_info.get("depth_counter", 0)
    color_counter = frame_info.get("color_counter", 0)
    dt_ms = frame_info.get("dt_ms")
    skip_count = frame_info.get("skip_count", 0)

    # 处理 dt_ms 可能为 None 的情况
    dt_str = f"{dt_ms:.1f}ms" if dt_ms is not None else "N/A"

    text1 = f"Depth#{depth_idx} | Counter: D={depth_counter}, C={color_counter} | dt={dt_str}"
    text2 = f"Tianmou#{tianmou_idx} | Time: {ts_str} | Skip: {skip_count}"

    cv2.putText(info_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(info_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # 垂直拼接
    result = np.vstack([comparison, info_img])

    return result


def export_triple_aligned_frames(
    index_file: str,
    tianmou_dir: str,
    bag_path: str,
    output_dir: str,
    max_frames: int = -1,
    use_raw_depth: bool = False
) -> bool:
    """导出三路对齐的帧"""
    print(f"\n{'='*60}")
    print(f"三路对齐帧导出")
    print(f"{'='*60}")
    print(f"输入:")
    print(f"  索引文件: {index_file}")
    print(f"  天眸目录: {tianmou_dir}")
    print(f"  Bag文件: {bag_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  原始深度模式: {'是 (16位Z16)' if use_raw_depth else '否 (处理后深度)'}")
    print(f"{'='*60}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    tianmou_output = os.path.join(output_dir, "tianmou")
    color_output = os.path.join(output_dir, "color")
    depth_output = os.path.join(output_dir, "depth")
    comparison_output = os.path.join(output_dir, "comparison")
    os.makedirs(tianmou_output, exist_ok=True)
    os.makedirs(color_output, exist_ok=True)
    os.makedirs(depth_output, exist_ok=True)
    os.makedirs(comparison_output, exist_ok=True)

    # 读取索引文件
    print("\n读取索引文件...")
    with open(index_file, 'r') as f:
        index_data = json.load(f)

    frames = index_data.get("frames", [])
    anchor = index_data.get("anchor", {})
    statistics = index_data.get("statistics", {})

    print(f"  总帧数: {len(frames)}")
    print(f"  锚定点: 天眸第{anchor.get('tianmou_frame_index', '?')}帧")
    print(f"  统计信息:")
    print(f"    - 天眸总帧: {statistics.get('total_tianmou_frames', '?')}")
    print(f"    - Depth总帧: {statistics.get('total_depth_frames', '?')}")
    print(f"    - Color总帧: {statistics.get('total_color_frames', '?')}")
    print(f"    - 有效三路对齐帧: {statistics.get('valid_triple_aligned', '?')}")

    # 过滤有效帧（只导出三路都有的帧）
    valid_frames = [f for f in frames if f.get("is_valid", False) and f.get("has_color", False)]
    print(f"\n  有效帧（含 Color）: {len(valid_frames)}")

    # 限制导出数量
    if max_frames > 0:
        valid_frames = valid_frames[:max_frames]
        print(f"  限制导出帧数: {max_frames}")
    print(f"  实际导出帧数: {len(valid_frames)}")

    # 初始化天眸读取器
    print("\n初始化天眸读取器...")
    if not HAS_TIANMOUCV:
        print("Error: tianmoucv 未安装")
        return False

    try:
        tianmou_reader = TianmoucDataReader(tianmou_dir, print_info=False, N=1, camera_idx=0)
        print(f"  天眸总帧数: {len(tianmou_reader)}")
    except Exception as e:
        print(f"Error: 无法初始化天眸读取器: {e}")
        return False

    # 提取RealSense图像
    print("\n提取RealSense Color和Depth图像...")
    realsense_raw_dir = os.path.join(output_dir, "realsense_raw")
    rs_result, metadata_dir, _ = extract_realsense_images(
        bag_path, realsense_raw_dir, use_raw_depth=use_raw_depth
    )

    # 建立 Frame Counter 到图像的映射
    color_dir = os.path.join(realsense_raw_dir, "color")
    depth_dir = os.path.join(realsense_raw_dir, "depth")
    color_meta_dir = os.path.join(metadata_dir, "color")
    depth_meta_dir = os.path.join(metadata_dir, "depth")

    # 从所有帧收集已知的 Counter 列表（用于按顺序匹配）
    all_color_counters = sorted(set(f.get("color_counter") for f in frames if f.get("color_counter") is not None))
    all_depth_counters = sorted(set(f.get("depth_counter") for f in frames if f.get("depth_counter") is not None))
    print(f"  索引文件中 Color Counter 范围: {all_color_counters[0] if all_color_counters else 'N/A'} - {all_color_counters[-1] if all_color_counters else 'N/A'} (共{len(all_color_counters)}个)")
    print(f"  索引文件中 Depth Counter 范围: {all_depth_counters[0] if all_depth_counters else 'N/A'} - {all_depth_counters[-1] if all_depth_counters else 'N/A'} (共{len(all_depth_counters)}个)")

    color_counter_map = {}
    depth_counter_map = {}

    if os.path.exists(color_dir):
        color_counter_map = find_realsense_images_by_counter(color_dir, "Color", color_meta_dir, all_color_counters)
    if os.path.exists(depth_dir):
        depth_counter_map = find_realsense_images_by_counter(
            depth_dir, "Depth", depth_meta_dir, all_depth_counters, is_raw_depth=use_raw_depth
        )

    # 导出帧
    print("\n开始导出帧...")
    start_time = time.time()
    exported_count = 0
    error_count = 0

    for i, frame in enumerate(valid_frames):
        depth_idx = frame.get("depth_idx", i)
        tianmou_idx = frame.get("tianmou_idx", 0)
        depth_counter = frame.get("depth_counter", 0)
        color_counter = frame.get("color_counter", 0)

        # 读取天眸图像
        tianmou_img = get_tianmou_rgb(tianmou_reader, tianmou_idx)
        if tianmou_img is None:
            print(f"Warning: 无法读取天眸第{tianmou_idx}帧")
            error_count += 1
            continue

        # 获取天眸时间戳
        ts_us, ts_str = get_tianmou_timestamp(tianmou_reader, tianmou_idx)

        # 保存天眸图像
        tianmou_with_info = tianmou_img.copy()
        cv2.putText(tianmou_with_info, f"Tianmou #{tianmou_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(tianmou_with_info, f"Time: {ts_str}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        tianmou_path = os.path.join(tianmou_output, f"tianmou_{i:04d}.png")
        cv2.imwrite(tianmou_path, tianmou_with_info)

        # 读取Color图像（根据 Frame Counter 匹配）
        color_img = get_realsense_image_by_counter(color_counter, color_counter_map)
        if color_img is None:
            color_img = np.zeros_like(tianmou_img)
            cv2.putText(color_img, f"No Color (Counter={color_counter})", (50, tianmou_img.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        color_path = os.path.join(color_output, f"color_{i:04d}.png")
        cv2.imwrite(color_path, color_img)

        # 读取Depth图像（根据 Frame Counter 匹配）
        depth_img = get_realsense_image_by_counter(depth_counter, depth_counter_map, is_raw_depth=use_raw_depth)
        if depth_img is None:
            depth_img = np.zeros((tianmou_img.shape[0], tianmou_img.shape[1]), dtype=np.uint8)
            cv2.putText(depth_img, f"No Depth (Counter={depth_counter})", (50, tianmou_img.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        depth_path = os.path.join(depth_output, f"depth_{i:04d}.png")
        if use_raw_depth and depth_img.dtype == np.uint16:
            # 原始16位深度直接保存
            cv2.imwrite(depth_path, depth_img)
        else:
            cv2.imwrite(depth_path, depth_img)

        # 创建三路对比图像
        comparison = create_triple_comparison_image(
            tianmou_img, color_img, depth_img, frame, ts_str, is_raw_depth=use_raw_depth
        )
        comp_path = os.path.join(comparison_output, f"compare_{i:04d}.png")
        cv2.imwrite(comp_path, comparison)

        exported_count += 1

        # 进度输出
        if (i + 1) % 50 == 0 or i == len(valid_frames) - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(valid_frames) - i - 1) if i > 0 else 0
            print(f"  进度: {i+1}/{len(valid_frames)} ({100*(i+1)/len(valid_frames):.1f}%) | 耗时: {elapsed:.1f}s | ETA: {eta:.1f}s")

    print(f"\n{'='*60}")
    print(f"导出完成!")
    print(f"  成功导出: {exported_count} 帧")
    print(f"  错误: {error_count} 帧")
    print(f"  输出目录: {output_dir}")
    print(f"    - 天眸图像: {tianmou_output}")
    print(f"    - Color图像: {color_output}")
    print(f"    - Depth图像: {depth_output}")
    print(f"    - 对比图像: {comparison_output}")
    print(f"{'='*60}")

    return True


def main():
    parser = argparse.ArgumentParser(description="天眸、Depth、Color 三路对齐帧导出")
    parser.add_argument("--index", "-i", required=True, help="索引文件 (JSON)")
    parser.add_argument("--tianmou-dir", "-t", required=True, help="天眸数据目录")
    parser.add_argument("--bag", "-b", required=True, help="RealSense bag文件")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--max-frames", "-m", type=int, default=-1, help="最大导出帧数 (-1表示全部)")
    parser.add_argument("--use-raw-depth", "-r", action="store_true",
                       help="使用 rs-convert -r 提取原始 16 位深度 (Z16)，而不是处理后的深度")

    args = parser.parse_args()

    success = export_triple_aligned_frames(
        args.index,
        args.tianmou_dir,
        args.bag,
        args.output,
        max_frames=args.max_frames,
        use_raw_depth=args.use_raw_depth
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
