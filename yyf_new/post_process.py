#!/usr/bin/env python3
"""
天眸与RealSense后处理对齐脚本（修复版：三路对齐）

功能：
1. 读取物理基准时间戳、天眸帧时间戳
2. 找到锚定点（天眸第n帧 = 物理时间戳对应的天眸帧）
3. Depth 与天眸对齐（所有帧根据 dt 检测跳帧）
4. Color 与 Depth 对齐（基于 Frame Counter + 相位偏移）
5. 输出三路对齐的 JSON 索引文件

对齐逻辑：
- Depth 与天眸：dt ≈ 33ms 正常，dt ≈ 66ms 跳1帧，dt ≈ 99ms 跳2帧，dt ≈ 0 重复
- Color 与 Depth：
  1. 先找共有 Frame Counter 的帧
  2. 计算 Diff = |Color_Timestamp - Depth_Timestamp| / 1000 (ms)
  3. 相位偏移 = round(Diff / 33)
  4. 根据偏移将 Color 对到对应的 Depth
  5. 丢弃没有共有 Frame Counter 的帧
  6. 丢弃没有 Color 匹配的 Depth 帧
  7. 最终只输出三路都有的帧
"""

import os
import sys
import json
import re
import glob
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# 常量配置
DEFAULT_TEMPORAL_DIR = "/projects/cxr_data/temporal_files"
DEFAULT_CXR_DATA_DIR = "/projects/cxr_data"

# 硬件触发参数
EXPECTED_INTERVAL_MS = 33.0  # 理论帧间隔（ms）


# ========== 进度追踪类 ==========
class ProgressTracker:
    """进度追踪器"""

    def __init__(self):
        self.stages = []
        self.current_stage = None
        self.start_time = time.time()
        self.stage_start_time = None

    def start_stage(self, stage_name: str, description: str = ""):
        if self.current_stage is not None:
            self._finish_stage()

        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.stages.append({
            "name": stage_name,
            "description": description,
            "start_time": self.stage_start_time,
            "status": "running"
        })

        elapsed = time.time() - self.start_time
        print(f"\n[{elapsed:.1f}s] {'='*50}")
        print(f"[{elapsed:.1f}s] 阶段 {len(self.stages)}: {stage_name}")
        if description:
            print(f"[{elapsed:.1f}s] {description}")
        print(f"[{elapsed:.1f}s] {'='*50}")

    def update_progress(self, current: int, total: int, prefix: str = "进度"):
        if total == 0:
            return
        percent = current / total * 100
        elapsed = time.time() - (self.stage_start_time or time.time())
        if current > 0:
            eta = elapsed / current * (total - current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = ""
        print(f"\r[{time.time()-self.start_time:.1f}s] {prefix}: {current}/{total} ({percent:.1f}%) {eta_str}", end="", flush=True)

    def _finish_stage(self):
        if self.current_stage and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            for stage in reversed(self.stages):
                if stage["name"] == self.current_stage and stage["status"] == "running":
                    stage["status"] = "completed"
                    stage["elapsed"] = elapsed
                    break
            print(f"\n[{time.time()-self.start_time:.1f}s] ✓ {self.current_stage} 完成 (耗时: {elapsed:.1f}s)")

    def finish(self):
        if self.current_stage is not None:
            self._finish_stage()
        total_elapsed = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"总耗时: {total_elapsed:.1f}s")
        print(f"{'='*50}")
        print("\n各阶段用时:")
        for i, stage in enumerate(self.stages, 1):
            elapsed = stage.get("elapsed", 0)
            print(f"  {i}. {stage['name']}: {elapsed:.1f}s")


progress = ProgressTracker()


def read_temporal_timestamp(filepath: str) -> Optional[int]:
    """读取物理基准时间戳（微秒）"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[0] == 'ok':
                    return int(parts[1])
    except Exception as e:
        print(f"Error: 读取时间戳文件失败: {e}")
    return None


def read_tianmou_timestamps_from_info(info_file: str) -> List[int]:
    """从info.txt文件读取天眸帧时间戳（备选方案）"""
    timestamps = []
    with open(info_file, 'r') as f:
        for line in f:
            if 'unix standard time stamp:' in line:
                match = re.search(r'unix standard time stamp:(\d+)s,(\d+)us', line)
                if match:
                    sec = int(match.group(1))
                    us = int(match.group(2))
                    ts_us = sec * 1_000_000 + us
                    timestamps.append(ts_us)
    return timestamps


def read_tianmou_timestamps_from_reader(tianmou_dir: str, camera_idx: int = 0) -> List[int]:
    """使用 TianmoucDataReader 读取天眸帧时间戳"""
    global progress

    try:
        from tianmoucv.data import TianmoucDataReader
    except (ImportError, AttributeError, Exception) as e:
        print(f"Warning: 无法导入 tianmoucv ({e})，使用 info.txt 作为备选")
        cone_dir = os.path.join(tianmou_dir, "cone")
        rod_dir = os.path.join(tianmou_dir, "rod")

        info_file = None
        if os.path.exists(cone_dir):
            info_files = glob.glob(f"{cone_dir}/info_*.txt")
            if info_files:
                info_file = info_files[0]

        if not info_file and os.path.exists(rod_dir):
            info_files = glob.glob(f"{rod_dir}/info_*.txt")
            if info_files:
                info_file = info_files[0]

        if info_file:
            print(f"使用 info.txt 作为备选: {info_file}")
            return read_tianmou_timestamps_from_info(info_file)
        return []

    print(f"初始化 TianmoucDataReader...")
    reader = TianmoucDataReader(tianmou_dir, print_info=False, N=1, camera_idx=camera_idx, strict=True)
    total_frames = len(reader)
    print(f"总帧数: {total_frames}")

    timestamps = []
    start_time = time.time()

    for i in range(total_frames):
        sample = reader[i]
        meta = sample.get("meta", {})
        Cts = meta.get("C_timestamp", None)

        if Cts is None or len(Cts) < 2:
            ts_us = 0
        else:
            ts_start_10us = int(Cts[0])
            ts_us = ts_start_10us * 10

        timestamps.append(ts_us)

        if (i + 1) % 50 == 0 or i == total_frames - 1:
            elapsed = time.time() - start_time
            percent = (i + 1) / total_frames * 100
            eta = elapsed / (i + 1) * (total_frames - i - 1) if i > 0 else 0
            print(f"  天眸帧读取: {i+1}/{total_frames} ({percent:.1f}%) | 耗时: {elapsed:.1f}s | ETA: {eta:.1f}s")

    print(f"✓ 天眸帧读取完成, 共 {len(timestamps)} 帧")
    return timestamps


def find_anchor_tianmou_frame(tianmou_timestamps: List[int], physical_ts_us: int) -> Optional[int]:
    """在天眸时间戳序列中找到与物理时间戳最接近的帧"""
    if not tianmou_timestamps:
        return None

    min_diff = float('inf')
    closest_idx = 0

    for i, ts in enumerate(tianmou_timestamps):
        diff = abs(ts - physical_ts_us)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    print(f"找到锚定点: 天眸第{closest_idx}帧 ≈ 物理时间戳")
    print(f"  天眸帧时间戳: {tianmou_timestamps[closest_idx]} us")
    print(f"  物理时间戳: {physical_ts_us} us")
    print(f"  差值: {min_diff} us ({min_diff/1000:.3f} ms)")

    return closest_idx


def extract_realsense_metadata(bag_path: str, output_dir: str = "/projects/cxr_data/metadata_temp") -> str:
    """使用rs-convert提取RealSense Color和Depth metadata"""
    global progress

    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")

    # 方案1: 每次提取前清空临时目录，确保干净的环境
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"已清空临时目录: {output_dir}")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"正在提取RealSense metadata...")
    print(f"  Bag文件: {bag_path}")
    bag_size_mb = os.path.getsize(bag_path) / (1024 * 1024)
    print(f"  Bag大小: {bag_size_mb:.1f} MB")

    start_time = time.time()

    cmd_color = ['rs-convert', '-i', bag_path, '-c', '-p', os.path.join(color_dir, 'frame') + '_']
    print(f"  执行命令: rs-convert -i {bag_path} -c -p {color_dir}/frame_")
    result = subprocess.run(cmd_color, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"Warning: rs-convert -c 失败: {result.stderr}")

    cmd_depth = ['rs-convert', '-i', bag_path, '-d', '-p', os.path.join(depth_dir, 'frame') + '_']
    print(f"  执行命令: rs-convert -i {bag_path} -d -p {depth_dir}/frame_")
    result = subprocess.run(cmd_depth, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"Warning: rs-convert -d 失败: {result.stderr}")

    elapsed = time.time() - start_time

    color_meta = sorted(glob.glob(os.path.join(color_dir, "*_Color_metadata_*.txt")))
    depth_meta = sorted(glob.glob(os.path.join(depth_dir, "*_Depth_metadata_*.txt")))
    print(f"✓ RealSense metadata提取完成")
    print(f"  Color metadata: {len(color_meta)} 个文件")
    print(f"  Depth metadata: {len(depth_meta)} 个文件")
    print(f"  耗时: {elapsed:.1f}s")
    return output_dir


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


def read_realsense_frame_data(metadata_dir: str) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
    """
    读取 RealSense Color 和 Depth 的 Frame Counter 和 Frame Timestamp

    Returns:
        (color_counters, depth_counters, color_data, depth_data)
        - counters: 按时间排序的 Frame Counter 列表
        - data: {frame_counter: frame_timestamp_us}
    """
    color_dir = os.path.join(metadata_dir, "color")
    depth_dir = os.path.join(metadata_dir, "depth")

    # 用 dict 存储，key 是 Frame Counter，value 是 timestamp
    # 方案2: 用 dict 存储，key 是 Frame Counter
    # 同一个 Frame Counter 可能对应多个文件，保留最后一个（最新）
    color_data = {}
    depth_data = {}

    color_files = sorted(glob.glob(os.path.join(color_dir, "*_Color_metadata_*.txt")))
    for f in color_files:
        m = parse_metadata_file(f)
        if m["counter"] is not None and m["timestamp"] is not None:
            color_data[m["counter"]] = m["timestamp"]

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*_Depth_metadata_*.txt")))
    for f in depth_files:
        m = parse_metadata_file(f)
        if m["counter"] is not None and m["timestamp"] is not None:
            depth_data[m["counter"]] = m["timestamp"]

    # 按时间戳排序得到 Counter 列表
    color_counters = [cnt for cnt, _ in sorted(color_data.items(), key=lambda x: x[1])]
    depth_counters = [cnt for cnt, _ in sorted(depth_data.items(), key=lambda x: x[1])]

    return color_counters, depth_counters, color_data, depth_data


def build_depth_tianmou_mapping(
    anchor_n: int,
    depth_counters: List[int],
    depth_data: Dict[int, int],
    tianmou_timestamps: List[int]
) -> List[Dict[str, Any]]:
    """
    Depth 与天眸对齐（简化版，所有帧统一处理）

    逻辑：
    - 以 Depth 帧为基准，从锚定点开始逐帧匹配天眸
    - dt ≈ 33ms: 正常帧，天眸+1, Depth+1
    - dt ≈ 66ms: 跳帧，天眸+2, Depth+1
    - dt ≈ 99ms: 跳帧，天眸+3, Depth+1
    - dt ≈ 0: 重复帧，天眸不变, Depth+1

    Returns:
        列表，每个元素包含：
        - depth_idx: 对齐后的 Depth 帧索引（从0开始）
        - depth_counter: Depth 的 Frame Counter
        - tianmou_idx: 对应的天眸帧索引
        - dt_ms: 帧间隔
        - skip_count: 跳过的天眸帧数
        - is_valid: 是否有效
    """
    frames = []

    tianmou_idx = anchor_n
    depth_idx = 0

    while depth_idx < len(depth_counters) and tianmou_idx < len(tianmou_timestamps):
        depth_counter = depth_counters[depth_idx]
        depth_ts = depth_data[depth_counter]

        dt_ms = None
        if depth_idx > 0:
            prev_depth_ts = depth_data[depth_counters[depth_idx - 1]]
            dt_ms = (depth_ts - prev_depth_ts) / 1000.0  # 转为 ms

        # 计算跳帧数
        skip_count = 0
        if dt_ms is not None:
            if abs(dt_ms) < 1.0:
                # dt ≈ 0: 重复帧
                skip_count = 0
            else:
                # dt / 33 - 1 得到跳过的帧数
                skip_count = round(dt_ms / EXPECTED_INTERVAL_MS) - 1
                skip_count = max(0, skip_count)

        frames.append({
            "depth_idx": depth_idx,
            "depth_counter": depth_counter,
            "tianmou_idx": tianmou_idx,
            "dt_ms": round(dt_ms, 3) if dt_ms is not None else None,
            "skip_count": skip_count,
            "is_valid": True  # 初始标记为有效，后面会根据 Color 对齐情况更新
        })

        # 更新索引
        tianmou_idx += 1 + skip_count
        depth_idx += 1

    return frames


def build_color_depth_mapping(depth_frames: List[Dict], depth_data: Dict[int, int], color_data: Dict[int, int]) -> Dict[int, int]:
    """
    建立 Color Frame Counter 到 Depth Frame Counter 的映射

    逻辑：
    1. 找共有 Frame Counter 的帧
    2. 计算 Diff = |Color_Timestamp - Depth_Timestamp| / 1000 (ms)
    3. 相位偏移 = round(Diff / 33)
    4. 返回映射: {depth_counter: color_counter}

    示例：
    Counter=0, Diff=292.18ms, 292.18/33=8.85≈9
    → Counter=0 的 Color 对应 Counter=9 的 Depth
    → 映射: {9: 0} (depth_counter=9 对应 color_counter=0)

    Returns:
        映射 dict: {depth_counter: color_counter}
    """
    common_counters = sorted(set(depth_data.keys()) & set(color_data.keys()))

    print(f"  共有 Frame Counter 数量: {len(common_counters)}")

    if not common_counters:
        print(f"  Warning: 没有共有的 Frame Counter！")
        return {}

    # 计算每个共有帧的相位偏移（保留方向：正=Color晚于Depth，负=Color早于Depth）
    offset_map = {}  # {color_counter: offset}
    for cnt in common_counters:
        d_ts = depth_data[cnt]
        c_ts = color_data[cnt]
        diff_ms = (c_ts - d_ts) / 1000.0   # 不再取 abs()
        offset = round(diff_ms / EXPECTED_INTERVAL_MS)
        offset_map[cnt] = offset

    # 打印前几个偏移情况
    print(f"  前10个共有帧的偏移情况:")
    print(f"  {'Counter':<10} | {'Diff(ms)':<10} | {'偏移(帧)':<10} | {'说明':<25}")
    print(f"  {'-'*65}")
    for cnt in common_counters[:10]:
        d_ts = depth_data[cnt]
        c_ts = color_data[cnt]
        diff_ms = (c_ts - d_ts) / 1000.0
        offset = offset_map[cnt]
        if offset == 0:
            note = "无偏移"
        elif offset > 0:
            note = f"Color晚于Depth {offset} 帧"
        else:
            note = f"Color早于Depth {-offset} 帧"
        print(f"  {cnt:<10} | {diff_ms:<10.2f} | {offset:<10} | {note:<25}")

    # 建立映射: depth_counter -> color_counter
    # diff = c_ts - d_ts（保留方向）
    # offset = round(diff / 33)
    # depth_cnt = color_cnt + offset
    #   offset > 0: Color 晚于 Depth，depth counter 比 color counter 大
    #   offset < 0: Color 早于 Depth，depth counter 比 color counter 小
    mapping = {}  # {depth_counter: color_counter}
    for color_cnt in common_counters:
        offset = offset_map[color_cnt]
        depth_cnt = color_cnt + offset
        if depth_cnt in depth_data:  # 确保这个 depth counter 存在
            mapping[depth_cnt] = color_cnt

    # 统计偏移分布
    offset_counts = {}
    for cnt in common_counters:
        off = offset_map[cnt]
        offset_counts[off] = offset_counts.get(off, 0) + 1
    print(f"\n  偏移分布: {offset_counts}")

    return mapping


def mark_invalid_frames(depth_frames: List[Dict], color_depth_mapping: Dict[int, int]):
    """
    根据 Color-Depth 映射标记无效帧

    规则：
    1. 如果 depth_counter 不在 mapping 中，标记为无效（只有 Depth 没有 Color）
    2. 只有共有的 Frame Counter 才有 Color，所以已经在 mapping 中的帧是有效的
    """
    valid_count = 0
    invalid_count = 0

    for frame in depth_frames:
        depth_counter = frame["depth_counter"]
        if depth_counter in color_depth_mapping:
            frame["has_color"] = True
            frame["color_counter"] = color_depth_mapping[depth_counter]
            valid_count += 1
        else:
            frame["has_color"] = False
            frame["color_counter"] = None
            frame["is_valid"] = False  # 标记为无效
            invalid_count += 1

    print(f"\n  Color 对齐结果:")
    print(f"    有效帧（含 Color）: {valid_count}")
    print(f"    无效帧（无 Color）: {invalid_count}")

    return depth_frames


def process_session(tianmou_dir: str, bag_path: str, temporal_file: str,
                   output_path: Optional[str] = None) -> Dict[str, Any]:
    """处理一对天眸和RealSense数据（三路对齐）"""
    global progress

    print(f"\n{'='*60}")
    print(f"天眸与RealSense后处理对齐 (三路对齐版)")
    print(f"{'='*60}")
    print(f"输入参数:")
    print(f"  天眸目录: {tianmou_dir}")
    print(f"  Bag文件: {bag_path}")
    print(f"  时间戳文件: {temporal_file}")
    print(f"  输出文件: {output_path or 'None'}")
    print(f"{'='*60}")

    # ========== 阶段1: 读取物理基准时间戳 ==========
    progress.start_stage("读取物理基准时间戳", "从temporal文件读取外部电路触发时间")
    physical_ts_us = read_temporal_timestamp(temporal_file)
    if not physical_ts_us:
        print("Error: 无法读取物理时间戳")
        progress.stages[-1]["status"] = "failed"
        return {}
    print(f"  物理基准时间戳: {physical_ts_us} us")
    progress._finish_stage()

    # ========== 阶段2: 读取天眸帧时间戳 ==========
    progress.start_stage("读取天眸帧时间戳", "使用TianmoucDataReader读取C_timestamp")
    print(f"  天眸目录: {tianmou_dir}")
    tianmou_timestamps = read_tianmou_timestamps_from_reader(tianmou_dir, camera_idx=0)
    if not tianmou_timestamps:
        print("Error: 无法读取天眸时间戳")
        progress.stages[-1]["status"] = "failed"
        return {}
    print(f"  天眸总帧数: {len(tianmou_timestamps)}")
    print(f"  天眸首帧时间戳: {tianmou_timestamps[0]} us")
    print(f"  天眸末帧时间戳: {tianmou_timestamps[-1]} us")
    progress._finish_stage()

    # ========== 阶段3: 提取RealSense metadata ==========
    progress.start_stage("提取RealSense Metadata", "使用rs-convert从bag文件提取Color和Depth metadata")
    print(f"  Bag文件: {bag_path}")
    metadata_dir = extract_realsense_metadata(bag_path)
    if not metadata_dir:
        print("Error: 无法提取RealSense metadata")
        progress.stages[-1]["status"] = "failed"
        return {}
    progress._finish_stage()

    # ========== 阶段4: 读取RealSense帧数据 ==========
    progress.start_stage("读取RealSense帧数据", "解析Frame Counter和Frame Timestamp")
    color_counters, depth_counters, color_data, depth_data = read_realsense_frame_data(metadata_dir)
    print(f"  Color帧数据: {len(color_data)} 帧 (去重后)")
    print(f"  Depth帧数据: {len(depth_data)} 帧 (去重后)")
    print(f"  Color Frame Counter 范围: {min(color_counters)} - {max(color_counters)}" if color_counters else "  Color: 无")
    print(f"  Depth Frame Counter 范围: {min(depth_counters)} - {max(depth_counters)}" if depth_counters else "  Depth: 无")
    progress._finish_stage()

    # ========== 阶段5: 找到锚定点 ==========
    progress.start_stage("锚定点匹配", "寻找天眸帧与RealSense帧的起始对齐点")
    anchor_n = find_anchor_tianmou_frame(tianmou_timestamps, physical_ts_us)
    if anchor_n is None:
        print("Error: 无法找到锚定点")
        progress.stages[-1]["status"] = "failed"
        return {}
    print(f"  锚定点: 天眸第{anchor_n}帧 <-> Depth第0帧")
    progress._finish_stage()

    # ========== 阶段6: Depth与天眸对齐 ==========
    progress.start_stage("Depth与天眸对齐", "以Depth为基准，所有帧统一检测跳帧")

    # 使用实际的 Frame Counter 列表
    print(f"  Depth帧数: {len(depth_counters)}")
    if len(depth_counters) > 1:
        # 计算帧间隔
        dts = []
        for i in range(1, min(10, len(depth_counters))):
            dt_us = depth_data[depth_counters[i]] - depth_data[depth_counters[i-1]]
            dts.append(dt_us / 1000.0)
        avg_ms = sum(dts) / len(dts) if dts else 0
        print(f"  平均帧间隔: {avg_ms:.2f} ms")

    depth_frames = build_depth_tianmou_mapping(anchor_n, depth_counters, depth_data, tianmou_timestamps)
    print(f"  ✓ Depth-天眸对齐完成，共 {len(depth_frames)} 个映射")
    progress._finish_stage()

    # ========== 阶段7: Color与Depth对齐 ==========
    progress.start_stage("Color与Depth对齐", "基于Frame Counter和相位偏移")

    # 构建 depth_counter -> color_counter 映射
    color_depth_mapping = build_color_depth_mapping(depth_frames, depth_data, color_data)

    # 标记无效帧
    depth_frames = mark_invalid_frames(depth_frames, color_depth_mapping)
    progress._finish_stage()

    # ========== 阶段8: 输出最终结果 ==========
    progress.start_stage("生成最终结果", "只保留三路都有的帧")

    # 统计信息
    valid_frames = [f for f in depth_frames if f.get("is_valid", False) and f.get("has_color", False)]
    invalid_depth_no_color = sum(1 for f in depth_frames if not f.get("has_color", False))

    statistics = {
        "total_tianmou_frames": len(tianmou_timestamps),
        "total_depth_frames": len(depth_frames),
        "total_color_frames": len(color_data),
        "common_frame_counters": len(set(depth_data.keys()) & set(color_data.keys())),
        "valid_triple_aligned": len(valid_frames),
        "invalid_no_color": invalid_depth_no_color
    }

    # 构建结果
    result = {
        "input": {
            "tianmou_dir": tianmou_dir,
            "bag_path": bag_path,
            "temporal_file": temporal_file,
            "timestamp_source": "TianmoucDataReader (C_timestamp[0] * 10)"
        },
        "anchor": {
            "tianmou_frame_index": anchor_n,
            "depth_frame_index": 0,
            "physical_timestamp_us": physical_ts_us
        },
        "statistics": statistics,
        "frames": valid_frames  # 只输出三路都有的帧
    }

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存到: {output_path}")

    progress._finish_stage()
    progress.finish()

    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  天眸帧: {statistics['total_tianmou_frames']}")
    print(f"  Depth帧: {statistics['total_depth_frames']}")
    print(f"  Color帧: {statistics['total_color_frames']}")
    print(f"  锚定点: 天眸第{anchor_n}帧")
    print(f"  有效三路对齐帧: {statistics['valid_triple_aligned']}")
    print(f"  无效（无Color）: {statistics['invalid_no_color']}")
    print(f"{'='*60}")

    return result


def find_data_in_root(root_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从根目录自动探测三路数据路径。

    Returns:
        (tianmou_dir, bag_path, temporal_file)
    """
    tianmou_dir = os.path.join(root_dir, "tianmou")
    if not os.path.isdir(tianmou_dir):
        print(f"Error: 天眸目录不存在: {tianmou_dir}")
        return None, None, None

    bag_files = glob.glob(os.path.join(root_dir, "**/*.bag"), recursive=True)
    if not bag_files:
        print(f"Error: 未找到 bag 文件 in {root_dir}")
        return None, None, None
    if len(bag_files) > 1:
        print(f"Warning: 发现多个 bag 文件，取第一个: {bag_files[0]}")
    bag_path = bag_files[0]

    temporal_files = glob.glob(os.path.join(root_dir, "temporal_files/tianmou_timestamp_*.csv"))
    if not temporal_files:
        temporal_files = glob.glob(os.path.join(root_dir, "**/tianmou_timestamp_*.csv"), recursive=True)
    if not temporal_files:
        print(f"Error: 未找到 temporal 时间戳文件 in {root_dir}")
        return None, None, None
    if len(temporal_files) > 1:
        print(f"Warning: 发现多个 temporal 文件，取第一个: {temporal_files[0]}")
    temporal_file = temporal_files[0]

    print(f"自动探测结果:")
    print(f"  根目录: {root_dir}")
    print(f"  天眸:   {tianmou_dir}")
    print(f"  Bag:    {bag_path}")
    print(f"  Temporal: {temporal_file}")

    return tianmou_dir, bag_path, temporal_file


def find_matching_files(data_dir: str, temporal_file: str) -> Tuple[Optional[str], Optional[str]]:
    """根据temporal文件找到对应的天眸目录和bag文件"""
    basename = os.path.basename(temporal_file)
    match = re.search(r'tianmou_timestamp_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.csv', basename)
    if not match:
        return None, None

    date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}_{match.group(4)}{match.group(5)}"

    tianmou_dir = os.path.join(data_dir, date_str)
    if not os.path.exists(tianmou_dir):
        print(f"Warning: 天眸目录不存在: {tianmou_dir}")
        return None, None

    date_ymd = f"2026-{match.group(2)}-{match.group(3)}"
    bag_dir = os.path.join(data_dir, date_ymd)

    if not os.path.exists(bag_dir):
        print(f"Warning: bag目录不存在: {bag_dir}")
        return tianmou_dir, None

    target_hour = int(match.group(4))
    target_min = int(match.group(5))
    target_time = target_hour * 60 + target_min

    bag_files = glob.glob(f"{bag_dir}/*.bag")
    best_bag = None
    best_diff = float('inf')

    for bf in bag_files:
        bf_name = os.path.basename(bf)
        bag_match = re.match(r'(\d{1,2})-(\d{2})-(\d{2})\.bag', bf_name)
        if bag_match:
            bh = int(bag_match.group(1))
            bm = int(bag_match.group(2))
            bag_time = bh * 60 + bm
            diff = abs(bag_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_bag = bf

    return tianmou_dir, best_bag


def main():
    parser = argparse.ArgumentParser(description="天眸与RealSense后处理对齐 (三路对齐版)")
    parser.add_argument("--root", help="数据根目录（自动探测天眸、bag、temporal）")
    parser.add_argument("--tianmou-dir", help="天眸数据目录")
    parser.add_argument("--bag", help="RealSense bag文件")
    parser.add_argument("--temporal", help="物理基准时间戳文件")
    parser.add_argument("--auto", action="store_true", help="自动匹配文件")
    parser.add_argument("--data-dir", default=DEFAULT_CXR_DATA_DIR, help="数据根目录（auto模式用）")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")

    args = parser.parse_args()

    # 模式1: --root 模式，自动探测
    if args.root:
        tianmou_dir, bag_path, temporal_file = find_data_in_root(args.root)
        if not all([tianmou_dir, bag_path, temporal_file]):
            print("Error: --root 模式无法找到全部三路数据，请检查目录结构")
            return
    # 模式2: --auto 模式，根据 temporal 文件名推算路径
    elif args.auto:
        if not args.temporal:
            temporal_files = sorted(glob.glob(f"{DEFAULT_TEMPORAL_DIR}/tianmou_timestamp_*.csv"))
            if not temporal_files:
                print("Error: 未找到temporal文件")
                return
            args.temporal = temporal_files[-1]
            print(f"使用最新的temporal文件: {args.temporal}")

        tianmou_dir, bag_path = find_matching_files(args.data_dir, args.temporal)
        if not tianmou_dir:
            print("Error: 无法找到对应的天眸目录")
            return
        if not bag_path:
            print("Warning: 未找到对应的bag文件")
        temporal_file = args.temporal
    # 模式3: 三件套手动指定
    else:
        if not all([args.tianmou_dir, args.bag, args.temporal]):
            parser.error("需要提供 --root, 或 --tianmou-dir + --bag + --temporal, 或使用 --auto")
            return
        tianmou_dir = args.tianmou_dir
        bag_path = args.bag
        temporal_file = args.temporal

    result = process_session(tianmou_dir, bag_path, temporal_file, args.output)

    if result:
        print("\n处理完成!")
    else:
        print("\n处理失败!")


if __name__ == "__main__":
    main()
