#!/usr/bin/env python3
"""
天眸与RealSense Color后处理对齐脚本（简化版，仅处理Color流）

功能：
1. 读取物理基准时间戳、天眸帧时间戳、RealSense Color帧时间戳
2. 找到锚定点（天眸第n帧 = RealSense Color第1帧）
3. 检测跳帧和重复帧
4. 输出JSON索引文件

跳帧检测逻辑（基于dt与33ms的比值）：
- dt = 0    → duplicate（两个Color对应一个天眸，跳过这个Color帧）
- dt ≈ 33  → 正常帧
- dt ≈ 66  → skip 1（天眸拍了两帧，Color只拍了一帧）
- dt ≈ 99  → skip 2（天眸拍了两帧，Color只拍了一帧）
- ...
- 公式：skip_count = round(dt / 33) - 1
"""

import os
import sys
import json
import re
import glob
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
        """开始一个新阶段"""
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
        """更新进度"""
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
        """结束当前阶段"""
        if self.current_stage and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            for stage in reversed(self.stages):
                if stage["name"] == self.current_stage and stage["status"] == "running":
                    stage["status"] = "completed"
                    stage["elapsed"] = elapsed
                    break

            print(f"\n[{time.time()-self.start_time:.1f}s] ✓ {self.current_stage} 完成 (耗时: {elapsed:.1f}s)")

    def finish(self):
        """结束所有阶段"""
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


# 全局进度追踪器
progress = ProgressTracker()


def read_temporal_timestamp(filepath: str) -> Optional[int]:
    """
    读取物理基准时间戳

    Args:
        filepath: 时间戳文件路径

    Returns:
        Unix微秒时间戳，如果没有效数据则返回None
    """
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


def find_matching_temporal_file(temporal_dir: str, target_time_sec: float) -> Optional[str]:
    """
    在temporal_files目录中找到与目标时间匹配的时间戳文件
    """
    pattern = os.path.join(temporal_dir, "tianmou_timestamp_*.csv")
    files = glob.glob(pattern)

    best_match = None
    best_diff = float('inf')

    for filepath in files:
        try:
            ts_us = read_temporal_timestamp(filepath)
            if ts_us is None:
                continue
            ts_sec = ts_us / 1_000_000
            diff = abs(ts_sec - target_time_sec)
            if diff < best_diff:
                best_diff = diff
                best_match = filepath
        except Exception as e:
            continue

    if best_match and best_diff < 60:
        print(f"找到匹配的时间戳文件: {os.path.basename(best_match)}, 差值: {best_diff:.2f}秒")
        return best_match

    return None


def read_tianmou_timestamps_from_info(info_file: str) -> List[int]:
    """
    从info.txt文件读取天眸帧时间戳（备选方案）
    """
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
    """
    使用 TianmoucDataReader 读取天眸帧时间戳

    这是更精确的方法，C_timestamp[0] 与物理时间戳完全匹配

    Args:
        tianmou_dir: 天眸数据目录
        camera_idx: 相机索引 (0=cone, 1=rod)

    Returns:
        时间戳列表(微秒)
    """
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
    """
    在天眸时间戳序列中找到与物理时间戳最接近的帧

    Args:
        tianmou_timestamps: 天眸帧时间戳列表(微秒)
        physical_ts_us: 物理基准时间戳(微秒)

    Returns:
        匹配的天眸帧索引
    """
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


def extract_color_metadata(bag_path: str, output_dir: str = "/projects/cxr_data/metadata_color_temp") -> str:
    """
    使用rs-convert提取RealSense Color metadata

    Args:
        bag_path: bag文件路径
        output_dir: 输出目录

    Returns:
        metadata文件所在目录
    """
    global progress

    os.makedirs(output_dir, exist_ok=True)

    color_meta_files = sorted(glob.glob(f"{output_dir}/*_Color_metadata_*.txt"))
    if color_meta_files:
        print(f"找到已提取的Color metadata文件: {len(color_meta_files)} 个")
        return output_dir

    print(f"正在提取RealSense Color metadata...")
    print(f"  Bag文件: {bag_path}")
    bag_size_mb = os.path.getsize(bag_path) / (1024 * 1024)
    print(f"  Bag大小: {bag_size_mb:.1f} MB")

    start_time = time.time()

    # 只提取 Color
    cmd_color = ['rs-convert', '-i', bag_path, '-c', '-p', os.path.join(output_dir, 'frame') + '_']
    print(f"  执行命令: rs-convert -i {bag_path} -c -p {output_dir}/frame_")
    result = subprocess.run(cmd_color, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"Warning: rs-convert -c 失败: {result.stderr}")

    elapsed = time.time() - start_time

    color_meta_files = sorted(glob.glob(f"{output_dir}/*_Color_metadata_*.txt"))
    print(f"✓ RealSense Color metadata提取完成")
    print(f"  Color metadata: {len(color_meta_files)} 个文件")
    print(f"  耗时: {elapsed:.1f}s")
    return output_dir


def read_color_timestamps(metadata_dir: str, temporal_timestamp_us: int = None) -> Tuple[List[float], List[int]]:
    """
    从Color metadata文件读取时间戳

    Args:
        metadata_dir: metadata文件所在目录
        temporal_timestamp_us: 物理基准时间戳（微秒），用于过滤只保留该时间点之后的帧

    Returns:
        (Frame Timestamp列表[毫秒], 帧号列表)
    """
    color_metadata_files = sorted(glob.glob(f"{metadata_dir}/*_Color_metadata_*.txt"))
    if not color_metadata_files:
        color_metadata_files = sorted(glob.glob(f"{metadata_dir}/timestamps_*_Color_metadata_*.txt"))

    print(f"  找到 Color metadata: {len(color_metadata_files)} 个")

    if not color_metadata_files:
        print(f"Warning: 未找到metadata文件 in {metadata_dir}")
        return [], []

    print(f"  过滤前metadata数量: {len(color_metadata_files)}")

    frame_timestamps_ms = []
    frame_numbers = []

    for f in color_metadata_files:
        with open(f, 'r') as fp:
            content = fp.read()

            # 提取Frame Timestamp (微秒，需要转为毫秒)
            match = re.search(r'Frame Timestamp:\s*(\d+)', content)
            if match:
                ft_us = int(match.group(1))
                frame_timestamps_ms.append(ft_us / 1000.0)

            # 提取Frame Counter
            fc_match = re.search(r'Frame Counter:\s*(\d+)', content)
            fc = int(fc_match.group(1)) if fc_match else len(frame_numbers)
            frame_numbers.append(fc)

    # 按Frame Timestamp排序（确保帧是按时间顺序）
    if frame_timestamps_ms and len(frame_timestamps_ms) == len(frame_numbers):
        sorted_pairs = sorted(zip(frame_timestamps_ms, frame_numbers))
        frame_timestamps_ms, frame_numbers = zip(*sorted_pairs)
        frame_timestamps_ms = list(frame_timestamps_ms)
        frame_numbers = list(frame_numbers)

    total_before_filter = len(frame_timestamps_ms)

    # 过滤：如果提供了temporal_timestamp，只保留temporal之后的帧
    if temporal_timestamp_us is not None:
        temporal_ms = temporal_timestamp_us / 1000

        filtered_frames = []
        for i, ft in enumerate(frame_timestamps_ms):
            if ft >= temporal_ms - 500:
                filtered_frames.append(i)

        if filtered_frames:
            print(f"  过滤后保留 {len(filtered_frames)} 帧 (从temporal时间点开始)")
            frame_timestamps_ms = [frame_timestamps_ms[i] for i in filtered_frames]
            frame_numbers = [frame_numbers[i] for i in filtered_frames]

    if frame_timestamps_ms:
        print(f"  Color帧数: {len(frame_timestamps_ms)}")
        if len(frame_timestamps_ms) > 1:
            dts = [frame_timestamps_ms[i+1] - frame_timestamps_ms[i] for i in range(min(10, len(frame_timestamps_ms)-1))]
            dts_valid = [d for d in dts if d > 0]
            if dts_valid:
                avg_ms = sum(dts_valid) / len(dts_valid)
                print(f"  平均帧间隔: {avg_ms:.2f} ms")
                print(f"  对应帧率: {1000/avg_ms:.1f} fps")

    return frame_timestamps_ms, frame_numbers


def detect_skipped_frames_color(color_ts_ms: List[float], start_idx: int = 0) -> List[dict]:
    """
    检测跳帧和重复帧

    逻辑：
    - dt = 0 → duplicate（两个Color帧对应同一个天眸帧，跳过这个Color帧）
    - dt ≈ 33 → 正常帧
    - dt ≈ 66 → skip 1（天眸拍了两帧，Color只拍了一帧）
    - dt ≈ 99 → skip 2（天眸拍了两帧，Color只拍了一帧）
    - 公式：skip_count = round(dt / 33) - 1

    Args:
        color_ts_ms: Color Frame Timestamp列表(毫秒)
        start_idx: 开始检查的索引

    Returns:
        跳帧信息列表，每项包含 {frame_idx, dt, skip_count, type}
            - type: "skip" 表示跳帧（天眸多帧），"duplicate" 表示重复帧
    """
    skipped = []

    for i in range(start_idx + 1, len(color_ts_ms)):
        dt = color_ts_ms[i] - color_ts_ms[i-1]

        # 考虑浮点数精度，dt < 1ms 视为 0
        if abs(dt) < 1.0:
            # dt ≈ 0: 重复帧，两个Color帧对应同一个天眸帧
            skipped.append({"frame_idx": i, "dt": dt, "skip_count": 1, "type": "duplicate"})
            print(f"  检测到重复帧: Color第{i}帧, dt={dt:.2f}ms (两个Color对应一帧天眸)")
        else:
            # 计算跳帧数：dt / 33 并四舍五入，然后减1得到跳过的帧数
            skip_count = round(dt / EXPECTED_INTERVAL_MS) - 1

            if skip_count > 0:
                # 跳帧：天眸拍了 skip_count+1 帧，Color只拍了一帧
                skipped.append({"frame_idx": i, "dt": dt, "skip_count": skip_count, "type": "skip"})
                print(f"  检测到跳帧: Color第{i}帧, dt={dt:.2f}ms, 跳{skip_count}帧")

    return skipped


def build_frame_mapping_color(
    anchor_n: int,
    color_ts_ms: List[float],
    tianmou_timestamps: List[int],
    skipped_frames: List[dict]
) -> List[Dict[str, Any]]:
    """
    构建天眸与Color的帧对齐映射

    处理逻辑：
    - dt = 0 (duplicate): 跳过这个Color帧，天眸帧不变
    - dt > 33 (skip): Color帧跳过了 skip_count 帧

    Args:
        anchor_n: 锚定点天眸帧索引
        color_ts_ms: Color Frame Timestamp列表
        tianmou_timestamps: 天眸时间戳列表
        skipped_frames: 跳帧信息列表

    Returns:
        帧映射列表
    """
    frames = []

    # 将跳帧信息转换为索引->信息的映射
    skip_info = {sf["frame_idx"]: {"skip_count": sf["skip_count"], "type": sf["type"]}
                 for sf in skipped_frames}

    frame_id = 0
    tianmou_idx = anchor_n
    color_idx = 0

    while color_idx < len(color_ts_ms) and tianmou_idx < len(tianmou_timestamps):
        skip_info_frame = skip_info.get(color_idx, {"skip_count": 0, "type": None})
        skip_count = skip_info_frame["skip_count"]
        skip_type = skip_info_frame["type"]

        # dt_ms 计算
        dt_ms = color_ts_ms[color_idx] - color_ts_ms[color_idx-1] if color_idx > 0 else None

        if skip_type == "duplicate":
            # dt ≈ 0: 重复帧，跳过这个Color帧
            frames.append({
                "frame_id": frame_id,
                "tianmou_idx": tianmou_idx,
                "color_idx": color_idx,
                "dt_ms": round(dt_ms, 3) if dt_ms is not None else None,
                "skip": True,
                "skip_type": "duplicate",
                "skip_count": 1
            })
            # 只增加color_idx，不增加tianmou_idx
            color_idx += 1
            frame_id += 1
        elif skip_count > 0:
            # 跳帧：跳过 skip_count 个天眸帧
            frames.append({
                "frame_id": frame_id,
                "tianmou_idx": tianmou_idx,
                "color_idx": color_idx,
                "dt_ms": round(dt_ms, 3) if dt_ms is not None else None,
                "skip": False,
                "skip_type": None,
                "skip_count": 0,
                "note": f"跳帧前，下一帧需要跳过{skip_count}个天眸帧"
            })
            # Color帧 +1，天眸帧跳过 skip_count + 1 帧
            tianmou_idx += skip_count + 1
            color_idx += 1
            frame_id += 1
        else:
            # 正常帧
            frames.append({
                "frame_id": frame_id,
                "tianmou_idx": tianmou_idx,
                "color_idx": color_idx,
                "dt_ms": round(dt_ms, 3) if dt_ms is not None else None,
                "skip": False,
                "skip_type": None,
                "skip_count": 0
            })
            tianmou_idx += 1
            color_idx += 1
            frame_id += 1

    return frames


def process_session(tianmou_dir: str, bag_path: str, temporal_file: str,
                   output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    处理一对天眸和RealSense Color数据

    Args:
        tianmou_dir: 天眸数据目录
        bag_path: RealSense bag文件路径
        temporal_file: 物理基准时间戳文件
        output_path: 输出JSON文件路径

    Returns:
        处理结果字典
    """
    global progress

    print(f"\n{'='*60}")
    print(f"天眸与RealSense Color后处理对齐 (简化版)")
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

    # ========== 阶段3: 提取Color时间戳 ==========
    progress.start_stage("提取RealSense Color时间戳", "使用rs-convert从bag文件提取Color metadata")
    print(f"  Bag文件: {bag_path}")
    metadata_dir = extract_color_metadata(bag_path)
    if not metadata_dir:
        print("Error: 无法提取RealSense Color metadata")
        progress.stages[-1]["status"] = "failed"
        return {}

    color_ts_ms, color_frame_nums = read_color_timestamps(metadata_dir, physical_ts_us)
    if not color_ts_ms:
        print("Error: 无法读取Color时间戳")
        progress.stages[-1]["status"] = "failed"
        return {}
    print(f"  Color总帧数: {len(color_ts_ms)}")
    print(f"  Color首帧时间戳: {color_ts_ms[0]:.3f} ms")
    print(f"  Color末帧时间戳: {color_ts_ms[-1]:.3f} ms")
    progress._finish_stage()

    # ========== 阶段4: 找到锚定点 ==========
    progress.start_stage("锚定点匹配", "寻找天眸帧与Color帧的起始对齐点")
    anchor_n = find_anchor_tianmou_frame(tianmou_timestamps, physical_ts_us)
    if anchor_n is None:
        print("Error: 无法找到锚定点")
        progress.stages[-1]["status"] = "failed"
        return {}
    print(f"  锚定点: 天眸第{anchor_n}帧 <-> Color第0帧")
    progress._finish_stage()

    # ========== 阶段5: 检测跳帧 ==========
    progress.start_stage("检测跳帧", "分析Color帧间隔，检测跳帧和重复帧")
    print(f"  从锚定点开始检测跳帧...")
    skipped_frames = detect_skipped_frames_color(color_ts_ms, start_idx=0)
    print(f"\n  跳帧检测结果:")
    print(f"    检测到跳帧数: {len(skipped_frames)}")
    duplicate_count = sum(1 for sf in skipped_frames if sf["type"] == "duplicate")
    skip_count = sum(1 for sf in skipped_frames if sf["type"] == "skip")
    print(f"    - 重复帧 (duplicate): {duplicate_count}")
    print(f"    - 跳帧 (skip): {skip_count}")
    progress._finish_stage()

    # ========== 阶段6: 构建帧映射 ==========
    progress.start_stage("构建帧映射", "生成对齐的帧索引文件")
    frames = build_frame_mapping_color(anchor_n, color_ts_ms, tianmou_timestamps, skipped_frames)
    print(f"  ✓ 帧映射构建完成，共 {len(frames)} 个映射")
    progress._finish_stage()

    # ========== 阶段7: 保存结果 ==========
    progress.start_stage("保存结果", "将结果写入JSON文件")

    # 统计信息
    total_skipped = sum(f.get("skip_count", 0) for f in frames if f.get("skip_type") == "skip")
    total_duplicate = sum(1 for f in frames if f.get("skip_type") == "duplicate")

    result = {
        "input": {
            "tianmou_dir": tianmou_dir,
            "bag_path": bag_path,
            "temporal_file": temporal_file,
            "timestamp_source": "TianmoucDataReader (C_timestamp[0] * 10)"
        },
        "anchor": {
            "tianmou_frame_index": anchor_n,
            "color_frame_index": 0,
            "physical_timestamp_us": physical_ts_us
        },
        "statistics": {
            "total_tianmou_frames": len(tianmou_timestamps),
            "total_color_frames": len(color_ts_ms),
            "skip_frame_count": len(skipped_frames),
            "duplicate_count": duplicate_count,
            "skip_total": total_skipped,
            "matched_frames_count": len(frames)
        },
        "color_timestamps_ms": color_ts_ms,
        "frames": frames
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存到: {output_path}")
    progress._finish_stage()

    # 打印最终统计
    progress.finish()

    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  天眸帧: {len(tianmou_timestamps)}")
    print(f"  Color帧: {len(color_ts_ms)}")
    print(f"  锚定点: 天眸第{anchor_n}帧")
    print(f"  重复帧 (duplicate): {duplicate_count}")
    print(f"  跳帧 (skip): {skip_count}")
    print(f"  有效映射: {len(frames)}")
    print(f"{'='*60}")

    return result


def find_matching_files(data_dir: str, temporal_file: str) -> Tuple[Optional[str], Optional[str]]:
    """
    根据temporal文件找到对应的天眸目录和bag文件
    """
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
    parser = argparse.ArgumentParser(description="天眸与RealSense Color后处理对齐 (简化版)")
    parser.add_argument("--tianmou-dir", help="天眸数据目录 (如 /projects/cxr_data/20260316_1749/)")
    parser.add_argument("--bag", help="RealSense bag文件")
    parser.add_argument("--temporal", help="物理基准时间戳文件")
    parser.add_argument("--auto", action="store_true", help="自动匹配文件")
    parser.add_argument("--data-dir", default=DEFAULT_CXR_DATA_DIR, help="数据根目录")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")

    args = parser.parse_args()

    if args.auto:
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
    else:
        if not all([args.tianmou_dir, args.bag, args.temporal]):
            parser.error("需要提供 --tianmou-dir, --bag, --temporal 或使用 --auto")
            return
        tianmou_dir = args.tianmou_dir
        bag_path = args.bag

    result = process_session(tianmou_dir, bag_path, args.temporal, args.output)

    if result:
        print("\n处理完成!")
    else:
        print("\n处理失败!")


if __name__ == "__main__":
    main()
