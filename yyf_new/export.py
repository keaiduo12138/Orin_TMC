#!/usr/bin/env python3
"""
天眸与RealSense后处理对齐与标定导出脚本

功能：
1. 读取物理基准时间戳、天眸帧时间戳
2. 解析Depth和Color的Metadata，按Frame Counter强制匹配
3. 支持Color与Depth的物理帧偏移修正 (-n)
4. 检测稳定期跳帧，剔除无效帧
5. 一键导出完全对齐的(Tianmou, Depth, Color)图像集用于相机标定
"""

import os
import sys
import json
import re
import glob
import subprocess
import argparse
import time
import shutil
from typing import Optional, Tuple, List, Dict, Any

# 常量配置
DEFAULT_TEMPORAL_DIR = "/projects/cxr_data/temporal_files"
DEFAULT_CXR_DATA_DIR = "/projects/cxr_data"

DT_NORMAL_MS = 33.33  # 正常帧间隔 (~30fps)
DT_THRESHOLD_MS = DT_NORMAL_MS * 1.5  # 超过则判定跳帧
UNSTABLE_FRAME_COUNT = 150  # 不稳定期帧数
DT_MATCH_THRESHOLD_MS = 0.1  # 不稳定期对齐阈值


# ========== 进度追踪类 ==========
class ProgressTracker:
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
        self.stages.append({"name": stage_name, "status": "running"})
        elapsed = time.time() - self.start_time
        print(f"\n[{elapsed:.1f}s] {'='*50}")
        print(f"[{elapsed:.1f}s] 阶段 {len(self.stages)}: {stage_name} - {description}")
    
    def update_progress(self, current: int, total: int, prefix: str = "进度"):
        """动态刷新进度条"""
        if total == 0:
            return
        percent = current / total * 100
        elapsed = time.time() - (self.stage_start_time or self.start_time)
        eta = (elapsed / current * (total - current)) if current > 0 else 0
        # 使用 \r 回车符实现同行刷新
        print(f"\r[{time.time()-self.start_time:.1f}s] {prefix}: [{current}/{total}] {percent:.1f}% | ETA: {eta:.1f}s", end="", flush=True)
    
    def _finish_stage(self):
        if self.current_stage and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            # 先换行，避免覆盖进度条
            print(f"\n[{time.time()-self.start_time:.1f}s] ✓ {self.current_stage} 完成 (耗时: {elapsed:.1f}s)")
    
    def finish(self):
        if self.current_stage is not None:
            self._finish_stage()
        print(f"\n{'='*50}\n总耗时: {time.time() - self.start_time:.1f}s\n{'='*50}")


progress = ProgressTracker()


def read_temporal_timestamp(filepath: str) -> Optional[int]:
    try:
        with open(filepath, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[0] == 'ok':
                    return int(parts[1])
    except Exception as e:
        print(f"Error: 读取时间戳文件失败: {e}")
    return None

def read_tianmou_timestamps_from_reader(tianmou_dir: str, camera_idx: int = 0) -> List[int]:
    try:
        from tianmoucv.data import TianmoucDataReader
    except ImportError:
        print("Error: 无法导入 tianmoucv，必须依赖其读取天眸时间戳")
        return []
    
    print(f"初始化 TianmoucDataReader...")
    reader = TianmoucDataReader(tianmou_dir, print_info=False, N=1, camera_idx=camera_idx, strict=True)
    total_frames = len(reader)
    timestamps = []
    for i in range(total_frames):
        sample = reader[i]
        Cts = sample.get("meta", {}).get("C_timestamp", None)
        ts_us = int(Cts[0]) * 10 if Cts and len(Cts) >= 2 else 0
        timestamps.append(ts_us)
        # 每解析10帧刷新一次进度条
        if (i + 1) % 10 == 0 or i == total_frames - 1:
            progress.update_progress(i + 1, total_frames, "解析天眸帧")
            
    print(f"\n✓ 天眸帧读取完成, 共 {len(timestamps)} 帧")
    return timestamps

def find_anchor_tianmou_frame(tianmou_timestamps: List[int], physical_ts_us: int) -> Optional[int]:
    if not tianmou_timestamps: return None
    min_diff = float('inf')
    closest_idx = 0
    for i, ts in enumerate(tianmou_timestamps):
        diff = abs(ts - physical_ts_us)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    print(f"找到锚定点: 天眸第{closest_idx}帧, 差值: {min_diff/1000:.3f} ms")
    return closest_idx

def extract_realsense_metadata(bag_path: str, output_dir: str = "/projects/cxr_data/metadata_temp") -> str:
    os.makedirs(output_dir, exist_ok=True)
    bag_name = os.path.basename(bag_path).replace('.bag', '')
    prefix_d = os.path.join(output_dir, f"{bag_name}_Depth")
    prefix_c = os.path.join(output_dir, f"{bag_name}_Color")
    
    if glob.glob(f"{prefix_d}_metadata_*.txt") and glob.glob(f"{prefix_c}_metadata_*.txt"):
        print(f"找到已提取的metadata文件: {output_dir}")
        return output_dir
    
    print(f"正在提取RealSense metadata (可能需要几秒钟)...")
    subprocess.run(['rs-convert', '-i', bag_path, '-d', '-p', output_dir + '/'], capture_output=True)
    subprocess.run(['rs-convert', '-i', bag_path, '-c', '-p', output_dir + '/'], capture_output=True)
    return output_dir

def parse_metadata_stream(metadata_dir: str, stream_type: str) -> Dict[int, dict]:
    """解析单个流的Metadata，返回 {Counter: info_dict}"""
    # 兼容有没有前缀的各种文件名
    files = glob.glob(f"{metadata_dir}/*{stream_type}_metadata_*.txt")
    stream_data = {}
    total_files = len(files)
    
    for i, f in enumerate(files):
        # 提取文件序号，去掉了前面死板的下划线要求
        idx_match = re.search(rf'{stream_type}_metadata_(\d+)\.txt', f)
        if not idx_match: continue
        file_idx = int(idx_match.group(1))
        
        with open(f, 'r') as fp:
            content = fp.read()
            
        fc_match = re.search(r'Frame Counter:\s*(\d+)', content)
        ts_match = re.search(r'Frame Timestamp:\s*(\d+)', content)
        toa_match = re.search(r'Time Of Arrival:\s*(\d+)', content)
        
        if fc_match and ts_match and toa_match:
            counter = int(fc_match.group(1))
            stream_data[counter] = {
                "file_idx": file_idx,
                "counter": counter,
                "ts_ms": int(ts_match.group(1)) / 1000.0,
                "toa": int(toa_match.group(1))
            }
            
        if (i + 1) % 50 == 0 or i == total_files - 1:
            progress.update_progress(i + 1, total_files, f"解析 {stream_type} Metadata")
            
    print("") 
    return stream_data

def read_and_match_realsense(metadata_dir: str, shift_n: int, temporal_ts_us: int) -> List[dict]:
    """
    强制通过Frame Counter匹配Depth和Color
    Color_Counter = Depth_Counter + shift_n
    """
    depth_data = parse_metadata_stream(metadata_dir, "Depth")
    color_data = parse_metadata_stream(metadata_dir, "Color")
    
    temporal_ms = temporal_ts_us / 1000.0 if temporal_ts_us else 0
    matched_pairs = []
    
    for d_cnt, d_info in depth_data.items():
        c_cnt = d_cnt + shift_n
        if c_cnt in color_data:
            c_info = color_data[c_cnt]
            # 根据Depth的ToA过滤时间点之前的脏数据
            if d_info["toa"] < temporal_ms - 100:
                continue
            matched_pairs.append({
                "ts_ms": d_info["ts_ms"], # 以Depth的时间戳作为主要基准
                "depth": d_info,
                "color": c_info
            })
            
    matched_pairs.sort(key=lambda x: x["ts_ms"])
    print(f"  解析到 Depth 帧: {len(depth_data)}")
    print(f"  解析到 Color 帧: {len(color_data)}")
    print(f"  (基于移位 n={shift_n}) 严格匹配成功的完整帧: {len(matched_pairs)}")
    return matched_pairs

def find_unstable_end_y(tianmou_timestamps: List[int], anchor_n: int,
                        matched_rs: List[dict], x: int = UNSTABLE_FRAME_COUNT) -> int:
    if len(matched_rs) <= x:
        x = len(matched_rs) - 1
    rs_delta_ms = matched_rs[x]["ts_ms"] - matched_rs[0]["ts_ms"]
    
    anchor_ts_us = tianmou_timestamps[anchor_n]
    best_y, best_diff = x, float('inf')
    
    for y in range(x, len(tianmou_timestamps) - anchor_n):
        tm_delta_ms = (tianmou_timestamps[anchor_n + y] - anchor_ts_us) / 1000.0
        diff = abs(tm_delta_ms - rs_delta_ms)
        if diff < best_diff:
            best_diff, best_y = diff, y
        if diff < DT_MATCH_THRESHOLD_MS:
            return y
    return best_y

def detect_skipped_frames(matched_rs: List[dict], start_idx: int) -> List[dict]:
    skipped = []
    for i in range(start_idx + 1, len(matched_rs)):
        dt = matched_rs[i]["ts_ms"] - matched_rs[i-1]["ts_ms"]
        if dt == 0:
            skipped.append({"idx": i, "dt": dt, "skip_count": 1, "type": "duplicate"})
        elif dt > DT_THRESHOLD_MS:
            skip_count = max(1, round(dt / DT_NORMAL_MS) - 1)
            skipped.append({"idx": i, "dt": dt, "skip_count": skip_count, "type": "skip"})
    return skipped

def build_frame_mapping(anchor_n: int, unstable_end_y: int, unstable_end_rs_idx: int,
                        tianmou_timestamps: List[int], matched_rs: List[dict],
                        skipped_frames: List[dict]) -> List[Dict]:
    frames = []
    skip_info = {sf["idx"]: sf for sf in skipped_frames}
    
    tianmou_idx = anchor_n
    rs_idx = 0
    max_frames = max(len(tianmou_timestamps) - anchor_n, len(matched_rs))
    
    for i in range(max_frames):
        if rs_idx >= len(matched_rs): break
        
        rs_item = matched_rs[rs_idx]
        is_unstable = rs_idx <= unstable_end_rs_idx
        dt_ms = None if rs_idx == 0 else rs_item["ts_ms"] - matched_rs[rs_idx-1]["ts_ms"]
        
        base_info = {
            "tianmou_idx": tianmou_idx,
            "rs_match_idx": rs_idx,
            "depth_file_idx": rs_item["depth"]["file_idx"],
            "color_file_idx": rs_item["color"]["file_idx"],
            "depth_counter": rs_item["depth"]["counter"],
            "color_counter": rs_item["color"]["counter"],
            "dt_ms": round(dt_ms, 3) if dt_ms else None,
            "phase": "unstable" if is_unstable else "stable"
        }
        
        if is_unstable:
            frames.append({**base_info, "skip": False})
            tianmou_idx += 1
            rs_idx += 1
        else:
            sf = skip_info.get(rs_idx, {"skip_count": 0, "type": None})
            if sf["type"] == "duplicate":
                frames.append({**base_info, "skip": True, "skip_reason": "duplicate"})
                rs_idx += 1
            else:
                is_skip = sf["skip_count"] > 0
                frames.append({**base_info, "skip": is_skip, "skip_reason": "hardware_drop" if is_skip else None})
                tianmou_idx += sf["skip_count"] + 1
                rs_idx += 1
                
    return frames

def export_aligned_frames(frames: List[dict], bag_path: str, tianmou_dir: str, export_dir: str):
    import cv2
    tm_dir = os.path.join(export_dir, "tianmou")
    depth_dir = os.path.join(export_dir, "depth")
    color_dir = os.path.join(export_dir, "color")
    rs_raw_dir = os.path.join(export_dir, "rs_raw")
    
    for d in [tm_dir, depth_dir, color_dir, rs_raw_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 提取 RS 图片
    progress.start_stage("导出RealSense物理帧", "使用rs-convert提取PNG至临时目录")
    subprocess.run(['rs-convert', '-i', bag_path, '-d', '-p', os.path.join(rs_raw_dir, "frame")], stdout=subprocess.DEVNULL)
    subprocess.run(['rs-convert', '-i', bag_path, '-c', '-p', os.path.join(rs_raw_dir, "frame")], stdout=subprocess.DEVNULL)
    progress._finish_stage()
    
    # 提取 Tianmou 并搬运
    progress.start_stage("生成纯净标定集", "丢弃无效帧，重命名并对齐三组图片")
    from tianmoucv.data import TianmoucDataReader
    reader = TianmoucDataReader(tianmou_dir, print_info=False, N=1, camera_idx=0, strict=True)
    
    export_count = 0
    # 计算需要导出的稳定帧总数，以便提供进度条
    total_to_export = sum(1 for f in frames if not (f["skip"] or f["phase"] != "stable"))
    
    for frm in frames:
        if frm["skip"] or frm["phase"] != "stable":
            continue  # 丢弃跳帧和不稳定帧
            
        name = f"{export_count:05d}.png"
        src_depth = os.path.join(rs_raw_dir, f"frame_Depth_{frm['depth_file_idx']:05d}.png")
        src_color = os.path.join(rs_raw_dir, f"frame_Color_{frm['color_file_idx']:05d}.png")
        
        if not (os.path.exists(src_depth) and os.path.exists(src_color)):
            continue
            
        # 复制RS图像
        shutil.copy(src_depth, os.path.join(depth_dir, name))
        shutil.copy(src_color, os.path.join(color_dir, name))
        
        # 提取Tianmou并转为常规图像
        try:
            tm_sample = reader[frm["tianmou_idx"]]
            tm_img = tm_sample['F0'].cpu().numpy()
            if tm_img.max() <= 1.0:
                tm_img = (tm_img * 255).astype(np.uint8)
            else:
                tm_img = tm_img.astype(np.uint8)
            tm_img = cv2.cvtColor(tm_img, cv2.COLOR_RGB2BGR) # 天眸库默认RGB，保存需转BGR
            cv2.imwrite(os.path.join(tm_dir, name), tm_img)
            
            export_count += 1
            # 刷新导出进度
            if export_count % 5 == 0 or export_count == total_to_export:
                progress.update_progress(export_count, total_to_export, "组装对齐数据集")
                
        except Exception as e:
            print(f"\nWarning: 导出天眸帧 {frm['tianmou_idx']} 失败: {e}")
            
    # 清理临时解压出来的全部冗余图片
    shutil.rmtree(rs_raw_dir, ignore_errors=True)
    print(f"\n🎉 导出完成！共提取 {export_count} 组完美的对齐标定图像。")
    print(f"   路径: {export_dir}")

def process_session(tianmou_dir, bag_path, temporal_file, output_path, shift_n, export_dir):
    global progress
    progress.start_stage("读取物理基准时间戳")
    physical_ts_us = read_temporal_timestamp(temporal_file)
    progress._finish_stage()
    
    progress.start_stage("读取天眸帧时间戳")
    tianmou_timestamps = read_tianmou_timestamps_from_reader(tianmou_dir, camera_idx=0)
    progress._finish_stage()
    
    progress.start_stage("提取并强制匹配RealSense双流")
    # ================= 核心修复点 =================
    temp_meta_dir = "/projects/cxr_data/metadata_temp"
    os.makedirs(temp_meta_dir, exist_ok=True)
    
    # 强制指定输出前缀包含 "frame"，确保生成文件名为 frame_Depth_metadata_xxx.txt
    out_prefix = os.path.join(temp_meta_dir, "frame")
    print("⏳ 正在调用 rs-convert 提取底层 Metadata...")
    subprocess.run(['rs-convert', '-i', bag_path, '-d', '-p', out_prefix], capture_output=True)
    subprocess.run(['rs-convert', '-i', bag_path, '-c', '-p', out_prefix], capture_output=True)
    
    matched_rs = read_and_match_realsense(temp_meta_dir, shift_n, physical_ts_us)
    # ==============================================
    progress._finish_stage()
    
    if not matched_rs:
        print("错误：未找到任何匹配的RS帧对，请检查 bag 文件或 shift_n 的设置。")
        return
        
    progress.start_stage("寻找锚定点与不稳定期")
    anchor_n = find_anchor_tianmou_frame(tianmou_timestamps, physical_ts_us)
    x = min(UNSTABLE_FRAME_COUNT, len(matched_rs) - 1)
    unstable_end_y = find_unstable_end_y(tianmou_timestamps, anchor_n, matched_rs, x)
    progress._finish_stage()
    
    progress.start_stage("检测稳定期硬件跳帧")
    skipped_frames = detect_skipped_frames(matched_rs, start_idx=x)
    progress._finish_stage()
    
    progress.start_stage("构建整体映射矩阵")
    frames = build_frame_mapping(anchor_n, unstable_end_y, x, tianmou_timestamps, matched_rs, skipped_frames)
    progress._finish_stage()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({"shift_n_applied": shift_n, "frames": frames}, f, indent=2)
            
    if export_dir:
        export_aligned_frames(frames, bag_path, tianmou_dir, export_dir)
        
    progress.finish()


def main():
    parser = argparse.ArgumentParser(description="天眸与RealSense极致对齐与标定导出")
    parser.add_argument("--tianmou-dir", required=True, help="天眸数据目录")
    parser.add_argument("--bag", required=True, help="RealSense bag文件")
    parser.add_argument("--temporal", required=True, help="物理基准时间戳文件")
    parser.add_argument("-o", "--output", help="输出JSON映射文件路径")
    
    # 新增核心参数
    parser.add_argument("-n", "--shift-n", type=int, default=0, 
                        help="Color相对于Depth的物理滞后补偿。例如Color慢了8帧，此处输入 -8")
    parser.add_argument("-e", "--export-dir", type=str, 
                        help="一键导出对齐图像的目录。若不填则只输出映射关系不解压图像。")
    
    args = parser.parse_args()
    process_session(args.tianmou_dir, args.bag, args.temporal, args.output, args.shift_n, args.export_dir)

if __name__ == "__main__":
    main()