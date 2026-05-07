#!/usr/bin/env python3
"""
RealSense 纯净标定集导出工具 (按 Frame Counter 绝对对齐)
功能：
1. 从 .bag 文件中提取 Depth 和 Color 图像及 Metadata。
2. 扫描 Metadata，基于 Frame Counter 进行绝对匹配。
3. 支持设置物理相差帧数 (-n)。
4. 将匹配成功的图像重命名并导出到独立文件夹，方便标定程序直接读取。
"""

import os
import re
import tempfile
import shutil
import subprocess
import argparse
import time

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.stage_start_time = None
        self.current_stage = None

    def start_stage(self, stage_name: str):
        if self.current_stage:
            self._finish_stage()
        print(f"\n[{time.time()-self.start_time:.1f}s] {'='*50}")
        print(f"[{time.time()-self.start_time:.1f}s] 开始阶段: {stage_name}")
        self.current_stage = stage_name
        self.stage_start_time = time.time()

    def update_progress(self, current: int, total: int, prefix: str = "处理中"):
        if total == 0: return
        percent = current / total * 100
        elapsed = time.time() - self.stage_start_time
        eta = (elapsed / current * (total - current)) if current > 0 else 0
        print(f"\r[{time.time()-self.start_time:.1f}s] {prefix}: [{current}/{total}] {percent:.1f}% | ETA: {eta:.1f}s", end="", flush=True)

    def _finish_stage(self):
        elapsed = time.time() - self.stage_start_time
        print(f"\n[{time.time()-self.start_time:.1f}s] ✓ {self.current_stage} 完成 (耗时: {elapsed:.1f}s)")

    def finish(self):
        if self.current_stage:
            self._finish_stage()
        print(f"\n{'='*50}\n🎉 导出任务完成! 总耗时: {time.time() - self.start_time:.1f}s\n{'='*50}")

def parse_metadata_file(filepath: str) -> dict:
    """与 check_bag_sync 完全一致的宽松解析逻辑"""
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

def parse_metadata_stream(temp_dir: str, stream_type: str) -> dict:
    """暴力遍历目录，不再使用脆弱的 glob 路径拼接"""
    mapping = {}
    all_files = os.listdir(temp_dir)
    
    # 筛选对应的 txt 文件
    target_files = [f for f in all_files if f"{stream_type}_metadata_" in f and f.endswith(".txt")]
    
    for filename in target_files:
        filepath = os.path.join(temp_dir, filename)
        
        # 提取序号：直接提取文件名里的所有数字，取最后那一组 (完全免疫格式变化)
        digits = re.findall(r'\d+', filename)
        if not digits:
            continue
        idx_str = digits[-1]
        
        # 读取文件内容获取 Counter
        m = parse_metadata_file(filepath)
        if m["counter"] is not None:
            mapping[m["counter"]] = idx_str
            
    return mapping

def export_aligned_bag(bag_path: str, export_dir: str, shift_n: int):
    if not os.path.exists(bag_path):
        print(f"❌ 错误: 找不到文件 {bag_path}")
        return

    progress = ProgressTracker()
    temp_dir = tempfile.mkdtemp(prefix="rs_calib_extract_")
    output_prefix = os.path.join(temp_dir, "frame")

    try:
        # 1. 提取图像和 Metadata
        progress.start_stage("从 Bag 提取原始帧 (可能需要几十秒)")
        subprocess.run(['rs-convert', '-i', bag_path, '-d', '-p', output_prefix], stdout=subprocess.DEVNULL)
        subprocess.run(['rs-convert', '-i', bag_path, '-c', '-p', output_prefix], stdout=subprocess.DEVNULL)
        
        # 2. 解析 Metadata
        progress.start_stage("解析 Metadata 寻找 Counter")
        depth_map = parse_metadata_stream(temp_dir, "Depth")
        color_map = parse_metadata_stream(temp_dir, "Color")
        
        print(f"\n   提取到 Depth 帧: {len(depth_map)}")
        print(f"   提取到 Color 帧: {len(color_map)}")

        # 3. 寻找匹配对 (Color_Counter = Depth_Counter + shift_n)
        progress.start_stage(f"执行匹配 (Color_Counter = Depth_Counter + {shift_n})")
        matched_pairs = []
        for d_cnt, d_idx in depth_map.items():
            c_cnt = d_cnt + shift_n
            if c_cnt in color_map:
                matched_pairs.append({
                    "depth_idx": d_idx,
                    "color_idx": color_map[c_cnt],
                    "counter": d_cnt
                })
        
        matched_pairs.sort(key=lambda x: x["counter"])
        print(f"\n   严格匹配成功的完整帧对: {len(matched_pairs)}")

        if not matched_pairs:
            print("❌ 错误：未找到任何匹配的帧对，请检查 shift_n 是否设置正确。")
            return

        # 4. 导出 (增强型鲁棒搜索)
        progress.start_stage("导出纯净对齐图片")
        depth_out_dir = os.path.join(export_dir, "depth")
        color_out_dir = os.path.join(export_dir, "color")
        os.makedirs(depth_out_dir, exist_ok=True)
        os.makedirs(color_out_dir, exist_ok=True)
        
        # 辅助函数：在临时目录中暴力寻找匹配的图像文件
        def find_image_file(stream_type, idx_str):
            for ext in [".png", ".jpg", ".jpeg", ".raw"]:
                guess = os.path.join(temp_dir, f"frame_{stream_type}_{idx_str}{ext}")
                if os.path.exists(guess): return guess
            # 如果常规命名都找不到，直接遍历文件夹搜索
            for f in os.listdir(temp_dir):
                if stream_type in f and idx_str in f and not f.endswith('.txt'):
                    return os.path.join(temp_dir, f)
            return None

        total_export = len(matched_pairs)
        success_count = 0
        
        for i, pair in enumerate(matched_pairs):
            d_src = find_image_file("Depth", pair['depth_idx'])
            c_src = find_image_file("Color", pair['color_idx'])
            
            if d_src and c_src:
                new_name = f"{success_count:05d}.png" # 统一下发为 png 后缀名称（即使原图是 jpg）
                shutil.copy(d_src, os.path.join(depth_out_dir, new_name))
                shutil.copy(c_src, os.path.join(color_out_dir, new_name))
                success_count += 1
            else:
                if i == 0:  # 只在第一次找不到时打印警告，避免刷屏
                    print(f"\n   [警告] 找不到对应的图片！期望寻找 Depth_{pair['depth_idx']} 和 Color_{pair['color_idx']}")
                    print(f"   [调试] 临时目录内的前 5 个文件示例: {os.listdir(temp_dir)[:5]}")
            
            if (i + 1) % 10 == 0 or i == total_export - 1:
                progress.update_progress(i + 1, total_export, f"搬运对齐图像 ({success_count} 成功)")

        print(f"\n   导出路径: {export_dir}")
        print(f"   最终成功导出: {success_count} 组")
        progress.finish()

    finally:
        # 清理几百MB的临时解压文件
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense 纯净标定集导出工具")
    parser.add_argument("--bag", required=True, help="输入 bag 文件路径")
    parser.add_argument("-e", "--export-dir", required=True, help="导出对齐图像的目录")
    parser.add_argument("-n", "--shift-n", type=int, default=0, help="Color滞后Depth的帧数偏移(例如 -8)")
    
    args = parser.parse_args()
    export_aligned_bag(args.bag, args.export_dir, args.shift_n)