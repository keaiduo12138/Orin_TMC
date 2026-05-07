#!/usr/bin/env python3
"""
RealSense Genlock 深度同步分析工具
增加：前10帧明细对比、时间戳差值趋势分析
"""

import os
import re
import glob
import tempfile
import shutil
import subprocess
import argparse

def parse_metadata_file(filepath):
    res = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            fc_match = re.search(r"Frame Counter:\s*(\d+)", content)
            if fc_match:
                res["counter"] = int(fc_match.group(1))
            
            # 优先匹配 Frame Timestamp (硬件时钟)
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

def check_bag_sync(bag_path: str, max_count: int = 0):
    print(f"{'='*70}")
    print(f"深度同步分析: {os.path.basename(bag_path)}")
    print(f"{'='*70}")

    if not os.path.exists(bag_path):
        print(f"错误: 找不到文件 {bag_path}")
        return

    temp_dir = tempfile.mkdtemp(prefix="rs_sync_check_")
    output_prefix = os.path.join(temp_dir, "frame")

    try:
        print("正在提取 Metadata...")
        subprocess.run(['rs-convert', '-i', bag_path, '-d', '-p', output_prefix], stdout=subprocess.DEVNULL)
        subprocess.run(['rs-convert', '-i', bag_path, '-c', '-p', output_prefix], stdout=subprocess.DEVNULL)

        depth_files = glob.glob(f"{output_prefix}_Depth_metadata_*.txt")
        color_files = glob.glob(f"{output_prefix}_Color_metadata_*.txt")

        depth_data = {}
        for f in depth_files:
            m = parse_metadata_file(f)
            if m["counter"] is not None:
                depth_data[m["counter"]] = m["timestamp"]

        color_data = {}
        for f in color_files:
            m = parse_metadata_file(f)
            if m["counter"] is not None:
                color_data[m["counter"]] = m["timestamp"]

        common_counters = sorted(list(set(depth_data.keys()) & set(color_data.keys())))
        
        # --- 前 N 帧明细对比 ---
        count_limit = max_count if max_count > 0 else len(common_counters)
        print(f"\n前 {count_limit} 帧同步明细 (单位: us):")
        print(f"{'-'*75}")
        print(f"{'Counter':<10} | {'Depth Timestamp':<18} | {'Color Timestamp':<18} | {'Diff (ms)':<10}")
        print(f"{'-'*75}")

        for count in common_counters[:count_limit]:
            d_ts = depth_data[count]
            c_ts = color_data[count]
            diff_ms = abs(d_ts - c_ts) / 1000.0
            print(f"{count:<10} | {d_ts:<18} | {c_ts:<18} | {diff_ms:<10.2f}")
        print(f"{'-'*75}")

        # --- 基础统计 ---
        diffs = [abs(depth_data[c] - color_data[c]) for c in common_counters if depth_data[c] and color_data[c]]

        if diffs:
            print(f"\n统计汇总:")
            print(f"   共有帧数: {len(common_counters)}")
            print(f"   平均偏差: {sum(diffs)/len(diffs):.2f} us")
            print(f"   最大偏差: {max(diffs)} us")
            print(f"   最小偏差: {min(diffs)} us")

            # 判定偏差稳定性
            if max(diffs) - min(diffs) < 1000: # 波动小于 1ms
                print(f"   结论: 偏差非常【稳定】，存在固定时差（约 {sum(diffs)/len(diffs)/1000:.1f} ms）。")
            else:
                print(f"   结论: 偏差【不稳定】，同步链路存在抖动。")

        # --- 找出与最后一帧 Depth 时间戳最接近的 Color 帧 ---
        if common_counters:
            last_depth_counter = max(depth_data.keys())
            last_depth_ts = depth_data[last_depth_counter]

            closest_color_counter = None
            closest_diff = None
            for c_counter, c_ts in color_data.items():
                diff = abs(c_ts - last_depth_ts)
                if closest_diff is None or diff < closest_diff:
                    closest_diff = diff
                    closest_color_counter = c_counter

            print(f"\n与最后一帧 Depth 对齐的 Color 帧:")
            print(f"   - 最后一帧 Depth: Counter={last_depth_counter}, Timestamp={last_depth_ts}")
            print(f"   - 最接近的 Color: Counter={closest_color_counter}, Timestamp={color_data[closest_color_counter]}")
            print(f"   - 时间差: {closest_diff / 1000:.2f} ms")

        d_only = set(depth_data.keys()) - set(color_data.keys())
        c_only = set(color_data.keys()) - set(depth_data.keys())
        if d_only or c_only:
            print(f"\n独立帧 (丢帧) 分析:")
            if d_only: print(f"   - 只有 Depth 有的 Counter: {sorted(list(d_only))[:5]} ... (共{len(d_only)}个)")
            if c_only: print(f"   - 只有 Color 有的 Counter: {sorted(list(c_only))[:5]} ... (共{len(c_only)}个)")

    finally:
        shutil.rmtree(temp_dir)
        print(f"\n临时目录已清理。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense Genlock 深度同步分析工具")
    parser.add_argument("bag_path", help="bag 文件路径")
    parser.add_argument("--max", "-m", type=int, default=0, help="限制输出前 N 帧（默认 0 表示全部）")
    args = parser.parse_args()
    check_bag_sync(args.bag_path, max_count=args.max)