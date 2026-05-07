#!/usr/bin/env python3
"""
后处理测试脚本 - 导出对齐帧进行验证

功能：
1. 从 RealSense bag 文件提取深度帧图像
2. 从天眸 tmdat 文件读取 RGB 帧图像
3. 按照 output_*.json 的索引将对齐帧导出到同一目录
4. 可选：将两个图像合成为一张便于对比

用法：
    python3 yyf/post_process_test.py output_0317_2.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


class PostProcessTester:
    def __init__(self, json_path: str):
        """初始化测试器"""
        self.json_path = json_path
        self.output_dir = json_path.replace('.json', '_frames')
        
        # 加载索引文件
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.input_info = self.data.get('input', {})
        self.anchor = self.data.get('anchor', {})
        self.unstable = self.data.get('unstable_period', {})
        self.frames = self.data.get('frames', [])
        
        # 输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'tianmou'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'realsense'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'merged'), exist_ok=True)
        
        # 初始化天眸读取器 (使用 rod 相机，camera_idx=1)
        self.tianmou_reader = None
        self.total_tianmou_frames = 0
        self.init_tianmou_reader(camera_idx=1)
        
        print(f"输出目录: {self.output_dir}")
    
    def get_realsense_stable_start_idx(self) -> int:
        """获取稳定期开始的帧索引"""
        return self.unstable.get('realsense_stable_frame', 150)
    
    def extract_realsense_depth_frames(self) -> str:
        """使用 rs-convert 提取 RealSense 深度帧"""
        bag_path = self.input_info.get('bag_path')
        if not bag_path or not os.path.exists(bag_path):
            print(f"Warning: Bag 文件不存在: {bag_path}")
            return None
        
        depth_dir = bag_path.replace('.bag', '_depth_frames')
        
        if os.path.exists(depth_dir) and len(os.listdir(depth_dir)) > 0:
            print(f"RealSense 深度帧已存在于: {depth_dir}")
            return depth_dir
        
        print(f"正在使用 rs-convert 提取 RealSense 深度帧...")
        print(f"  输入: {bag_path}")
        print(f"  输出: {depth_dir}/")
        
        cmd = ['rs-convert', '-i', bag_path, '-p', depth_dir + '/']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"  ✓ 提取完成")
                return depth_dir
            else:
                print(f"  ✗ 提取失败: {result.stderr}")
                return None
        except FileNotFoundError:
            print("  ✗ rs-convert 未找到，请安装 Intel RealSense SDK")
            return None
        except subprocess.TimeoutExpired:
            print("  ✗ rs-convert 超时")
            return None
    
    def init_tianmou_reader(self, camera_idx: int = 1):
        """初始化天眸读取器（使用 TianmoucDataReader）
        
        Args:
            camera_idx: 相机索引 (0=cone, 1=rod)
        """
        tianmou_dir = self.input_info.get('tianmou_dir')
        if not tianmou_dir:
            self.tianmou_reader = None
            self.total_tianmou_frames = 0
            return
        
        try:
            from tianmoucv.data import TianmoucDataReader
            print(f"初始化 TianmoucDataReader (camera_idx={camera_idx})...")
            self.tianmou_reader = TianmoucDataReader(tianmou_dir, print_info=False, N=1, camera_idx=camera_idx)
            self.total_tianmou_frames = len(self.tianmou_reader)
            print(f"天眸总帧数: {self.total_tianmou_frames}")
        except ImportError:
            print("Warning: 无法导入 tianmoucv")
            self.tianmou_reader = None
            self.total_tianmou_frames = 0
    
    def read_tianmou_frame(self, tianmou_idx: int) -> Optional[np.ndarray]:
        """读取天眸单帧图像
        
        Args:
            tianmou_idx: 天眸帧索引
        
        Returns:
            RGB 图像 numpy 数组
        """
        if self.tianmou_reader is None:
            return None
        
        if tianmou_idx >= self.total_tianmou_frames:
            print(f"  Warning: 天眸索引 {tianmou_idx} 超出范围 ({self.total_tianmou_frames} 帧)")
            return None
        
        try:
            sample = self.tianmou_reader[tianmou_idx]
            # 获取 RGB 图像
            rgb = sample.get('rgb')
            if rgb is not None:
                return rgb
            
            # 如果没有 RGB，尝试从其他通道获取
            # TianmoucDataReader 返回的格式可能需要检查
            for key in ['image', 'frame', 'data']:
                if key in sample:
                    img = sample[key]
                    if isinstance(img, np.ndarray) and len(img.shape) == 3:
                        return img
            
            return None
        except Exception as e:
            print(f"  Error 读取天眸帧 {tianmou_idx}: {e}")
            return None
    
    def read_realsense_frame(self, realsense_idx: int, depth_dir: str) -> Optional[np.ndarray]:
        """读取 RealSense 深度帧
        
        Args:
            realsense_idx: RealSense 帧索引
            depth_dir: 深度帧目录
        
        Returns:
            深度图像 numpy 数组
        """
        # 获取所有 png 文件并排序
        png_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        
        if realsense_idx >= len(png_files):
            print(f"  Warning: RealSense 索引 {realsense_idx} 超出范围 ({len(png_files)} 帧)")
            return None
        
        png_path = os.path.join(depth_dir, png_files[realsense_idx])
        
        try:
            img = Image.open(png_path)
            return np.array(img)
        except Exception as e:
            print(f"  Error 读取 RealSense 帧: {e}")
            return None
    
    def merge_images(self, tianmou_img: np.ndarray, realsense_img: np.ndarray, 
                     max_height: int = 1080) -> np.ndarray:
        """合并天眸和 RealSense 图像
        
        左边放天眸，右边放 RealSense
        
        Args:
            tianmou_img: 天眸 RGB 图像
            realsense_img: RealSense 深度图像
            max_height: 最大高度（用于缩放）
        
        Returns:
            合并后的图像
        """
        # 调整大小到相同高度
        h = min(tianmou_img.shape[0], realsense_img.shape[0], max_height)
        
        # 缩放天眸图像
        t_h, t_w = tianmou_img.shape[:2]
        t_scale = h / t_h
        t_w_new = int(t_w * t_scale)
        tianmou_resized = np.array(Image.fromarray(tianmou_img).resize((t_w_new, h)))
        
        # 缩放 RealSense 图像
        r_h, r_w = realsense_img.shape[:2]
        r_scale = h / r_h
        r_w_new = int(r_w * r_scale)
        realsense_resized = np.array(Image.fromarray(realsense_img).resize((r_w_new, h)))
        
        # 合并
        merged = np.hstack([tianmou_resized, realsense_resized])
        
        # 添加分隔线（红色）
        separator = np.full((h, 2, 3), [0, 0, 255], dtype=np.uint8)
        merged = np.hstack([tianmou_resized, separator, realsense_resized])
        
        return merged
    
    def export_aligned_frames(self, max_frames: int = 100, merge: bool = True):
        """导出对齐的帧进行验证
        
        Args:
            max_frames: 最大导出帧数
            merge: 是否合并图像
        """
        stable_start = self.get_realsense_stable_start_idx()
        print(f"\n稳定期从第 {stable_start} 帧开始")
        print(f"总帧数: {len(self.frames)}")
        
        # 提取 RealSense 深度帧
        bag_path = self.input_info.get('bag_path')
        depth_dir = bag_path.replace('.bag', '_depth_frames') if bag_path else None
        
        if depth_dir and os.path.exists(depth_dir) and len(os.listdir(depth_dir)) > 0:
            print(f"使用已提取的 RealSense 帧: {depth_dir}")
        else:
            depth_dir = self.extract_realsense_depth_frames()
        
        if not depth_dir:
            print("无法提取 RealSense 帧，退出")
            return
        
        # 导出稳定期的帧
        exported = 0
        skipped = 0
        
        # 找到稳定期的第一帧
        stable_frame_start = None
        for i, frame in enumerate(self.frames):
            if frame.get('phase') == 'stable':
                stable_frame_start = i
                break
        
        if stable_frame_start is None:
            print("未找到稳定期帧")
            return
        
        print(f"\n开始导出帧 (从第 {stable_frame_start} 帧开始)...")
        
        for i in range(stable_frame_start, min(stable_frame_start + max_frames, len(self.frames))):
            frame = self.frames[i]
            tianmou_idx = frame.get('tianmou_idx')
            realsense_idx = frame.get('realsense_idx')
            is_skip = frame.get('skip', False)
            skip_count = frame.get('skip_count', 0)
            
            # 读取天眸图像
            tianmou_img = self.read_tianmou_frame(tianmou_idx)
            if tianmou_img is None:
                skipped += 1
                continue
            
            # 读取 RealSense 图像
            realsense_img = self.read_realsense_frame(realsense_idx, depth_dir)
            if realsense_img is None:
                skipped += 1
                continue
            
            # 保存天眸图像
            tianmou_path = os.path.join(self.output_dir, 'tianmou', f'frame_{exported:04d}_tm{tianmou_idx}.png')
            Image.fromarray(tianmou_img).save(tianmou_path)
            
            # 保存 RealSense 图像
            realsense_path = os.path.join(self.output_dir, 'realsense', f'frame_{exported:04d}_rs{realsense_idx}.png')
            Image.fromarray(realsense_img).save(realsense_path)
            
            # 合并图像
            if merge:
                try:
                    merged = self.merge_images(tianmou_img, realsense_img)
                    merged_path = os.path.join(self.output_dir, 'merged', f'frame_{exported:04d}.png')
                    Image.fromarray(merged).save(merged_path)
                except Exception as e:
                    print(f"  Warning: 合并失败: {e}")
            
            # 打印信息
            skip_info = f" [跳帧: {skip_count}]" if is_skip else ""
            print(f"  帧 {exported}: 天眸[{tianmou_idx}] <-> RealSense[{realsense_idx}]{skip_info}")
            
            exported += 1
            if exported >= max_frames:
                break
        
        print(f"\n导出完成!")
        print(f"  成功: {exported} 帧")
        print(f"  跳过: {skipped} 帧")
        print(f"  输出目录: {self.output_dir}")
        
        if merge:
            print(f"  合并图像: {os.path.join(self.output_dir, 'merged')}")


def main():
    parser = argparse.ArgumentParser(description='后处理测试 - 导出对齐帧')
    parser.add_argument('json_file', help='output_*.json 索引文件路径')
    parser.add_argument('--max-frames', '-n', type=int, default=100, 
                        help='最大导出帧数 (默认: 100)')
    parser.add_argument('--no-merge', action='store_true', 
                        help='不合并图像')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"文件不存在: {args.json_file}")
        sys.exit(1)
    
    tester = PostProcessTester(args.json_file)
    tester.export_aligned_frames(max_frames=args.max_frames, merge=not args.no_merge)


if __name__ == '__main__':
    main()
