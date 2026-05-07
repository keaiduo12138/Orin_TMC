#!/usr/bin/env python3
"""
calib/crop_color_frames.py
==========================
将 RealSense Color 图像从 640×480 裁剪为 640×320（顶部80px + 底部80px），
与天眸分辨率对齐，便于后续外参标定。

使用方式:
    python3 yyf/calib/crop_color_frames.py \
        --input /projects/calib_data/0324_calibration_depth_16_debug/color/ \
        --output /projects/calib_data/0324_calibration_depth_16_debug/color_cropped/

输出:
    color_cropped/color_0000.png  (640×320)
    ...
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


def crop_image(img: np.ndarray, crop_top: int, crop_bottom: int) -> np.ndarray:
    """裁剪图像顶部和底部"""
    h = img.shape[0]
    return img[crop_top:h - crop_bottom, :]


def main():
    parser = argparse.ArgumentParser(description="裁剪 RealSense Color 图像到 640×320")
    parser.add_argument("--input", "-i", required=True,
                        help="输入图像目录（含 color_XXXX.png 或 frame__Color_*.png）")
    parser.add_argument("--output", "-o", required=True,
                        help="输出目录")
    parser.add_argument("--crop-top", type=int, default=80,
                        help="顶部裁剪像素（默认80）")
    parser.add_argument("--crop-bottom", type=int, default=80,
                        help="底部裁剪像素（默认80）")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 查找图像文件（支持多种命名格式）
    patterns = [
        os.path.join(args.input, "color_*.png"),
        os.path.join(args.input, "frame__Color_*.png"),
    ]
    files = []
    for p in patterns:
        files.extend(glob(p))

    if not files:
        print(f"未找到图像文件，请检查路径: {args.input}")
        return

    # 按文件名排序
    files.sort()

    print(f"找到 {len(files)} 张图像")
    print(f"裁剪: 顶部 {args.crop_top}px, 底部 {args.crop_bottom}px")
    print(f"输入:  {args.input}")
    print(f"输出:  {args.output}")

    # 验证第一张图像的尺寸
    sample = cv2.imread(files[0])
    if sample is None:
        print(f"无法读取样本图像: {files[0]}")
        return
    orig_h, orig_w = sample.shape[:2]
    expected_h = 480
    expected_w = 640
    if orig_h != expected_h or orig_w != expected_w:
        print(f"警告: 图像尺寸为 {orig_w}×{orig_h}，不是预期的 {expected_w}×{expected_h}")
    else:
        print(f"原始尺寸: {orig_w}×{orig_h}")

    target_h = orig_h - args.crop_top - args.crop_bottom
    target_w = orig_w
    print(f"目标尺寸: {target_w}×{target_h}")

    cropped_count = 0
    skipped_count = 0

    for f in tqdm(files, desc="裁剪中", unit="帧"):
        img = cv2.imread(f)
        if img is None:
            skipped_count += 1
            continue

        cropped = crop_image(img, args.crop_top, args.crop_bottom)

        # 保持原文件名
        basename = os.path.basename(f)
        out_path = os.path.join(args.output, basename)
        cv2.imwrite(out_path, cropped)
        cropped_count += 1

    print(f"\n完成: 裁剪 {cropped_count} 张，跳过 {skipped_count} 张")
    print(f"输出目录: {args.output}")

    # 验证输出
    out_files = glob(os.path.join(args.output, "*.png"))
    if out_files:
        sample_out = cv2.imread(out_files[0])
        h, w = sample_out.shape[:2]
        print(f"输出图像尺寸验证: {w}×{h}")
        if w == 640 and h == 320:
            print("✓ 尺寸正确: 640×320")
        else:
            print(f"⚠ 尺寸异常: 期望 640×320，实际 {w}×{h}")


if __name__ == "__main__":
    main()
