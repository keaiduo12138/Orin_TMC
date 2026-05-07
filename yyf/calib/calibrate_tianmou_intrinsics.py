#!/usr/bin/env python3
"""
calib/calibrate_tianmou_intrinsics.py
=====================================
标定天眸相机内参（焦距、主点、畸变）。

使用方式:
    python3 calibrate_tianmou_intrinsics.py --data-dir /projects/calib_data/0325_calibration

流程:
    1. 读取全部天眸图像，检测棋盘格角点
    2. 运行 cv2.calibrateCamera 获取 K 和 dist
    3. 对比标定结果与 TM_K_REF（参考内参）
    4. 输出: intrinsic_tianmou.json + corner 可视化
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob
from typing import List, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    BOARD_COLS, BOARD_ROWS, SQUARE_SIZE, TM_K_REF,
    read_tianmou_image, to_gray, build_3d_objpoints,
    detect_corners, draw_corners,
    ensure_dir, save_json, print_header,
    print_matrix, print_vector,
)


def load_tianmou_frames(data_dir: str) -> List[Tuple[int, str]]:
    """加载天眸目录下的全部 PNG 帧，返回 [(帧索引, 路径)]。"""
    # 支持扁平目录（0325_N_tianmou.png）和子目录（tianmou/tianmou_NNNN.png）
    flat = glob(os.path.join(data_dir, "*_tianmou.png"))
    subdir = glob(os.path.join(data_dir, "tianmou", "tianmou_*.png"))

    files = flat if flat else subdir
    if not files:
        raise FileNotFoundError(f"未找到天眸图像: {data_dir}/<N>_tianmou.png 或 .../tianmou/tianmou_*.png")

    def extract_idx(path: str) -> int:
        basename = os.path.basename(path)
        # "0325_1_tianmou.png" → 1, "tianmou_0001.png" → 1
        parts = basename.replace("_tianmou", "").replace("tianmou_", "").split("_")
        return int(parts[-1].replace(".png", ""))

    files.sort(key=extract_idx)
    return [(extract_idx(f), f) for f in files]


def get_image_size(data_dir: str) -> Tuple[int, int]:
    """读取第一帧获取图像分辨率。"""
    frames = load_tianmou_frames(data_dir)
    img = read_tianmou_image(frames[0][1])
    h, w = img.shape[:2]
    return w, h   # OpenCV 惯例: (宽, 高)


def main():
    parser = argparse.ArgumentParser(description="天眸相机内参标定")
    parser.add_argument("--data-dir", "-d", required=True, help="标定数据目录")
    parser.add_argument("--frames", "-f", type=int, nargs="+", default=None,
                        help="指定帧索引（可选，默认使用全部）")
    parser.add_argument("--output", "-o", default="./calibration_output",
                        help="输出目录")
    parser.add_argument("--select-frames", action="store_true",
                        help="交互式选择帧（按空格跳过，q 退出）")
    args = parser.parse_args()

    ensure_dir(args.output)
    ensure_dir(os.path.join(args.output, "visualization"))

    # ── 加载帧 ───────────────────────────────────────────────
    frames = load_tianmou_frames(args.data_dir)
    all_ids = [f[0] for f in frames]
    path_map = {f[0]: f[1] for f in frames}

    if args.frames:
        raise ValueError("--frames 参数在内参标定中已废弃，请使用全量检测")
    elif args.select_frames:
        print("[交互模式暂不支持大数据集，建议使用全量检测]")
    print(f"  全量可用帧: {len(all_ids)}")

    # ── 全量角点检测 ──────────────────────────────────────────
    print_header("检测天眸棋盘格角点（全量）")
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)

    # Step 1: 检测所有帧
    all_detected_ids = []
    for idx, path in tqdm(frames, desc="[1/3] 全量检测", unit="帧"):
        img = read_tianmou_image(path)
        if img is None:
            continue
        gray = to_gray(img)
        ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
        if ret:
            all_detected_ids.append(idx)

    print(f"  全量帧: {len(frames)}, 检测成功: {len(all_detected_ids)}")

    if len(all_detected_ids) < 4:
        print("  检测到棋盘格的帧不足 4 帧，退出")
        return

    # Step 2: 均匀采样（最多 MAX_INTRINSIC_FRAMES=300 帧，避免标定过慢）
    MAX_INTRINSIC_FRAMES = 300
    if len(all_detected_ids) <= MAX_INTRINSIC_FRAMES:
        sampled_ids = all_detected_ids
    else:
        # 从全量中均匀抽取
        indices = np.linspace(0, len(all_detected_ids) - 1, MAX_INTRINSIC_FRAMES, dtype=int)
        sampled_ids = [all_detected_ids[i] for i in sorted(set(indices))]

    print(f"  均匀采样: {len(sampled_ids)} 帧（均匀覆盖 {all_detected_ids[0]}~{all_detected_ids[-1]}）")

    # Step 3: 在采样帧上重新检测（确保 imgpoints 顺序与 objpoints 一致）
    path_map = {f[0]: f[1] for f in frames}
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    valid_ids: List[int] = []

    for idx in tqdm(sampled_ids, desc="[2/3] 采样帧检测", unit="帧"):
        img = read_tianmou_image(path_map[idx])
        if img is None:
            continue
        gray = to_gray(img)
        ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            valid_ids.append(idx)
            # 可视化
            vis = draw_corners(img, corners)
            cv2.imwrite(os.path.join(args.output, "visualization", f"corners_{idx:04d}.png"), vis)

    print(f"  最终有效帧: {len(valid_ids)}/{len(sampled_ids)}")

    if len(objpoints) < 4:
        print("  帧数不足（至少需要 4 帧），退出")
        return

    # ── 内参标定 ──────────────────────────────────────────────
    print_header("天眸内参标定")
    w, h = get_image_size(args.data_dir)
    print(f"  图像尺寸: {w}×{h}")
    print(f"  使用帧数: {len(objpoints)}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None, criteria=criteria)

    # ── 重投影误差 ───────────────────────────────────────────
    per_frame_errors = []
    for i, (objp_i, imgp_i, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(objp_i, rv, tv, K, dist)
        err = float(np.linalg.norm(proj.squeeze().astype(np.float64)
                                - imgp_i.squeeze().astype(np.float64)))
        per_frame_errors.append({
            "frame_idx": valid_ids[i],
            "rvec": rv.ravel().tolist(),
            "tvec": tv.ravel().tolist(),
            "reproj_error": err,
        })

    mean_error = np.mean([e["reproj_error"] for e in per_frame_errors])
    print(f"\n  标定结果 (分辨率 {w}×{h}):")
    print_matrix("K", K)
    print(f"  dist: {dist.ravel().round(6).tolist()}")
    print(f"\n  平均重投影误差: {mean_error:.4f} pix")

    # ── 与参考内参对比 ────────────────────────────────────────
    print_header("与参考内参 TM_K_REF 对比")
    print_matrix("TM_K_REF", TM_K_REF)
    diff_fro = np.linalg.norm(K - TM_K_REF, "fro") / np.linalg.norm(TM_K_REF) * 100
    print(f"\n  Frobenius 偏差: {diff_fro:.1f}%")
    if diff_fro < 10:
        print("  偏差 < 10%%，标定结果与参考内参吻合良好")
    elif diff_fro < 30:
        print("  偏差 10%%~30%%，有差异，注意检查")
    else:
        print("  偏差 > 30%%，建议检查标定质量或棋盘格尺寸设置")

    # ── 估算棋盘格距离 ───────────────────────────────────────
    print_header("各帧棋盘格距离（辅助验证）")
    print(f"  {'帧':>4} | {'tx':>8} {'ty':>8} {'tz':>8} | {'|t|':>7} mm")
    print("  " + "-" * 50)
    tvecs_mm = [np.array(e["tvec"]) for e in per_frame_errors]
    for i, err_info in enumerate(per_frame_errors):
        t = tvecs_mm[i]
        print(f"  {valid_ids[i]:>4} | {t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f} | {float(np.linalg.norm(t)):7.0f}")
    mean_dist = float(np.mean([np.linalg.norm(t) for t in tvecs_mm]))
    print(f"\n  平均距离: {mean_dist:.0f} mm")
    if 300 < mean_dist < 2000:
        print("  距离范围合理（30cm~2m）")
    else:
        print(f"  警告: 距离 {mean_dist:.0f}mm 超出常规范围，检查棋盘格尺寸是否正确")

    # ── 保存结果 ──────────────────────────────────────────────
    result = {
        "camera": "tianmou",
        "resolution": {"width": w, "height": h},
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rvecs": [e["rvec"] for e in per_frame_errors],
        "tvecs": [e["tvec"] for e in per_frame_errors],
        "reproj_errors": per_frame_errors,
        "mean_reproj_error": mean_error,
        "valid_frame_indices": valid_ids,
        "n_frames": len(valid_ids),
        "square_size_mm": SQUARE_SIZE,
        "frobenius_deviation_from_ref": float(diff_fro),
    }
    save_json(result, os.path.join(args.output, "intrinsic_tianmou.json"))
    print(f"\n  结果已保存: {args.output}/intrinsic_tianmou.json")


if __name__ == "__main__":
    main()
