#!/usr/bin/env python3
"""
calib/compare_realsense_intrinsics.py
=====================================
对比 RealSense 出厂内参 vs 棋盘格实测内参，帮助判断哪种更可靠。

使用方式:
    # 方式1: 直接指定 bag 文件
    python3 compare_realsense_intrinsics.py \
        --data-dir /projects/calib_data/0325_calibration \
        --bag /projects/cxr_data/2026-3-24/17-26-26.bag

    # 方式2: 只用棋盘格（无 bag 文件时使用默认出厂内参）
    python3 compare_realsense_intrinsics.py \
        --data-dir /projects/calib_data/0325_calibration

流程:
    1. 读取 bag 文件获取出厂内参（或使用 D455 默认值）
    2. 读取棋盘格帧，标定实测内参
    3. 对比两个 K 和 dist
    4. 给出推荐：出厂 vs 实测

判断标准:
    - 出厂内参已由 Intel 工厂校准，精度约 1%
    - 实测内参受限于帧数、姿态多样性和畸变模型自由度
    - 经验法则: |Δfx/fx| < 2% → 出厂内参可信
    - 经验法则: 实测 k3 > 0.5 → 畸变过拟合，使用出厂更安全
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    BOARD_COLS, BOARD_ROWS, SQUARE_SIZE,
    RS_K_DEFAULT, RS_D_DEFAULT,
    to_gray, build_3d_objpoints,
    detect_corners,
    ensure_dir, save_json, print_header,
    print_matrix,
)


def load_realsense_frames(data_dir: str) -> List[Tuple[int, str]]:
    """加载 RealSense Color 帧，支持多种命名格式。"""
    subdir = os.path.join(data_dir, "color")
    # frame__Color_TIMESTAMP.png
    files = glob(os.path.join(subdir, "frame__Color_*.png"))
    if not files:
        # color_N.png 或 N_color.png
        files = glob(os.path.join(subdir, "*_color.png")) or glob(os.path.join(subdir, "color_*.png"))
    if not files:
        raise FileNotFoundError(f"未找到 RealSense Color 图像: {subdir}/")

    def extract_idx(path: str) -> int:
        basename = os.path.basename(path)
        # "frame__Color_1774348397716.18530273437500.png" → 提取时间戳数字部分
        if "frame__Color_" in basename:
            ts = basename.replace("frame__Color_", "").replace(".png", "")
            return int(ts.split(".")[0])
        # "0325_1_color.png" 或 "color_1.png"
        parts = basename.replace("_color", "").replace("color_", "").split("_")
        return int(parts[-1].replace(".png", ""))

    files.sort(key=extract_idx)
    return [(extract_idx(f), f) for f in files]


def read_rs_intrinsics_from_bag(bag_path: str) -> Optional[Dict]:
    """从 bag 文件读取 RealSense 出厂内参。"""
    try:
        import pyrealsense2 as rs2
        pipe = rs2.pipeline()
        cfg = rs2.config()
        cfg.enable_device_from_file(bag_path, repeat_playback=True)
        profile = pipe.start(cfg)
        rs = profile.get_device()
        sn = rs.get_info(rs2.camera_info.serial_number)
        playback = rs.as_playback()
        playback.pause()
        profile = pipe.get_active_profile()
        vs = profile.get_stream(rs2.stream.color)
        intr = vs.as_video_stream_profile().get_intrinsics()
        pipe.stop()
        return {
            "K": [
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1],
            ],
            "dist": list(intr.coeffs),
            "width": intr.width,
            "height": intr.height,
            "model": str(intr.model),
            "serial": sn,
        }
    except Exception as e:
        print(f"  [警告] 无法读取 bag 内参: {e}")
        return None


def compare_intrinsics(K1: np.ndarray, dist1: np.ndarray,
                        K2: np.ndarray, dist2: np.ndarray) -> Dict:
    """比较两套内参，给出差异量化指标。"""
    diff_K = np.linalg.norm(K1 - K2, "fro") / np.linalg.norm(K1) * 100
    diff_fx = abs(K1[0, 0] - K2[0, 0]) / K1[0, 0] * 100
    diff_fy = abs(K1[1, 1] - K2[1, 1]) / K1[1, 1] * 100
    diff_cx = abs(K1[0, 2] - K2[0, 2])
    diff_cy = abs(K1[1, 2] - K2[1, 2])
    diff_k1 = abs(dist1[0] - dist2[0])
    diff_k2 = abs(dist1[1] - dist2[1])
    diff_k3 = abs(dist1[4] - dist2[4])
    return {
        "diff_K_frobenius_pct": diff_K,
        "diff_fx_pct": diff_fx,
        "diff_fy_pct": diff_fy,
        "diff_cx_px": diff_cx,
        "diff_cy_px": diff_cy,
        "diff_k1": diff_k1,
        "diff_k2": diff_k2,
        "diff_k3": diff_k3,
    }


def main():
    parser = argparse.ArgumentParser(description="对比 RealSense 出厂 vs 实测内参")
    parser.add_argument("--data-dir", "-d", required=True)
    parser.add_argument("--bag", "-b", default=None)
    parser.add_argument("--frames", "-f", type=int, nargs="+", default=None)
    parser.add_argument("--output", "-o", default="./calibration_output")
    args = parser.parse_args()

    ensure_dir(args.output)

    # ── 出厂内参 ─────────────────────────────────────────────
    print_header("RealSense 出厂内参")
    bag_intrinsics = None
    if args.bag and os.path.exists(args.bag):
        print(f"  从 bag 读取: {args.bag}")
        bag_intrinsics = read_rs_intrinsics_from_bag(args.bag)
    else:
        print("  未提供 bag 文件，使用 D455 默认出厂值")

    K_factory = (np.array(bag_intrinsics["K"], dtype=np.float64)
                 if bag_intrinsics else RS_K_DEFAULT.copy())
    dist_factory = (np.array(bag_intrinsics["dist"], dtype=np.float64)
                    if bag_intrinsics else RS_D_DEFAULT.copy())

    print(f"  图像尺寸: {bag_intrinsics['width']}×{bag_intrinsics['height']}" if bag_intrinsics else "  图像尺寸: 640×480")
    print_matrix("K_factory", K_factory)
    print(f"  dist_factory: {dist_factory.ravel().round(6).tolist()}")

    # ── 全量角点检测 ──────────────────────────────────────────
    print_header("检测 RealSense 棋盘格角点（全量）")
    frames = load_realsense_frames(args.data_dir)

    # Step 1: 检测所有帧
    all_detected_ids = []
    for idx, path in tqdm(frames, desc="[1/3] 全量检测", unit="帧"):
        img = cv2.imread(path)
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

    # Step 2: 均匀采样（最多 MAX_INTRINSIC_FRAMES=300 帧）
    MAX_INTRINSIC_FRAMES = 300
    if len(all_detected_ids) <= MAX_INTRINSIC_FRAMES:
        sampled_ids = all_detected_ids
    else:
        indices = np.linspace(0, len(all_detected_ids) - 1, MAX_INTRINSIC_FRAMES, dtype=int)
        sampled_ids = [all_detected_ids[i] for i in sorted(set(indices))]

    print(f"  均匀采样: {len(sampled_ids)} 帧（覆盖 {all_detected_ids[0]}~{all_detected_ids[-1]}）")

    # Step 3: 在采样帧上提取 objpoints/imgpoints
    path_map = {f[0]: f[1] for f in frames}
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)
    objpoints, imgpoints, valid_ids = [], [], []

    for idx in tqdm(sampled_ids, desc="[2/3] 采样帧检测", unit="帧"):
        img = cv2.imread(path_map[idx])
        if img is None:
            continue
        gray = to_gray(img)
        ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            valid_ids.append(idx)

    print(f"  最终有效帧: {len(valid_ids)}/{len(sampled_ids)}")

    if len(objpoints) < 4:
        print("  帧数不足（至少需要 4 帧），退出")
        return

    # ── 实测内参标定 ─────────────────────────────────────────
    print_header("RealSense 棋盘格实测内参标定")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # 用出厂内参作为初始值（CALIB_USE_INTRINSIC_GUESS）
    ret, K_calib, dist_calib, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (640, 480),
        K_factory.copy(), RS_D_DEFAULT.copy(),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=criteria,
    )

    # 计算每帧重投影误差
    per_frame_errors = []
    for i, (objp_i, imgp_i, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        proj, _ = cv2.projectPoints(objp_i, rv, tv, K_calib, dist_calib)
        err = float(np.linalg.norm(
            proj.squeeze().astype(np.float64)
            - imgp_i.squeeze().astype(np.float64)))
        per_frame_errors.append({"frame_idx": valid_ids[i], "error": err})

    mean_error = np.mean([e["error"] for e in per_frame_errors])
    print(f"  使用帧数: {len(objpoints)}")
    print(f"  平均重投影误差: {mean_error:.4f} pix")
    print(f"  K_calib:")
    print_matrix("K_calib", K_calib)
    print(f"  dist_calib: {dist_calib.ravel().round(6).tolist()}")

    # ── 对比分析 ─────────────────────────────────────────────
    print_header("出厂 vs 实测内参对比")
    comp = compare_intrinsics(K_factory, dist_factory, K_calib, dist_calib)
    print(f"  K Frobenius 偏差:    {comp['diff_K_frobenius_pct']:.2f}%")
    print(f"  fx 偏差:             {comp['diff_fx_pct']:.2f}%")
    print(f"  fy 偏差:             {comp['diff_fy_pct']:.2f}%")
    print(f"  cx 偏差:             {comp['diff_cx_px']:.2f} px")
    print(f"  cy 偏差:             {comp['diff_cy_px']:.2f} px")
    print(f"  k1 偏差:             {comp['diff_k1']:.4f}")
    print(f"  k2 偏差:             {comp['diff_k2']:.4f}")
    print(f"  k3 偏差:             {comp['diff_k3']:.4f}")

    # ── 判断推荐 ──────────────────────────────────────────────
    print_header("推荐方案")

    recommendation = "use_factory"
    reasons = []

    if comp["diff_K_frobenius_pct"] > 10:
        recommendation = "factory_only"
        reasons.append(f"K 偏差 {comp['diff_K_frobenius_pct']:.1f}% > 10%%，实测结果不可信")
    elif comp["diff_fx_pct"] > 5:
        recommendation = "use_factory"
        reasons.append(f"fx 偏差 {comp['diff_fx_pct']:.1f}% > 5%%，建议使用出厂内参")
    elif comp["diff_k3"] > 0.5:
        recommendation = "use_factory"
        reasons.append(f"|k3| = {comp['diff_k3']:.2f} > 0.5，高阶畸变过拟合，使用出厂更安全")
    elif comp["diff_k2"] > 0.3:
        recommendation = "use_factory"
        reasons.append(f"|k2| 偏差 = {comp['diff_k2']:.2f} > 0.3，畸变模型过拟合")
    else:
        recommendation = "use_factory"
        reasons.append("即使实测结果良好，也推荐使用出厂内参（Intel 工厂校准更权威）")

    if recommendation == "use_factory":
        print("  ★ 推荐: 使用出厂内参（bag 读取或 D455 默认值）")
    else:
        print("  ★ 推荐: 使用棋盘格实测内参")

    for r in reasons:
        print(f"    原因: {r}")

    # ── 畸变稳定性检查 ───────────────────────────────────────
    print_header("畸变系数健康检查")
    print(f"  出厂: k1={dist_factory[0]:+.4f}, k2={dist_factory[1]:+.4f}, "
          f"p1={dist_factory[2]:+.4f}, p2={dist_factory[3]:+.4f}, k3={dist_factory[4]:+.4f}")
    print(f"  实测: k1={dist_calib[0]:+.4f}, k2={dist_calib[1]:+.4f}, "
          f"p1={dist_calib[2]:+.4f}, p2={dist_calib[3]:+.4f}, k3={dist_calib[4]:+.4f}")
    dist_warnings = []
    if abs(dist_calib[4]) > 1.0:
        dist_warnings.append("k3 过大（>1.0），畸变模型过拟合")
    if abs(dist_calib[1]) > 0.3:
        dist_warnings.append("k2 偏差过大（>0.3），畸变模型不稳定")
    if dist_warnings:
        for w in dist_warnings:
            print(f"  ⚠ {w}")
    else:
        print("  ✓ 畸变系数在正常范围内")

    # ── 估算棋盘格距离 ───────────────────────────────────────
    print_header("各帧棋盘格距离（辅助验证）")
    tvecs_mm = [np.array(tv.ravel()) for tv in tvecs]
    for i, fid in enumerate(valid_ids):
        t = tvecs_mm[i]
        print(f"  帧{fid:4d}: |t|={float(np.linalg.norm(t)):7.0f}mm  "
              f"(tx={t[0]:7.1f} ty={t[1]:7.1f} tz={t[2]:7.1f})")
    mean_dist = float(np.mean([np.linalg.norm(t) for t in tvecs_mm]))
    print(f"\n  平均距离: {mean_dist:.0f}mm")
    if 300 < mean_dist < 2000:
        print("  ✓ 距离范围合理（30cm~2m）")
    else:
        print(f"  ⚠ 距离 {mean_dist:.0f}mm 超出常规范围")

    # ── 保存 ────────────────────────────────────────────────
    result = {
        "factory": {
            "K": K_factory.tolist(),
            "dist": dist_factory.tolist(),
            "source": "bag" if bag_intrinsics else "d455_default",
        },
        "calibrated": {
            "K": K_calib.tolist(),
            "dist": dist_calib.tolist(),
            "reproj_errors": per_frame_errors,
            "mean_reproj_error": mean_error,
            "valid_frame_indices": valid_ids,
            "n_frames": len(valid_ids),
        },
        "comparison": comp,
        "recommendation": recommendation,
        "reasons": reasons,
    }
    save_json(result, os.path.join(args.output, "compare_realsense_intrinsics.json"))
    print(f"\n  结果已保存: {args.output}/compare_realsense_intrinsics.json")

    print("\n  ★ 总结:")
    print(f"  factory K_fx={K_factory[0,0]:.1f}  calibrated K_fx={K_calib[0,0]:.1f}")
    print(f"  推荐使用: {'出厂内参' if recommendation == 'use_factory' else '实测内参'}")


if __name__ == "__main__":
    main()
