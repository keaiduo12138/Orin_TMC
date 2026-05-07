#!/usr/bin/env python3
"""
calib/calibrate_realsense_cropped.py
=====================================
对裁剪后的 RealSense Color 图像（640×480 → 640×320）重新进行内参标定。

注意：裁剪后的图像 cx 不变（宽度不变），cy 需要减去顶部裁剪量。
本脚本直接用棋盘格实测标定，得到新的内参 K 和 dist。

使用方式:
    # 单数据集
    python3 calibrate_realsense_cropped.py \
        --data-dirs /projects/calib_data/dataset_A/color_cropped/ \
        -o ./calibration_output

    # 多数据集联合（各数据集 n 帧取 1 帧）
    python3 calibrate_realsense_cropped.py \
        --data-dirs /projects/calib_data/dataset_A/color_cropped/ \
                   /projects/calib_data/dataset_B/color_cropped/ \
        --sample-steps 5 8 \
        -o ./calibration_output

流程:
    1. 读取 RS 图像，检测棋盘格角点（各数据集独立全量检测）
    2. 各数据集按步长均匀采样（--sample-steps 分别控制）
    3. 迭代剔除坏帧（重投影误差 > --reproj-threshold px），多轮优化
    4. 对比标定结果与出厂内参
    5. 输出: intrinsic_realsense_cropped.json
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob
from typing import List, Dict, Tuple
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
    """加载 RealSense Color 帧。"""
    patterns = [
        os.path.join(data_dir, "color_*.png"),
        os.path.join(data_dir, "frame__Color_*.png"),
    ]
    files = []
    for p in patterns:
        files.extend(glob(p))

    if not files:
        raise FileNotFoundError(f"未找到图像: {data_dir}")

    def extract_idx(path: str) -> int:
        basename = os.path.basename(path)
        if "frame__Color_" in basename:
            ts = basename.replace("frame__Color_", "").replace(".png", "")
            return int(ts.split(".")[0])
        parts = basename.replace("_color", "").replace("color_", "").split("_")
        return int(parts[-1].replace(".png", ""))

    files.sort(key=extract_idx)
    return [(extract_idx(f), f) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="裁剪后 RealSense Color 内参标定（支持多数据集联合，640×480 → 640×320）")
    parser.add_argument("--data-dirs", "-d", nargs="+", required=True,
                        help="数据集目录（支持多个目录）")
    parser.add_argument("--sample-steps", type=int, nargs="+", default=None,
                        help="每个数据集的抽样步长（默认各为5）。"
                             "例如 --sample-steps 5 8 表示数据集A每5帧取1、数据集B每8帧取1。"
                             "数据集数量必须与目录数量一致。")
    parser.add_argument("--max-frames-per-dataset", type=int, default=300,
                        help="每个数据集最多使用帧数（默认300）")
    parser.add_argument("--output", "-o", default="./calibration_output",
                        help="输出目录")
    parser.add_argument("--init-k", action="store_true", default=True,
                        help="用 cx 调整后的出厂值初始化（默认开启）")
    parser.add_argument("--no-init-k", dest="init_k", action="store_false",
                        help="关闭出厂值初始化")
    parser.add_argument("--crop-top", type=int, default=80,
                        help="顶部裁剪像素（用于初始化 cx，默认80）")
    parser.add_argument("--rational", action="store_true",
                        help="使用有理畸变模型（较慢但更精确，默认关闭使用标准模型）")
    parser.add_argument("--reproj-threshold", type=float, default=5.0,
                        help="迭代剔除阈值(px)：重投影误差超过此值的帧将被剔除，"
                             "然后重新标定直到无新帧被剔除（默认5.0px）")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="最大迭代剔除轮数（默认10）")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="剔除后最少保留帧数，低于此数时停止剔除（默认10）")
    args = parser.parse_args()

    ensure_dir(args.output)

    # ── 参数规范化 ────────────────────────────────────────────
    data_dirs = args.data_dirs
    n_datasets = len(data_dirs)

    if args.sample_steps is None:
        sample_steps = [5] * n_datasets
    elif len(args.sample_steps) == 1:
        sample_steps = args.sample_steps * n_datasets
    elif len(args.sample_steps) != n_datasets:
        raise ValueError(f"--sample-steps 数量({len(args.sample_steps)}) "
                         f"必须等于 --data-dirs 数量({n_datasets})，"
                         f"或只提供 1 个值（所有数据集共用）")
    else:
        sample_steps = args.sample_steps

    print(f"\n{'='*60}")
    print(f"  RS 内参标定（多数据集联合，裁剪后 640×320）")
    print(f"{'='*60}")
    print(f"  数据集数量: {n_datasets}")
    for i, (dd, ss) in enumerate(zip(data_dirs, sample_steps)):
        print(f"  数据集{i+1}: {dd}  (每{ss}帧取1)")
    print(f"  每数据集最大帧数: {args.max_frames_per_dataset}")
    print(f"  初始化: {'出厂值(crop调整)' if args.init_k else '零初始化'}")

    # ── 验证图像尺寸 ────────────────────────────────────────
    for dd in data_dirs:
        frames = load_realsense_frames(dd)
        if frames:
            _, sample_path = frames[0]
            sample = cv2.imread(sample_path)
            if sample is not None:
                h, w = sample.shape[:2]
                print(f"  图像尺寸: {w}×{h}（应为 640×320）")
                if w != 640 or h != 320:
                    print(f"  ⚠ 尺寸警告: 期望 640×320，实际 {w}×{h}")
            break

    # ── 逐数据集全量检测 ────────────────────────────────────
    print_header("逐数据集全量棋盘格检测")
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)

    all_objpoints: List[np.ndarray] = []
    all_imgpoints: List[np.ndarray] = []
    all_valid_ids: List[Tuple[int, int]] = []

    for di, data_dir in enumerate(data_dirs):
        print(f"\n  ── 数据集 {di+1}/{n_datasets}: {data_dir}")
        frames = load_realsense_frames(data_dir)
        print(f"    全量可用帧: {len(frames)}")

        # 全量检测
        detected = []
        for idx, path in tqdm(frames, desc=f"[{di+1}] 检测", unit="帧"):
            img = cv2.imread(path)
            if img is None:
                continue
            gray = to_gray(img)
            ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
            if ret:
                detected.append((idx, path))

        print(f"    检测成功: {len(detected)}/{len(frames)}")

        # 均匀采样
        step = sample_steps[di]
        sampled = detected[::step]
        if len(sampled) > args.max_frames_per_dataset:
            idxs = np.linspace(0, len(sampled)-1, args.max_frames_per_dataset, dtype=int)
            sampled = [sampled[i] for i in sorted(set(idxs))]

        print(f"    采样步长={step}，采样帧数={len(sampled)}")

        for idx, path in sampled:
            img = cv2.imread(path)
            if img is None:
                continue
            gray = to_gray(img)
            ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
            if ret:
                all_objpoints.append(objp.copy())
                all_imgpoints.append(corners)
                all_valid_ids.append((di, idx))

    print(f"\n  所有数据集合计: {len(all_objpoints)} 帧用于标定")

    if len(all_objpoints) < 4:
        print("  帧数不足（至少需要 4 帧），退出")
        return

    # ── 迭代剔除坏帧 + RS 内参标定 ────────────────────────
    print_header("RS 裁剪后内参标定（联合优化 + 迭代剔除）")
    w, h = 640, 320  # 固定分辨率
    print(f"  图像尺寸: {w}×{h}")
    print(f"  初始帧数: {len(all_objpoints)}")
    print(f"  剔除阈值: {args.reproj_threshold} px")
    print(f"  最大迭代轮数: {args.max_iterations}")
    print(f"  最少保留帧数: {args.min_frames}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # 初始化 K
    if args.init_k:
        K_init = RS_K_DEFAULT.copy()
        K_init[1, 2] = RS_K_DEFAULT[1, 2] - args.crop_top
        print(f"  初始化 K（裁剪后出厂值）:")
        print_matrix("K_init", K_init)
    else:
        K_init = None

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL

    dist_len = 8 if args.rational else 5
    dist_init = np.zeros(dist_len, dtype=np.float64)

    # 复制用于迭代
    objpoints_iter = list(all_objpoints)
    imgpoints_iter = list(all_imgpoints)
    valid_ids_iter = list(all_valid_ids)

    iteration = 0
    removed_in_round = []
    bad = []  # track last bad frames for final check

    while iteration < args.max_iterations:
        iteration += 1

        ret, K_calib, dist_calib, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints_iter,
            imagePoints=imgpoints_iter,
            imageSize=(w, h),
            cameraMatrix=K_init if iteration == 1 else K_calib,
            distCoeffs=dist_init if iteration == 1 else dist_calib,
            flags=flags,
            criteria=criteria,
        )

        # 计算每帧重投影误差
        per_iter_errors = []
        for i, (objp_i, imgp_i, rv, tv) in enumerate(
                zip(objpoints_iter, imgpoints_iter, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(objp_i, rv, tv, K_calib, dist_calib)
            err = float(np.linalg.norm(
                proj.squeeze().astype(np.float64)
                - imgp_i.squeeze().astype(np.float64)))
            per_iter_errors.append({"idx": i, "error": err})

        bad = [e for e in per_iter_errors if e["error"] > args.reproj_threshold]
        bad_idxs = set(e["idx"] for e in bad)

        mean_err = np.mean([e["error"] for e in per_iter_errors])
        print(f"\n  ── 迭代 {iteration}:")
        print(f"    帧数: {len(objpoints_iter)}，平均误差: {mean_err:.4f} px")
        print(f"    超阈值帧: {len(bad)}")
        if bad:
            for e in sorted(bad, key=lambda x: -x["error"])[:5]:
                vid = valid_ids_iter[e["idx"]]
                print(f"      ds{vid[0]+1}_{vid[1]:04d}: {e['error']:.2f} px")

        if not bad:
            print("    ✓ 无新帧被剔除，迭代收敛")
            break

        if len(objpoints_iter) - len(bad) < args.min_frames:
            print(f"    ⚠ 剔除后帧数 {len(objpoints_iter)-len(bad)} < {args.min_frames}，停止剔除")
            break

        removed_in_round.append(
            [(valid_ids_iter[e["idx"]], e["error"]) for e in sorted(bad, key=lambda x: -x["error"])]
        )
        bad_idxs_sorted = sorted(bad_idxs, reverse=True)
        for bi in bad_idxs_sorted:
            del objpoints_iter[bi]
            del imgpoints_iter[bi]
            del valid_ids_iter[bi]

    # ── 最终标定结果 ─────────────────────────────────────
    removed_flat = [item for round_list in removed_in_round for item in round_list]
    print_header("RS 内参标定（最终结果）")
    print(f"  剔除轮数: {iteration - (1 if not bad else 0)}")
    print(f"  累计剔除: {len(removed_flat)} 帧")
    print(f"  最终帧数: {len(objpoints_iter)}（初始 {len(all_objpoints)} 帧）")

    ret_final, K_calib, dist_calib, rvecs_final, tvecs_final = cv2.calibrateCamera(
        objectPoints=objpoints_iter,
        imagePoints=imgpoints_iter,
        imageSize=(w, h),
        cameraMatrix=K_init if K_init is not None else K_calib,
        distCoeffs=dist_init if K_init is None else dist_calib,
        flags=flags,
        criteria=criteria,
    )

    # ── 最终重投影误差 ─────────────────────────────────────
    per_frame_errors = []
    for i, (objp_i, imgp_i, rv, tv) in enumerate(
            zip(objpoints_iter, imgpoints_iter, rvecs_final, tvecs_final)):
        proj, _ = cv2.projectPoints(objp_i, rv, tv, K_calib, dist_calib)
        err = float(np.linalg.norm(
            proj.squeeze().astype(np.float64)
            - imgp_i.squeeze().astype(np.float64)))
        vid = valid_ids_iter[i]
        per_frame_errors.append({"dataset": vid[0], "frame_idx": vid[1], "error_px": err})

    mean_error = float(np.mean([e["error_px"] for e in per_frame_errors]))
    print(f"\n  标定返回值: {ret_final:.4f}")
    print(f"  有效帧数: {len(per_frame_errors)}")
    print(f"  平均重投影误差: {mean_error:.4f} px")
    print(f"  各帧误差范围: {min(e['error_px'] for e in per_frame_errors):.3f}"
          f" ~ {max(e['error_px'] for e in per_frame_errors):.3f} px")
    print(f"\n  K_calib (裁剪后 640×320):")
    print_matrix("K", K_calib)
    print(f"\n  dist_calib: {dist_calib.ravel().round(6).tolist()}")

    # ── 剔除记录 ─────────────────────────────────────────
    if removed_flat:
        print(f"\n  剔除帧详情（共 {len(removed_flat)} 帧）:")
        print(f"  {'数据集':>2} | {'帧':>4} | {'重投影误差(px)':>14}")
        print("  " + "-" * 30)
        for vid, err in sorted(removed_flat, key=lambda x: -x[1]):
            print(f"  {vid[0]+1:>2} | {vid[1]:>4} | {err:>14.2f}")

    # ── 对比出厂值 ────────────────────────────────────────────
    print_header("与出厂内参对比（D455 640×480）")
    factory_K_480 = RS_K_DEFAULT
    print(f"  出厂 K (640×480):")
    print_matrix("K_factory_480", factory_K_480)

    diff_fx = abs(K_calib[0, 0] - factory_K_480[0, 0]) / factory_K_480[0, 0] * 100
    diff_fy = abs(K_calib[1, 1] - factory_K_480[1, 1]) / factory_K_480[1, 1] * 100
    diff_cx = abs(K_calib[0, 2] - factory_K_480[0, 2])
    expected_cy = factory_K_480[1, 2] - args.crop_top
    diff_cy = abs(K_calib[1, 2] - expected_cy)
    print(f"\n  偏差分析:")
    print(f"    fx 偏差: {diff_fx:.2f}%")
    print(f"    fy 偏差: {diff_fy:.2f}%")
    print(f"    cx 偏差: {diff_cx:.2f} px（宽度不变，cx 应接近）")
    print(f"    cy 偏差: {diff_cy:.2f} px（预期 cy≈{expected_cy:.1f}，实际 {K_calib[1,2]:.1f}）")

    # ── 保存结果 ──────────────────────────────────────────────
    result = {
        "resolution": {"width": w, "height": h},
        "K": K_calib.tolist(),
        "dist": dist_calib.tolist(),
        "mean_reproj_error_px": mean_error,
        "per_frame_errors": per_frame_errors,
        "valid_frame_indices": [e["frame_idx"] for e in per_frame_errors],
        "n_frames": len(objpoints_iter),
        "n_frames_initial": len(all_objpoints),
        "n_frames_removed": len(removed_flat),
        "removed_frames": [{"dataset": v[0], "frame_idx": v[1], "error": e}
                          for v, e in removed_flat],
        "n_datasets": n_datasets,
        "data_dirs": data_dirs,
        "sample_steps": sample_steps,
        "init_method": "crop_adjusted_factory" if args.init_k else "zero_init",
        "crop_top_px": args.crop_top,
        "factory_K_480": factory_K_480.tolist(),
        "iterations": iteration - (1 if not bad else 0),
        "comparison": {
            "diff_fx_pct": diff_fx,
            "diff_fy_pct": diff_fy,
            "diff_cx_px": diff_cx,
            "diff_cy_px": diff_cy,
            "expected_cy": expected_cy,
        }
    }

    out_path = os.path.join(args.output, "intrinsic_realsense_cropped.json")
    save_json(result, out_path)
    print(f"\n  ✓ 结果已保存: {out_path}")

    print(f"\n  总结:")
    print(f"    分辨率: {w}×{h}")
    print(f"    fx={K_calib[0,0]:.1f}, fy={K_calib[1,1]:.1f}, cx={K_calib[0,2]:.1f}, cy={K_calib[1,2]:.1f}")
    print(f"    平均重投影误差: {mean_error:.3f} px")


if __name__ == "__main__":
    main()
