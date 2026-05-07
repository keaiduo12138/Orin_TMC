#!/usr/bin/env python3
"""
calib/calibrate_tianmou_intrinsics.py
=====================================
标定天眸相机内参（焦距、主点、畸变）。

支持多数据集联合标定：多个数据集共用同一套内参，数据越多精度越高。

使用方式:
    # 单数据集
    python3 calibrate_tianmou_intrinsics.py --data-dirs /projects/calib_data/dataset_A

    # 多数据集联合（各数据集 n 帧取 1 帧）
    python3 calibrate_tianmou_intrinsics.py \
        --data-dirs /projects/calib_data/dataset_A /projects/calib_data/dataset_B \
        --sample-steps 5 8 \
        -o ./calibration_output

流程:
    1. 读取全部天眸图像，检测棋盘格角点（各数据集独立全量检测）
    2. 各数据集按步长均匀采样（--sample-steps 分别控制）
    3. 迭代剔除坏帧（重投影误差 > --reproj-threshold px），多轮优化
    4. 对比标定结果与 TM_K_REF（参考内参）
    5. 输出: intrinsic_tianmou.json + corner 可视化
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
    """加载天眸目录下的全部 PNG 帧，返回 [(帧索引, 路径)]。

    支持的文件命名格式：
      - tianmou_0000.png  （前缀格式，直接在目录下）
      - tianmou/tianmou_0000.png  （前缀格式，子目录）
      - 0325_1_tianmou.png  （后缀格式）
    """
    flat_prefix = glob(os.path.join(data_dir, "tianmou_*.png"))
    subdir_prefix = glob(os.path.join(data_dir, "tianmou", "tianmou_*.png"))
    flat_suffix = glob(os.path.join(data_dir, "*_tianmou.png"))

    files = flat_prefix or subdir_prefix or flat_suffix
    if not files:
        raise FileNotFoundError(
            f"未找到天眸图像: {data_dir}/tianmou_*.png 或 .../tianmou/tianmou_*.png 或 .../*_tianmou.png")

    def extract_idx(path: str) -> int:
        basename = os.path.basename(path)
        # "tianmou_0001.png" → 1, "0325_1_tianmou.png" → 1
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
    parser = argparse.ArgumentParser(description="天眸相机内参标定（支持多数据集联合）")
    parser.add_argument("--data-dirs", "-d", nargs="+", required=True,
                        help="数据集目录（支持多个，逗号分隔或空格分隔）")
    parser.add_argument("--sample-steps", type=int, nargs="+", default=None,
                        help="每个数据集的抽样步长（默认各为5）。"
                             "例如 --sample-steps 5 8 表示数据集A每5帧取1、数据集B每8帧取1。"
                             "数据集数量必须与目录数量一致。")
    parser.add_argument("--max-frames-per-dataset", type=int, default=300,
                        help="每个数据集最多使用帧数（默认300）")
    parser.add_argument("--output", "-o", default="./calibration_output",
                        help="输出目录")
    parser.add_argument("--select-frames", action="store_true",
                        help="交互式选择帧（按空格跳过，q 退出）")
    parser.add_argument("--reproj-threshold", type=float, default=5.0,
                        help="迭代剔除阈值(px)：重投影误差超过此值的帧将被剔除，"
                             "然后重新标定直到无新帧被剔除（默认5.0px）")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="最大迭代剔除轮数（默认10）")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="剔除后最少保留帧数，低于此数时停止剔除（默认10）")
    args = parser.parse_args()

    ensure_dir(args.output)
    ensure_dir(os.path.join(args.output, "visualization"))

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
    print(f"  天眸内参标定（多数据集联合）")
    print(f"{'='*60}")
    print(f"  数据集数量: {n_datasets}")
    for i, (dd, ss) in enumerate(zip(data_dirs, sample_steps)):
        print(f"  数据集{i+1}: {dd}  (每{ss}帧取1)")
    print(f"  每数据集最大帧数: {args.max_frames_per_dataset}")

    # ── 逐数据集加载 & 全量检测 ──────────────────────────────
    print_header("逐数据集全量棋盘格检测")
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)

    all_objpoints: List[np.ndarray] = []
    all_imgpoints: List[np.ndarray] = []
    all_valid_ids: List[int] = []

    for di, data_dir in enumerate(data_dirs):
        print(f"\n  ── 数据集 {di+1}/{n_datasets}: {data_dir}")
        frames = load_tianmou_frames(data_dir)
        print(f"    全量可用帧: {len(frames)}")

        # 全量检测
        detected_ids = []
        for idx, path in tqdm(frames, desc=f"[{di+1}] 检测", unit="帧"):
            img = read_tianmou_image(path)
            if img is None:
                continue
            gray = to_gray(img)
            ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
            if ret:
                detected_ids.append((idx, path))

        print(f"    检测成功: {len(detected_ids)}/{len(frames)}")

        # 均匀采样
        step = sample_steps[di]
        # 收集检测成功的帧路径
        path_map = {idx: p for idx, p in detected_ids}
        sampled_ids = detected_ids[::step]
        if len(sampled_ids) > args.max_frames_per_dataset:
            idxs = np.linspace(0, len(sampled_ids)-1, args.max_frames_per_dataset, dtype=int)
            sampled_ids = [sampled_ids[i] for i in sorted(set(idxs))]

        print(f"    采样步长={step}，采样帧数={len(sampled_ids)}")

        # 在采样帧上提取角点
        for idx, path in sampled_ids:
            img = read_tianmou_image(path)
            if img is None:
                continue
            gray = to_gray(img)
            ret, corners = detect_corners(gray, BOARD_COLS, BOARD_ROWS)
            if ret:
                all_objpoints.append(objp.copy())
                all_imgpoints.append(corners)
                all_valid_ids.append((di, idx))
                # 可视化（加前缀区分数据集）
                vis = draw_corners(img, corners)
                cv2.imwrite(os.path.join(args.output, "visualization",
                                         f"ds{di+1}_corners_{idx:04d}.png"), vis)

        print(f"    有效帧: {len([x for x in all_valid_ids if x[0]==di])}/{len(sampled_ids)}")

    print(f"\n  所有数据集合计: {len(all_objpoints)} 帧用于标定")

    if len(all_objpoints) < 4:
        print("  帧数不足（至少需要 4 帧），退出")
        return

    # ── 迭代剔除坏帧 + 内参标定 ─────────────────────────────
    print_header("天眸内参标定（联合优化 + 迭代剔除）")

    w, h = 640, 320  # 天眸固定分辨率
    print(f"  图像尺寸: {w}×{h}")
    print(f"  初始帧数: {len(all_objpoints)}")
    print(f"  剔除阈值: {args.reproj_threshold} px")
    print(f"  最大迭代轮数: {args.max_iterations}")
    print(f"  最少保留帧数: {args.min_frames}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # 复制一份用于迭代
    objpoints_iter = list(all_objpoints)
    imgpoints_iter = list(all_imgpoints)
    valid_ids_iter = list(all_valid_ids)

    iteration = 0
    removed_in_round = []

    while iteration < args.max_iterations:
        iteration += 1

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints_iter, imgpoints_iter, (w, h), None, None, criteria=criteria)

        # 计算每帧重投影误差
        per_iter_errors = []
        for i, (objp_i, imgp_i, rv, tv) in enumerate(
                zip(objpoints_iter, imgpoints_iter, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(objp_i, rv, tv, K, dist)
            err = float(np.linalg.norm(
                proj.squeeze().astype(np.float64)
                - imgp_i.squeeze().astype(np.float64)))
            per_iter_errors.append({"idx": i, "error": err})

        # 找出超阈值的帧
        bad = [e for e in per_iter_errors if e["error"] > args.reproj_threshold]
        bad_ids = set(e["idx"] for e in bad)

        # 打印本轮情况
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

        # 剔除坏帧（从后往前删，保持索引稳定）
        removed_in_round.append(
            [(valid_ids_iter[e["idx"]], e["error"]) for e in sorted(bad, key=lambda x: -x["error"])]
        )
        bad_idxs_sorted = sorted(bad_ids, reverse=True)
        for bi in bad_idxs_sorted:
            del objpoints_iter[bi]
            del imgpoints_iter[bi]
            del valid_ids_iter[bi]

    # ── 最终标定结果 ───────────────────────────────────────
    print_header("天眸内参标定（最终结果）")
    print(f"  剔除轮数: {iteration - (1 if not bad else 0)}")
    removed_flat = [item for round_list in removed_in_round for item in round_list]
    print(f"  累计剔除: {len(removed_flat)} 帧")
    print(f"  最终帧数: {len(objpoints_iter)}（初始 {len(all_objpoints)} 帧）")

    ret_final, K, dist, rvecs_final, tvecs_final = cv2.calibrateCamera(
        objpoints_iter, imgpoints_iter, (w, h), None, None, criteria=criteria)

    # ── 最终重投影误差 ─────────────────────────────────────
    per_frame_errors = []
    for i, (objp_i, imgp_i, rv, tv) in enumerate(
            zip(objpoints_iter, imgpoints_iter, rvecs_final, tvecs_final)):
        proj, _ = cv2.projectPoints(objp_i, rv, tv, K, dist)
        err = float(np.linalg.norm(
            proj.squeeze().astype(np.float64)
            - imgp_i.squeeze().astype(np.float64)))
        vid = valid_ids_iter[i]
        per_frame_errors.append({
            "dataset": vid[0],
            "frame_idx": vid[1],
            "rvec": rv.ravel().tolist(),
            "tvec": tv.ravel().tolist(),
            "reproj_error": err,
        })

    mean_error = float(np.mean([e["reproj_error"] for e in per_frame_errors]))
    print(f"\n  标定结果 (分辨率 {w}×{h}):")
    print_matrix("K", K)
    print(f"  dist: {dist.ravel().round(6).tolist()}")
    print(f"  平均重投影误差: {mean_error:.4f} pix")
    print(f"  各帧误差范围: {min(e['reproj_error'] for e in per_frame_errors):.3f}"
          f" ~ {max(e['reproj_error'] for e in per_frame_errors):.3f} px")

    # ── 剔除记录 ───────────────────────────────────────────
    if removed_flat:
        print(f"\n  剔除帧详情（共 {len(removed_flat)} 帧）:")
        print(f"  {'数据集':>2} | {'帧':>4} | {'重投影误差(px)':>14}")
        print("  " + "-" * 30)
        for vid, err in sorted(removed_flat, key=lambda x: -x[1]):
            print(f"  {vid[0]+1:>2} | {vid[1]:>4} | {err:>14.2f}")

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
    print(f"  {'数据集':>2} | {'帧':>4} | {'tx':>8} {'ty':>8} {'tz':>8} | {'|t|':>7} mm")
    print("  " + "-" * 55)
    tvecs_mm = [np.array(e["tvec"]) for e in per_frame_errors]
    for i, err_info in enumerate(per_frame_errors):
        t = tvecs_mm[i]
        print(f"  {err_info['dataset']+1:>2} | {err_info['frame_idx']:>4} | "
              f"{t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f} | {float(np.linalg.norm(t)):7.0f}")
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
        "valid_frame_indices": [e["frame_idx"] for e in per_frame_errors],
        "n_frames": len(objpoints_iter),
        "n_frames_initial": len(all_objpoints),
        "n_frames_removed": len(removed_flat),
        "removed_frames": [{"dataset": v[0], "frame_idx": v[1], "error": e}
                          for v, e in removed_flat],
        "n_datasets": n_datasets,
        "data_dirs": data_dirs,
        "sample_steps": sample_steps,
        "square_size_mm": SQUARE_SIZE,
        "frobenius_deviation_from_ref": float(diff_fro),
        "iterations": iteration - (1 if not bad else 0),
    }
    save_json(result, os.path.join(args.output, "intrinsic_tianmou.json"))
    print(f"\n  结果已保存: {args.output}/intrinsic_tianmou.json")


if __name__ == "__main__":
    main()
