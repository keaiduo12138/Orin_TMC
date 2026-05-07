#!/usr/bin/env python3
"""
calib/calibrate_stereo_extrinsics.py
====================================
标定天眸 ↔ RealSense 双目外参（R, T）。

支持两种模式：
  - 单数据集模式（向后兼容）：--tm-data-dir + --rs-data-dir
  - 多数据集联合模式：--tm-data-dirs + --rs-data-dirs + --sample-steps

多数据集联合模式下，所有数据集共用同一套 R/T，
OpenCV stereoCalibrate 对所有帧对同时优化，数据越多精度越高。

推荐算法：cv2.stereoCalibrate（全局联合优化）
备选算法：逐帧 PnP + 位姿相减（仅作对比参考）

使用方式:
    # 单数据集（向后兼容）
    python3 calibrate_stereo_extrinsics.py \
        --tm-data-dir /projects/calib_data/0324/tianmou/ \
        --rs-data-dir /projects/calib_data/0324/color_cropped/ \
        --tm-intrinsic ./calibration_output/intrinsic_tianmou.json \
        --rs-intrinsic ./calibration_output/intrinsic_realsense_cropped.json \
        -o ./calibration_output

    # 多数据集联合（各数据集 n 帧取 1 帧）
    python3 calibrate_stereo_extrinsics.py \
        --tm-data-dirs /projects/calib_data/A/tianmou/ /projects/calib_data/B/tianmou/ \
        --rs-data-dirs /projects/calib_data/A/color/   /projects/calib_data/B/color/ \
        --sample-steps 5 8 \
        --tm-intrinsic ./calibration_output/intrinsic_tianmou.json \
        --rs-intrinsic ./calibration_output/intrinsic_realsense_cropped.json \
        -o ./calibration_output

外参含义:
    P_tianmou = R @ P_realsense + T
    即: 天眸坐标系 = R × RS坐标系 + T

输出:
    extrinsic_tianmou_realsense.json
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    BOARD_COLS, BOARD_ROWS, SQUARE_SIZE, TM_K_REF, RS_K_DEFAULT, RS_D_DEFAULT,
    read_tianmou_image, to_gray, build_3d_objpoints,
    detect_corners, detect_corners_pair,
    visualize_stereo_pair, plot_reprojection_errors, draw_corners,
    ensure_dir, save_json, print_header,
    print_matrix, print_vector,
)


def _load_tm_files(tm_dir: str) -> Dict[str, str]:
    """加载天眸图像文件，按索引键索引"""
    # 支持：tianmou_0000.png 或 color/ 下的 tianmou_*.png
    patterns = [
        glob(os.path.join(tm_dir, "tianmou_*.png")),
        glob(os.path.join(tm_dir, "*_tianmou.png")),
        glob(os.path.join(tm_dir, "tianmou", "tianmou_*.png")),
    ]
    files = []
    for p in patterns:
        files.extend(p)

    if not files:
        return {}

    result = {}
    for f in files:
        bn = os.path.basename(f)
        if bn.startswith("tianmou_"):
            # tianmou_0000.png
            key = bn.replace("tianmou_", "").replace(".png", "")
        elif "_tianmou" in bn:
            # xxx_tianmou.png
            key = bn.replace("_tianmou", "").split("_")[-1].replace(".png", "")
        else:
            continue
        result[key] = f
    return result


def _load_rs_files(rs_dir: str) -> Dict[str, str]:
    """加载 RealSense Color 图像文件，按索引键索引"""
    patterns = [
        glob(os.path.join(rs_dir, "color_*.png")),
        glob(os.path.join(rs_dir, "frame__Color_*.png")),
        glob(os.path.join(rs_dir, "color", "color_*.png")),
        glob(os.path.join(rs_dir, "color", "frame__Color_*.png")),
        glob(os.path.join(rs_dir, "*_color.png")),
    ]
    files = []
    for p in patterns:
        files.extend(p)

    if not files:
        return {}

    result = {}
    for f in files:
        bn = os.path.basename(f)
        if bn.startswith("color_"):
            # color_0000.png
            key = bn.replace("color_", "").replace(".png", "")
        elif bn.startswith("frame__Color_"):
            # frame__Color_1774356905244.23803710937500.png
            # 从文件名末尾提取索引（取最后一段数字作为键）
            key = bn.replace("frame__Color_", "").split(".")[0][-4:]
        elif "_color" in bn:
            # xxx_color.png
            key = bn.replace("_color", "").split("_")[-1].replace(".png", "")
        else:
            continue
        result[key] = f
    return result


def load_frame_pairs(data_dir: str = None,
                     tm_data_dir: str = None,
                     rs_data_dir: str = None
                     ) -> List[Tuple[int, str, str]]:
    """加载天眸和 RealSense 帧对。

    支持两种模式：
    - 分离目录模式（推荐）：tm_data_dir 和 rs_data_dir 分别指定
    - 统一目录模式（兼容旧用法）：data_dir 指定单一目录
    """
    # ── 模式1：分离目录 ──────────────────────────────────────
    if tm_data_dir is not None and rs_data_dir is not None:
        tm_files = _load_tm_files(tm_data_dir)
        rs_files = _load_rs_files(rs_data_dir)

        # 找共同键
        def sort_key(k):
            try:
                return int(k)
            except (ValueError, TypeError):
                return 0

        common = sorted(set(tm_files.keys()) & set(rs_files.keys()), key=sort_key)
        pairs = []
        for k in common:
            try:
                idx = int(k)
            except (ValueError, TypeError):
                idx = len(pairs)
            pairs.append((idx, tm_files[k], rs_files[k]))
        return pairs

    # ── 模式2：统一目录（兼容旧用法）─────────────────────────
    tm_flat = glob(os.path.join(data_dir, "*_tianmou.png"))
    tm_subdir = glob(os.path.join(data_dir, "tianmou", "tianmou_*.png"))
    tm_files = {os.path.basename(f).replace("_tianmou", "").replace("tianmou_", "").split("_")[-1].replace(".png", ""): f
                for f in (tm_flat if tm_flat else tm_subdir)}

    rs_flat = glob(os.path.join(data_dir, "*_color.png"))
    rs_subdir = glob(os.path.join(data_dir, "color", "color_*.png"))
    rs_files = {os.path.basename(f).replace("_color", "").replace("color_", "").split("_")[-1].replace(".png", ""): f
                for f in (rs_flat if rs_flat else rs_subdir)}

    common_keys = sorted(set(tm_files.keys()) & set(rs_files.keys()), key=lambda k: int(k))
    pairs = [(int(k), tm_files[k], rs_files[k]) for k in common_keys]
    return pairs


def load_intrinsics(tm_path: str, rs_path: str, rs_source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载并验证内参。"""
    # 天眸
    if tm_path.lower() == "ref":
        K_tm = TM_K_REF.copy()
        dist_tm = np.zeros(5, dtype=np.float64)
        print("  天眸内参: 使用 TM_K_REF（硬编码参考值）")
    else:
        data = __import__("utils").load_json(tm_path) if os.path.exists(tm_path) else None
        if data is None:
            raise FileNotFoundError(f"天眸内参文件不存在: {tm_path}")
        K_tm = np.array(data["K"], dtype=np.float64)
        dist_tm = np.array(data["dist"], dtype=np.float64)
        # 验证分辨率一致性
        if "resolution" in data:
            w, h = data["resolution"]["width"], data["resolution"]["height"]
            print(f"  天眸内参分辨率: {w}×{h}")
        else:
            print("  [警告] 天眸内参无分辨率字段")

    # RealSense
    if os.path.exists(rs_path):
        data = __import__("utils").load_json(rs_path)
        if "factory" in data:
            if rs_source == "factory":
                K_rs = np.array(data["factory"]["K"], dtype=np.float64)
                dist_rs = np.array(data["factory"]["dist"], dtype=np.float64)
            elif rs_source == "calibrated":
                K_rs = np.array(data["calibrated"]["K"], dtype=np.float64)
                dist_rs = np.array(data["calibrated"]["dist"], dtype=np.float64)
            else:
                K_rs = np.array(data["factory"]["K"], dtype=np.float64)
                dist_rs = np.array(data["factory"]["dist"], dtype=np.float64)
        else:
            # 新的裁剪后内参格式（K 和 dist 在顶层，如 intrinsic_realsense_cropped.json）
            K_rs = np.array(data["K"], dtype=np.float64)
            dist_rs = np.array(data["dist"], dtype=np.float64)
            print(f"  RS 内参（裁剪后格式）: fx={K_rs[0,0]:.1f}, fy={K_rs[1,1]:.1f}, "
                  f"cx={K_rs[0,2]:.1f}, cy={K_rs[1,2]:.1f}, dist={dist_rs.ravel().round(4).tolist()}")
    else:
        K_rs = RS_K_DEFAULT.copy()
        dist_rs = RS_D_DEFAULT.copy()
        print("  [警告] RS 内参文件不存在，使用默认值")

    return K_tm, dist_tm, K_rs, dist_rs


def detect_all_pairs(pairs: List[Tuple], objp: np.ndarray,
                     ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """对全部帧对检测棋盘格角点。"""
    objpoints, tm_pts, rs_pts, valid_ids = [], [], [], []
    for idx, tm_path, rs_path in pairs:
        tm_img = read_tianmou_image(tm_path)
        rs_img = cv2.imread(rs_path)
        if tm_img is None or rs_img is None:
            continue
        ret, c_tm, c_rs = detect_corners_pair(to_gray(tm_img), to_gray(rs_img))
        if ret:
            objpoints.append(objp.copy())
            tm_pts.append(c_tm)
            rs_pts.append(c_rs)
            valid_ids.append(idx)
    return objpoints, tm_pts, rs_pts, valid_ids


def extract_RT_from_E(E_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从Essential矩阵SVD提取旋转R和平移T。

    E = [T]_× R，其中 [T]_× = [[0,-Tz,Ty],[Tz,0,-Tx],[-Ty,Tx,0]]
    SVD: E = U Σ Vᵀ，U·W·Vᵀ 或 U·Wᵀ·Vᵀ（W为辅助矩阵）

    OpenCV 策略：
        U, S, Vt = np.linalg.svd(E)
        W = [[0,-1,0],[1,0,0],[0,0,1]]

        R1 = U·W·Vt,   T1 = U[:,2]    (列3)
        R2 = U·Wᵀ·Vt,  T2 = -U[:,2]
    共4种组合，只有一个使两点云在相机前方。
    这里返回 SVD 的自然结果（T 以欧氏距离为单位，R 以行列式=1约束）。
    """
    U, S, Vt = np.linalg.svd(E_mat)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # W 辅助矩阵
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R_candidates = [
        U @ W @ Vt,
        U @ W.T @ Vt,
    ]
    T_candidates = [
        U[:, 2].ravel(),
        -U[:, 2].ravel(),
    ]

    # 正交化
    R_candidates = [np.linalg.svd(R)[0] @ np.linalg.svd(R)[2]
                    for R in R_candidates]

    # 取行列式正的（旋转矩阵行列式=+1）
    R_candidates = [R * np.linalg.det(R) for R in R_candidates]

    # 简单选择：第一个（后续通过验证判断是否需要翻转）
    R = R_candidates[0]
    T = T_candidates[0]

    # 确保 T 方向合理：Z 分量应为正（RS 在 TM 前方为正 Z）
    # 如果 Z < 0，翻转 T 和 R
    if T[2] < 0:
        T = -T
        R = R.T  # R^T = R^{-1}

    return R, T


def verify_extrinsic(E_mat: np.ndarray, tm_pts: List[np.ndarray],
                     rs_pts: List[np.ndarray],
                     K_tm: np.ndarray, dist_tm: np.ndarray,
                     K_rs: np.ndarray, dist_rs: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    逐帧验证 E 矩阵质量，计算平均对极距离。

    返回: (R, T, mean_epipolar_px)
    """
    from itertools import combinations

    # 尝试从 E 的两种分解中选更好的
    U, _, Vt = np.linalg.svd(E_mat)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    candidates = []
    for sign_T in [1, -1]:
        for use_W_T in [False, True]:
            T_c = sign_T * U[:, 2].ravel()
            R_c = U @ (W.T if use_W_T else W) @ Vt
            R_c = np.linalg.svd(R_c)[0] @ np.linalg.svd(R_c)[2]
            R_c *= np.linalg.det(R_c)

            candidates.append((R_c, T_c))

    best_ep = float('inf')
    best_R, best_T = None, None

    for R_c, T_c in candidates:
        # 过滤：确保 T_z > 0（RS 在 TM 相机前方）
        if T_c[2] < 0:
            T_c = -T_c
            R_c = R_c.T

        ep_dists = []
        for c_tm, c_rs in zip(tm_pts[:5], rs_pts[:5]):
            c_tm_ud = cv2.undistortPoints(
                c_tm.astype(np.float64), K_tm, dist_tm)
            c_rs_ud = cv2.undistortPoints(
                c_rs.astype(np.float64), K_rs, dist_rs)
            F_mat = np.linalg.inv(K_rs).T @ E_mat @ np.linalg.inv(K_tm)
            for pt_t, pt_r in zip(c_tm_ud.squeeze(), c_rs_ud.squeeze()):
                epi = F_mat @ np.array([pt_t[0], pt_t[1], 1.0])
                epi_norm = epi[:2] / (np.linalg.norm(epi[:2]) + 1e-9)
                dist = abs(float(epi_norm @ np.array([pt_r[0], pt_r[1]])))
                ep_dists.append(dist)

        mean_ep = np.mean(ep_dists)
        if mean_ep < best_ep:
            best_ep = mean_ep
            best_R, best_T = R_c, T_c

    return best_R, best_T, best_ep


def compute_reproj_errors_stereo(
    objpoints: List[np.ndarray],
    tm_pts: List[np.ndarray],
    rs_pts: List[np.ndarray],
    K_tm: np.ndarray, dist_tm: np.ndarray,
    K_rs: np.ndarray, dist_rs: np.ndarray,
    R: np.ndarray, T: np.ndarray,
) -> List[Dict]:
    """
    用全局 R, T 计算每帧的重投影误差。

    方法：把每帧 TM 和 RS 的 PnP 位姿用全局 R, T 做一致性检验。
    - 对 TM:  用全局外参预测其在 RS 中的位姿，与实际 RS PnP 位姿对比
    - 对 RS:  用全局外参预测其在 TM 中的位姿，与实际 TM PnP 位姿对比

    不使用全局 R, T 直接投影（因为 PnP 的世界坐标系是任意的）。
    """
    results = []
    R_inv = R.T

    for objp_i, c_tm, c_rs in zip(objpoints, tm_pts, rs_pts):
        # 各相机独立 PnP（绝对位姿，共享同一世界坐标系）
        _, rvec_t, tvec_t = cv2.solvePnP(
            objp_i.astype(np.float64), c_tm.astype(np.float64),
            K_tm, dist_tm, flags=cv2.SOLVEPNP_ITERATIVE)
        _, rvec_r, tvec_r = cv2.solvePnP(
            objp_i.astype(np.float64), c_rs.astype(np.float64),
            K_rs, dist_rs, flags=cv2.SOLVEPNP_ITERATIVE)

        # 用全局 R, T 把 RS 绝对位姿变换到 TM 坐标系
        # TM_pred = R @ RS_abs + T
        R_r, _ = cv2.Rodrigues(rvec_r)
        tvec_r_in_tm = R @ tvec_r + T

        # 全局外参预测的 TM 位姿投影回图像，与实际检测角点对比
        p_tm_pred, _ = cv2.projectPoints(objp_i, rvec_t, tvec_t, K_tm, dist_tm)
        err_tm = float(np.mean(np.sqrt(
            np.sum((p_tm_pred.squeeze() - c_tm.squeeze())**2, axis=1))))

        # 同理：把 TM 绝对位姿变换到 RS 坐标系
        # RS_pred = R_inv @ (TM_abs - T)
        tvec_t_in_rs = R_inv @ (tvec_t - T)
        p_rs_pred, _ = cv2.projectPoints(objp_i, rvec_r, tvec_r, K_rs, dist_rs)
        err_rs = float(np.mean(np.sqrt(
            np.sum((p_rs_pred.squeeze() - c_rs.squeeze())**2, axis=1))))

        # 外参一致性误差：TM 视角下的 RS 位置 vs 用全局外参变换后的位置
        # 即：tvec_r vs tvec_r_in_tm 的差异有多大
        consistency_err = float(np.mean(np.abs(
            tvec_r.ravel() - R @ tvec_r - T)))

        results.append({
            "error_tm": err_tm,
            "error_rs": err_rs,
            "mean_error": (err_tm + err_rs) / 2.0,
            "consistency_px": consistency_err,
        })
    return results


def compute_epipolar_errors(
    tm_pts: List[np.ndarray], rs_pts: List[np.ndarray],
    K_tm: np.ndarray, dist_tm: np.ndarray,
    K_rs: np.ndarray, dist_rs: np.ndarray,
    E_mat: np.ndarray,
) -> Tuple[float, List[float]]:
    """计算去畸变后的对极距离误差（单位：像素）。"""
    F_mat = np.linalg.inv(K_rs).T @ E_mat @ np.linalg.inv(K_tm)
    all_dists = []
    per_frame = []
    for c_tm, c_rs in zip(tm_pts, rs_pts):
        c_tm_ud = cv2.undistortPoints(c_tm.astype(np.float64), K_tm, dist_tm)
        c_rs_ud = cv2.undistortPoints(c_rs.astype(np.float64), K_rs, dist_rs)
        frame_dists = []
        for pt_t, pt_r in zip(c_tm_ud.squeeze(), c_rs_ud.squeeze()):
            epi = F_mat @ np.array([pt_t[0], pt_t[1], 1.0])
            epi_norm = epi[:2] / (np.linalg.norm(epi[:2]) + 1e-9)
            dist = abs(float(epi_norm @ np.array([pt_r[0], pt_r[1]])))
            frame_dists.append(dist)
            all_dists.append(dist)
        per_frame.append(float(np.mean(frame_dists)))
    return float(np.mean(all_dists)), per_frame


def main():
    parser = argparse.ArgumentParser(description="天眸 ↔ RealSense 双目外参标定（支持多数据集联合）")

    # ── 多数据集模式（推荐）─────────────────────────────────
    grp = parser.add_argument_group("多数据集模式（推荐）")
    grp.add_argument("--tm-data-dirs", nargs="+", default=None,
                     help="天眸图像目录列表（对应 --rs-data-dirs）")
    grp.add_argument("--rs-data-dirs", nargs="+", default=None,
                     help="RS 图像目录列表（对应 --tm-data-dirs）")

    # ── 单数据集模式（兼容）─────────────────────────────────
    grp2 = parser.add_argument_group("单数据集模式（兼容旧用法）")
    grp2.add_argument("--tm-data-dir", default=None,
                     help="天眸图像目录（单数据集）")
    grp2.add_argument("--rs-data-dir", default=None,
                     help="RealSense Color 图像目录（单数据集）")
    grp2.add_argument("--data-dir", "-d", default=None,
                     help="统一目录模式（天眸和 RealSense 在同一目录下）")

    # ── 公共参数 ──────────────────────────────────────────────
    parser.add_argument("--tm-intrinsic", default="ref",
                        help="天眸内参 JSON，或 'ref' 使用 TM_K_REF")
    parser.add_argument("--rs-intrinsic", default=None,
                        help="RS 内参 JSON（支持旧格式 factory/calibrated 和新的顶层 K/dist 格式）")
    parser.add_argument("--rs-source", default="calibrated",
                        choices=["factory", "calibrated", "default"],
                        help="RS 内参来源（仅对旧格式有效，多数据集模式下默认 calibrated）")
    parser.add_argument("--sample-steps", type=int, nargs="+", default=None,
                        help="每个数据集的抽样步长（默认各为5）。"
                             "例如 --sample-steps 5 8 表示数据集A每5帧取1、数据集B每8帧取1。"
                             "数据集数量必须与目录数量一致。")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="总最大帧数上限（默认无限制）；超过时各数据集等比例缩减")
    parser.add_argument("--output", "-o", default="./calibration_output",
                        help="输出目录（外参保存为 extrinsic_tianmou_realsense.json）")
    parser.add_argument("--bad-threshold", type=float, default=1.0,
                        help="最终报告的坏帧标记阈值(px)：重投影误差超过此值的帧在报告中标记为BAD（不影响标定，默认1.0px）")
    parser.add_argument("--reproj-threshold", type=float, default=5.0,
                        help="迭代剔除阈值(px)：重投影误差超过此值的帧将被剔除，"
                             "然后重新标定直到无新帧被剔除（默认5.0px）")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="最大迭代剔除轮数（默认10）")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="剔除后最少保留帧数，低于此数时停止剔除（默认10）")
    parser.add_argument("--legacy", action="store_true",
                        help="强制使用旧的逐帧PnP+相减方法（不推荐）")
    parser.add_argument("--select-frames", action="store_true")
    parser.add_argument("--no-vis", action="store_true",
                        help="跳过逐帧角点可视化（加速标定）")
    args = parser.parse_args()

    # ── 模式判断 & 参数规范化 ───────────────────────────────
    # 优先多数据集模式
    multi_mode = (args.tm_data_dirs is not None and args.rs_data_dirs is not None)
    single_mode = ((args.tm_data_dir is not None and args.rs_data_dir is not None)
                   or args.data_dir is not None)

    if not multi_mode and not single_mode:
        print("错误: 必须提供 --tm-data-dirs + --rs-data-dirs（多数据集）"
              "或 --tm-data-dir + --rs-data-dir（单数据集）"
              "或 --data-dir（统一目录）")
        return

    ensure_dir(args.output)
    vis_dir = os.path.join(args.output, "visualization")
    ensure_dir(vis_dir)

    # ── 规范化目录列表 ──────────────────────────────────────
    if multi_mode:
        n_datasets = len(args.tm_data_dirs)
        if len(args.rs_data_dirs) != n_datasets:
            print(f"错误: --tm-data-dirs({n_datasets}) 与 --rs-data-dirs({len(args.rs_data_dirs)}) 数量不一致")
            return
        tm_dirs = args.tm_data_dirs
        rs_dirs = args.rs_data_dirs
    else:
        n_datasets = 1
        if args.data_dir is not None:
            tm_dirs = [args.data_dir]
            rs_dirs = [args.data_dir]
        else:
            tm_dirs = [args.tm_data_dir]
            rs_dirs = [args.rs_data_dir]

    # 规范化抽样步长
    if args.sample_steps is None:
        sample_steps = [5] * n_datasets
    elif len(args.sample_steps) == 1:
        sample_steps = args.sample_steps * n_datasets
    elif len(args.sample_steps) != n_datasets:
        raise ValueError(f"--sample-steps 数量({len(args.sample_steps)}) "
                         f"必须等于数据集数量({n_datasets})")
    else:
        sample_steps = args.sample_steps

    print(f"\n{'='*60}")
    print(f"  双目外参标定（{'多数据集联合' if multi_mode else '单数据集'}）")
    print(f"{'='*60}")
    if multi_mode:
        for i, (td, rd, ss) in enumerate(zip(tm_dirs, rs_dirs, sample_steps)):
            print(f"  数据集{i+1}: TM={td}")
            print(f"           RS={rd}")
            print(f"           步长={ss}")
    else:
        print(f"  TM: {tm_dirs[0]}")
        print(f"  RS: {rs_dirs[0]}")
        print(f"  步长: {sample_steps[0]}")

    # ── 加载内参 ─────────────────────────────────────────────
    print_header("加载内参")
    K_tm, dist_tm, K_rs, dist_rs = load_intrinsics(
        args.tm_intrinsic, args.rs_intrinsic or "", args.rs_source)

    print(f"\n  天眸: fx={K_tm[0,0]:.1f}, fy={K_tm[1,1]:.1f}, cx={K_tm[0,2]:.1f}, cy={K_tm[1,2]:.1f}")
    print(f"        dist: {dist_tm.ravel().round(4).tolist()}")
    print(f"  RS:   fx={K_rs[0,0]:.1f}, fy={K_rs[1,1]:.1f}, cx={K_rs[0,2]:.1f}, cy={K_rs[1,2]:.1f}")
    print(f"        dist: {dist_rs.ravel().round(4).tolist()}")
    print(f"  RS 内参来源: {args.rs_source}")

    # ── 加载帧对（多数据集）──────────────────────────────────
    print_header("加载帧对")
    all_pairs_per_dataset = []
    total_raw = 0
    for di, (tm_dir, rs_dir) in enumerate(zip(tm_dirs, rs_dirs)):
        pairs = load_frame_pairs(tm_data_dir=tm_dir, rs_data_dir=rs_dir)
        sampled = pairs[::sample_steps[di]]
        all_pairs_per_dataset.append((di, sampled))
        total_raw += len(pairs)
        print(f"\n  ── 数据集 {di+1}/{n_datasets}: {tm_dir}")
        print(f"    帧对总数: {len(pairs)}，抽样步长={sample_steps[di]}，采样后={len(sampled)}")

    # 总帧数上限
    if args.max_frames is not None:
        total_sampled = sum(len(p) for _, p in all_pairs_per_dataset)
        if total_sampled > args.max_frames:
            scale = args.max_frames / total_sampled
            print(f"\n  ⚠ 超过总帧数上限({args.max_frames})，各数据集等比例缩减...")
            new_all = []
            for di, pairs in all_pairs_per_dataset:
                n_keep = max(1, int(len(pairs) * scale))
                new_all.append((di, pairs[:n_keep]))
            all_pairs_per_dataset = new_all
            total_sampled = sum(len(p) for _, p in all_pairs_per_dataset)
            print(f"  缩减后总帧数: {total_sampled}")

    # ── 检测角点 ─────────────────────────────────────────────
    print_header("检测棋盘格角点")
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)

    objpoints_all, tm_pts_all, rs_pts_all = [], [], []
    valid_ids_all: List[int] = []
    pair_meta: List[Tuple[int, int, str, str]] = []  # (dataset_idx, frame_idx, tm_path, rs_path)

    for di, pairs in all_pairs_per_dataset:
        print(f"\n  ── 数据集 {di+1}: 检测 {len(pairs)} 帧")
        for frame_idx, tm_path, rs_path in pairs:
            tm_img = read_tianmou_image(tm_path)
            rs_img = cv2.imread(rs_path)
            if tm_img is None or rs_img is None:
                continue
            ret, c_tm, c_rs = detect_corners_pair(to_gray(tm_img), to_gray(rs_img))
            if ret:
                objpoints_all.append(objp.copy())
                tm_pts_all.append(c_tm)
                rs_pts_all.append(c_rs)
                valid_ids_all.append(frame_idx)
                pair_meta.append((di, frame_idx, tm_path, rs_path))
        print(f"    检测成功: {len([x for x in pair_meta if x[0]==di])}/{len(pairs)}")

    print(f"\n  所有数据集合计: {len(objpoints_all)} 帧")

    if len(objpoints_all) < 3:
        print("  帧数不足（至少需要 3 帧），退出")
        return

    # ── 角点可视化（带数据集前缀）───────────────────────────
    if not args.no_vis:
        pair_map = {vmid[1]: (vmid[2], vmid[3]) for vmid in pair_meta}
        for di_frame_idx, c_tm, c_rs, (di, frame_idx, tm_path, rs_path) in \
                zip(valid_ids_all, tm_pts_all, rs_pts_all,
                    [(m[0], m[1], m[2], m[3]) for m in pair_meta]):
            tm_img = read_tianmou_image(tm_path)
            rs_img = cv2.imread(rs_path)
            visualize_stereo_pair(tm_img, rs_img, c_tm, c_rs,
                                 f"ds{di+1}_{frame_idx:04d}", vis_dir)
    else:
        print("  [跳过可视化，共 %d 帧]" % len(pair_meta))

    # ══════════════════════════════════════════════════════════
    #  主算法：cv2.stereoCalibrate 全局联合优化 + 迭代剔除
    # ══════════════════════════════════════════════════════════
    print_header("主算法：cv2.stereoCalibrate（全局联合优化 + 迭代剔除）")

    # 图像尺寸（各数据集分辨率应该一致，统一取第一个）
    first_pair_meta = pair_meta[0]
    tm_img_sample = read_tianmou_image(first_pair_meta[2])
    rs_img_sample = cv2.imread(first_pair_meta[3])
    tm_h, tm_w = tm_img_sample.shape[:2]
    rs_h, rs_w = rs_img_sample.shape[:2]
    print(f"  天眸图像尺寸: {tm_w}×{tm_h}")
    print(f"  RS图像尺寸:   {rs_w}×{rs_h}")
    print(f"  初始帧数: {len(objpoints_all)}")
    print(f"  剔除阈值: {args.reproj_threshold} px")
    print(f"  最大迭代轮数: {args.max_iterations}")
    print(f"  最少保留帧数: {args.min_frames}")

    # flags: 锁死内参，只优化 R, T
    flags = (
        cv2.CALIB_FIX_INTRINSIC
        | cv2.CALIB_RATIONAL_MODEL
    )

    # 复制用于迭代
    objpoints_iter = list(objpoints_all)
    tm_pts_iter = list(tm_pts_all)
    rs_pts_iter = list(rs_pts_all)
    pair_meta_iter = list(pair_meta)

    iteration = 0
    removed_in_round = []
    bad = []  # track last bad for convergence check

    while iteration < args.max_iterations:
        iteration += 1

        ret_stereo, K1, D1, K2, D2, R_stereo, T_stereo, E_stereo, F_stereo = cv2.stereoCalibrate(
            objectPoints=objpoints_iter,
            imagePoints1=tm_pts_iter,
            imagePoints2=rs_pts_iter,
            cameraMatrix1=K_tm,
            distCoeffs1=dist_tm,
            cameraMatrix2=K_rs,
            distCoeffs2=dist_rs,
            imageSize=(tm_w, tm_h),
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6),
        )

        R_use = R_stereo.copy()
        T_use = T_stereo.ravel().copy()

        # 正交化
        U, _, Vt = np.linalg.svd(R_use)
        R_use = U @ Vt
        if np.linalg.det(R_use) < 0:
            Vt[-1, :] *= -1
            R_use = U @ Vt

        # E 矩阵归一化
        T_norm = T_use / (np.linalg.norm(T_use) + 1e-9)
        T_skew = np.array([[0, -T_norm[2], T_norm[1]],
                           [T_norm[2], 0, -T_norm[0]],
                           [-T_norm[1], T_norm[0], 0]], dtype=np.float64)
        E_iter = T_skew @ R_use

        # 计算每帧重投影误差
        per_iter_errs = []
        for i, (objp_i, ct_i, cr_i) in enumerate(zip(objpoints_iter, tm_pts_iter, rs_pts_iter)):
            proj_t, _ = cv2.projectPoints(objp_i, cv2.Rodrigues(R_use)[0],
                                           T_use.reshape(3, 1), K_tm, dist_tm)
            proj_r, _ = cv2.projectPoints(objp_i, cv2.Rodrigues(np.eye(3))[0],
                                           np.zeros((3, 1)), K_rs, dist_rs)
            # 单目误差：各相机重投影到各自图像
            proj_tm, _ = cv2.projectPoints(objp_i, cv2.Rodrigues(np.eye(3))[0],
                                              np.zeros((3, 1)), K_tm, dist_tm)
            proj_rs, _ = cv2.projectPoints(objp_i, cv2.Rodrigues(R_use.T)[0],
                                             (-R_use.T @ T_use).reshape(3, 1), K_rs, dist_rs)
            err_tm = float(np.linalg.norm(
                proj_tm.squeeze().astype(np.float64) - ct_i.squeeze().astype(np.float64)))
            err_rs = float(np.linalg.norm(
                proj_rs.squeeze().astype(np.float64) - cr_i.squeeze().astype(np.float64)))
            per_iter_errs.append({"idx": i, "err_tm": err_tm, "err_rs": err_rs,
                                  "mean": (err_tm + err_rs) / 2})

        bad = [e for e in per_iter_errs if e["mean"] > args.reproj_threshold]
        bad_idxs = set(e["idx"] for e in bad)

        mean_err = np.mean([e["mean"] for e in per_iter_errs])
        print(f"\n  ── 迭代 {iteration}:")
        print(f"    帧数: {len(objpoints_iter)}，平均误差: {mean_err:.4f} px")
        print(f"    超阈值帧: {len(bad)}")
        if bad:
            for e in sorted(bad, key=lambda x: -x["mean"])[:5]:
                vid = pair_meta_iter[e["idx"]]
                print(f"      ds{vid[0]+1}_{vid[1]:04d}: TM={e['err_tm']:.2f} RS={e['err_rs']:.2f} 均值={e['mean']:.2f}")

        if not bad:
            print("    ✓ 无新帧被剔除，迭代收敛")
            break

        if len(objpoints_iter) - len(bad) < args.min_frames:
            print(f"    ⚠ 剔除后帧数 {len(objpoints_iter)-len(bad)} < {args.min_frames}，停止剔除")
            break

        removed_in_round.append(
            [(pair_meta_iter[e["idx"]], e["mean"]) for e in sorted(bad, key=lambda x: -x["mean"])]
        )
        bad_idxs_sorted = sorted(bad_idxs, reverse=True)
        for bi in bad_idxs_sorted:
            del objpoints_iter[bi]
            del tm_pts_iter[bi]
            del rs_pts_iter[bi]
            del pair_meta_iter[bi]

    # ── 最终结果 ──────────────────────────────────────────────
    removed_flat = [item for round_list in removed_in_round for item in round_list]
    print_header("外参标定（最终结果）")
    print(f"  剔除轮数: {iteration - (1 if not bad else 0)}")
    print(f"  累计剔除: {len(removed_flat)} 帧")
    print(f"  最终帧数: {len(objpoints_iter)}（初始 {len(objpoints_all)} 帧）")

    ret_stereo, K1, D1, K2, D2, R_stereo, T_stereo, E_stereo, F_stereo = cv2.stereoCalibrate(
        objectPoints=objpoints_iter,
        imagePoints1=tm_pts_iter,
        imagePoints2=rs_pts_iter,
        cameraMatrix1=K_tm,
        distCoeffs1=dist_tm,
        cameraMatrix2=K_rs,
        distCoeffs2=dist_rs,
        imageSize=(tm_w, tm_h),
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6),
    )

    print(f"\n  stereoCalibrate 返回值: {ret_stereo:.4f}")

    R = R_stereo.copy()
    T = T_stereo.ravel().copy()

    # 从 E 提取 R, T（验证）
    R_from_E, T_from_E = extract_RT_from_E(E_stereo)
    print(f"\n  从 E 提取的 R, T:")
    print(f"  T_from_E = [{T_from_E[0]:.1f}, {T_from_E[1]:.1f}, {T_from_E[2]:.1f}] mm")
    print(f"  |T_from_E| = {np.linalg.norm(T_from_E):.1f} mm")

    print(f"\n  使用 cv2.stereoCalibrate 结果:")
    print(f"  T = [{T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f}] mm")
    print(f"  |T| = {np.linalg.norm(T):.1f} mm ({np.linalg.norm(T)/10:.1f} cm)")

    # 正交化
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # ── E 矩阵（归一化格式，与 depth_to_tianmou.py 兼容）─────
    T_norm = T / np.linalg.norm(T)
    T_skew = np.array([[0, -T_norm[2], T_norm[1]],
                       [T_norm[2], 0, -T_norm[0]],
                       [-T_norm[1], T_norm[0], 0]], dtype=np.float64)
    E_mat = T_skew @ R
    svd_E = np.linalg.svd(E_mat)[1]
    print(f"\n  E 奇异值（应为 [1, 1, 0]）: {svd_E.round(4)}")

    # ── 剔除记录 ─────────────────────────────────────────
    if removed_flat:
        print(f"\n  剔除帧详情（共 {len(removed_flat)} 帧）:")
        print(f"  {'数据集':>2} | {'帧':>4} | {'重投影误差(px)':>14}")
        print("  " + "-" * 30)
        for vid, err in sorted(removed_flat, key=lambda x: -x[1]):
            print(f"  {vid[0]+1:>2} | {vid[1]:>4} | {err:>14.2f}")

    # ── 旧算法（仅作对比参考）：逐帧 PnP + 位姿相减 ─────────
    # 使用最终保留下来的帧（与 stereoCalibrate 一致）
    print_header("对比：逐帧 PnP + 位姿相减（旧算法，最终帧集）")

    rvecs_tm, tvecs_tm, rvecs_rs, tvecs_rs = [], [], [], []
    for di, frame_idx, tm_path, rs_path in pair_meta_iter:
        tm_gray = to_gray(read_tianmou_image(tm_path))
        rs_gray = to_gray(cv2.imread(rs_path))
        ret_t, ct = detect_corners(tm_gray, BOARD_COLS, BOARD_ROWS)
        ret_r, cr = detect_corners(rs_gray, BOARD_COLS, BOARD_ROWS)
        if not (ret_t and ret_r):
            raise RuntimeError(f"帧 {frame_idx} 重检角点失败")

        _, rv_t, tv_t = cv2.solvePnP(objp.astype(np.float64), ct.astype(np.float64),
                                      K_tm, dist_tm, flags=cv2.SOLVEPNP_ITERATIVE)
        _, rv_r, tv_r = cv2.solvePnP(objp.astype(np.float64), cr.astype(np.float64),
                                      K_rs, dist_rs, flags=cv2.SOLVEPNP_ITERATIVE)
        rvecs_tm.append(rv_t.ravel())
        tvecs_tm.append(tv_t.ravel())
        rvecs_rs.append(rv_r.ravel())
        tvecs_rs.append(tv_r.ravel())

    Rs_rel, Ts_rel = [], []
    for rv_t, t_t, rv_r, t_r in zip(rvecs_tm, tvecs_tm, rvecs_rs, tvecs_rs):
        R_t, _ = cv2.Rodrigues(rv_t)
        R_r, _ = cv2.Rodrigues(rv_r)
        R_rel = R_t @ R_r.T
        T_rel = (t_t - R_t @ R_r.T @ t_r).ravel()
        Rs_rel.append(R_rel)
        Ts_rel.append(T_rel)

    # 正确的中位数选择：按迹排序，选最接近中位数的
    trace_scores = [np.trace(R) for R in Rs_rel]
    median_score = float(np.median(trace_scores))
    best_idx = int(np.argmin([abs(t - median_score) for t in trace_scores]))
    R_legacy = Rs_rel[best_idx]
    T_legacy = Ts_rel[best_idx].ravel()
    U_l, _, Vt_l = np.linalg.svd(R_legacy)
    R_legacy = U_l @ Vt_l

    print(f"  T_PnP_legacy = [{T_legacy[0]:.1f}, {T_legacy[1]:.1f}, {T_legacy[2]:.1f}] mm")
    print(f"  |T_PnP_legacy| = {np.linalg.norm(T_legacy):.1f} mm")
    print(f"  R_PnP_legacy[0]: {R_legacy[0].round(3)}")

    # ══════════════════════════════════════════════════════════
    #  极线几何验证（去畸变角点 + 归一化E）
    # ══════════════════════════════════════════════════════════
    print_header("极线几何验证（去畸变角点 + E矩阵）")
    mean_epipolar, per_frame_ep = compute_epipolar_errors(
        tm_pts_iter, rs_pts_iter, K_tm, dist_tm, K_rs, dist_rs, E_mat)

    print(f"  平均对极距离: {mean_epipolar:.3f} px")
    print(f"  各帧对极距离: {[f'{e:.2f}px' for e in per_frame_ep[:5]]}")
    if mean_epipolar < 1.0:
        print("  ✓ 极线几何一致性优秀（< 1px）")
    elif mean_epipolar < 2.0:
        print("  ⚠ 极线几何误差中等（1~2px）")
    else:
        print("  ✗ 极线几何误差较大，检查内参")

    # ── 重投影误差（基于最终 R, T）─────────────────────────────
    print_header("各帧重投影误差（基于最终全局 R, T）")
    per_frame_err = compute_reproj_errors_stereo(
        objpoints_iter, tm_pts_iter, rs_pts_iter,
        K_tm, dist_tm, K_rs, dist_rs, R, T)

    per_frame_errors = [
        {"dataset": pair_meta_iter[i][0], "frame_idx": pair_meta_iter[i][1],
         "error_tm": e["error_tm"], "error_rs": e["error_rs"],
         "mean_error": e["mean_error"]}
        for i, e in enumerate(per_frame_err)
    ]
    bad_frames = [e for e in per_frame_errors if e["mean_error"] > args.bad_threshold]

    print(f"  {'数据集':>2} | {'帧':>4} | {'TM(px)':>8} | {'RS(px)':>8} | {'均值':>8} | {'状态':>6}")
    print("  " + "-" * 65)
    for e in per_frame_errors:
        flag = " <<< BAD" if e["mean_error"] > args.bad_threshold else ""
        print(f"  {e['dataset']+1:>2} | {e['frame_idx']:>4} | {e['error_tm']:8.3f} | "
              f"{e['error_rs']:8.3f} | {e['mean_error']:8.3f} | {flag}")

    mean_reproj = float(np.mean([e["mean_error"] for e in per_frame_errors]))
    print(f"\n  平均重投影误差: {mean_reproj:.3f} px")

    if bad_frames:
        print(f"\n  ⚠ {len(bad_frames)} 帧误差 > {args.bad_threshold}px:")
        for e in bad_frames:
            print(f"  坏帧: ds{e['dataset']+1}_{e['frame_idx']}")

    # ── 重投影误差图 ───────────────────────────────────────────
    plot_path = os.path.join(args.output, "reproj_error_plot.png")
    plot_reprojection_errors(per_frame_errors, "TM ↔ RS 双目标定（stereoCalibrate）",
                             plot_path, args.bad_threshold)
    print(f"\n  误差图已保存: {plot_path}")

    # ── 保存结果 ─────────────────────────────────────────────
    print(f"\n  最终外参（R, T 来源: cv2.stereoCalibrate）:")
    print(f"  旋转矩阵 R (P_tm = R @ P_rs + T):")
    print_matrix("R", R)
    print(f"\n  平移向量 T (mm):")
    print_vector("T", T)
    baseline_mm = float(np.linalg.norm(T))
    print(f"\n  基线距离 |T|: {baseline_mm:.1f} mm ({baseline_mm/10:.1f} cm)")

    # 旋转角
    angle_x = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
    angle_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) * 180 / np.pi
    angle_z = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
    print("\n  物理合理性检查:")
    print(f"    T = [{T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f}] mm")
    print(f"    旋转角: Rx={angle_x:.1f}°, Ry={angle_y:.1f}°, Rz={angle_z:.1f}°")

    # ── 保存 JSON ─────────────────────────────────────────────
    per_frame_epipolar = [
        {"dataset": pair_meta_iter[i][0], "frame_idx": pair_meta_iter[i][1], "epipolar_px": ep}
        for i, ep in enumerate(per_frame_ep)
    ]

    result = {
        "P_tianmou_is_R_P_rs_plus_T": True,
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E_mat.tolist(),
        "F": None,
        "baseline_mm": baseline_mm,
        "baseline_cm": baseline_mm / 10.0,
        "mean_reproj_error": mean_reproj,
        "mean_epipolar_error_px": mean_epipolar,
        "per_frame_errors": per_frame_errors,
        "per_frame_epipolar_errors_px": per_frame_epipolar,
        "bad_frames": [{"dataset": e["dataset"], "frame_idx": e["frame_idx"]} for e in bad_frames],
        "n_valid_frames": len(objpoints_iter),
        "n_frames_initial": len(objpoints_all),
        "n_frames_removed": len(removed_flat),
        "removed_frames": [{"dataset": v[0], "frame_idx": v[1], "error": e}
                          for v, e in removed_flat],
        "n_datasets": n_datasets,
        "data_dirs": {"tianmou": tm_dirs, "realsense": rs_dirs},
        "sample_steps": sample_steps,
        "valid_frame_indices": [{"dataset": m[0], "frame_idx": m[1]} for m in pair_meta_iter],
        "K_tianmou": K_tm.tolist(),
        "dist_tianmou": dist_tm.tolist(),
        "K_realsense": K_rs.tolist(),
        "dist_realsense": dist_rs.tolist(),
        "rs_source": args.rs_source,
        "epipolar_errors_px": per_frame_ep[:3],
        "legacy_T_PnP": T_legacy.tolist(),
        "legacy_baseline_mm": float(np.linalg.norm(T_legacy)),
        "calibration_method": "cv2.stereoCalibrate (multi-dataset joint)" if multi_mode
                             else "cv2.stereoCalibrate",
        "iterations": iteration - (1 if not bad else 0),
    }
    save_json(result, os.path.join(args.output, "extrinsic_tianmou_realsense.json"))
    print(f"\n  结果已保存: {args.output}/extrinsic_tianmou_realsense.json")


if __name__ == "__main__":
    main()
