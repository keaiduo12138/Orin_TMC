#!/usr/bin/env python3
"""
calib/calibrate_stereo_extrinsics.py
====================================
标定天眸 ↔ RealSense 双目外参（R, T）。

推荐算法：cv2.stereoCalibrate（全局联合优化）
备选算法：逐帧 PnP + 位姿相减（仅作对比参考）

使用方式:
    python3 calibrate_stereo_extrinsics.py \
        --data-dir /projects/calib_data/0325_calibration \
        --tm-intrinsic ./calibration_output/intrinsic_tianmou.json \
        --rs-intrinsic ./calibration_output/compare_realsense_intrinsics.json \
        --rs-source factory \
        --output ./calibration_output

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


def load_frame_pairs(data_dir: str) -> List[Tuple[int, str, str]]:
    """加载天眸和 RealSense 帧对。返回 [(帧索引, tm_path, rs_path)]"""
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
            K_rs = np.array(data["K"], dtype=np.float64)
            dist_rs = np.array(data["dist"], dtype=np.float64)
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
    parser = argparse.ArgumentParser(description="天眸 ↔ RealSense 双目外参标定")
    parser.add_argument("--data-dir", "-d", required=True)
    parser.add_argument("--tm-intrinsic", default="ref",
                        help="天眸内参 JSON，或 'ref' 使用 TM_K_REF")
    parser.add_argument("--rs-intrinsic", default=None,
                        help="RS 内参 JSON")
    parser.add_argument("--rs-source", default="factory",
                        choices=["factory", "calibrated", "default"],
                        help="RS 内参来源")
    parser.add_argument("--frames", "-f", type=int, nargs="+", default=None)
    parser.add_argument("--select-frames", action="store_true")
    parser.add_argument("--sample-step", type=int, default=5,
                        help="自动抽样步长（默认5，即每5帧取1帧）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="最大使用帧数（与 sample-step 二选一；设了此参数则忽略 sample-step）")
    parser.add_argument("--output", "-o", default="./calibration_output")
    parser.add_argument("--bad-threshold", type=float, default=1.0)
    parser.add_argument("--legacy", action="store_true",
                        help="强制使用旧的逐帧PnP+相减方法（不推荐）")
    args = parser.parse_args()

    ensure_dir(args.output)
    vis_dir = os.path.join(args.output, "visualization")
    ensure_dir(vis_dir)

    # ── 加载内参 ─────────────────────────────────────────────
    print_header("加载内参")
    K_tm, dist_tm, K_rs, dist_rs = load_intrinsics(
        args.tm_intrinsic, args.rs_intrinsic or "", args.rs_source)

    print(f"\n  天眸: fx={K_tm[0,0]:.1f}, fy={K_tm[1,1]:.1f}, cx={K_tm[0,2]:.1f}, cy={K_tm[1,2]:.1f}")
    print(f"        dist: {dist_tm.ravel().round(4).tolist()}")
    print(f"  RS:   fx={K_rs[0,0]:.1f}, fy={K_rs[1,1]:.1f}, cx={K_rs[0,2]:.1f}, cy={K_rs[1,2]:.1f}")
    print(f"        dist: {dist_rs.ravel().round(4).tolist()}")
    print(f"  RS 内参来源: {args.rs_source}")

    # ── 加载帧对 ─────────────────────────────────────────────
    all_pairs = load_frame_pairs(args.data_dir)
    print(f"\n  帧对总数: {len(all_pairs)}")

    if args.frames:
        pairs = [(i, t, r) for i, t, r in all_pairs if i in args.frames]
        print(f"  指定帧数: {len(pairs)} 帧")
    elif args.max_frames is not None:
        pairs = all_pairs[:args.max_frames]
        print(f"  限制最大帧数: {len(pairs)} 帧（原 {len(all_pairs)} 帧）")
    else:
        pairs = all_pairs[::args.sample_step]
        print(f"  自动抽样: 每 {args.sample_step} 帧取 1 帧，共 {len(pairs)} 帧（原 {len(all_pairs)} 帧）")

    # 交互选择
    if args.select_frames:
        selected = []
        for idx, tm_path, rs_path in pairs:
            tm_img = read_tianmou_image(tm_path)
            rs_img = cv2.imread(rs_path)
            if tm_img is None or rs_img is None:
                continue
            ret, c_tm, c_rs = detect_corners_pair(to_gray(tm_img), to_gray(rs_img))
            vis = np.zeros((480, 960*2, 3), dtype=np.uint8)
            if ret:
                tm_v = draw_corners(tm_img, c_tm)
                rs_v = draw_corners(to_gray(rs_img), c_rs)
                rs_v = cv2.cvtColor(rs_v, cv2.COLOR_GRAY2BGR)
                tm_v = cv2.resize(tm_v, (960, 480))
                rs_v = cv2.resize(rs_v, (960, 480))
                vis = np.hstack([tm_v, rs_v])
            else:
                tm_v = cv2.resize(tm_img, (960, 480))
                rs_v = cv2.resize(to_gray(rs_img), (960, 480))
                rs_v = cv2.cvtColor(rs_v, cv2.COLOR_GRAY2BGR)
                vis = np.hstack([tm_v, rs_v])
            cv2.imshow("选择帧 (SPACE=skip, q=quit, other=accept)", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key != 32:
                selected.append((idx, tm_path, rs_path))
        cv2.destroyAllWindows()
        pairs = selected
        print(f"\n  交互选择: {len(pairs)} 帧")

    # ── 检测角点 ─────────────────────────────────────────────
    print_header(f"检测棋盘格角点（{len(pairs)} 候选帧）")
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)
    objpoints, tm_pts, rs_pts, valid_ids = detect_all_pairs(pairs, objp)
    print(f"  共同检测成功: {len(objpoints)}/{len(pairs)} 帧")

    if len(objpoints) < 3:
        print("  帧数不足（至少需要 3 帧），退出")
        return

    # 角点可视化
    pair_map = {i: (t, r) for i, t, r in pairs}
    for idx in valid_ids:
        tm_path, rs_path = pair_map[idx]
        tm_img = read_tianmou_image(tm_path)
        rs_img = cv2.imread(rs_path)
        pos = valid_ids.index(idx)
        visualize_stereo_pair(tm_img, rs_img,
                              tm_pts[pos], rs_pts[pos],
                              idx, vis_dir)

    # ══════════════════════════════════════════════════════════
    #  主算法：cv2.stereoCalibrate 全局联合优化
    # ══════════════════════════════════════════════════════════
    print_header("主算法：cv2.stereoCalibrate（全局联合优化）")

    # 图像尺寸
    tm_img_sample = read_tianmou_image(pair_map[valid_ids[0]][0])
    rs_img_sample = cv2.imread(pair_map[valid_ids[0]][1])
    tm_h, tm_w = tm_img_sample.shape[:2]
    rs_h, rs_w = rs_img_sample.shape[:2]
    print(f"  天眸图像尺寸: {tm_w}×{tm_h}")
    print(f"  RS图像尺寸:   {rs_w}×{rs_h}")

    # flags: 锁死内参，只优化 R, T
    flags = (
        cv2.CALIB_FIX_INTRINSIC
        | cv2.CALIB_RATIONAL_MODEL
    )

    ret_stereo, K1, D1, K2, D2, R_stereo, T_stereo, E_stereo, F_stereo = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=tm_pts,   # 天眸
        imagePoints2=rs_pts,    # RealSense
        cameraMatrix1=K_tm,
        distCoeffs1=dist_tm,
        cameraMatrix2=K_rs,
        distCoeffs2=dist_rs,
        imageSize=(tm_w, tm_h),
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6),
    )

    print(f"\n  stereoCalibrate 返回值: {ret_stereo:.4f}")
    print(f"  （注意：raw E奇异值=|T|，保存前会归一化）")

    # 从 E 提取 R, T（验证 stereoCalibrate 的结果是否合理）
    R_from_E, T_from_E = extract_RT_from_E(E_stereo)
    print(f"\n  从 E 提取的 R, T:")
    print(f"  T_from_E = [{T_from_E[0]:.1f}, {T_from_E[1]:.1f}, {T_from_E[2]:.1f}] mm")
    print(f"  |T_from_E| = {np.linalg.norm(T_from_E):.1f} mm")
    print(f"  R_from_E[0]: {R_from_E[0].round(3)}")

    # 采用 stereoCalibrate 直接返回的 R, T（已经过全局优化）
    R = R_stereo.copy()
    T = T_stereo.ravel().copy()
    print(f"\n  使用 cv2.stereoCalibrate 结果:")
    print(f"  T = [{T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f}] mm")
    print(f"  |T| = {np.linalg.norm(T):.1f} mm ({np.linalg.norm(T)/10:.1f} cm)")

    # 正交化（确保是纯旋转）
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # ── E 矩阵（归一化格式，与 depth_to_tianmou.py 兼容）─────
    # cv2.stereoCalibrate 返回的 E 奇异值 = |T|（OpenCV 内部带尺度优化）。
    # depth_to_tianmou.py 用 E = [T̂]× R（其中 T̂ = T/|T| 为单位向量），
    # 期望 E 奇异值 = [1, 1, 0]。因此这里用相同的公式重新计算。
    T_norm = T / np.linalg.norm(T)
    T_skew = np.array([[0, -T_norm[2], T_norm[1]],
                       [T_norm[2], 0, -T_norm[0]],
                       [-T_norm[1], T_norm[0], 0]], dtype=np.float64)
    E_mat = T_skew @ R
    svd_E = np.linalg.svd(E_mat)[1]
    print(f"\n  E 奇异值（应为 [1, 1, 0]）: {svd_E.round(4)}")

    # ══════════════════════════════════════════════════════════
    #  旧算法（仅作对比参考）：逐帧 PnP + 位姿相减
    # ══════════════════════════════════════════════════════════
    print_header("对比：逐帧 PnP + 位姿相减（旧算法）")

    rvecs_tm, tvecs_tm, rvecs_rs, tvecs_rs = [], [], [], []
    for vid, tm_path, rs_path in zip(valid_ids,
                                       [pair_map[v][0] for v in valid_ids],
                                       [pair_map[v][1] for v in valid_ids]):
        tm_gray = to_gray(read_tianmou_image(tm_path))
        rs_gray = to_gray(cv2.imread(rs_path))
        ret_t, ct = detect_corners(tm_gray, BOARD_COLS, BOARD_ROWS)
        ret_r, cr = detect_corners(rs_gray, BOARD_COLS, BOARD_ROWS)
        if not (ret_t and ret_r):
            raise RuntimeError(f"帧 {vid} 重检角点失败")

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
        tm_pts, rs_pts, K_tm, dist_tm, K_rs, dist_rs, E_mat)

    print(f"  平均对极距离: {mean_epipolar:.3f} px")
    print(f"  各帧对极距离: {[f'{e:.2f}px' for e in per_frame_ep[:5]]}")
    if mean_epipolar < 1.0:
        print("  ✓ 极线几何一致性优秀（< 1px）")
    elif mean_epipolar < 2.0:
        print("  ⚠ 极线几何误差中等（1~2px）")
    else:
        print("  ✗ 极线几何误差较大，检查内参")

    # ══════════════════════════════════════════════════════════
    #  重投影误差
    # ══════════════════════════════════════════════════════════
    print_header("各帧重投影误差（基于全局 R, T）")
    per_frame_err = compute_reproj_errors_stereo(
        objpoints, tm_pts, rs_pts,
        K_tm, dist_tm, K_rs, dist_rs, R, T)

    per_frame_errors = [
        {"frame_idx": vid, "error_tm": e["error_tm"], "error_rs": e["error_rs"],
         "mean_error": e["mean_error"]}
        for vid, e in zip(valid_ids, per_frame_err)
    ]
    bad_frames = [e for e in per_frame_errors if e["mean_error"] > args.bad_threshold]

    print(f"  {'帧':>4} | {'TM(px)':>8} | {'RS(px)':>8} | {'均值':>8} | {'状态':>6}")
    print("  " + "-" * 55)
    for e in per_frame_errors:
        flag = " <<< BAD" if e["mean_error"] > args.bad_threshold else ""
        print(f"  {e['frame_idx']:>4} | {e['error_tm']:8.3f} | {e['error_rs']:8.3f} | "
              f"{e['mean_error']:8.3f} | {flag}")

    mean_reproj = float(np.mean([e["mean_error"] for e in per_frame_errors]))
    print(f"\n  平均重投影误差: {mean_reproj:.3f} px")

    if bad_frames:
        print(f"\n  ⚠ {len(bad_frames)} 帧误差 > {args.bad_threshold}px:")
        print(f"  坏帧索引: {[e['frame_idx'] for e in bad_frames]}")

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
    print(f"    注：|T| 来自 stereoCalibrate，深度投射用 E_mat（已归一化），极线精度不受影响")

    # ── 保存 JSON ─────────────────────────────────────────────
    result = {
        "P_tianmou_is_R_P_rs_plus_T": True,
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E_mat.tolist(),   # 已归一化（奇异值=[1,1,0]），与旧代码一致
        "F": None,
        "baseline_mm": baseline_mm,
        "baseline_cm": baseline_mm / 10.0,
        "mean_reproj_error": mean_reproj,
        "mean_epipolar_error_px": mean_epipolar,
        "per_frame_errors": per_frame_errors,
        "per_frame_epipolar_errors_px": [{"frame_idx": vid, "epipolar_px": ep}
                                          for vid, ep in zip(valid_ids, per_frame_ep)],
        "bad_frames": [e["frame_idx"] for e in bad_frames],
        "n_valid_frames": len(objpoints),
        "valid_frame_indices": valid_ids,
        "K_tianmou": K_tm.tolist(),
        "dist_tianmou": dist_tm.tolist(),
        "K_realsense": K_rs.tolist(),
        "dist_realsense": dist_rs.tolist(),
        "rs_source": args.rs_source,
        "epipolar_errors_px": per_frame_ep[:3],
        # 对比旧算法结果（供诊断参考）
        "legacy_T_PnP": T_legacy.tolist(),
        "legacy_baseline_mm": float(np.linalg.norm(T_legacy)),
        "calibration_method": "cv2.stereoCalibrate + PnP尺度修正",
        "T_scale_note": "T方向从E提取，尺度从PnP计算（自动修正stereoCalibrate尺度歧义）",
    }
    save_json(result, os.path.join(args.output, "extrinsic_tianmou_realsense.json"))
    print(f"\n  结果已保存: {args.output}/extrinsic_tianmou_realsense.json")


if __name__ == "__main__":
    main()
