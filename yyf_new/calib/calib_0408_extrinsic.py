#!/usr/bin/env python3
"""
calib/calib_0408_extrinsic.py
对 /projects/calib_data/output_0326_2346 进行天眸 ↔ RealSense 外参标定。
分辨率不同：tianmou=640×320, realsense=640×480。

原理（基于PnP的双目外参标定）:
  1. 对每个同步帧对，分别在两个相机中独立检测棋盘格角点
  2. 用各自的内参 + PnP 估算该帧棋盘格在各自相机坐标系下的位姿 (R_i, t_i)
  3. 所有帧共用一个物理棋盘格，通过 Procrustes 问题求解两个相机间的刚体变换 (R, T)
  4. 迭代剔除误差较大的帧

外参含义: P_tianmou = R @ P_realsense + T
"""

import os, sys, cv2, numpy as np, json
from glob import glob
from tqdm import tqdm
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, os.path.dirname(__file__))
import utils

OUTPUT = "./calib_0408"
os.makedirs(os.path.join(OUTPUT, "visualization"), exist_ok=True)

BOARD_COLS = utils.BOARD_COLS          # 11
BOARD_ROWS = utils.BOARD_ROWS         # 8
SQUARE_SIZE = utils.SQUARE_SIZE       # 29.66 mm


def build_objp():
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def detect_corners(gray):
    h, w = gray.shape[:2]
    factor = max(2.0, 1280 / w)
    gray_big = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_NEAREST)
    ret, corners_big = cv2.findChessboardCorners(gray_big, (BOARD_COLS, BOARD_ROWS), None)
    if not ret:
        return False, np.array([])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray_big, corners_big, (11, 11), (-1, -1), criteria)
    corners = corners_big / factor
    return True, corners.astype(np.float32)


def load_intrinsics():
    """加载本次标定的内参"""
    tm_json = os.path.join(OUTPUT, "intrinsic_tianmou.json")
    rs_json = os.path.join(OUTPUT, "intrinsic_realsense.json")

    with open(tm_json) as f:
        d = json.load(f)
    K_tm = np.array(d["K"], dtype=np.float64)
    dist_tm = np.array(d["dist"], dtype=np.float64)

    with open(rs_json) as f:
        d = json.load(f)
    K_rs = np.array(d["K"], dtype=np.float64)
    dist_rs = np.array(d["dist"], dtype=np.float64)

    return K_tm, dist_tm, K_rs, dist_rs


def load_frame_pairs():
    """加载天眸和 RealSense 同步帧对（前800帧中均匀取50张，step=16）"""
    tm_dir = "/projects/calib_data/output_0326_2346/tianmou"
    rs_dir = "/projects/calib_data/output_0326_2346/color_orig"

    tm_files = sorted(glob(os.path.join(tm_dir, "tianmou_*.png")))[:800][::16][:50]
    rs_files = sorted(glob(os.path.join(rs_dir, "color_*.png")))[:800][::16][:50]

    # 按帧索引配对
    pairs = []
    for tm_path in tm_files:
        bn = os.path.basename(tm_path)
        idx_str = bn.replace("tianmou_", "").replace(".png", "")
        # 找对应的 RS 文件
        rs_candidate = os.path.join(rs_dir, f"color_{idx_str}.png")
        if os.path.exists(rs_candidate):
            pairs.append((int(idx_str), tm_path, rs_candidate))

    return sorted(pairs, key=lambda x: x[0])


def estimate_board_pose_K(K: np.ndarray, dist: np.ndarray, objp: np.ndarray,
                          corners: np.ndarray):
    """用 PnP 估算棋盘格在相机坐标系下的位姿"""
    ret, rvec, tvec = cv2.solvePnP(
        objp.astype(np.float64), corners.astype(np.float64),
        K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ret:
        return None, None
    R_cam, _ = cv2.Rodrigues(rvec)
    return R_cam, tvec.ravel()


def solve_rigid_transform_umeyama(X: np.ndarray, Y: np.ndarray):
    """
    Umeyama 算法：从对应点集 Y -> X 求刚体变换 X = R @ Y + T
    X: (N, 3) 目标坐标
    Y: (N, 3) 源坐标
    返回: R (3,3), T (3,)
    """
    N, D = X.shape
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    sigma_X2 = (Xc ** 2).sum() / N
    sigma_Y2 = (Yc ** 2).sum() / N

    cov = Yc.T @ Xc / N

    U, D_svd, Vt = np.linalg.svd(cov)
    S = np.eye(D)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[D - 1, D - 1] = -1

    R = Vt.T @ S @ U.T
    c = (sigma_X2 / (sigma_Y2 + 1e-10)) * np.trace(np.diag(D_svd) @ S) / D
    T = mu_X - c * (R @ mu_Y)

    # 正交化
    U2, _, Vt2 = np.linalg.svd(R)
    R = U2 @ Vt2
    if np.linalg.det(R) < 0:
        Vt2[-1, :] *= -1
        R = U2 @ Vt2

    return R, T


def compute_reproj_errors(K_tm, dist_tm, K_rs, dist_rs,
                          objp, R, T, frames_data):
    """
    给定外参 R, T，计算每帧的重投影误差。
    frames_data: [(rvec_tm, t_tm, rvec_rs, t_rs), ...]
    """
    R_inv = R.T
    errors = []
    for i, (R_tm, t_tm, R_rs, t_rs) in enumerate(frames_data):
        # 天眸重投影误差（用全局 R, T 变换后的位姿）
        t_rs_in_tm = R @ t_rs + T
        # 对 RS 位姿做一致性检查：R_tm @ t_rs_in_tm 应等于 t_tm
        err_consistency = float(np.linalg.norm(t_tm - R_tm @ t_rs_in_tm))

        # 直接用独立 PnP 的重投影误差
        proj_tm, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_tm)[0], t_tm, K_tm, dist_tm)
        proj_rs, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_rs)[0], t_rs, K_rs, dist_rs)

        errors.append({
            "idx": i,
            "error": err_consistency,
            "frame_idx": frames_data[i][4] if len(frames_data[i]) > 4 else i,
        })
    return errors


def compute_epipolar_error(K_tm, dist_tm, K_rs, dist_rs, E_mat,
                            corners_tm, corners_rs):
    """计算去畸变后的对极距离误差（单位：像素）"""
    c_tm_ud = cv2.undistortPoints(corners_tm.astype(np.float64), K_tm, dist_tm)
    c_rs_ud = cv2.undistortPoints(corners_rs.astype(np.float64), K_rs, dist_rs)
    F_mat = np.linalg.inv(K_rs).T @ E_mat @ np.linalg.inv(K_tm)

    dists = []
    for pt_t, pt_r in zip(c_tm_ud.squeeze(), c_rs_ud.squeeze()):
        epi = F_mat @ np.array([pt_t[0], pt_t[1], 1.0])
        epi_norm = epi[:2] / (np.linalg.norm(epi[:2]) + 1e-9)
        dist = abs(float(epi_norm @ np.array([pt_r[0], pt_r[1]])))
        dists.append(dist)
    return float(np.mean(dists))


def main():
    print("\n" + "="*60)
    print("  天眸 ↔ RealSense 双目外参标定（不同分辨率：640×320 vs 640×480）")
    print("="*60)

    # ── 加载内参 ──────────────────────────────────────────────
    print("\n[1/5] 加载内参...")
    K_tm, dist_tm, K_rs, dist_rs = load_intrinsics()
    print(f"  天眸 (640×320): fx={K_tm[0,0]:.1f}, fy={K_tm[1,1]:.1f}, cx={K_tm[0,2]:.1f}, cy={K_tm[1,2]:.1f}")
    print(f"  RS   (640×480): fx={K_rs[0,0]:.1f}, fy={K_rs[1,1]:.1f}, cx={K_rs[0,2]:.1f}, cy={K_rs[1,2]:.1f}")

    # ── 加载帧对 & 检测角点 ───────────────────────────────────
    print("\n[2/5] 加载帧对 & 检测角点（前800帧均匀取50张）...")
    pairs = load_frame_pairs()
    print(f"  帧对总数（index匹配）: {len(pairs)}")

    objp = build_objp()
    frames_data = []   # [(R_tm, t_tm, R_rs, t_rs, frame_idx), ...]
    corners_tm_all, corners_rs_all = [], []

    for frame_idx, tm_path, rs_path in tqdm(pairs, desc="[检测角点]"):
        # 天眸图像
        img_tm = cv2.imread(tm_path)
        if img_tm is None:
            continue
        if utils.TM_FLIP_HORIZONTAL:
            img_tm = cv2.flip(img_tm, 1)
        gray_tm = cv2.cvtColor(img_tm, cv2.COLOR_BGR2GRAY)

        # RealSense 图像
        img_rs = cv2.imread(rs_path)
        if img_rs is None:
            continue
        gray_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)

        ret_t, c_tm = detect_corners(gray_tm)
        ret_r, c_rs = detect_corners(gray_rs)
        if not (ret_t and ret_r):
            continue

        # 各自 PnP 估算棋盘格位姿
        R_tm, t_tm = estimate_board_pose_K(K_tm, dist_tm, objp, c_tm)
        R_rs, t_rs = estimate_board_pose_K(K_rs, dist_rs, objp, c_rs)
        if R_tm is None or R_rs is None:
            continue

        frames_data.append((R_tm.copy(), t_tm.copy(), R_rs.copy(), t_rs.copy(), frame_idx))
        corners_tm_all.append(c_tm.copy())
        corners_rs_all.append(c_rs.copy())

        # 可视化
        vis_tm = img_tm.copy()
        vis_rs = img_rs.copy()
        cv2.drawChessboardCorners(vis_tm, (BOARD_COLS, BOARD_ROWS), c_tm, True)
        cv2.drawChessboardCorners(vis_rs, (BOARD_COLS, BOARD_ROWS), c_rs, True)
        cv2.imwrite(os.path.join(OUTPUT, "visualization", f"ex_tm_{frame_idx:04d}.png"), vis_tm)
        cv2.imwrite(os.path.join(OUTPUT, "visualization", f"ex_rs_{frame_idx:04d}.png"), vis_rs)

    print(f"  有效帧对: {len(frames_data)}")
    if len(frames_data) < 3:
        print("  帧数不足，退出")
        return

    # ── 迭代求解外参 + 剔除坏帧 ───────────────────────────────
    print("\n[3/5] 迭代求解外参 + 剔除坏帧...")
    THRESHOLD = 50.0   # 位姿一致性误差阈值 (mm)
    MAX_ITER = 10

    # 初始化：用 Umeyama 从所有帧计算初始 R, T
    data_iter = list(frames_data)
    corners_tm_iter = list(corners_tm_all)
    corners_rs_iter = list(corners_rs_all)
    removed = []
    iteration = 0

    while iteration < MAX_ITER:
        iteration += 1

        # 构建对应点集：TM = R @ RS + T
        # 即 TM_translation = R @ RS_translation + T
        t_tm_arr = np.array([d[1] for d in data_iter])   # (N, 3)
        t_rs_arr = np.array([d[3] for d in data_iter])   # (N, 3)

        R_init, T_init = solve_rigid_transform_umeyama(t_tm_arr, t_rs_arr)

        # 计算每帧的位姿一致性误差
        per_err = []
        for i, (R_tm, t_tm, R_rs, t_rs, fidx) in enumerate(data_iter):
            # t_tm_pred = R @ t_rs + T
            t_tm_pred = R_init @ t_rs + T_init
            err = float(np.linalg.norm(t_tm - t_tm_pred))
            per_err.append((i, err, fidx))

        bad = [(i, e, f) for i, e, f in per_err if e > THRESHOLD]
        mean_err = np.mean([e for _, e, _ in per_err])
        print(f"  迭代 {iteration}: 帧数={len(data_iter)}, "
              f"平均位姿误差={mean_err:.2f}mm, 剔除={len(bad)}")

        if not bad:
            print("  ✓ 收敛")
            break
        if len(data_iter) - len(bad) < 10:
            print("  ⚠ 帧数不足，停止")
            break

        for i, e, f in sorted(bad, key=lambda x: -x[1]):
            removed.append((f, e))
        bad_idxs = sorted([i for i, _, _ in bad], reverse=True)
        for bi in bad_idxs:
            del data_iter[bi]
            del corners_tm_iter[bi]
            del corners_rs_iter[bi]

    # ── 最终外参 ───────────────────────────────────────────────
    print("\n[4/5] 计算最终外参...")
    t_tm_final = np.array([d[1] for d in data_iter])
    t_rs_final = np.array([d[3] for d in data_iter])
    R_final, T_final = solve_rigid_transform_umeyama(t_tm_final, t_rs_final)

    baseline = float(np.linalg.norm(T_final))
    print(f"\n  外参结果 (P_tianmou = R @ P_realsense + T):")
    print(f"  T = [{T_final[0]:.1f}, {T_final[1]:.1f}, {T_final[2]:.1f}] mm")
    print(f"  |T| = {baseline:.1f} mm ({baseline/10:.1f} cm)")

    # 旋转角
    angle_x = np.arctan2(R_final[2, 1], R_final[2, 2]) * 180 / np.pi
    angle_y = np.arctan2(-R_final[2, 0], np.sqrt(R_final[2, 1]**2 + R_final[2, 2]**2)) * 180 / np.pi
    angle_z = np.arctan2(R_final[1, 0], R_final[0, 0]) * 180 / np.pi
    print(f"  旋转角: Rx={angle_x:.1f}°, Ry={angle_y:.1f}°, Rz={angle_z:.1f}°")

    # E 矩阵
    T_norm = T_final / (np.linalg.norm(T_final) + 1e-9)
    T_skew = np.array([
        [0, -T_norm[2], T_norm[1]],
        [T_norm[2], 0, -T_norm[0]],
        [-T_norm[1], T_norm[0], 0]], dtype=np.float64)
    E_mat = T_skew @ R_final

    # ── 对极几何验证 ───────────────────────────────────────────
    print("\n[5/5] 对极几何验证...")
    ep_errors = []
    for i, (c_tm, c_rs) in enumerate(zip(corners_tm_iter, corners_rs_iter)):
        ep_err = compute_epipolar_error(K_tm, dist_tm, K_rs, dist_rs, E_mat, c_tm, c_rs)
        ep_errors.append(ep_err)
    mean_ep = float(np.mean(ep_errors))
    print(f"  平均对极距离: {mean_ep:.3f} px")
    if mean_ep < 1.0:
        print("  ✓ 极线几何一致性优秀（< 1px）")
    elif mean_ep < 2.0:
        print("  ⚠ 极线几何误差中等（1~2px）")
    else:
        print("  ✗ 极线几何误差较大，检查内参")

    # ── 每帧误差汇总 ───────────────────────────────────────────
    print("\n  各帧位姿误差 (mm):")
    print(f"  {'帧':>4} | {'位姿误差':>10} | {'对极误差':>10} | {'状态':>6}")
    print("  " + "-" * 40)
    for i, (d, ep) in enumerate(zip(data_iter, ep_errors)):
        flag = " BAD" if ep > 3.0 or np.linalg.norm(d[1] - R_final @ d[3] - T_final) > THRESHOLD else ""
        pose_err = float(np.linalg.norm(d[1] - R_final @ d[3] - T_final))
        print(f"  {d[4]:>4} | {pose_err:10.2f} | {ep:10.3f} | {flag}")

    # ── 保存 ──────────────────────────────────────────────────
    frame_errors = []
    for i, (d, ep) in enumerate(zip(data_iter, ep_errors)):
        pose_err = float(np.linalg.norm(d[1] - R_final @ d[3] - T_final))
        frame_errors.append({
            "frame_idx": d[4],
            "pose_error_mm": pose_err,
            "epipolar_error_px": ep,
        })

    result = {
        "P_tianmou_is_R_P_rs_plus_T": True,
        "resolution_tianmou": {"width": 640, "height": 320},
        "resolution_realsense": {"width": 640, "height": 480},
        "R": R_final.tolist(),
        "T": T_final.tolist(),
        "E": E_mat.tolist(),
        "F": None,
        "baseline_mm": baseline,
        "baseline_cm": baseline / 10.0,
        "mean_epipolar_error_px": mean_ep,
        "per_frame_errors": frame_errors,
        "n_frames": len(data_iter),
        "n_frames_initial": len(frames_data),
        "n_frames_removed": len(removed),
        "removed_frames": [{"frame_idx": f, "error_mm": e} for f, e in removed],
        "valid_frame_indices": [d[4] for d in data_iter],
        "K_tianmou": K_tm.tolist(),
        "dist_tianmou": dist_tm.tolist(),
        "K_realsense": K_rs.tolist(),
        "dist_realsense": dist_rs.tolist(),
        "calibration_method": "PnP + Umeyama rigid transform (different resolutions)",
        "iterations": iteration - (1 if not bad else 0),
        "reproj_threshold_mm": THRESHOLD,
        "rotation_angles_deg": {"Rx": angle_x, "Ry": angle_y, "Rz": angle_z},
    }
    path = os.path.join(OUTPUT, "extrinsic_tianmou_realsense.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(f"\n  已保存: {path}")


if __name__ == "__main__":
    main()
