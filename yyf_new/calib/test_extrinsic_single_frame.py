#!/usr/bin/env python3
"""
calib/test_extrinsic_single_frame.py
======================================
用单帧图像标定天眸 ↔ RealSense 外参（R, T）的测试脚本。

使用方法:
    python3 test_extrinsic_single_frame.py

说明:
    - 使用 /projects/calib_data/output_0327_1624/tianmou/tianmou_0000.png
      和 /projects/calib_data/output_0327_1624/color_cropped/color_0000.png
    - 内参来自 calibration_output_1624/ 目录
    - 无任何过滤逻辑，直接对这一帧进行外参标定
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    BOARD_COLS, BOARD_ROWS, SQUARE_SIZE,
    read_tianmou_image, to_gray, build_3d_objpoints,
    detect_corners, detect_corners_pair, visualize_stereo_pair,
    ensure_dir, save_json, print_header,
    print_matrix, print_vector,
)


# ============================================================
#  内参（来自 calibration_output_1624/）
# ============================================================
CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration_output_1624")

TM_K = np.array([
    [707.313,   0.0,    405.064],
    [  0.0,  707.571,  266.083],
    [  0.0,    0.0,      1.0  ]], dtype=np.float64)

TM_DIST = np.array([-0.3285, 0.2797, -0.000077, 0.000731, -0.2768], dtype=np.float64)

RS_K = np.array([
    [385.880,   0.0,    329.332],
    [  0.0,  385.681,  162.213],
    [  0.0,    0.0,      1.0  ]], dtype=np.float64)

RS_DIST = np.array([-0.0551, 0.0688, -0.0010, -0.0011, -0.0389], dtype=np.float64)


# ============================================================
#  图像路径
# ============================================================
TM_IMG_PATH = "/projects/calib_data/output_0327_1624/tianmou/tianmou_0000.png"
RS_IMG_PATH = "/projects/calib_data/output_0327_1624/color_cropped/color_0000.png"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "calibration_output_1624")


def main():
    ensure_dir(OUTPUT_DIR)
    vis_dir = os.path.join(OUTPUT_DIR, "visualization")
    ensure_dir(vis_dir)

    print(f"\n{'='*60}")
    print(f"  单帧外参标定测试（天眸 ↔ RealSense）")
    print(f"{'='*60}")
    print(f"  天眸图像: {TM_IMG_PATH}")
    print(f"  RS图像:   {RS_IMG_PATH}")

    # ── 1. 读取图像 ──────────────────────────────────────────
    print_header("读取图像")
    tm_img = read_tianmou_image(TM_IMG_PATH)
    rs_img = cv2.imread(RS_IMG_PATH)

    if tm_img is None:
        print(f"  错误: 无法读取天眸图像: {TM_IMG_PATH}")
        return
    if rs_img is None:
        print(f"  错误: 无法读取RS图像: {RS_IMG_PATH}")
        return

    print(f"  天眸图像尺寸: {tm_img.shape[1]}×{tm_img.shape[0]}")
    print(f"  RS图像尺寸:   {rs_img.shape[1]}×{rs_img.shape[0]}")

    # ── 2. 检测棋盘格角点 ─────────────────────────────────────
    print_header("检测棋盘格角点")
    tm_gray = to_gray(tm_img)
    rs_gray = to_gray(rs_img)

    ret_tm, c_tm = detect_corners(tm_gray, BOARD_COLS, BOARD_ROWS)
    ret_rs, c_rs = detect_corners(rs_gray, BOARD_COLS, BOARD_ROWS)
    print(f"  天眸角点检测: {'成功' if ret_tm else '失败'} (找到 {len(c_tm)} 个点)")
    print(f"  RS角点检测:   {'成功' if ret_rs else '失败'} (找到 {len(c_rs)} 个点)")

    if not (ret_tm and ret_rs):
        print("  角点检测失败，退出")
        return

    # 可视化
    vis_path = visualize_stereo_pair(tm_gray, rs_img, c_tm, c_rs, "extrinsic_single_frame", vis_dir)
    print(f"  可视化已保存: {vis_path}")

    # ── 3. 构建 3D 物体点 ─────────────────────────────────────
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)

    # ── 4. 独立 PnP：估算各相机位姿 ──────────────────────────
    print_header("独立 PnP 位姿估算")

    _, rvec_tm, tvec_tm = cv2.solvePnP(
        objp.astype(np.float64), c_tm.astype(np.float64),
        TM_K, TM_DIST, flags=cv2.SOLVEPNP_ITERATIVE)
    _, rvec_rs, tvec_rs = cv2.solvePnP(
        objp.astype(np.float64), c_rs.astype(np.float64),
        RS_K, RS_DIST, flags=cv2.SOLVEPNP_ITERATIVE)

    R_tm, _ = cv2.Rodrigues(rvec_tm)
    R_rs, _ = cv2.Rodrigues(rvec_rs)

    print(f"\n  天眸位姿:")
    print(f"    rvec_tm = {rvec_tm.ravel().round(4).tolist()}")
    print(f"    tvec_tm = {tvec_tm.ravel().round(2).tolist()} mm")
    print(f"    |tvec_tm| = {np.linalg.norm(tvec_tm):.2f} mm")

    print(f"\n  RealSense 位姿:")
    print(f"    rvec_rs = {rvec_rs.ravel().round(4).tolist()}")
    print(f"    tvec_rs = {tvec_rs.ravel().round(2).tolist()} mm")
    print(f"    |tvec_rs| = {np.linalg.norm(tvec_rs):.2f} mm")

    # ── 5. 计算外参：P_tm = R @ P_rs + T ──────────────────────
    print_header("外参计算（PnP 相减法）")
    print(f"  公式: P_tianmou = R @ P_rs + T")
    print(f"       R = R_tm @ R_rs.T")
    print(f"       T = tvec_tm - R @ tvec_rs")

    R_ext = R_tm @ R_rs.T
    T_ext = (tvec_tm - R_ext @ tvec_rs).ravel()

    # 正交化
    U, _, Vt = np.linalg.svd(R_ext)
    R_ext = U @ Vt
    if np.linalg.det(R_ext) < 0:
        Vt[-1, :] *= -1
        R_ext = U @ Vt

    baseline_mm = float(np.linalg.norm(T_ext))
    baseline_cm = baseline_mm / 10.0

    print(f"\n  旋转矩阵 R:")
    print_matrix("R", R_ext)
    print(f"\n  平移向量 T:")
    print_vector("T", T_ext)
    print(f"\n  基线距离 |T|: {baseline_mm:.2f} mm ({baseline_cm:.2f} cm)")

    # 旋转角
    angle_x = np.arctan2(R_ext[2, 1], R_ext[2, 2]) * 180 / np.pi
    angle_y = np.arctan2(-R_ext[2, 0], np.sqrt(R_ext[2, 1]**2 + R_ext[2, 2]**2)) * 180 / np.pi
    angle_z = np.arctan2(R_ext[1, 0], R_ext[0, 0]) * 180 / np.pi
    print(f"\n  旋转角: Rx={angle_x:.1f}°, Ry={angle_y:.1f}°, Rz={angle_z:.1f}°")

    # ── 6. E 矩阵 ────────────────────────────────────────────
    print_header("本质矩阵 E")
    T_norm = T_ext / (np.linalg.norm(T_ext) + 1e-9)
    T_skew = np.array([
        [  0,         -T_norm[2],  T_norm[1]],
        [  T_norm[2],  0,         -T_norm[0]],
        [ -T_norm[1],  T_norm[0],  0        ]], dtype=np.float64)
    E_mat = T_skew @ R_ext
    print(f"  E 奇异值（应为 [1, 1, 0]）: {np.linalg.svd(E_mat)[1].round(4)}")

    # ── 7. 重投影误差验证 ─────────────────────────────────────
    print_header("重投影误差验证")

    # 外参含义: P_tm = R_ext @ P_rs + T_ext
    #   即: 世界坐标系的棋盘格点经 R_ext, T_ext 从 RS 坐标系变换到 TM 坐标系
    #
    # 验证方式:
    #   (a) 把 RS PnP 求出的 3D 点 P_rs 投影到 TM 图像，与 TM 检测角点对比
    #   (b) 把 TM PnP 求出的 3D 点 P_tm 投影到 RS 图像，与 RS 检测角点对比
    #
    # P_tm = R_ext @ P_rs + T_ext  ⇒  cv2 Rodrigues 表示为
    #   rvec = Rodrigues(R_ext.T), tvec = -R_ext.T @ T_ext

    # (a) RS 3D 点 → TM 图像投影
    rvec_rs2tm = cv2.Rodrigues(R_ext.T)[0]
    tvec_rs2tm = (-R_ext.T @ T_ext.reshape(3, 1)).reshape(3, 1)
    proj_rs2tm, _ = cv2.projectPoints(objp, rvec_rs2tm, tvec_rs2tm, TM_K, TM_DIST)
    err_rs2tm = float(np.mean(np.linalg.norm(
        proj_rs2tm.squeeze().astype(np.float64)
        - c_tm.squeeze().astype(np.float64), axis=1)))

    # (b) TM 3D 点 → RS 图像投影
    rvec_tm2rs = cv2.Rodrigues(R_ext)[0]
    tvec_tm2rs = T_ext.reshape(3, 1)
    proj_tm2rs, _ = cv2.projectPoints(objp, rvec_tm2rs, tvec_tm2rs, RS_K, RS_DIST)
    err_tm2rs = float(np.mean(np.linalg.norm(
        proj_tm2rs.squeeze().astype(np.float64)
        - c_rs.squeeze().astype(np.float64), axis=1)))

    # 单目内参直接重投影（验证内参质量）
    proj_tm_direct, _ = cv2.projectPoints(objp, rvec_tm, tvec_tm, TM_K, TM_DIST)
    err_tm_direct = float(np.mean(np.linalg.norm(
        proj_tm_direct.squeeze().astype(np.float64)
        - c_tm.squeeze().astype(np.float64), axis=1)))
    proj_rs_direct, _ = cv2.projectPoints(objp, rvec_rs, tvec_rs, RS_K, RS_DIST)
    err_rs_direct = float(np.mean(np.linalg.norm(
        proj_rs_direct.squeeze().astype(np.float64)
        - c_rs.squeeze().astype(np.float64), axis=1)))

    print(f"  单目内参验证:")
    print(f"    TM 直接投影:  {err_tm_direct:.3f} px")
    print(f"    RS 直接投影:  {err_rs_direct:.3f} px")
    print(f"  双目外参验证:")
    print(f"    RS 3D→TM 投影: {err_rs2tm:.3f} px  (关键，外参正确性)")
    print(f"    TM 3D→RS 投影: {err_tm2rs:.3f} px  (关键，外参正确性)")

    # ── 8. 对极几何验证 ───────────────────────────────────────
    print_header("对极几何验证")
    F_mat = np.linalg.inv(RS_K).T @ E_mat @ np.linalg.inv(TM_K)
    c_tm_ud = cv2.undistortPoints(c_tm.astype(np.float64), TM_K, TM_DIST)
    c_rs_ud = cv2.undistortPoints(c_rs.astype(np.float64), RS_K, RS_DIST)
    ep_dists = []
    for pt_t, pt_r in zip(c_tm_ud.squeeze(), c_rs_ud.squeeze()):
        epi = F_mat @ np.array([pt_t[0], pt_t[1], 1.0])
        epi_norm = epi[:2] / (np.linalg.norm(epi[:2]) + 1e-9)
        dist = abs(float(epi_norm @ np.array([pt_r[0], pt_r[1]])))
        ep_dists.append(dist)
    mean_ep = float(np.mean(ep_dists))
    print(f"  平均对极距离: {mean_ep:.3f} px")
    print(f"  各角点对极距离: {[f'{e:.3f}' for e in ep_dists]}")

    # ── 9. 保存结果 ──────────────────────────────────────────
    print_header("保存结果")
    result = {
        "description": "单帧外参标定结果（PnP相减法）",
        "frame": "tianmou_0000.png + color_0000.png",
        "P_tianmou_is_R_P_rs_plus_T": True,
        "R": R_ext.tolist(),
        "T": T_ext.tolist(),
        "E": E_mat.tolist(),
        "baseline_mm": baseline_mm,
        "baseline_cm": baseline_cm,
        "rotation_angles_deg": {
            "Rx": float(angle_x),
            "Ry": float(angle_y),
            "Rz": float(angle_z),
        },
        "reprojection_errors": {
            "tianmou_direct_px": err_tm_direct,
            "realsense_direct_px": err_rs_direct,
            "extrinsic_rs2tm_px": err_rs2tm,
            "extrinsic_tm2rs_px": err_tm2rs,
        },
        "epipolar_distance_px": mean_ep,
        "per_point_epipolar_distances": ep_dists,
        "rvec_tianmou": rvec_tm.ravel().tolist(),
        "tvec_tianmou": tvec_tm.ravel().tolist(),
        "rvec_realsense": rvec_rs.ravel().tolist(),
        "tvec_realsense": tvec_rs.ravel().tolist(),
        "K_tianmou": TM_K.tolist(),
        "dist_tianmou": TM_DIST.tolist(),
        "K_realsense": RS_K.tolist(),
        "dist_realsense": RS_DIST.tolist(),
        "calibration_method": "single_frame_PnP_subtraction",
    }

    out_path = os.path.join(OUTPUT_DIR, "extrinsic_single_frame.json")
    save_json(result, out_path)
    print(f"  结果已保存: {out_path}")

    print(f"\n{'='*60}")
    print(f"  总结")
    print(f"{'='*60}")
    print(f"  天眸相对RS的姿态:")
    print(f"    平移: [{T_ext[0]:.1f}, {T_ext[1]:.1f}, {T_ext[2]:.1f}] mm")
    print(f"    基线: {baseline_cm:.1f} cm")
    print(f"    旋转: Rx={angle_x:.1f}°, Ry={angle_y:.1f}°, Rz={angle_z:.1f}°")
    print(f"  单目内参误差: TM={err_tm_direct:.3f}px, RS={err_rs_direct:.3f}px")
    print(f"  双目外参误差: RS→TM={err_rs2tm:.3f}px, TM→RS={err_tm2rs:.3f}px")
    print(f"  对极距离:     {mean_ep:.3f} px")


if __name__ == "__main__":
    main()
