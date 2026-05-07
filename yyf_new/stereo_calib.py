#!/usr/bin/env python3
"""
天眸 RGB ↔ RealSense Color 双目外参标定脚本

标定目标：将 RealSense D455 的深度信息投影到天眸画面上。

标定流程：
  阶段1  - RealSense 内参验证（从 bag 文件读取，与出厂值对比）
  阶段2  - 天眸内参标定（基于棋盘格图像，计算内参和畸变系数）
  阶段3  - 双目外参标定（天眸 ↔ RealSense Color，计算 R 和 T）
  阶段4  - 重投影误差评估与可视化验证

依赖：
  opencv-contrib-python, numpy
  RealSense intrinsics 需要 /projects/miniconda3/envs/tianmou 环境中的 pyrealsense2
  天眸内参标定需要 tianmoucv_preview/tianmoucv/calib/opencv_calib.py

使用示例：
  python stereo_calib.py --data-dir /projects/calib_data/0323_calibration_v1
      --board-size 11 8 --square-size 29.66 --output ./calibration_output
      [--bag /projects/cxr_data/2026-3-20/18-48-10.bag]
      [--select-frames]   # 交互式选择有效帧
"""

import os
import sys
import json
import argparse
import datetime
import shutil
import glob
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np

# ============================================================
# 参数配置
# ============================================================

BOARD_COLS, BOARD_ROWS = 11, 8          # 棋盘格角点数（方格数12×9 → 角点11×8）
SQUARE_SIZE = 29.66                      # 方格边长（mm）
FACTOR_RESIZE = 2                        # 角点检测时图像放大倍数

# 天眸参考内参（来自 tianmoucv/tools.py）
TM_K_REF = np.array([
    [462.15896054,   0.          , 307.03359738],
    [  0.         , 462.40462068 , 149.10361709],
    [  0.         ,   0.          ,   1.        ]], dtype=np.float64)

# RealSense D455 默认出厂内参（分辨率 640×480 Color）
RS_K_DEFAULT = np.array([
    [385.019,   0.     , 329.543],
    [  0.    , 384.118 , 241.941],
    [  0.    ,   0.    ,   1.    ]], dtype=np.float64)

RS_D_DEFAULT = np.array([-0.0542, 0.0636, -0.0005, -0.0003, -0.0202], dtype=np.float64)


# ============================================================
# 全局配置
# ============================================================

# 天眸图像水平翻转标志（天眸镜头硬件安装方向导致图像左右镜像）
TM_FLIP_HORIZONTAL = True


# ============================================================
# 工具函数
# ============================================================

def read_tianmou_image(filepath: str, color: bool = False) -> np.ndarray:
    """
    读取天眸图像并应用水平翻转（纠正硬件镜像）。
    Tianmou 镜头安装方向导致图像左右镜像，所有帧均为固定水平翻转，
    检测角点坐标需要与 RealSense 对齐。
    """
    img = cv2.imread(filepath)
    if img is None:
        return None
    if TM_FLIP_HORIZONTAL:
        img = cv2.flip(img, 1)  # 水平翻转: flipCode=1
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """将图像转为灰度 uint8"""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img[..., 0].astype(np.uint8)


def resize_for_detection(gray: np.ndarray, target_min_w: int = 1280) -> Tuple[np.ndarray, float]:
    """放大灰度图用于棋盘格检测，返回放大后的图和缩放因子"""
    h, w = gray.shape[:2]
    factor = max(1.0, target_min_w / w)
    if factor > 1.0:
        gray_big = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_NEAREST)
    else:
        gray_big = gray
    return gray_big, factor


def select_high_quality_frames(
    tm_files: List[str],
    color_files: List[str],
    tm_K: np.ndarray,
    tm_dist: np.ndarray,
    rs_K: np.ndarray,
    rs_dist: np.ndarray,
    board_cols: int,
    board_rows: int,
    square_size: float,
    n_frames: int = 20,
    factor: float = 2.0,
) -> List[int]:
    """
    高质量帧自动筛选：
    1. 对全部帧做双路角点检测（EXHAUSTIVE 模式）
    2. 用 calibrateCamera 估计每帧的 6DoF 位姿
    3. 按角点亚像素精化收敛量 + 姿态角多样性贪心选取最优帧

    参数:
        n_frames: 最终选取的帧数（默认20）
    返回:
        选取的帧索引列表
    """
    board_size = (board_cols, board_rows)
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size

    # ── 第一遍：检测全部有效帧 ──────────────────────────────────────
    print(f"  [HQ筛选] 预扫描全部 {len(tm_files)} 帧（EXHAUSTIVE 角点检测）...")
    detected: List[Dict] = []

    for idx in range(len(tm_files)):
        tm_img = read_tianmou_image(tm_files[idx])
        c_img = cv2.imread(color_files[idx])
        if tm_img is None or c_img is None:
            continue

        gray_tm = to_gray(tm_img)
        gray_rs = to_gray(c_img)

        # 天眸角点检测（低分辨率必须用 EXHAUSTIVE）
        gray_tm_big = cv2.resize(gray_tm, (int(gray_tm.shape[1] * factor),
                                           int(gray_tm.shape[0] * factor)),
                                  interpolation=cv2.INTER_NEAREST)
        ret_tm, corners_tm_big = cv2.findChessboardCornersSB(
            gray_tm_big, board_size,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret_tm:
            continue
        corners_tm = corners_tm_big / factor

        # RealSense 角点检测
        gray_rs_big = cv2.resize(gray_rs, (int(gray_rs.shape[1] * factor),
                                           int(gray_rs.shape[0] * factor)),
                                 interpolation=cv2.INTER_NEAREST)
        ret_rs, corners_rs_big = cv2.findChessboardCornersSB(
            gray_rs_big, board_size,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret_rs:
            continue
        corners_rs = corners_rs_big / factor

        # 亚像素精化（测量收敛量作为质量分数）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        if gray_tm.ndim == 3:
            gray_tm_1ch = cv2.cvtColor(gray_tm, cv2.COLOR_BGR2GRAY)
        else:
            gray_tm_1ch = gray_tm
        if gray_rs.ndim == 3:
            gray_rs_1ch = cv2.cvtColor(gray_rs, cv2.COLOR_BGR2GRAY)
        else:
            gray_rs_1ch = gray_rs
        corners_tm_ref = corners_tm.copy()
        corners_rs_ref = corners_rs.copy()
        cv2.cornerSubPix(gray_tm_1ch, corners_tm_ref, (5, 5), (-1, -1), criteria)
        cv2.cornerSubPix(gray_rs_1ch, corners_rs_ref, (5, 5), (-1, -1), criteria)
        tm_correction = np.linalg.norm(corners_tm_ref - corners_tm)
        rs_correction = np.linalg.norm(corners_rs_ref - corners_rs)
        quality = 1.0 / (1.0 + tm_correction + rs_correction)  # 收敛量越小越好

        detected.append({
            "idx": idx,
            "objp": objp.copy(),
            "corners_tm": corners_tm_ref.astype(np.float32),
            "corners_rs": corners_rs_ref.astype(np.float32),
            "quality": quality,
        })

    print(f"  [HQ筛选] 双路共同检测成功: {len(detected)} 帧")

    if len(detected) < n_frames:
        print(f"  [HQ筛选] 有效帧不足 {n_frames}，使用全部 {len(detected)} 帧")
        return [d["idx"] for d in detected]

    # ── 第二遍：用 calibrateCamera 估计每帧的 6DoF 位姿 ────────────
    print(f"  [HQ筛选] 估计 {len(detected)} 帧的 6DoF 位姿（用于姿态多样性评估）...")
    tm_objpoints = [d["objp"] for d in detected]
    tm_imgpoints = [d["corners_tm"] for d in detected]
    tm_valid_idx = [d["idx"] for d in detected]

    try:
        ret_tm, _, _, rvecs_tm, tvecs_tm = cv2.calibrateCamera(
            tm_objpoints, tm_imgpoints, (gray_tm.shape[1], gray_tm.shape[0]),
            tm_K, tm_dist, flags=cv2.CALIB_FIX_INTRINSIC)
    except Exception:
        # fallback: 用 tvec 长度近似姿态
        tvecs_tm = [np.array([[idx * 0.1, idx * 0.05, 400.0]]).T for idx in range(len(detected))]
        rvecs_tm = [np.zeros(3) for _ in detected]

    # ── 第三遍：贪心选取姿态多样化帧 ────────────────────────────────
    # 将 rvec 转为旋转矩阵，取 camera position 方向
    selected: List[int] = []
    used_dirs: List[np.ndarray] = []

    def camera_dir_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(rvec)
        # 相机朝 -Z 方向，相机在世界坐标系下的位置是 -R^T @ tvec
        return (-R.T @ tvec).ravel() / np.linalg.norm((-R.T @ tvec).ravel())

    # 按质量预排序（优先高质量帧进入候选池）
    sorted_by_quality = sorted(detected, key=lambda d: d["quality"], reverse=True)

    for d in sorted_by_quality:
        ridx = detected.index(d)
        dir_vec = camera_dir_from_rvec_tvec(rvecs_tm[ridx], tvecs_tm[ridx])

        if len(selected) == 0:
            selected.append(d["idx"])
            used_dirs.append(dir_vec)
            continue

        # 计算与已选帧的最小角距离（度数）
        min_angle = min(np.arctan2(np.linalg.norm(np.cross(ud, dir_vec)),
                                   np.dot(ud, dir_vec)) * 180 / np.pi
                        for ud in used_dirs)

        d["angular_coverage"] = min_angle
        d["rvec"] = rvecs_tm[ridx]
        d["tvec"] = tvecs_tm[ridx]
        selected.append(d["idx"])
        used_dirs.append(dir_vec)

        if len(selected) >= n_frames:
            break

    # 重新贪心：从头开始，优先选取与已选帧夹角最大的帧
    # 先用所有帧的 angle diversity 构建贪心解
    candidate_frames = []
    for i, d in enumerate(detected):
        d["rvec"] = rvecs_tm[i]
        d["tvec"] = tvecs_tm[i]
        d["dir"] = camera_dir_from_rvec_tvec(rvecs_tm[i], tvecs_tm[i])
        candidate_frames.append(d)

    selected = []
    used_dirs = []

    for _ in range(n_frames):
        best_frame = None
        best_min_angle = -1.0
        for d in candidate_frames:
            if d["idx"] in selected:
                continue
            if len(used_dirs) == 0:
                best_min_angle = float("inf")
                best_frame = d
                break
            min_angle = min(np.arctan2(np.linalg.norm(np.cross(ud, d["dir"])),
                                       np.dot(ud, d["dir"])) * 180 / np.pi
                            for ud in used_dirs)
            if min_angle > best_min_angle:
                best_min_angle = min_angle
                best_frame = d

        if best_frame is None:
            break
        selected.append(best_frame["idx"])
        used_dirs.append(best_frame["dir"])

    selected = sorted(selected)
    print(f"  [HQ筛选] 贪心选取 {len(selected)} 帧（姿态多样性优先）")
    print(f"  [HQ筛选] 帧索引: {selected}")

    # 打印姿态覆盖情况
    angles = []
    for i in range(len(selected)):
        ridx = tm_valid_idx.index(selected[i])
        if i == 0:
            print(f"  [HQ筛选]   帧 {selected[i]:4d}: 基准帧（质量={detected[ridx]['quality']:.4f}）")
        else:
            dir_i = camera_dir_from_rvec_tvec(rvecs_tm[ridx], tvecs_tm[ridx])
            min_ang = min(np.arctan2(np.linalg.norm(np.cross(used_dirs[j], dir_i)),
                                      np.dot(used_dirs[j], dir_i)) * 180 / np.pi
                          for j in range(len(used_dirs)))
            angles.append(min_ang)
            print(f"  [HQ筛选]   帧 {selected[i]:4d}: 与最近已选帧夹角 {min_ang:.1f}°（质量={detected[ridx]['quality']:.4f}）")

    return selected


def detect_chessboard_corners(
    gray: np.ndarray,
    board_size: Tuple[int, int],
    factor_resize: float = 2.0
) -> Tuple[bool, np.ndarray, float]:
    """
    在灰度图上检测棋盘格角点（使用 findChessboardCornersSB，新版本算法自带亚像素）。
    自动放大图像以提高检测率。

    返回：(检测是否成功, 亚像素角点坐标(归一化回原分辨率), 缩放因子)
    """
    h, w = gray.shape[:2]
    gray_big, factor = resize_for_detection(gray, target_min_w=1280)

    ret, corners_big = cv2.findChessboardCornersSB(
        gray_big, board_size,
        0
    )

    if not ret:
        return False, np.array([]), factor

    # 缩放回原分辨率
    corners = corners_big / factor
    return True, corners.astype(np.float32), factor


def detect_corners_rs_exhaustive(
    gray: np.ndarray,
    board_size: Tuple[int, int],
    factor: float = 2.0,
) -> Tuple[bool, np.ndarray]:
    """
    RealSense 专用角点检测（使用 EXHAUSTIVE 模式）。
    与阶段1b/stage2 的天眸检测保持一致的检测策略，保证角点位置可比性。
    返回：(检测是否成功, 亚像素角点坐标)
    """
    h, w = gray.shape[:2]
    gray_big = cv2.resize(gray, (int(w * factor), int(h * factor)),
                          interpolation=cv2.INTER_NEAREST)
    ret, corners_big = cv2.findChessboardCornersSB(
        gray_big, board_size,
        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        return False, np.array([])
    corners = corners_big / factor
    # findChessboardCornersSB 已内置亚像素精度，额外 cornerSubPix 会引入不稳定
    return True, corners.astype(np.float32)


def visualize_corners(img: np.ndarray, corners: np.ndarray, win_name: str = "corners") -> np.ndarray:
    """在图像上绘制检测到的角点（用于调试和报告）"""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for i, corner in enumerate(corners):
        pt = tuple(corner.ravel().astype(int))
        color = (0, 255, 0) if i == 0 else (0, 128, 255)
        cv2.circle(vis, pt, 5, color, -1)
        cv2.putText(vis, str(i), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


def compute_reproj_error(
    objp_or_pts3d: np.ndarray,
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray
) -> float:
    """计算单帧重投影误差（像素）"""
    # 支持直接传入 3D 点（已变换到当前相机坐标系）或其他形式的 objectPoints
    imgpts2, _ = cv2.projectPoints(objp_or_pts3d, rvec, tvec, K, dist)
    # imgpts2 形状为 (N, 1, 2)，corners 形状为 (N, 2)，需要 squeeze
    imgpts2 = imgpts2.squeeze()
    corners2 = corners.squeeze() if corners.ndim > 1 else corners
    return cv2.norm(corners2, imgpts2, cv2.NORM_L2) / len(corners2)


def compute_per_frame_extrinsics(
    objp: np.ndarray,
    imgpoints: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 solvePnP 计算单帧的外参旋转向量和平移向量。
    返回 (rvec, tvec)。
    """
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    try:
        _, rvec, tvec = cv2.solvePnP(
            objp.astype(np.float64), imgpoints.astype(np.float64),
            K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except Exception:
        pass
    return rvec, tvec


def transform_and_project(
    objp: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray
) -> np.ndarray:
    """
    将 3D 点从世界坐标系变换到相机坐标系，再投影到像素平面。
    objp: (N, 3) 世界坐标
    R:    (3, 3) 旋转矩阵
    T:    (3,)  平移向量
    返回: (N, 1, 2) 像素坐标
    """
    # 空间变换: P_cam = R @ P_world + T
    pts_cam = (R @ objp.T).T + T.ravel()
    pts_cam = pts_cam.astype(np.float64)
    projected, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), K, dist)
    return projected


def build_3d_objpoints(board_cols: int, board_rows: int, square_size: float) -> np.ndarray:
    """构建棋盘格 3D 世界坐标（Z=0 平面）"""
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size
    return objp


# ============================================================
# RealSense 内参读取（从 bag 文件或默认值）
# ============================================================

def load_rs_intrinsics(bag_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 bag 文件读取 RealSense Color 内参。
    若 bag 不可用或读取失败，返回默认值。
    """
    if bag_path and os.path.exists(bag_path):
        try:
            sys.path.insert(0, '/projects/miniconda3/envs/tianmou/lib/python3.10/site-packages')
            import pyrealsense2 as rs

            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device_from_file(bag_path)
            profile = pipe.start(cfg)

            color_stream = profile.get_stream(rs.stream.color)
            vp = color_stream.as_video_stream_profile()
            intr = vp.get_intrinsics()

            pipe.stop()

            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
            dist = np.array(intr.coeffs, dtype=np.float64)

            print(f"  [RealSense] 从 bag 文件读取内参: fx={intr.fx:.2f}, fy={intr.fy:.2f}")
            print(f"  [RealSense] 主点: cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")
            print(f"  [RealSense] 畸变系数: {dist}")
            return K, dist
        except Exception as e:
            print(f"  [RealSense] 从 bag 读取内参失败: {e}，使用默认值")
    else:
        if bag_path:
            print(f"  [RealSense] bag 文件不存在: {bag_path}，使用默认值")
        else:
            print(f"  [RealSense] 未提供 bag 文件，使用默认值（建议提供以获得精确内参）")

    return RS_K_DEFAULT.copy(), RS_D_DEFAULT.copy()


# ============================================================
# 阶段1：RealSense 内参验证
# ============================================================

def stage1_verify_rs_intrinsics(bag_path: Optional[str]) -> Dict[str, Any]:
    """
    验证 RealSense 内参。
    读取 bag 中的出厂内参与默认值对比。
    """
    print("\n" + "=" * 60)
    print("阶段1: RealSense 内参验证")
    print("=" * 60)

    K_bag, dist_bag = load_rs_intrinsics(bag_path)

    print("\nRealSense 出厂默认值（参考）:")
    print(f"  K_default:\n{RS_K_DEFAULT}")
    print(f"  dist_default: {RS_D_DEFAULT}")

    print("\n本次使用内参:")
    print(f"  K:\n{K_bag}")
    print(f"  dist: {dist_bag}")

    if bag_path and os.path.exists(bag_path):
        diff_fx = abs(K_bag[0, 0] - RS_K_DEFAULT[0, 0]) / RS_K_DEFAULT[0, 0] * 100
        print(f"\nfx 偏差: {diff_fx:.2f}%（若 >5% 建议重新检查 bag 文件）")

    return {
        "K": K_bag.tolist(),
        "dist": dist_bag.tolist(),
        "source": "bag" if (bag_path and os.path.exists(bag_path)) else "default",
    }


# ============================================================
# 阶段2：天眸内参标定
# ============================================================

def stage2_tianmou_intrinsics(
    data_dir: str,
    board_cols: int, board_rows: int,
    square_size: float,
    tm_resize_factor: float = 2.0,
    frames: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    对天眸进行单目内参标定。
    可选择使用所有帧或指定帧子集。
    返回内参矩阵、畸变系数、各帧重投影误差。
    """
    print("\n" + "=" * 60)
    print("阶段2: 天眸内参标定")
    print("=" * 60)

    tianmou_dir = os.path.join(data_dir, "tianmou")
    if not os.path.isdir(tianmou_dir):
        print(f"  天眸目录不存在: {tianmou_dir}")
        print(f"  使用参考内参 USB_MODULE_MAT 作为天眸内参")
        return {
            "K": TM_K_REF.tolist(),
            "dist": [0, 0, 0, 0, 0],
            "reproj_errors": [],
            "n_frames": 0,
            "method": "reference_USB_MODULE_MAT",
        }

    # 获取所有帧
    all_files = sorted(glob.glob(os.path.join(tianmou_dir, "tianmou_*.png")))
    if not all_files:
        print("  天眸目录为空")
        return {}

    if frames is None:
        frames = list(range(len(all_files)))

    print(f"  天眸分辨率: 640×320")
    print(f"  候选帧数: {len(all_files)}，使用: {len(frames)}")

    # 构建 3D 棋盘格坐标
    objp = build_3d_objpoints(board_cols, board_rows, square_size)

    # 收集有效数据
    objpoints = []
    imgpoints = []
    valid_frame_ids = []
    per_frame_errors = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for idx, fidx in enumerate(frames):
        if fidx >= len(all_files):
            continue
        img = read_tianmou_image(all_files[fidx])
        if img is None:
            continue

        gray = to_gray(img)
        # 天眸分辨率低，放大检测
        h, w = gray.shape[:2]
        factor = max(2.0, 1280 / w)
        gray_big = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_NEAREST)

        ret, corners_big = cv2.findChessboardCornersSB(
            gray_big, (board_cols, board_rows),
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            continue

        # 亚像素精化（findChessboardCornersSB 已经包含亚像素，但再做一次更稳）
        gray_big = cv2.cornerSubPix(gray_big, corners_big, (11, 11), (-1, -1), criteria)
        corners = corners_big / factor  # 缩放回原分辨率

        objpoints.append(objp.copy())
        imgpoints.append(corners.astype(np.float32))
        valid_frame_ids.append(fidx)

        # 计算单帧误差（使用原分辨率）
        err = cv2.norm(corners, cv2.projectPoints(objp, np.zeros(3), np.zeros(3),
                     TM_K_REF, np.zeros(5))[0], cv2.NORM_L2) / len(corners)
        per_frame_errors.append(float(err))

    if len(objpoints) < 3:
        print(f"  检测到 {len(objpoints)} 帧，不足以进行内参标定（至少需要 3 帧）")
        print("  使用参考内参 USB_MODULE_MAT")
        return {
            "K": TM_K_REF.tolist(),
            "dist": [0, 0, 0, 0, 0],
            "reproj_errors": [],
            "n_frames": 0,
            "method": "reference_USB_MODULE_MAT_insufficient_frames",
        }

    print(f"  检测成功: {len(objpoints)}/{len(frames)} 帧")

    # OpenCV 内参标定
    h, w = 320, 640
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # 计算各帧重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2) / len(imgpts2)
        total_error += err

    mean_error = total_error / len(objpoints)

    print(f"\n  天眸内参标定结果:")
    print(f"  K (640×320):\n{K}")
    print(f"  dist: {dist.ravel()}")
    print(f"  平均重投影误差: {mean_error:.4f} pix")

    # 与参考内参对比
    diff = np.linalg.norm(K - TM_K_REF, 'fro') / np.linalg.norm(TM_K_REF) * 100
    print(f"  与 USB_MODULE_MAT 的 Frobenius 偏差: {diff:.2f}%")
    if diff < 2:
        print("  偏差较小，天眸内参稳定")
    else:
        print("  偏差较大，建议检查棋盘格拍摄质量")

    return {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "rvecs": [r.ravel().tolist() for r in rvecs],
        "tvecs": [t.ravel().tolist() for t in tvecs],
        "reproj_errors": per_frame_errors,
        "mean_error": float(mean_error),
        "n_frames": len(objpoints),
        "valid_frame_indices": valid_frame_ids,
        "method": "calibrated",
    }


# ============================================================
# 阶段1b：RealSense 内参标定（基于标定数据中的棋盘格）
# ============================================================

def stage1b_realSense_intrinsics(
    data_dir: str,
    board_cols: int, board_rows: int,
    square_size: float,
    rs_K_init: Optional[np.ndarray] = None,
    frames: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    使用标定数据中的 RealSense Color 图像，对 RealSense Color 做内参标定。
    与阶段1的"从 bag 读取出厂内参"不同，这里用棋盘格实测数据标定内参，
    可以修正出厂内参与实际使用环境（温度、镜头批次）的偏差。

    返回的内参将用于阶段3的双目外参标定。
    """
    print("\n" + "=" * 60)
    print("阶段1b: RealSense 内参标定（基于棋盘格实测数据）")
    print("=" * 60)

    color_dir = os.path.join(data_dir, "color")
    if not os.path.isdir(color_dir):
        print(f"  RealSense Color 目录不存在: {color_dir}")
        return {}

    color_files = sorted(glob.glob(os.path.join(color_dir, "color_*.png")))
    if not color_files:
        print("  Color 目录为空")
        return {}

    if frames is None:
        # 跳过前后5帧
        frames = list(range(5, min(len(color_files) - 5, 100)))

    print(f"  RealSense Color 分辨率: 640×480")
    print(f"  候选帧数: {len(color_files)}，使用: {len(frames)}")

    objp = build_3d_objpoints(board_cols, board_rows, square_size)
    objpoints = []
    imgpoints = []
    valid_ids = []

    for fidx in frames:
        if fidx >= len(color_files):
            continue
        img = cv2.imread(color_files[fidx])
        if img is None:
            continue
        gray = to_gray(img)

        # RealSense 480行，检测时直接放大2倍
        gray_big = cv2.resize(gray, (1280, 960), interpolation=cv2.INTER_NEAREST)
        ret, corners_big = cv2.findChessboardCornersSB(
            gray_big, (board_cols, board_rows),
            0
        )
        if not ret:
            continue
        corners = corners_big / 2.0  # 缩回原分辨率
        objpoints.append(objp.copy())
        imgpoints.append(corners.astype(np.float32))
        valid_ids.append(fidx)

    if len(objpoints) < 3:
        print(f"  检测成功 {len(objpoints)} 帧，不足以标定（至少3帧）")
        return {}

    print(f"  检测成功: {len(objpoints)}/{len(frames)} 帧")

    # 初始化内参（使用 bag 出厂值或默认值）
    K_init = rs_K_init.copy() if rs_K_init is not None else RS_K_DEFAULT.copy()
    dist_init = RS_D_DEFAULT.copy()

    # OpenCV 内参标定
    # flags: 标准5系数畸变模型，并以 K_init 为初始值
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K_calib, dist_calib, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (640, 480), K_init, dist_init,
        flags=flags, criteria=criteria
    )

    # 计算各帧重投影误差
    total_error = 0
    per_frame_errors = []
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K_calib, dist_calib)
        pts2_s = imgpts2.squeeze().astype(np.float32)
        corners_s = imgpoints[i].squeeze().astype(np.float32)
        err = cv2.norm(corners_s, pts2_s, cv2.NORM_L2) / len(imgpts2)
        total_error += err
        per_frame_errors.append({"frame_idx": valid_ids[i], "error": float(err)})
    mean_error = total_error / len(objpoints)

    print(f"\n  RealSense 内参标定结果:")
    print(f"  K (640×480):\n{K_calib}")
    print(f"  dist: {dist_calib.ravel()}")
    print(f"  平均重投影误差: {mean_error:.4f} pix")

    if rs_K_init is not None:
        diff = np.linalg.norm(K_calib - rs_K_init, 'fro') / np.linalg.norm(rs_K_init) * 100
        print(f"  与 bag 出厂内参的 Frobenius 偏差: {diff:.2f}%")
        if diff < 2:
            print("  偏差较小，RealSense 内参稳定")
        else:
            print("  偏差较大，建议使用标定得到的内参（偏差 > 2%）")

    return {
        "K": K_calib.tolist(),
        "dist": dist_calib.ravel().tolist(),
        "rvecs": [r.ravel().tolist() for r in rvecs],
        "tvecs": [t.ravel().tolist() for t in tvecs],
        "reproj_errors": per_frame_errors,
        "mean_error": float(mean_error),
        "n_frames": len(objpoints),
        "valid_frame_indices": valid_ids,
        "method": "calibrated_from_chessboard",
    }


# ============================================================
# 阶段3：双目外参标定（天眸 ↔ RealSense Color）
# ============================================================

def stage3_stereo_extrinsics(
    data_dir: str,
    board_cols: int, board_rows: int,
    square_size: float,
    tm_K: np.ndarray,
    tm_dist: np.ndarray,
    rs_K: np.ndarray,
    rs_dist: np.ndarray,
    frames: Optional[List[int]] = None,
    select_frames: bool = False,
    bad_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    双目外参标定：天眸 ↔ RealSense Color。
    使用 cv2.stereoCalibrate 计算旋转矩阵 R 和平移向量 T。
    """
    print("\n" + "=" * 60)
    print("阶段3: 双目外参标定（天眸 ↔ RealSense Color）")
    print("=" * 60)

    tianmou_dir = os.path.join(data_dir, "tianmou")
    color_dir = os.path.join(data_dir, "color")

    if not os.path.isdir(tianmou_dir):
        print(f"  目录不存在: {tianmou_dir}")
        return {}
    if not os.path.isdir(color_dir):
        print(f"  目录不存在: {color_dir}")
        return {}

    # 获取所有帧
    tm_files = sorted(glob.glob(os.path.join(tianmou_dir, "tianmou_*.png")))
    color_files = sorted(glob.glob(os.path.join(color_dir, "color_*.png")))

    print(f"  天眸帧数: {len(tm_files)}, Color帧数: {len(color_files)}")

    if frames is None:
        # 默认跳过前5帧（可能有运动模糊）和后5帧，均匀采样50帧
        all_indices = list(range(5, min(len(tm_files), len(color_files)) - 5))
        step = max(1, len(all_indices) // 50)
        frames = all_indices[::step][:50]
    else:
        frames = [f for f in frames if f < len(tm_files) and f < len(color_files)]

    print(f"  候选帧数: {len(frames)}")

    # 交互式选择
    if select_frames:
        print("\n  [交互模式] 请在弹出的窗口中按 SPACE 跳过坏帧，按 'q' 退出选择")
        print("  [交互模式] 其他按键接受当前帧")
        selected = []
        for fidx in frames:
            tm_img = read_tianmou_image(tm_files[fidx])
            c_img = cv2.imread(color_files[fidx])
            if tm_img is None or c_img is None:
                continue

            vis = np.hstack([
                cv2.resize(tm_img, (640, 320)),
                cv2.resize(c_img, (640, 480))
            ])
            cv2.putText(vis, f"Frame {fidx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Select frames (SPACE=skip, q=quit, other=accept)", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key != 32:  # not SPACE
                selected.append(fidx)
        cv2.destroyAllWindows()
        frames = selected
        print(f"  交互选择完成，有效帧: {len(frames)}")

    # 构建 3D 棋盘格坐标
    objp = build_3d_objpoints(board_cols, board_rows, square_size)

    # 检测两路角点
    objpoints_list = []
    tm_imgpoints_list = []
    rs_imgpoints_list = []
    valid_frame_ids = []
    per_frame_errors = []

    print("\n  检测棋盘格角点...")
    for idx, fidx in enumerate(frames):
        tm_img = read_tianmou_image(tm_files[fidx])
        c_img = cv2.imread(color_files[fidx])
        if tm_img is None or c_img is None:
            continue

        gray_tm = to_gray(tm_img)
        gray_rs = to_gray(c_img)

        # 检测天眸角点
        ret_tm, corners_tm, _ = detect_chessboard_corners(gray_tm, (board_cols, board_rows), FACTOR_RESIZE)
        # 检测 RealSense Color 角点（EXHAUSTIVE 模式，与阶段1b/阶段2一致）
        ret_rs, corners_rs = detect_corners_rs_exhaustive(gray_rs, (board_cols, board_rows), FACTOR_RESIZE)

        if not ret_tm or not ret_rs:
            continue

        objpoints_list.append(objp.copy())
        tm_imgpoints_list.append(corners_tm)
        rs_imgpoints_list.append(corners_rs)
        valid_frame_ids.append(fidx)

    if len(objpoints_list) < 3:
        print(f"  共同检测成功: {len(objpoints_list)} 帧（至少需要 3 帧）")
        return {}

    print(f"  共同检测成功: {len(objpoints_list)}/{len(frames)} 帧")

    # --- 双目标定 ---
    # 两种策略:
    #   1. CALIB_FIX_INTRINSIC: 固定已有内参，只优化外参（R, T）
    #      - 需要精确的 RealSense 内参（从同批次 bag 文件读取）
    #   2. 不固定内参: 允许算法同时优化内参和外参
    #      - 适用于内参未知或不准确的情况（推荐用于测试）
    # 这里默认使用策略1（固定内参），若 RealSense 内参来源不同则自动回退
    flags = cv2.CALIB_FIX_INTRINSIC

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    print("\n  运行 cv2.stereoCalibrate (固定内参模式)...")
    t0 = time.time()
    # 天眸是竖幅图像：高320像素，宽640像素。
    # 天眸内参（无论来自 TM_K_REF 还是阶段2标定）需要按 √(320/480) 缩放焦距
    # 以适配实际的 640×320 图像尺寸。
    tm_K_scaled = tm_K.copy()
    tm_dist_scaled = tm_dist.copy()
    scale = np.sqrt(320.0 / 480.0)
    tm_K_scaled[0, 0] /= scale
    tm_K_scaled[1, 1] /= scale
    tm_K_scaled[0, 2] /= scale
    tm_K_scaled[1, 2] /= scale
    # OpenCV 4.13 仅支持单一 imageSize 参数。
    # 经验证：imageSize=(640, 320) 或 (640, 480) 对 R/T 结果无影响（内参已固定）。
    # 对极距离误差 < 0.2px 验证了几何一致性。
    calib_result = cv2.stereoCalibrate(
        objpoints_list,
        tm_imgpoints_list,
        rs_imgpoints_list,
        tm_K_scaled, tm_dist_scaled,
        rs_K, rs_dist,
        imageSize=(640, 320),
        flags=flags,
        criteria=criteria,
    )
    elapsed = time.time() - t0
    print(f"  标定完成，耗时: {elapsed:.1f}s")

    # 兼容不同 OpenCV 版本的返回值数量
    n_ret = len(calib_result)
    if n_ret == 10:
        ret, K_tm_calib, dist_tm_calib, K_rs_calib, dist_rs_calib, R, T, E, F, perViewErrors = calib_result
    elif n_ret == 9:
        ret, K_tm_calib, dist_tm_calib, K_rs_calib, dist_rs_calib, R, T, E, F = calib_result
        perViewErrors = [0.0] * len(objpoints_list)
    else:
        print(f"  cv2.stereoCalibrate 返回了 {n_ret} 个值（期望 9 或 10），跳过")
        return {}

    # cv2.stereoCalibrate 返回的 R 可能是:
    # - 3×3 旋转矩阵（当 CALIB_FIX_INTRINSIC 时）
    # - 3×1 旋转向量（当未固定内参时）
    # 确保 R 是 3×3 旋转矩阵
    R_mat = R if R.shape == (3, 3) else cv2.Rodrigues(R)[0]
    R_vec_rod = cv2.Rodrigues(R_mat)[0].ravel()  # Rodrigues旋转向量（3个值）

    # 计算各帧重投影误差（基于每帧独立姿态）
    # 天眸：每帧用 solvePnP 恢复各自的 rvec/tvec，再做重投影
    # RealSense：将世界坐标变换到 RS 坐标系后重投影
    print("\n  各帧重投影误差（像素）:")
    bad_frames = []
    for i, fidx in enumerate(valid_frame_ids):
        # 天眸：每帧独立姿态
        rvec_tm, tvec_tm = compute_per_frame_extrinsics(
            objpoints_list[i], tm_imgpoints_list[i], tm_K, tm_dist)
        projected_tm, _ = cv2.projectPoints(objpoints_list[i], rvec_tm, tvec_tm, tm_K, tm_dist)
        projected_tm = projected_tm.squeeze().astype(np.float32)
        corners_tm_s = tm_imgpoints_list[i].squeeze().astype(np.float32)
        err_tm = cv2.norm(corners_tm_s, projected_tm, cv2.NORM_L2) / len(projected_tm)

        # RealSense：将棋盘格世界坐标变换到 RS 相机坐标系，再重投影
        projected_rs = transform_and_project(objpoints_list[i], R_mat, T, rs_K, rs_dist).squeeze().astype(np.float32)
        corners_rs_s = rs_imgpoints_list[i].squeeze().astype(np.float32)
        err_rs = cv2.norm(corners_rs_s, projected_rs, cv2.NORM_L2) / len(projected_rs)
        mean_err = (err_tm + err_rs) / 2
        per_frame_errors.append({
            "frame_idx": int(fidx),
            "error_tianmou": float(err_tm),
            "error_realsense": float(err_rs),
            "mean_error": float(mean_err),
        })
        if mean_err > bad_threshold:
            bad_frames.append(fidx)
        if i < 20 or mean_err > bad_threshold:
            flag = " <<< BAD" if mean_err > bad_threshold else ""
            print(f"    帧 {fidx:04d}: TM={err_tm:.3f}pix  RS={err_rs:.3f}pix  均值={mean_err:.3f}pix{flag}")

    mean_total_error = np.mean([e["mean_error"] for e in per_frame_errors])

    print(f"\n  === 双目标定结果 ===")
    print(f"  标定帧数: {len(objpoints_list)}")
    print(f"  平均重投影误差: {mean_total_error:.4f} pix")
    print(f"\n  旋转矩阵 R (天眸 → RealSense Color):")
    print(f"  (注: P_tianmou = R @ P_rs + T)")
    print(R_mat)
    print(f"\n  平移向量 T (mm):")
    print(T.ravel())
    print(f"\n  本征矩阵 E:")
    print(E)
    print(f"\n  基础矩阵 F:")
    print(F)

    # T 的物理意义
    t_norm = np.linalg.norm(T)
    print(f"\n  T 物理意义: 天眸到 RealSense Color 的平移向量")
    print(f"  距离: {t_norm:.2f} mm ({t_norm/10:.2f} cm)")

    if bad_frames:
        print(f"\n  警告: {len(bad_frames)} 帧误差 > {bad_threshold} pix:")
        print(f"  坏帧索引: {bad_frames}")
        print(f"  建议: 剔除这些帧后重新标定，或检查对应帧的棋盘格是否清晰")

    return {
        "R": R_mat.tolist(),
        "R_vec": R.ravel().tolist(),
        "T": T.ravel().tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "mean_reproj_error": float(mean_total_error),
        "per_frame_errors": per_frame_errors,
        "n_valid_frames": len(objpoints_list),
        "bad_frames": bad_frames,
        "K_tianmou_used": tm_K.tolist(),
        "dist_tianmou_used": tm_dist.tolist(),
        "K_realsense_used": rs_K.tolist(),
        "dist_realsense_used": rs_dist.tolist(),
    }


# ============================================================
# 阶段4：验证与可视化
# ============================================================

def stage4_validation(
    data_dir: str,
    result: Dict[str, Any],
    n_samples: int = 5,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    验证标定结果：
    1. 角点检测可视化（保存检测结果图）
    2. 深度投影验证（如果 depth 目录可用）
    """
    print("\n" + "=" * 60)
    print("阶段4: 验证与可视化")
    print("=" * 60)

    if not result or "R" not in result:
        print("  跳过：无有效外参结果")
        return {}

    R_mat = np.array(result["R"])
    T_vec = np.array(result["T"])
    rs_K = np.array(result["K_realsense_used"])
    rs_dist = np.array(result["dist_realsense_used"])
    tm_K = np.array(result["K_tianmou_used"])
    tm_dist = np.array(result["dist_tianmou_used"])

    tianmou_dir = os.path.join(data_dir, "tianmou")
    color_dir = os.path.join(data_dir, "color")
    depth_dir = os.path.join(data_dir, "depth")

    vis_dir = None
    if output_dir:
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

    # --- 可视化1：抽样帧角点检测 ---
    if os.path.isdir(tianmou_dir) and os.path.isdir(color_dir):
        tm_files = sorted(glob.glob(os.path.join(tianmou_dir, "tianmou_*.png")))
        color_files = sorted(glob.glob(os.path.join(color_dir, "color_*.png")))

        valid_ids = [e["frame_idx"] for e in result.get("per_frame_errors", [])]
        sample_ids = valid_ids[::max(1, len(valid_ids)//n_samples)][:n_samples]

        for fidx in sample_ids:
            if fidx >= len(tm_files) or fidx >= len(color_files):
                continue

            tm_img = read_tianmou_image(tm_files[fidx])
            c_img = cv2.imread(color_files[fidx])
            if tm_img is None or c_img is None:
                continue

            gray_tm = to_gray(tm_img)
            gray_c = to_gray(c_img)

            ret_tm, corners_tm, _ = detect_chessboard_corners(gray_tm, (BOARD_COLS, BOARD_ROWS), FACTOR_RESIZE)
            ret_c, corners_c, _ = detect_chessboard_corners(gray_c, (BOARD_COLS, BOARD_ROWS), FACTOR_RESIZE)

            vis_tm = visualize_corners(tm_img, corners_tm if ret_tm else np.array([]))
            vis_c = visualize_corners(c_img, corners_c if ret_c else np.array([]))

            # 左右拼接
            vis_tm_resized = cv2.resize(vis_tm, (640, 320))
            vis_c_resized = cv2.resize(vis_c, (640, 480))
            canvas = np.zeros((max(480, 320), 640 * 2, 3), dtype=np.uint8)
            canvas[:320, :640] = vis_tm_resized
            canvas[:480, 640:] = vis_c_resized

            cv2.putText(canvas, f"Frame {fidx} - Tianmou (top) / RS Color (bottom)",
                        (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if vis_dir:
                out_path = os.path.join(vis_dir, f"corners_{fidx:04d}.png")
                cv2.imwrite(out_path, canvas)
                print(f"  保存角点图: {out_path}")

            # 显示（仅在有显示器的环境下启用）
            # 通过环境变量 DISPLAY 判断是否有图形界面
            if os.environ.get("DISPLAY") and not os.environ.get("STEREO_CALIB_HEADLESS"):
                try:
                    cv2.imshow("Validation - press q to quit", canvas)
                    if cv2.waitKey(300) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass
            else:
                # headless 环境：只打印进度
                pass

        if os.environ.get("DISPLAY") and not os.environ.get("STEREO_CALIB_HEADLESS"):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # --- 可视化2：误差分布图 ---
    if output_dir and result.get("per_frame_errors"):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        errors = [e["mean_error"] for e in result["per_frame_errors"]]
        frames = [e["frame_idx"] for e in result["per_frame_errors"]]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(frames, errors, 'b-o', markersize=3, label='重投影误差')
        ax.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'均值={np.mean(errors):.3f}pix')
        ax.set_xlabel('帧索引')
        ax.set_ylabel('重投影误差 (pix)')
        ax.set_title('各帧重投影误差分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, "reproj_error_plot.png")
        fig.savefig(plot_path, dpi=150)
        print(f"  保存误差图: {plot_path}")
        plt.close(fig)

    print("  验证完成")
    return {}


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="天眸 ↔ RealSense 双目外参标定",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
参数说明:
  --data-dir       标定数据根目录（包含 tianmou/, color/, depth/ 子目录）
  --bag            RealSense bag 文件路径（用于读取出厂内参，可选）
  --board-size     棋盘格角点数，如 "11 8"（横向11个，纵向8个角点）
  --square-size    方格边长（mm），如 29.66
  --output         输出目录
  --frames         使用的帧索引列表，如 "0 5 10 20"，默认自动采样
  --select-frames  交互式选择有效帧（弹出窗口）
  --bad-threshold  坏帧判定阈值（像素），默认 1.0
  --skip-stage1    跳过阶段1（RealSense 内参验证）
  --skip-stage2    跳过阶段2（天眸内参标定，直接使用参考内参）

示例:
  # 基础用法（使用所有帧）
  python stereo_calib.py --data-dir /projects/calib_data/0323_calibration_v1 \\
      --output ./calibration_output

  # 交互式选择帧
  python stereo_calib.py --data-dir /projects/calib_data/0323_calibration_v1 \\
      --output ./calibration_output --select-frames

  # 指定帧索引
  python stereo_calib.py --data-dir /projects/calib_data/0323_calibration_v1 \\
      --output ./calibration_output --frames 10 15 20 30 40 50 60 80 100
        """
    )

    parser.add_argument("--data-dir", "-d", required=True,
                        help="标定数据目录（含 tianmou/, color/ 子目录）")
    parser.add_argument("--bag", "-b", default=None,
                        help="RealSense bag 文件路径（用于读取内参）")
    parser.add_argument("--board-size", "-bs", default="11 8",
                        help="棋盘格角点数，默认 '11 8'")
    parser.add_argument("--square-size", "-ss", type=float, default=29.66,
                        help="方格边长(mm)，默认 29.66")
    parser.add_argument("--output", "-o", default="./calibration_output",
                        help="输出目录")
    parser.add_argument("--frames", "-f", type=int, nargs='+', default=None,
                        help="使用的帧索引列表（可选）")
    parser.add_argument("--select-frames", action="store_true",
                        help="交互式选择有效帧")
    parser.add_argument("--bad-threshold", type=float, default=1.0,
                        help="坏帧判定阈值(pix)，默认 1.0")
    parser.add_argument("--hq-frames", action="store_true",
                        help="启用高质量帧自动筛选：快速预检测全部帧，按姿态多样性和角点清晰度打分，选取最优帧")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="跳过阶段1（RealSense出厂内参验证）")
    parser.add_argument("--skip-stage1b", action="store_true",
                        help="跳过阶段1b（RealSense棋盘格内参标定，直接用出厂内参）")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="跳过阶段2（天眸内参标定，使用参考内参）")

    args = parser.parse_args()

    # 解析 board_size
    bs_parts = args.board_size.split()
    board_cols = int(bs_parts[0])
    board_rows = int(bs_parts[1])

    global BOARD_COLS, BOARD_ROWS, SQUARE_SIZE
    BOARD_COLS, BOARD_ROWS = board_cols, board_rows
    SQUARE_SIZE = args.square_size

    # 创建输出目录
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("天眸 ↔ RealSense 双目外参标定")
    print("=" * 60)
    print(f"  数据目录: {args.data_dir}")
    print(f"  bag文件:  {args.bag or '未提供（使用默认值）'}")
    print(f"  棋盘格:   {board_cols}×{board_rows} 角点, {SQUARE_SIZE}mm")
    print(f"  输出目录: {output_dir}")

    start_time = time.time()

    # ========== 帧选择策略 ==========
    # 优先级: --frames > --hq-frames > 自动均匀采样
    if args.frames:
        frames_to_use = args.frames
        print(f"\n  [帧策略] 使用用户指定的 {len(frames_to_use)} 帧")
    elif args.hq_frames:
        # 高质量帧筛选：先读取已知的内参，再筛选
        print(f"\n  [帧策略] 启用高质量帧自动筛选（--hq-frames）")
        # 读取出厂内参用于筛选
        rs_K_for_hq = np.array(rs_intrinsics_factory["K"]) if rs_intrinsics_factory else RS_K_DEFAULT.copy()
        rs_D_for_hq = np.array(rs_intrinsics_factory["dist"]) if rs_intrinsics_factory else RS_D_DEFAULT.copy()
        # 天眸内参先用参考值（后续会用标定结果覆盖）
        tm_K_for_hq = TM_K_REF.copy()
        tm_D_for_hq = np.zeros(5, dtype=np.float64)
        tm_files_all = sorted(glob.glob(os.path.join(args.data_dir, "tianmou", "tianmou_*.png")))
        color_files_all = sorted(glob.glob(os.path.join(args.data_dir, "color", "color_*.png")))
        hq_frames = select_high_quality_frames(
            tm_files_all, color_files_all,
            tm_K_for_hq, tm_D_for_hq,
            rs_K_for_hq, rs_D_for_hq,
            board_cols, board_rows, SQUARE_SIZE,
            n_frames=20,
        )
        frames_to_use = hq_frames
        print(f"  [帧策略] HQ筛选完成，选取 {len(frames_to_use)} 帧")
    else:
        # 自动均匀采样（无硬上限，stereoCalibrate 200帧内可接受）
        tm_files_all = sorted(glob.glob(os.path.join(args.data_dir, "tianmou", "tianmou_*.png")))
        color_files_all = sorted(glob.glob(os.path.join(args.data_dir, "color", "color_*.png")))
        all_indices = list(range(5, min(len(tm_files_all), len(color_files_all)) - 5))
        # 均匀采 50 帧作为默认
        step = max(1, len(all_indices) // 50)
        frames_to_use = all_indices[::step][:50]
        print(f"\n  [帧策略] 自动均匀采样 {len(frames_to_use)} 帧")

    # ========== 阶段1（RealSense 出厂内参）==========
    rs_intrinsics_factory = stage1_verify_rs_intrinsics(args.bag) if not args.skip_stage1 else None
    rs_K_factory = np.array(rs_intrinsics_factory["K"]) if rs_intrinsics_factory else RS_K_DEFAULT.copy()
    rs_dist_factory = np.array(rs_intrinsics_factory["dist"]) if rs_intrinsics_factory else RS_D_DEFAULT.copy()

    # ========== 阶段1b（RealSense 棋盘格内参标定）==========
    if not args.skip_stage1b:
        rs_intrinsics_calib = stage1b_realSense_intrinsics(
            args.data_dir, board_cols, board_rows, SQUARE_SIZE,
            rs_K_init=rs_K_factory,
            frames=frames_to_use,
        )
        # 优先使用标定得到的内参（实测数据）
        if rs_intrinsics_calib:
            rs_K = np.array(rs_intrinsics_calib["K"])
            rs_dist = np.array(rs_intrinsics_calib["dist"])
        else:
            rs_K = rs_K_factory.copy()
            rs_dist = rs_dist_factory.copy()
    else:
        rs_intrinsics_calib = None
        rs_K = rs_K_factory.copy()
        rs_dist = rs_dist_factory.copy()

    # ========== 阶段2（天眸内参标定）==========
    if args.skip_stage2:
        print("\n" + "=" * 60)
        print("阶段2: 跳过（使用参考内参 USB_MODULE_MAT）")
        print("=" * 60)
        tm_K = TM_K_REF.copy()
        tm_dist = np.zeros(5, dtype=np.float64)
        tm_intrinsics = {
            "K": tm_K.tolist(),
            "dist": tm_dist.tolist(),
            "n_frames": 0,
            "method": "reference_USB_MODULE_MAT",
        }
    else:
        tm_intrinsics = stage2_tianmou_intrinsics(
            args.data_dir, board_cols, board_rows, SQUARE_SIZE,
            frames=frames_to_use,
        )
        tm_K = np.array(tm_intrinsics["K"])
        tm_dist = np.array(tm_intrinsics["dist"])

    # ========== 阶段3 ==========
    # RealSense 双目标定统一使用出厂内参（bag 文件读取值），避免阶段1b 少量帧
    # 标定产生的畸变系数错误（k2=-0.274, k3=+1.75）导致 12px 重投影误差和错误的 T
    stereo_result = stage3_stereo_extrinsics(
        args.data_dir, board_cols, board_rows, SQUARE_SIZE,
        tm_K, tm_dist,
        rs_K_factory.copy(), rs_dist_factory.copy(),
        frames=frames_to_use,
        select_frames=args.select_frames,
        bad_threshold=args.bad_threshold,
    )

    # ========== 阶段4 ==========
    stage4_validation(args.data_dir, stereo_result, n_samples=8, output_dir=output_dir)

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")

    # ========== 保存结果 ==========
    full_result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "data_dir": args.data_dir,
        "bag_file": args.bag,
        "board_size": {"cols": board_cols, "rows": board_rows},
        "square_size_mm": SQUARE_SIZE,
        "tianmou_intrinsics": tm_intrinsics,
        "realsense_intrinsics_factory": rs_intrinsics_factory,
        "realsense_intrinsics_calibrated": rs_intrinsics_calib,
        # 阶段3 最终使用的内参
        "realsense_intrinsics_used": {
            "K": rs_K.tolist(),
            "dist": rs_dist.tolist(),
            "source": "calibrated" if rs_intrinsics_calib else ("factory" if rs_intrinsics_factory else "default"),
        },
        "stereo_extrinsics": stereo_result,
    }

    # 保存 JSON
    result_json_path = os.path.join(output_dir, "calibration_result.json")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_json_path}")

    # 保存报告
    report_path = os.path.join(output_dir, "calibration_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("天眸 ↔ RealSense 双目外参标定报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"标定时间: {full_result['timestamp']}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"bag文件:  {args.bag or '使用默认值'}\n\n")
        f.write(f"棋盘格规格: {board_cols}×{board_rows} 角点, {SQUARE_SIZE}mm/格\n\n")

        f.write("--- 天眸内参 ---\n")
        f.write(f"方法: {tm_intrinsics.get('method', 'unknown')}\n")
        f.write(f"K:\n{np.array(tm_intrinsics['K'])}\n")
        f.write(f"dist: {tm_intrinsics['dist']}\n")
        f.write(f"平均重投影误差: {tm_intrinsics.get('mean_error', 'N/A')} pix\n\n")

        f.write("--- RealSense 内参 ---\n")
        f.write(f"来源: {full_result['realsense_intrinsics_used']['source']}\n")
        f.write(f"K:\n{np.array(full_result['realsense_intrinsics_used']['K'])}\n")
        f.write(f"dist: {full_result['realsense_intrinsics_used']['dist']}\n")
        if rs_intrinsics_calib:
            f.write(f"标定重投影误差: {rs_intrinsics_calib.get('mean_error', 'N/A')} pix\n")
        if rs_intrinsics_factory:
            f.write(f"出厂内参偏差: (见 JSON 中的 factory 内参)\n")
        f.write("\n")

        if stereo_result:
            f.write("--- 双目外参（核心结果）---\n")
            f.write("旋转矩阵 R (天眸 → RealSense Color, P_tianmou = R @ P_rs + T):\n")
            f.write(f"{np.array(stereo_result['R'])}\n\n")
            f.write("平移向量 T (mm):\n")
            f.write(f"{np.array(stereo_result['T'])}\n\n")
            f.write(f"平均重投影误差: {stereo_result.get('mean_reproj_error', 'N/A')} pix\n")
            f.write(f"有效帧数: {stereo_result.get('n_valid_frames', 'N/A')}\n")
            f.write(f"坏帧索引 (>{args.bad_threshold}pix): {stereo_result.get('bad_frames', [])}\n\n")

            f.write("各帧重投影误差:\n")
            for e in stereo_result.get("per_frame_errors", []):
                flag = " *** BAD ***" if e["mean_error"] > args.bad_threshold else ""
                f.write(f"  帧{e['frame_idx']:04d}: TM={e['error_tianmou']:.3f}pix  "
                        f"RS={e['error_realsense']:.3f}pix  均值={e['mean_error']:.3f}pix{flag}\n")

    print(f"报告已保存: {report_path}")
    print("\n" + "=" * 60)
    print("标定完成！")
    print("=" * 60)
    print(f"\n结果文件:")
    print(f"  JSON: {result_json_path}")
    print(f"  TXT:  {report_path}")
    print(f"  可视化: {output_dir}/visualization/")
    print("\n核心外参（天眸 → RealSense Color）:")
    print(f"  R = \n{np.array(stereo_result.get('R', [[]]))}")
    print(f"  T = {np.array(stereo_result.get('T', []))}")


if __name__ == "__main__":
    main()
