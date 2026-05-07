"""
calib/utils.py — 标定工具集（所有标定脚本共享）
"""

import os
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from glob import glob
from typing import Tuple, List, Dict, Any, Optional


# ============================================================
# 棋盘格参数
# ============================================================
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE = 29.66          # mm（须与采集时一致）
FACTOR_RESIZE = 2.0         # 角点检测放大倍率

# 天眸参考内参（640×320, v2镜头 2026-03）
TM_K_REF = np.array([
    [713.0134,   0.      , 224.1754],
    [  0.    , 712.7342 , 269.0033],
    [  0.    ,   0.     ,   1.     ]], dtype=np.float64)

# RealSense D455 出厂默认（640×480 Color）
RS_K_DEFAULT = np.array([
    [385.019,   0.     , 329.543],
    [  0.    , 384.118 , 241.941],
    [  0.    ,   0.    ,   1.    ]], dtype=np.float64)

RS_D_DEFAULT = np.array([-0.0542, 0.0636, -0.0005, -0.0003, -0.0202], dtype=np.float64)


# ============================================================
# 图像读取
# ============================================================

TM_FLIP_HORIZONTAL = True  # 天眸镜头硬件安装方向导致图像左右镜像，标定和后续使用均须先翻转


def read_tianmou_image(path: str) -> Optional[np.ndarray]:
    """读取天眸图像，自动去黑边（如果是 TIFF 16bit），并水平镜像纠正硬件安装导致的左右翻转。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if TM_FLIP_HORIZONTAL:
        img = cv2.flip(img, 1)   # flipCode=1: 水平翻转
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ============================================================
# 角点检测
# ============================================================

def build_3d_objpoints(
    board_cols: int = BOARD_COLS,
    board_rows: int = BOARD_ROWS,
    square_size: float = SQUARE_SIZE,
) -> np.ndarray:
    """构建 3D 世界坐标（Z=0 平面）。"""
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size
    return objp


def detect_corners(
    gray: np.ndarray,
    board_cols: int = BOARD_COLS,
    board_rows: int = BOARD_ROWS,
) -> Tuple[bool, np.ndarray]:
    """
    检测棋盘格角点：findChessboardCorners + cornerSubPix。
    先将图像放大 FACTOR_RESIZE 倍检测，再缩回原分辨率。
    返回：(检测是否成功, 角点坐标 (N,1,2) float32)
    """
    h, w = gray.shape[:2]
    factor = max(FACTOR_RESIZE, 1280 / w)
    gray_big = cv2.resize(gray, (int(w * factor), int(h * factor)),
                          interpolation=cv2.INTER_NEAREST)

    ret, corners_big = cv2.findChessboardCorners(
        gray_big, (board_cols, board_rows), None)
    if not ret:
        return False, np.array([])

    # 亚像素精细化，检测时放大 factor 倍，坐标需除回原分辨率
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_big = cv2.cornerSubPix(gray_big, corners_big, (11, 11), (-1, -1), criteria)
    corners = corners_big / factor
    return True, corners.astype(np.float32)


def detect_corners_pair(
    tm_gray: np.ndarray,
    rs_gray: np.ndarray,
    board_cols: int = BOARD_COLS,
    board_rows: int = BOARD_ROWS,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """同时检测天眸和 RealSense 角点。"""
    ret_tm, c_tm = detect_corners(tm_gray, board_cols, board_rows)
    ret_rs, c_rs = detect_corners(rs_gray, board_cols, board_rows)
    if ret_tm and ret_rs:
        return True, c_tm, c_rs
    return False, np.array([]), np.array([])


# ============================================================
# 可视化
# ============================================================

def draw_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """在图像上绘制检测到的角点。"""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, (BOARD_COLS, BOARD_ROWS), corners, True)
    return vis


def visualize_stereo_pair(
    tm_gray: np.ndarray,
    rs_color: np.ndarray,
    corners_tm: np.ndarray,
    corners_rs: np.ndarray,
    frame_id: str,
    output_dir: str,
) -> str:
    """水平拼接天眸和 RealSense 图像并标注角点，保存为 PNG。"""
    tm_vis = draw_corners(tm_gray, corners_tm)
    rs_vis = draw_corners(to_gray(rs_color) if rs_color.ndim == 3 else rs_color, corners_rs)
    rs_vis = cv2.cvtColor(rs_vis, cv2.COLOR_GRAY2BGR) if rs_vis.ndim == 2 else rs_vis

    # 自适应高度：取两者最大高度，统一放大到480宽对应的高度
    vis_h = max(tm_vis.shape[0], rs_vis.shape[0])
    tm_vis = cv2.resize(tm_vis, (960, vis_h))
    rs_vis = cv2.resize(rs_vis, (960, vis_h))
    vis = np.hstack([tm_vis, rs_vis])
    path = os.path.join(output_dir, f"corners_{frame_id}.png")
    cv2.imwrite(path, vis)
    return path


def plot_reprojection_errors(
    per_frame_errors: List[Dict],
    title: str,
    output_path: str,
    bad_threshold: float = 1.0,
) -> str:
    """绘制每帧重投影误差柱状图。"""
    frame_ids = [e["frame_idx"] for e in per_frame_errors]
    errors = [e["mean_error"] for e in per_frame_errors]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(frame_ids, errors, color="steelblue", alpha=0.8)
    for bar, err in zip(bars, errors):
        if err > bad_threshold:
            bar.set_color("coral")
    ax.axhline(y=bad_threshold, color="red", linestyle="--", label=f"阈值 {bad_threshold}px")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Reprojection Error (px)")
    ax.set_title(f"{title}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# ============================================================
# 文件 I/O
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=True)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_K(K_list: List) -> np.ndarray:
    return np.array(K_list, dtype=np.float64)


def load_dist(d_list: List) -> np.ndarray:
    return np.array(d_list, dtype=np.float64)


# ============================================================
# 打印工具
# ============================================================

def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_matrix(label: str, M: np.ndarray, fmt: str = ".4f") -> None:
    print(f"  {label}:")
    for row in M:
        print("  [ " + "  ".join(f"{v:{fmt}}" for v in row) + " ]")


def print_vector(label: str, v: np.ndarray, fmt: str = ".4f") -> None:
    print(f"  {label}: " + "  ".join(f"{x:{fmt}}" for x in v.ravel()))


# ============================================================
# 棋盘格外参估算（辅助验证）
# ============================================================

def estimate_pose(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """对每帧独立估算 6DoF 位姿（用于分析棋盘格距离）。"""
    rvecs, tvecs = [], []
    for objp, imgp in zip(objpoints, imgpoints):
        try:
            _, rvec, tvec = cv2.solvePnP(
                objp.astype(np.float64), imgp.astype(np.float64),
                K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            rvecs.append(rvec.ravel())
            tvecs.append(tvec.ravel())
        except Exception:
            rvecs.append(np.zeros(3))
            tvecs.append(np.zeros(3))
    return rvecs, tvecs
