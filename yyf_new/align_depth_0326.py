#!/usr/bin/env python3
"""
Depth to TianMou Alignment - v3 (calibration_output_0326)

使用新的标定参数 calibration_output_0326，将 Realsense D455 原始深度图
(480×640, uint16 Z16) 对齐到 TianMou 事件相机视角 (640×320)

输入: /projects/calib_data/output_0326_2346/depth/
输出:
  - compare/ : 镜像 jet 可视化 | TianMou 原图 并排对比
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# =============================================================================
# 标定参数 (calibration_output_0326)
# =============================================================================

# --- TianMou 事件相机内参 (640x320) ---
K_reg = np.array([
    [709.8103861157124,  0.0,                411.55966548293605],
    [0.0,               708.456821072409,   264.81582935278465],
    [0.0,               0.0,                1.0]
], dtype=np.float64)

# --- Realsense D455 内参 (640x320, 裁剪后分辨率) ---
K_rgbd = np.array([
    [379.94175977462965,  0.0,               327.7954646713398],
    [0.0,                380.18957161465886,  161.4139429626936],
    [0.0,                0.0,                1.0]
], dtype=np.float64)

# --- 外参: Realsense -> TianMou (from extrinsic_tianmou_realsense.json) ---
# P_tianmou = R @ P_rs + T
R = np.array([
    [ 0.9994290144877676,  0.009057886316137699, -0.03255149298409764],
    [-0.009756706059484074, 0.9997239454619781, -0.021373805388946347],
    [ 0.03234890549738366,  0.021679196604571148, 0.9992414927071943]
], dtype=np.float64)

T = np.array([18.213214641293902, 108.02392529697205, 6.633026045168847], dtype=np.float64)

# 构建 4x4 齐次变换矩阵: P_tm = R @ P_rs + T
T_rs2tm = np.eye(4, dtype=np.float64)
T_rs2tm[:3, :3] = R
T_rs2tm[:3, 3] = T

# =============================================================================
# 路径配置
# =============================================================================

BASE_DIR    = "/projects/calib_data/output_0326_2346"
INPUT_DIR   = f"{BASE_DIR}/depth/"
TM_DIR      = f"{BASE_DIR}/tianmou/"   # TianMou 原图 (已镜像, 1333张)
OUTPUT_DIR  = f"{BASE_DIR}/compare/"   # 输出并排对比

# =============================================================================
# 深度对齐函数
# =============================================================================

def map_depth_to_tianmou(depth_raw, K_reg, K_rgbd, T_rs2tm,
                          valid_depth_min=50,
                          valid_depth_max=10000):
    """
    将 RGBD 深度图对齐到 TianMou 视角

    Parameters
    ----------
    depth_raw : np.ndarray (480, 640), dtype=uint16
        原始深度图，单位 mm（Z16）
    K_reg : np.ndarray (3, 3) - TianMou 内参 (640x320)
    K_rgbd : np.ndarray (3, 3) - Realsense 内参 (640x320)
    T_rs2tm : np.ndarray (4, 4) - Realsense -> TianMou 外参

    Returns
    -------
    aligned_depth : np.ndarray (320, 640), dtype=np.float32, 单位 mm, 无效值=65535
    """
    depth_mm = depth_raw.astype(np.float32)

    # 垂直裁剪: 480行 -> 320行 (与标定分辨率一致)
    h_raw, w_raw = depth_mm.shape   # 480, 640
    h_out, w_out = 320, 640         # 标定分辨率
    crop_top = (h_raw - h_out) // 2  # 80
    depth_cropped = depth_mm[crop_top:crop_top + h_out, :]  # (320, 640)

    aligned_depth = np.full((h_out, w_out), 65535.0, dtype=np.float32)

    # 有效深度过滤
    mask_valid = (depth_cropped > valid_depth_min) & (depth_cropped < valid_depth_max)

    # 反投影: 2D -> 3D (Realsense 相机坐标系)
    u, v = np.meshgrid(np.arange(w_out), np.arange(h_out))
    u_v = u[mask_valid].astype(np.float64)
    v_v = v[mask_valid].astype(np.float64)
    z_v = depth_cropped[mask_valid].astype(np.float64)

    x_rgbd = (u_v - K_rgbd[0, 2]) * z_v / K_rgbd[0, 0]
    y_rgbd = (v_v - K_rgbd[1, 2]) * z_v / K_rgbd[1, 1]
    p_rgbd = np.vstack((x_rgbd, y_rgbd, z_v, np.ones_like(z_v)))

    # 坐标变换: Realsense -> TianMou
    p_tm = T_rs2tm @ p_rgbd
    x_tm, y_tm, z_tm = p_tm[0], p_tm[1], p_tm[2]

    # 过滤 z <= 0 (相机后方)
    mask_front = z_tm > 0
    x_tm, y_tm, z_tm = x_tm[mask_front], y_tm[mask_front], z_tm[mask_front]

    # 投影: 3D -> 2D (TianMou 像素)
    u_tm = K_reg[0, 0] * x_tm / z_tm + K_reg[0, 2]
    v_tm = K_reg[1, 1] * y_tm / z_tm + K_reg[1, 2]

    # 边界检查
    u_idx = np.round(u_tm).astype(int)
    v_idx = np.round(v_tm).astype(int)
    in_view = (u_idx >= 0) & (u_idx < w_out) & (v_idx >= 0) & (v_idx < h_out)
    u_idx = u_idx[in_view]
    v_idx = v_idx[in_view]
    z_final = z_tm[in_view]

    # Z-buffer: 从远到近，近处覆盖远处
    sort_idx = np.argsort(z_final)[::-1]
    aligned_depth[v_idx[sort_idx], u_idx[sort_idx]] = z_final[sort_idx]

    return aligned_depth


# =============================================================================
# 可视化
# =============================================================================

def depth_to_jet_rgb(depth_map, valid_max=None):
    """深度图 -> jet 色图 RGB (uint8), 无效值显示为黑色"""
    import matplotlib.cm as cm
    vis = depth_map.copy()
    mask_invalid = (vis >= 65535) | (vis <= 0)
    if valid_max is None:
        valid_max = vis[~mask_invalid].max() if np.any(~mask_invalid) else 5000.0
    vis[mask_invalid] = 0
    normalized = np.clip(vis, 0, valid_max) / valid_max
    rgba = cm.jet(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


# =============================================================================
# 主循环
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(INPUT_DIR, "depth_*.png")))
    print(f"找到 {len(depth_files)} 张深度图\n")
    print(f"K_reg (TianMou 640x320):\n{K_reg}\n")
    print(f"K_rgbd (Realsense 640x320):\n{K_rgbd}\n")
    print(f"T_rs2tm:\n{T_rs2tm}\n")

    saved = 0
    for fpath in tqdm(depth_files, desc="对齐 + 生成对比图"):
        fname = os.path.basename(fpath)
        idx_str = fname.replace("depth_", "").replace(".png", "")

        depth_raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"  [WARN] 无法读取: {fname}")
            continue

        # --- 对齐 ---
        aligned = map_depth_to_tianmou(depth_raw, K_reg, K_rgbd, T_rs2tm)

        # --- jet 可视化 + 镜像 (与 TianMou 坐标系一致) ---
        vis_rgb = depth_to_jet_rgb(aligned)
        vis_mirrored = cv2.flip(vis_rgb, 1)  # 水平镜像

        # --- 读取 TianMou 原图 ---
        tm_path = os.path.join(TM_DIR, f"tianmou_{idx_str}.png")
        if os.path.exists(tm_path):
            tm_img = cv2.imread(tm_path)      # BGR
            tm_rgb = cv2.cvtColor(tm_img, cv2.COLOR_BGR2RGB)  # RGB
        else:
            tm_rgb = np.zeros((320, 640, 3), dtype=np.uint8)

        # --- 并排对比图 ---
        # 左: 镜像深度 jet, 右: TianMou 原图, 高: 320, 宽: 640*2
        pad = 16
        comparison = np.full((320 + pad, 640 * 2, 3), 0, dtype=np.uint8)
        comparison[:320, :640, :]  = vis_mirrored   # 左: 镜像深度 jet
        comparison[:320, 640:, :]  = tm_rgb          # 右: TianMou 原图
        comparison[:, 639:641, :]  = 255             # 中间分割线 (白)

        # 底部标签
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(comparison)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        draw.text((10,  325), "Aligned Depth (mirrored, jet colormap)", fill=(255, 255, 255), font=font)
        draw.text((650, 325), "TianMou (raw)",                           fill=(255, 255, 255), font=font)
        comparison = np.array(img_pil)

        out_path = os.path.join(OUTPUT_DIR, f"compare_{idx_str}.png")
        cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"\n完成! 共处理 {saved} 张 -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
