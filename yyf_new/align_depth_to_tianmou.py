#!/usr/bin/env python3
"""
Usage:
    python3 align_depth_to_tianmou.py /path/to/dataset
    python3 align_depth_to_tianmou.py /path/to/dataset --calib-dir ./yyf/calib/calibration_output_0326

Dataset 目录结构要求:
    dataset/
    ├── depth/       原始深度图 (480×640, uint16 Z16), 命名 depth_0000.png
    └── tianmou/     TianMou 原图 (320×640×3), 命名 tianmou_0000.png

输出:
    dataset/compare/  并排对比图 (336×1280×3)
        左: 镜像对齐深度 jet 色图
        右: TianMou 原图
"""

import argparse
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.cm as cm


# =============================================================================
# 默认标定参数 (calibration_output_0326)
# =============================================================================
CALIB_K_REG = np.array([
    [709.8103861157124,  0.0,              411.55966548293605],
    [0.0,               708.456821072409, 264.81582935278465],
    [0.0,               0.0,              1.0]
], dtype=np.float64)

CALIB_K_RGBD = np.array([
    [379.94175977462965, 0.0,             327.7954646713398],
    [0.0,               380.18957161465886, 161.4139429626936],
    [0.0,               0.0,               1.0]
], dtype=np.float64)

CALIB_R = np.array([
    [ 0.9994290144877676,  0.009057886316137699, -0.03255149298409764],
    [-0.009756706059484074, 0.9997239454619781, -0.021373805388946347],
    [ 0.03234890549738366,  0.021679196604571148, 0.9992414927071943]
], dtype=np.float64)

CALIB_T = np.array([18.213214641293902, 108.02392529697205, 6.633026045168847], dtype=np.float64)


def load_calib_from_dir(calib_dir):
    """从标定目录加载参数 JSON，覆盖默认值"""
    import json

    k = {}
    for name, path in [
        ("K_reg",   f"{calib_dir}/intrinsic_tianmou.json"),
        ("K_rgbd",  f"{calib_dir}/intrinsic_realsense_cropped.json"),
        ("extr",    f"{calib_dir}/extrinsic_tianmou_realsense.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                k[name] = json.load(f)
        else:
            print(f"[WARN] 标定文件不存在，跳过: {path}")
            return None

    K_reg  = np.array(k["K_reg"]["K"],   dtype=np.float64)
    K_rgbd = np.array(k["K_rgbd"]["K"],   dtype=np.float64)
    R      = np.array(k["extr"]["R"],     dtype=np.float64)
    T      = np.array(k["extr"]["T"],     dtype=np.float64)
    return K_reg, K_rgbd, R, T


def map_depth_to_tianmou(depth_raw, K_reg, K_rgbd, R, T,
                          valid_depth_min=50, valid_depth_max=10000):
    """
    Realsense 原始深度图 (480×640) -> TianMou 视角深度图 (320×640)

    步骤: 裁剪(480→320) -> 反投影(2D→3D) -> 坐标变换(RS→TM) -> 投影(3D→2D)
    """
    depth_mm = depth_raw.astype(np.float32)

    # 1. 垂直裁剪: 480行 -> 320行 (与标定分辨率一致)
    h_raw, w_raw = depth_mm.shape
    h_out, w_out = 320, 640
    crop_top = (h_raw - h_out) // 2
    depth_cropped = depth_mm[crop_top:crop_top + h_out, :]

    # 2. 初始化输出 (无效值 65535)
    aligned = np.full((h_out, w_out), 65535.0, dtype=np.float32)

    # 3. 有效深度过滤
    mask_valid = (depth_cropped > valid_depth_min) & (depth_cropped < valid_depth_max)

    # 4. 反投影: 2D pixel -> 3D (Realsense 相机坐标系, 单位 mm)
    u, v = np.meshgrid(np.arange(w_out), np.arange(h_out))
    u_v = u[mask_valid].astype(np.float64)
    v_v = v[mask_valid].astype(np.float64)
    z_v = depth_cropped[mask_valid].astype(np.float64)

    x_rgbd = (u_v - K_rgbd[0, 2]) * z_v / K_rgbd[0, 0]
    y_rgbd = (v_v - K_rgbd[1, 2]) * z_v / K_rgbd[1, 1]

    # 5. 坐标变换: Realsense -> TianMou (P_tm = R @ P_rs + T)
    T_mat = np.eye(4, dtype=np.float64)
    T_mat[:3, :3] = R
    T_mat[:3, 3]  = T
    p_rgbd_h = np.vstack((x_rgbd, y_rgbd, z_v, np.ones_like(z_v)))
    p_tm = T_mat @ p_rgbd_h
    x_tm, y_tm, z_tm = p_tm[0], p_tm[1], p_tm[2]

    # 6. 过滤 z <= 0 (相机后方)
    mask_front = z_tm > 0
    x_tm, y_tm, z_tm = x_tm[mask_front], y_tm[mask_front], z_tm[mask_front]

    # 7. 投影: 3D -> 2D (TianMou 像素)
    u_tm = K_reg[0, 0] * x_tm / z_tm + K_reg[0, 2]
    v_tm = K_reg[1, 1] * y_tm / z_tm + K_reg[1, 2]

    # 8. 边界检查
    u_idx = np.round(u_tm).astype(int)
    v_idx = np.round(v_tm).astype(int)
    in_view = (u_idx >= 0) & (u_idx < w_out) & (v_idx >= 0) & (v_idx < h_out)
    u_idx, v_idx, z_final = u_idx[in_view], v_idx[in_view], z_tm[in_view]

    # 9. Z-buffer: 从远到近，近处覆盖远处
    sort_idx = np.argsort(z_final)[::-1]
    aligned[v_idx[sort_idx], u_idx[sort_idx]] = z_final[sort_idx]

    return aligned


def depth_to_jet_rgb(depth_map):
    """深度图 -> jet 色图 RGB (uint8)，无效值(65535)显示为黑色"""
    vis = depth_map.copy()
    mask_invalid = (vis >= 65535) | (vis <= 0)
    valid_max = vis[~mask_invalid].max() if np.any(~mask_invalid) else 5000.0
    vis[mask_invalid] = 0
    normalized = np.clip(vis, 0, valid_max) / valid_max
    rgba = cm.jet(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def draw_label(img_array, text_left, text_right):
    """在图底部添加文字标签"""
    from PIL import Image, ImageDraw, ImageFont
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10,   img_pil.height - 20), text_left,  fill=(255, 255, 255), font=font)
    draw.text((650, img_pil.height - 20), text_right, fill=(255, 255, 255), font=font)
    return np.array(img_pil)


def main():
    parser = argparse.ArgumentParser(description="将 Realsense 深度图对齐到 TianMou 视角")
    parser.add_argument("dataset", help="数据集根目录，需包含 depth/ 和 tianmou/ 子目录")
    parser.add_argument("--calib-dir", default="",
                        help="标定参数目录 (默认: ./yyf/calib/calibration_output_0326)")
    parser.add_argument("--depth-dir", default="depth",
                        help="深度图子目录名 (默认: depth)")
    parser.add_argument("--tm-dir",   default="tianmou",
                        help="TianMou 图子目录名 (默认: tianmou)")
    parser.add_argument("--output",   default="compare",
                        help="输出目录名 (默认: compare)")
    args = parser.parse_args()

    # 标定参数
    calib_dir = args.calib_dir or \
        "/home/nvidia/Desktop/cxr_multi_sensor/yyf/calib/calibration_output_0326"

    if os.path.exists(f"{calib_dir}/intrinsic_tianmou.json"):
        K_reg, K_rgbd, R, T = load_calib_from_dir(calib_dir)
        print(f"[标定] 从 {calib_dir} 加载参数")
    else:
        K_reg, K_rgbd, R, T = CALIB_K_REG, CALIB_K_RGBD, CALIB_R, CALIB_T
        print(f"[标定] 使用内置默认参数 (calibration_output_0326)")

    # 路径
    dataset_dir = os.path.abspath(args.dataset)
    depth_dir   = os.path.join(dataset_dir, args.depth_dir)
    tm_dir      = os.path.join(dataset_dir, args.tm_dir)
    output_dir  = os.path.join(dataset_dir, args.output)

    # 校验
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"深度目录不存在: {depth_dir}")
    if not os.path.isdir(tm_dir):
        raise FileNotFoundError(f"TianMou 目录不存在: {tm_dir}")

    os.makedirs(output_dir, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.png")))
    if not depth_files:
        raise FileNotFoundError(f"未找到 depth_*.png 文件于: {depth_dir}")

    print(f"\n数据集  : {dataset_dir}")
    print(f"深度图  : {depth_dir} ({len(depth_files)} 张)")
    print(f"TianMou : {tm_dir}")
    print(f"输出    : {output_dir}")
    print(f"K_reg  (TM 640x320) :\n{K_reg}\n")
    print(f"K_rgbd (RS 640x320) :\n{K_rgbd}\n")
    print(f"R, T    (RS -> TM)  :\n{R}\n{T}\n")

    for fpath in tqdm(depth_files, desc="对齐 + 生成对比图"):
        fname = os.path.basename(fpath)
        idx_str = fname.replace("depth_", "").replace(".png", "")

        depth_raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"  [WARN] 读取失败: {fname}")
            continue

        # 对齐
        aligned = map_depth_to_tianmou(depth_raw, K_reg, K_rgbd, R, T)

        # jet 可视化 + 水平镜像
        vis_rgb     = depth_to_jet_rgb(aligned)
        vis_mirrored = cv2.flip(vis_rgb, 1)

        # TianMou 原图
        tm_path = os.path.join(tm_dir, f"tianmou_{idx_str}.png")
        if os.path.exists(tm_path):
            tm_rgb = cv2.cvtColor(cv2.imread(tm_path), cv2.COLOR_BGR2RGB)
        else:
            tm_rgb = np.zeros((320, 640, 3), dtype=np.uint8)

        # 并排合成: 左=镜像深度 jet, 右=TianMou
        pad = 16
        comparison = np.full((320 + pad, 640 * 2, 3), 0, dtype=np.uint8)
        comparison[:320, :640, :]  = vis_mirrored
        comparison[:320, 640:, :]  = tm_rgb
        comparison[:, 639:641, :]  = 255  # 中间白分割线
        comparison = draw_label(comparison,
                                "Aligned Depth (mirrored, jet)",
                                "TianMou (raw)")

        out_path = os.path.join(output_dir, f"compare_{idx_str}.png")
        cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print(f"\n完成! 共处理 {len(depth_files)} 张 -> {output_dir}/")


if __name__ == "__main__":
    main()
