#!/usr/bin/env python3
"""
depth_0408.py
基于 calib_0408 新标定结果，将 RealSense 深度图投射到天眸视角（640×320）。

与 depth_rec.py 的主要区别：
  1. 天眸和 RealSense 分辨率不同（TM: 640×320, RS: 640×480），
     不再裁剪，而是全分辨率反投影后过滤越界点。
  2. 外参取逆（R_inv = R.T, T_inv = -R.T @ T），因为标定结果是 TM→RS，
     投射深度需要 RS→TM。
  3. 内参直接使用 calib_0408 标定文件中的值。
  4. 对投影后超出 [0, W_out)×[0, H_out) 范围的像素做严格剔除（而非 clip）。
"""

import os, cv2, numpy as np, glob, json, argparse
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Align RealSense depth to TianMou view (calib_0408)")
    parser.add_argument("--base", type=str, default="/projects/calib_data/output_0326_2346",
                        help="Base directory containing depth/ and tianmou/ subdirectories")
    parser.add_argument("--index", type=str, default=None,
                        help="Path to post_process output JSON; if provided, generates depth_tm/frames.json")
    parser.add_argument("--no-speckle-filter", action="store_true",
                        help="Disable 5x5 median-based outlier filter")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Custom output directory (default: <base>/compare_0408/)")
    return parser.parse_args()

args = parse_args()

# ==============================================
# calib_0408 内参
# ==============================================
CALIB_DIR = "./calib_0408"

K_tm = np.array([
    [708.5435093177016,   0.0,              411.39820813805255],
    [  0.0,              708.4603025263957,  261.99171815297035],
    [  0.0,                0.0,                    1.0]
], dtype=np.float64)

K_rs = np.array([
    [385.36326763931373,   0.0,              328.78553294296944],
    [  0.0,               385.3195649220036,  240.31576660091955],
    [  0.0,                 0.0,                    1.0]
], dtype=np.float64)

# ==============================================
# calib_0408 外参（标定结果是 camera1=天眸, camera2=RealSense）
# 需要取逆得到 RS→TM 变换用于投射深度
# ==============================================
R_raw = np.array([
    [ 0.9999203721311448,  0.0044579992159311885, -0.011805746062984651],
    [-0.004417743928315652, 0.9999843465310002,   0.003433690660540546],
    [ 0.011820868652377069, -0.0033812624800819715, 0.9999244142075661]
], dtype=np.float64)

T_raw = np.array([-9.019156763129708, 77.40464249149463, 29.03800160434001], dtype=np.float64)

# R_inv = R^T, T_inv = -R^T @ T  →  RS → TM 变换
R_inv = R_raw.T
T_inv = -R_raw.T @ T_raw

print("外参（RS→TM）:")
print(f"  R_inv:\n{R_inv}")
print(f"  T_inv: {T_inv}")
print(f"  |T_inv|: {np.linalg.norm(T_inv):.2f} mm")

# 构建 4×4 变换矩阵
T_mat_rs_to_tm = np.eye(4, dtype=np.float64)
T_mat_rs_to_tm[:3, :3] = R_inv
T_mat_rs_to_tm[:3,  3] = T_inv

# ==============================================
# 输出尺寸（与天眸分辨率一致）
# ==============================================
H_OUT, W_OUT = 320, 640

# ==============================================
# 路径
# ==============================================
BASE  = args.base
DEPTH = f"{BASE}/depth/"
TM    = f"{BASE}/tianmou/"
OUT   = args.out_dir or f"{BASE}/compare_0408/"
TM_DEPTH = f"{BASE}/depth_tm_0408/"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TM_DEPTH, exist_ok=True)

# ==============================================
# 可选：加载 post_process JSON，构建 sidecar 索引
# ==============================================
frames_mapping = None
if args.index:
    print(f"加载索引文件: {args.index}")
    with open(args.index) as f:
        index_data = json.load(f)
    all_frames = index_data.get("frames", [])
    valid_frames = [f for f in all_frames if f.get("is_valid", False) and f.get("has_color", False)]
    frames_mapping = {}
    for i, frame in enumerate(valid_frames):
        seq_idx = f"{i:04d}"
        frames_mapping[seq_idx] = {
            "tianmou_idx":   frame.get("tianmou_idx", 0),
            "depth_counter": frame.get("depth_counter", 0),
            "color_counter": frame.get("color_counter", 0),
            "depth_idx":     frame.get("depth_idx", i),
            "dt_ms":         round(frame.get("dt_ms"), 2) if frame.get("dt_ms") is not None else None,
        }
    print(f"  原始帧数: {len(all_frames)}, 有效帧(含Color): {len(valid_frames)}")
else:
    print("未指定 --index，将不生成 depth_tm_0408/frames.json")

# ==============================================
# 飞点过滤参数（与 depth_rec.py 一致）
# ==============================================
SPECKLE_K_HALF = 3
SPECKLE_MIN_VALID_IN_WIN = 5
SPECKLE_MAX_MEDIAN_DIFF_MM = 350.0


# ==============================================
# 核心对齐函数
# ==============================================
def project_depth_to_tianmou(depth_raw, K_tm, K_rs, T_mat_rs_to_tm, H_out, W_out):
    """
    将 RealSense 深度图（640×480）投射到天眸视角（640×320）。

    关键差异（相比 depth_rec.py）：
      - 不裁剪深度图，全分辨率处理。
      - 对投影后超出 [0,W_out)×[0,H_out) 的像素严格剔除（而非 clip），
        避免越界区域的错误深度污染输出图。
    """
    depth_mm = depth_raw.astype(np.float32)

    # RealSense 深度图原始分辨率
    H_rs, W_rs = depth_mm.shape[:2]

    # 初始化输出（无效值=65535）
    aligned = np.full((H_out, W_out), 65535.0, dtype=np.float32)

    # 有效深度过滤（50mm ~ 10m）
    mask_v = (depth_mm > 50) & (depth_mm < 10000)
    u_rs = np.where(mask_v)[1].astype(np.float64)  # 列索引
    v_rs = np.where(mask_v)[0].astype(np.float64)  # 行索引
    z_rs = depth_mm[mask_v].astype(np.float64)

    # 反投影到 RS 相机坐标系 3D
    x_rs = (u_rs - K_rs[0, 2]) * z_rs / K_rs[0, 0]
    y_rs = (v_rs - K_rs[1, 2]) * z_rs / K_rs[1, 1]

    # 坐标变换：RS → TM
    p_rs_h = np.vstack([x_rs, y_rs, z_rs, np.ones_like(z_rs)])
    p_tm_h = T_mat_rs_to_tm @ p_rs_h
    x_tm, y_tm, z_tm = p_tm_h[0], p_tm_h[1], p_tm_h[2]

    # 过滤 z_tm <= 0（落在相机后方）
    ok = z_tm > 0
    x_tm, y_tm, z_tm = x_tm[ok], y_tm[ok], z_tm[ok]

    # 投影到 TM 像素坐标
    u_tm = K_tm[0, 0] * x_tm / z_tm + K_tm[0, 2]
    v_tm = K_tm[1, 1] * y_tm / z_tm + K_tm[1, 2]

    # ==========================================
    # 关键：严格边界剔除，np.floor 防止浮点精度导致 round 后越界
    # ==========================================
    valid = (u_tm >= 0) & (u_tm < W_out) & (v_tm >= 0) & (v_tm < H_out)
    u_tm = u_tm[valid]
    v_tm = v_tm[valid]
    z_tm = z_tm[valid]

    u_i = np.clip(np.floor(u_tm).astype(int), 0, W_out - 1)
    v_i = np.clip(np.floor(v_tm).astype(int), 0, H_out - 1)

    # 遮挡处理：每个 TM 像素只保留最近的 RS 深度（近处覆盖远处）
    np.minimum.at(aligned, (v_i, u_i), z_tm)

    return aligned


def filter_depth_median_outliers(aligned, invalid, k_half, min_valid_in_win, max_median_diff_mm):
    """
    剔除亚像素散射/视差导致的飞点：
    在 (2k+1)^2 窗口内取有效深度的中位数；
    若窗口内有效点太少则保留原值；
    若中心深度与中位数差过大则判为离群并抹除。
    """
    d = np.asarray(aligned, dtype=np.float32)
    valid = (d > 0) & (d < invalid)
    win = 2 * k_half + 1
    nan_depth = np.where(valid, d, np.nan).astype(np.float64)
    padded = np.pad(nan_depth, k_half, mode="constant", constant_values=np.nan)
    patches = sliding_window_view(padded, (win, win))
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(patches, axis=(2, 3))
    cnt = np.sum(np.isfinite(patches), axis=(2, 3))
    bad = valid & (cnt >= min_valid_in_win) & ((d.astype(np.float64) - med) > max_median_diff_mm)
    out = d.copy()
    out[bad] = invalid
    return out


# ==============================================
# 可视化工具
# ==============================================
def depth_to_jet(d):
    vis = d.copy()
    mask = (vis >= 65535) | (vis <= 0)
    vmax = vis[~mask].max() if np.any(~mask) else 5000.0
    vis[mask] = 0
    norm = np.clip(vis, 0, vmax) / vmax
    return (cm.jet(norm)[..., :3] * 255).astype(np.uint8)


def label_img(img, left, right=""):
    p = Image.fromarray(img)
    d = ImageDraw.Draw(p)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    d.text((10, img.shape[0] - 20), left, fill=(255, 255, 255), font=font)
    if right:
        d.text((650, img.shape[0] - 20), right, fill=(255, 255, 255), font=font)
    return np.array(p)


# ==============================================
# 主程序
# ==============================================
print(f"数据目录: {BASE}")
print(f"输出目录: {OUT}")
print(f"对齐深度目录: {TM_DEPTH}")
print(f"天眸分辨率: {W_OUT}×{H_OUT}, RealSense深度分辨率: 640×480")

if args.no_speckle_filter:
    print("已关闭飞点过滤")
else:
    print(f"飞点过滤: {2 * SPECKLE_K_HALF + 1}×{2 * SPECKLE_K_HALF + 1} 窗口, "
          f"有效点≥{SPECKLE_MIN_VALID_IN_WIN} 且 d-median>{SPECKLE_MAX_MEDIAN_DIFF_MM}mm → 剔除")

depth_files = sorted(glob.glob(os.path.join(DEPTH, "depth_*.png")))
print(f"找到 {len(depth_files)} 个深度文件")

processed_records = []

for fpath in tqdm(depth_files, desc="对齐中"):
    fname = os.path.basename(fpath)
    idx = fname.replace("depth_", "").replace(".png", "")

    depth_raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        continue

    # 对齐
    aligned = project_depth_to_tianmou(depth_raw, K_tm, K_rs, T_mat_rs_to_tm, H_OUT, W_OUT)

    if not args.no_speckle_filter:
        aligned = filter_depth_median_outliers(
            aligned, 65535.0, SPECKLE_K_HALF, SPECKLE_MIN_VALID_IN_WIN, SPECKLE_MAX_MEDIAN_DIFF_MM
        )

    # 可视化（对齐深度 + 天眸图像）
    vis = cv2.flip(depth_to_jet(aligned), 1)
    tm_path = os.path.join(TM, f"tianmou_{idx}.png")
    tm_rgb = cv2.cvtColor(cv2.imread(tm_path), cv2.COLOR_BGR2RGB) if os.path.exists(tm_path) \
             else np.zeros((H_OUT, W_OUT, 3), np.uint8)

    out_img = np.zeros((H_OUT, W_OUT * 2, 3), dtype=np.uint8)
    out_img[:H_OUT, :W_OUT] = vis
    out_img[:H_OUT, W_OUT:] = tm_rgb
    out_img = label_img(out_img, "Aligned Depth (RS→TM) calib_0408", "TianMou")

    cv2.imwrite(os.path.join(OUT, f"compare_{idx}.png"),
                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # 保存对齐后的深度图（Z16 格式，与天眸镜像一致）
    cv2.imwrite(os.path.join(TM_DEPTH, f"depth_{idx}.png"),
                cv2.flip(aligned, 1).astype(np.uint16))

    # 记录索引信息
    if frames_mapping and idx in frames_mapping:
        record = frames_mapping[idx].copy()
        record["seq_idx"] = idx
        processed_records.append(record)

# 生成 frames.json
if frames_mapping:
    frames_json_path = os.path.join(TM_DEPTH, "frames.json")
    sidecar = {
        "source": "depth_0408.py",
        "calibration": "./calib_0408",
        "base_index_file": args.index,
        "description": "depth_tm_0408/ 目录下的 depth_XXXX.png 对应的原始索引信息",
        "total_frames": len(processed_records),
        "input": index_data.get("input", {}),
        "frames": processed_records,
    }
    with open(frames_json_path, "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    print(f"\nframes.json 已生成: {frames_json_path}")

print(f"完成！可视化 → {OUT}，对齐深度 → {TM_DEPTH}")
