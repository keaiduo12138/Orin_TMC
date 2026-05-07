import os, cv2, numpy as np, glob, json
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Align depth images to TianMou view")
    parser.add_argument("--base", type=str, default="/projects/calib_data/output_1970",
                        help="Base directory containing depth/ and tianmou/ subdirectories")
    parser.add_argument("--index", type=str, default=None,
                        help="Path to post_process output JSON (e.g. output_1970.json). "
                             "If provided, will generate depth_tm/frames.json sidecar index.")
    parser.add_argument("--no-speckle-filter", action="store_true",
                        help="Disable 5x5 neighbor-consistency filter (removes flying pixels / background spray)")
    return parser.parse_args()

args = parse_args()

# ==============================================
# ✅ 你的正确外参：RS → TM 变换
# P_tm = R_inv @ P_rs + T_inv
# ==============================================
R_inv = np.array([
    [ 0.9994290144877676, -0.009756706059484074,  0.03234890549738366],
    [ 0.009057886316137699,  0.9997239454619781,  0.021679196604571148],
    [-0.03255149298409764, -0.021373805388946347, 0.9992414927071943]
], dtype=np.float64)

T_inv = np.array([-19.324158264221295, -107.79101099086113, -5.730018510908119], dtype=np.float64)

# ==============================================
# 内参（保持不变）
# ==============================================
K_tm = np.array([
    [707.3128679575574,   0.0,             405.0635769116221],
    [0.0,            707.5707312160775,   266.08271420592877],
    [0.0,                0.0,                  1.0]
], dtype=np.float64)

K_rs = np.array([
    [385.8800518938313,   0.0,             329.3323298300865],
    [0.0,            385.6813030560857,   162.21271852833553],
    [0.0,                0.0,                  1.0]
], dtype=np.float64)

# ==============================================
# ✅ 关键：直接构建 RS→TM 变换矩阵（不做多余求逆）
# ==============================================
T_mat_rs_to_tm = np.eye(4, dtype=np.float64)
T_mat_rs_to_tm[:3, :3] = R_inv   # 旋转
T_mat_rs_to_tm[:3,  3] = T_inv   # 平移

# ==============================================
# 路径
# ==============================================
BASE  = args.base
DEPTH = f"{BASE}/depth/"
TM    = f"{BASE}/tianmou/"
OUT   = f"{BASE}/compare/"
TM_DEPTH = f"{BASE}/depth_tm/"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TM_DEPTH, exist_ok=True)

# ==============================================
# 可选：加载 post_process JSON，构建 sidecar 索引
# ==============================================
frames_mapping = None
if args.index:
    print(f"📋 加载索引文件: {args.index}")
    with open(args.index) as f:
        index_data = json.load(f)

    all_frames = index_data.get("frames", [])
    # 过滤条件与 export_aligned_frames.py 保持一致：必须 is_valid 且 has_color
    valid_frames = [f for f in all_frames if f.get("is_valid", False) and f.get("has_color", False)]

    anchor = index_data.get("anchor", {})
    statistics = index_data.get("statistics", {})
    print(f"  原始帧数: {len(all_frames)}, 有效帧(含Color): {len(valid_frames)}")

    # 构建 seq_idx (PNG 顺序) → 原始帧信息 的映射
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
    print(f"  索引构建完成: {len(frames_mapping)} 条记录")
else:
    print("⚠️  未指定 --index，将不生成 depth_tm/frames.json")

# ==============================================
# 配置
# ==============================================
CROP_TOP = 80
H_OUT, W_OUT = 320, 640

# 飞点 / 背景溅射：局部中位数离群（稀疏投影下“数一致邻居”会误删整图，故用 median）
SPECKLE_K_HALF = 3
SPECKLE_MIN_VALID_IN_WIN = 5
SPECKLE_MAX_MEDIAN_DIFF_MM = 350.0

# ==============================================
# ✅ 核心对齐函数（100%正确）
# ==============================================
def project_depth_to_tianmou(depth_raw, K_tm, K_rs, T_mat_rs_to_tm):
    depth_mm = depth_raw.astype(np.float32)

    # 1. 裁剪深度图 480→320
    depth_crop = depth_mm[CROP_TOP:CROP_TOP + H_OUT, :]

    # 2. 初始化输出（无效值=65535）
    aligned = np.full((H_OUT, W_OUT), 65535.0, dtype=np.float32)

    # 3. 有效深度过滤
    mask_v = (depth_crop > 50) & (depth_crop < 10000)
    u, v = np.meshgrid(np.arange(W_OUT), np.arange(H_OUT))
    u_v = u[mask_v].astype(np.float64)
    v_v = v[mask_v].astype(np.float64)
    z_v = depth_crop[mask_v].astype(np.float64)

    # 4. 反投影到 RS 相机坐标系 3D
    x_rs = (u_v - K_rs[0, 2]) * z_v / K_rs[0, 0]
    y_rs = (v_v - K_rs[1, 2]) * z_v / K_rs[1, 1]

    # 5. 坐标变换：RS → TM
    p_rs_h = np.vstack([x_rs, y_rs, z_v, np.ones_like(z_v)])
    p_tm_h = T_mat_rs_to_tm @ p_rs_h
    x_tm, y_tm, z_tm = p_tm_h[0], p_tm_h[1], p_tm_h[2]

    # 6. 过滤 z_tm <= 0（相机后方）
    ok = z_tm > 0
    x_tm, y_tm, z_tm = x_tm[ok], y_tm[ok], z_tm[ok]

    # 7. 投影到 TM 像素坐标
    u_tm = K_tm[0, 0] * x_tm / z_tm + K_tm[0, 2]
    v_tm = K_tm[1, 1] * y_tm / z_tm + K_tm[1, 2]

    # 8. 边界裁剪到 [0, W_OUT) × [0, H_OUT)
    u_i = np.clip(np.round(u_tm).astype(int), 0, W_OUT - 1)
    v_i = np.clip(np.round(v_tm).astype(int), 0, H_OUT - 1)

    # 9. ✅ 遮挡处理：np.minimum.at 对每个 TM 像素只保留最近的 RS 深度
    #    如果多个 RS 像素落在同一 TM 像素，远的会被近的覆盖（正确遮挡）
    np.minimum.at(aligned, (v_i, u_i), z_tm)

    return aligned


def filter_depth_median_outliers(aligned, invalid, k_half, min_valid_in_win, max_median_diff_mm):
    """
    剔除亚像素散射/视差导致的飞点：在 (2k+1)^2 窗口内取有效深度的中位数；
    若窗口内有效点太少则保留原值（避免稀疏区域被整片删掉）；
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
    # 仅当中心深度显著大于局部中位数时标记为离群飞点（背景溅射使深度值偏大）
    # 深度值偏小（更近）的情况不处理，保留原值（可能是前景有效点）
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
    return (cm.jet(norm)[...,:3] * 255).astype(np.uint8)

def label_img(img, left, right=""):
    p = Image.fromarray(img)
    d = ImageDraw.Draw(p)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    d.text((10, img.shape[0]-20), left, fill=(255,255,255), font=font)
    if right:
        d.text((650, img.shape[0]-20), right, fill=(255,255,255), font=font)
    return np.array(p)

# ==============================================
# 主程序
# ==============================================
print("✅ 使用正确外参：RS → TM")
if args.no_speckle_filter:
    print("⚠️  已关闭飞点过滤（--no-speckle-filter）")
else:
    print(f"✅ 飞点过滤: {2 * SPECKLE_K_HALF + 1}x{2 * SPECKLE_K_HALF + 1} 窗口内有效深度中位数, "
          f"有效点数≥{SPECKLE_MIN_VALID_IN_WIN} 且 d-median>{SPECKLE_MAX_MEDIAN_DIFF_MM}mm → 剔除")
depth_files = sorted(glob.glob(os.path.join(DEPTH, "depth_*.png")))

# 收集所有处理记录，用于生成 sidecar frames.json
processed_records = []

for fpath in tqdm(depth_files, desc="对齐中"):
    fname = os.path.basename(fpath)
    idx = fname.replace("depth_","").replace(".png","")

    depth_raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if depth_raw is None: continue

    # ✅ 对齐
    aligned = project_depth_to_tianmou(depth_raw, K_tm, K_rs, T_mat_rs_to_tm)
    if not args.no_speckle_filter:
        aligned = filter_depth_median_outliers(
            aligned, 65535.0, SPECKLE_K_HALF, SPECKLE_MIN_VALID_IN_WIN, SPECKLE_MAX_MEDIAN_DIFF_MM
        )

    # 可视化
    vis = cv2.flip(depth_to_jet(aligned), 1)
    tm_path = os.path.join(TM, f"tianmou_{idx}.png")
    tm_rgb = cv2.cvtColor(cv2.imread(tm_path), cv2.COLOR_BGR2RGB) if os.path.exists(tm_path) else np.zeros((320,640,3),np.uint8)

    # 对比图
    out_img = np.zeros((320, 1280, 3), dtype=np.uint8)
    out_img[:320, :640] = vis
    out_img[:320, 640:] = tm_rgb
    out_img = label_img(out_img, "Aligned Depth (RS→TM)", "TianMou")

    cv2.imwrite(os.path.join(OUT, f"compare_{idx}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # ✅ 保存对齐后的深度图（Z16 格式，裁剪到 320x640，与 compare 镜像一致）
    cv2.imwrite(os.path.join(TM_DEPTH, f"depth_{idx}.png"),
                cv2.flip(aligned, 1).astype(np.uint16))

    # 记录该帧的索引信息（用于生成 frames.json）
    if frames_mapping and idx in frames_mapping:
        record = frames_mapping[idx].copy()
        record["seq_idx"] = idx
        processed_records.append(record)

# ==============================================
# 生成 depth_tm/frames.json
# ==============================================
if frames_mapping:
    frames_json_path = os.path.join(TM_DEPTH, "frames.json")
    sidecar = {
        "source": "depth_rec.py",
        "base_index_file": args.index,
        "description": "depth_tm/ 目录下的 depth_XXXX.png 对应的原始索引信息",
        "total_frames": len(processed_records),
        "input": index_data.get("input", {}),
        "frames": processed_records,
    }
    with open(frames_json_path, "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    print(f"\n✅ frames.json 已生成: {frames_json_path}")

print(f"✅ 完成！可视化→{OUT}，对齐深度→{TM_DEPTH}")
