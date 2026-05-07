#!/usr/bin/env python3
"""
calib/calib_0408_calibrate.py
对 /projects/calib_data/output_0326_2346 的天眸和 RealSense 分别做内参标定。
- 天眸：从800帧中均匀抽取50张做标定，分辨率 640×320。
- RealSense：从800帧中均匀抽取50张做标定，分辨率 640×480。
"""

import os, sys, cv2, numpy as np
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import utils

OUTPUT = "./calib_0408"
os.makedirs(os.path.join(OUTPUT, "visualization"), exist_ok=True)

# ───────────────────────────── 参数 ─────────────────────────────
BOARD_COLS = utils.BOARD_COLS          # 11
BOARD_ROWS = utils.BOARD_ROWS         # 8
SQUARE_SIZE = utils.SQUARE_SIZE       # 29.66 mm
TM_K_REF = utils.TM_K_REF
RS_K_DEFAULT = utils.RS_K_DEFAULT
RS_D_DEFAULT = utils.RS_D_DEFAULT

# ───────────────────────────── 工具函数 ─────────────────────────────
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

def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=True)

# ───────────────────────────── 1. 天眸内参 ─────────────────────────────
def calibrate_tianmou():
    print("\n" + "="*60)
    print("  天眸内参标定（从800帧中抽取50张）")
    print("="*60)

    data_dir = "/projects/calib_data/output_0326_2346/tianmou"
    files = sorted(glob(os.path.join(data_dir, "tianmou_*.png")))[:800]

    # 均匀采样50张，step=16
    sampled = files[::16][:50]
    print(f"  图像分辨率: 640×320")
    print(f"  采样帧数: {len(sampled)}")

    objp = build_objp()
    objpoints, imgpoints = [], []

    for path in tqdm(sampled, desc="[天眸] 检测角点"):
        img = cv2.imread(path)
        if img is None:
            continue
        if utils.TM_FLIP_HORIZONTAL:
            img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = detect_corners(gray)
        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            vis = img.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, (BOARD_COLS, BOARD_ROWS), corners, True)
            frame_id = os.path.basename(path).replace("tianmou_", "").replace(".png", "")
            cv2.imwrite(os.path.join(OUTPUT, "visualization", f"tm_corners_{frame_id}.png"), vis)

    print(f"  检测成功: {len(objpoints)}/{len(sampled)}")
    if len(objpoints) < 4:
        print("  帧数不足，退出")
        return

    # 迭代剔除
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    objp_iter = list(objpoints)
    imgp_iter = list(imgpoints)
    iteration = 0
    removed = []

    while iteration < 10:
        iteration += 1
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp_iter, imgp_iter, (640, 320), None, None, criteria=criteria)

        per_err = []
        for i, (op, ip, rv, tv) in enumerate(zip(objp_iter, imgp_iter, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
            err = float(np.linalg.norm(proj.squeeze().astype(np.float64) - ip.squeeze().astype(np.float64)))
            per_err.append((i, err))

        bad = [(i, e) for i, e in per_err if e > 5.0]
        mean_err = np.mean([e for _, e in per_err])
        print(f"  迭代 {iteration}: 帧数={len(objp_iter)}, 平均误差={mean_err:.4f}px, 剔除={len(bad)}")

        if not bad:
            print("  ✓ 收敛")
            break
        if len(objp_iter) - len(bad) < 10:
            print("  ⚠ 帧数不足，停止")
            break

        for i, e in sorted(bad, key=lambda x: -x[1]):
            removed.append(e)
        bad_idxs = sorted([i for i, _ in bad], reverse=True)
        for bi in bad_idxs:
            del objp_iter[bi]; del imgp_iter[bi]

    # 最终结果
    ret, K, dist, rvecs_final, tvecs_final = cv2.calibrateCamera(
        objp_iter, imgp_iter, (640, 320), None, None, criteria=criteria)

    per_errors = []
    for i, (op, ip, rv, tv) in enumerate(zip(objp_iter, imgp_iter, rvecs_final, tvecs_final)):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = float(np.linalg.norm(proj.squeeze().astype(np.float64) - ip.squeeze().astype(np.float64)))
        per_errors.append({"reproj_error": err})

    mean_err = float(np.mean([e["reproj_error"] for e in per_errors]))
    diff_fro = np.linalg.norm(K - TM_K_REF, "fro") / np.linalg.norm(TM_K_REF) * 100

    print(f"\n  天眸内参 (640×320):")
    print(f"  K = {K.tolist()}")
    print(f"  dist = {dist.ravel().round(6).tolist()}")
    print(f"  平均重投影误差: {mean_err:.4f} px")
    print(f"  与参考内参偏差: {diff_fro:.1f}%")

    result = {
        "camera": "tianmou",
        "resolution": {"width": 640, "height": 320},
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rvecs": [r.ravel().tolist() for r in rvecs_final],
        "tvecs": [t.ravel().tolist() for t in tvecs_final],
        "reproj_errors": [e["reproj_error"] for e in per_errors],
        "mean_reproj_error": mean_err,
        "frobenius_deviation_from_ref": float(diff_fro),
        "n_frames": len(objp_iter),
        "n_frames_initial": len(objpoints),
        "n_frames_removed": len(removed),
        "square_size_mm": SQUARE_SIZE,
    }
    save_json(result, os.path.join(OUTPUT, "intrinsic_tianmou.json"))
    print(f"  已保存: {OUTPUT}/intrinsic_tianmou.json")
    return K, dist

# ───────────────────────────── 2. RealSense 内参 ─────────────────────────────
def calibrate_realsense():
    print("\n" + "="*60)
    print("  RealSense 内参标定（从800帧中抽取50张）")
    print("="*60)

    data_dir = "/projects/calib_data/output_0326_2346/color_orig"
    files = sorted(glob(os.path.join(data_dir, "color_*.png")))[:800]

    # 均匀采样50张，step=16
    sampled = files[::16][:50]
    print(f"  图像分辨率: 640×480")
    print(f"  采样帧数: {len(sampled)}")

    objp = build_objp()
    objpoints, imgpoints = [], []

    for path in tqdm(sampled, desc="[RS] 检测角点"):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = detect_corners(gray)
        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (BOARD_COLS, BOARD_ROWS), corners, True)
            frame_id = os.path.basename(path).replace("color_", "").replace(".png", "")
            cv2.imwrite(os.path.join(OUTPUT, "visualization", f"rs_corners_{frame_id}.png"), vis)

    print(f"  检测成功: {len(objpoints)}/{len(sampled)}")
    if len(objpoints) < 4:
        print("  帧数不足，退出")
        return

    # 迭代剔除
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    objp_iter = list(objpoints)
    imgp_iter = list(imgpoints)
    iteration = 0
    removed = []

    while iteration < 10:
        iteration += 1
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp_iter, imgp_iter, (640, 480), None, None, criteria=criteria)

        per_err = []
        for i, (op, ip, rv, tv) in enumerate(zip(objp_iter, imgp_iter, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
            err = float(np.linalg.norm(proj.squeeze().astype(np.float64) - ip.squeeze().astype(np.float64)))
            per_err.append((i, err))

        bad = [(i, e) for i, e in per_err if e > 5.0]
        mean_err = np.mean([e for _, e in per_err])
        print(f"  迭代 {iteration}: 帧数={len(objp_iter)}, 平均误差={mean_err:.4f}px, 剔除={len(bad)}")

        if not bad:
            print("  ✓ 收敛")
            break
        if len(objp_iter) - len(bad) < 10:
            print("  ⚠ 帧数不足，停止")
            break

        for i, e in sorted(bad, key=lambda x: -x[1]):
            removed.append(e)
        bad_idxs = sorted([i for i, _ in bad], reverse=True)
        for bi in bad_idxs:
            del objp_iter[bi]; del imgp_iter[bi]

    # 最终结果
    ret, K, dist, rvecs_final, tvecs_final = cv2.calibrateCamera(
        objp_iter, imgp_iter, (640, 480), RS_K_DEFAULT.copy(), RS_D_DEFAULT.copy(),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=criteria)

    per_errors = []
    for i, (op, ip, rv, tv) in enumerate(zip(objp_iter, imgp_iter, rvecs_final, tvecs_final)):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        err = float(np.linalg.norm(proj.squeeze().astype(np.float64) - ip.squeeze().astype(np.float64)))
        per_errors.append({"reproj_error": err})

    mean_err = float(np.mean([e["reproj_error"] for e in per_errors]))
    diff_fro = np.linalg.norm(K - RS_K_DEFAULT, "fro") / np.linalg.norm(RS_K_DEFAULT) * 100

    print(f"\n  RealSense 内参 (640×480):")
    print(f"  K = {K.tolist()}")
    print(f"  dist = {dist.ravel().round(6).tolist()}")
    print(f"  平均重投影误差: {mean_err:.4f} px")
    print(f"  与出厂内参偏差: {diff_fro:.1f}%")

    result = {
        "camera": "realsense_color",
        "resolution": {"width": 640, "height": 480},
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rvecs": [r.ravel().tolist() for r in rvecs_final],
        "tvecs": [t.ravel().tolist() for t in tvecs_final],
        "reproj_errors": [e["reproj_error"] for e in per_errors],
        "mean_reproj_error": mean_err,
        "frobenius_deviation_from_factory": float(diff_fro),
        "n_frames": len(objp_iter),
        "n_frames_initial": len(objpoints),
        "n_frames_removed": len(removed),
        "square_size_mm": SQUARE_SIZE,
        "factory_K": RS_K_DEFAULT.tolist(),
        "factory_dist": RS_D_DEFAULT.tolist(),
    }
    save_json(result, os.path.join(OUTPUT, "intrinsic_realsense.json"))
    print(f"  已保存: {OUTPUT}/intrinsic_realsense.json")
    return K, dist

# ───────────────────────────── 主程序 ─────────────────────────────
if __name__ == "__main__":
    import json
    K_tm, dist_tm = calibrate_tianmou()
    K_rs, dist_rs = calibrate_realsense()

    print("\n" + "="*60)
    print("  标定完成")
    print("="*60)
    print(f"  天眸内参: K=\n{K_tm}\n  dist={dist_tm.ravel().round(6).tolist()}")
    print(f"  RealSense内参: K=\n{K_rs}\n  dist={dist_rs.ravel().round(6).tolist()}")