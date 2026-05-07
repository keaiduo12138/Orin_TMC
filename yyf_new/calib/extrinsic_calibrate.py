#!/usr/bin/env python3
"""
calib/extrinsic_calibrate.py
对天眸(CVS, 640×320) 和 RealSense Color (640×480) 做外参标定。
从已保存的内参文件加载 K 和 dist，
从 /projects/calib_data/output_0326_2346 前800帧中均匀抽取50对同步图像，
调用 cv2.stereoCalibrate 联合求解 R, T。
"""

import os, sys, cv2, numpy as np, json
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import utils

OUTPUT = "./calib_0408"

BOARD_COLS = utils.BOARD_COLS   # 11
BOARD_ROWS = utils.BOARD_ROWS   # 8
SQUARE_SIZE = utils.SQUARE_SIZE # 29.66 mm


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


def main():
    # 1. 加载内参
    K_tm_list = utils.load_json(os.path.join(OUTPUT, "intrinsic_tianmou.json"))
    K_tm  = utils.load_K(K_tm_list["K"])
    dist_tm = utils.load_dist(K_tm_list["dist"])

    K_rs_list = utils.load_json(os.path.join(OUTPUT, "intrinsic_realsense.json"))
    K_rs  = utils.load_K(K_rs_list["K"])
    dist_rs = utils.load_dist(K_rs_list["dist"])

    print(f"  天眸内参:\n{K_tm}")
    print(f"  RealSense内参:\n{K_rs}")

    # 2. 采集角点对
    tm_dir = "/projects/calib_data/output_0326_2346/tianmou"
    rs_dir = "/projects/calib_data/output_0326_2346/color_orig"

    tm_files = sorted(glob(os.path.join(tm_dir, "tianmou_*.png")))[:800]
    rs_files = sorted(glob(os.path.join(rs_dir, "color_*.png")))[:800]

    # 均匀采样50对
    sampled_tm = tm_files[::16][:50]
    sampled_rs = rs_files[::16][:50]

    print(f"\n  天眸: {tm_dir}")
    print(f"  RealSense: {rs_dir}")
    print(f"  采样帧数: {len(sampled_tm)}")

    objp  = build_objp()
    objpoints, imgpoints_tm, imgpoints_rs = [], [], []

    for tm_path, rs_path in tqdm(zip(sampled_tm, sampled_rs), total=len(sampled_tm), desc="[外参] 检测角点对"):
        tm_img = cv2.imread(tm_path)
        rs_img = cv2.imread(rs_path)
        if tm_img is None or rs_img is None:
            continue

        if utils.TM_FLIP_HORIZONTAL:
            tm_img = cv2.flip(tm_img, 1)

        tm_gray = cv2.cvtColor(tm_img, cv2.COLOR_BGR2GRAY)
        rs_gray = cv2.cvtColor(rs_img, cv2.COLOR_BGR2GRAY)

        ret_tm, c_tm = detect_corners(tm_gray)
        ret_rs, c_rs = detect_corners(rs_gray)

        if ret_tm and ret_rs:
            objpoints.append(objp.copy())
            imgpoints_tm.append(c_tm)
            imgpoints_rs.append(c_rs)

    print(f"  检测成功: {len(objpoints)}/{len(sampled_tm)} 对")
    if len(objpoints) < 5:
        print("  帧数不足，无法做外参标定")
        return

    # 3. stereoCalibrate
    # OpenCV要求传入imageSize，对双目来说这是第一台相机的图像尺寸
    imageSize_tm = (640, 320)
    imageSize_rs = (640, 480)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，只优化外参

    print("\n" + "="*60)
    print("  调用 cv2.stereoCalibrate（固定内参）")
    print("="*60)

    ret, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_tm, imgpoints_rs,
        K_tm, dist_tm, K_rs, dist_rs,
        imageSize_tm,
        R=np.eye(3, dtype=np.float64),
        T=np.zeros(3, dtype=np.float64),
        flags=flags,
        criteria=criteria,
    )

    # K1_out 对应天眸，K2_out 对应RealSense（与输入顺序一致）
    print(f"\n  重投影误差: {ret:.4f} px")

    # 4. 迭代剔除离群帧
    objp_iter = list(objpoints)
    imgp_tm_iter = list(imgpoints_tm)
    imgp_rs_iter = list(imgpoints_rs)
    iteration = 0
    removed = []

    while iteration < 10:
        iteration += 1
        result = cv2.stereoCalibrate(
            objp_iter, imgp_tm_iter, imgp_rs_iter,
            K_tm, dist_tm, K_rs, dist_rs,
            imageSize_tm,
            R=R, T=T,
            flags=flags,
            criteria=criteria,
        )
        # cv2.stereoCalibrate 返回值数量因 OpenCV 版本而异
        if len(result) == 10:
            ret, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F, perViewErr = result
        else:
            ret, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = result
            perViewErr = None
        if perViewErr is None:
            print("  当前 OpenCV 版本不支持逐帧误差，停止迭代")
            break
        per_err = np.array(perViewErr)
        bad = per_err > 2.0
        mean_err = float(per_err.mean())
        print(f"  迭代 {iteration}: 帧数={len(objp_iter)}, 平均误差={mean_err:.4f}px, 剔除={bad.sum()}")

        if not bad.any():
            print("  收敛")
            break
        if len(objp_iter) - bad.sum() < 10:
            print("  帧数不足，停止")
            break

        for idx in reversed(np.where(bad)[0]):
            removed.append(float(per_err[idx]))
            del objp_iter[idx]; del imgp_tm_iter[idx]; del imgp_rs_iter[idx]

    # 5. 极线误差
    print("\n  计算极线误差...")
    epipolar_errors = []
    for ip_tm, ip_rs in zip(imgp_tm_iter, imgp_rs_iter):
        for pt_tm, pt_rs in zip(ip_tm.squeeze(), ip_rs.squeeze()):
            line = cv2.computeCorrespondEpilines(
                np.array([[float(pt_tm[0]), float(pt_tm[1])]], dtype=np.float32), 1, F)
            err = abs(line[0, 0, 0] * pt_rs[0] + line[0, 0, 1] * pt_rs[1] + line[0, 0, 2])
            epipolar_errors.append(err)
    epipolar_err_mean = float(np.mean(epipolar_errors))
    epipolar_err_std = float(np.std(epipolar_errors))
    print(f"  极线误差: {epipolar_err_mean:.4f} +/- {epipolar_err_std:.4f} px")

    # 6. 保存结果
    R_mat = R.tolist()
    T_vec = T.ravel().tolist()
    rvec = cv2.Rodrigues(R)[0].ravel().tolist()

    result = {
        "camera1": "tianmou",
        "camera2": "realsense_color",
        "resolution1": {"width": 640, "height": 320},
        "resolution2": {"width": 640, "height": 480},
        "K1": K_tm.tolist(),
        "dist1": dist_tm.tolist(),
        "K2": K_rs.tolist(),
        "dist2": dist_rs.tolist(),
        "R": R_mat,
        "T": T_vec,
        "rvec": rvec,
        "E": E.tolist(),
        "F": F.tolist(),
        "reproj_error_stereo": float(ret),
        "epipolar_error_mean": epipolar_err_mean,
        "epipolar_error_std": epipolar_err_std,
        "n_frames": len(objp_iter),
        "n_frames_initial": len(objpoints),
        "n_frames_removed": len(removed),
        "square_size_mm": SQUARE_SIZE,
    }
    save_json(result, os.path.join(OUTPUT, "extrinsic_tianmou_realsense.json"))
    print(f"\n  已保存: {OUTPUT}/extrinsic_tianmou_realsense.json")

    # 7. 打印外参
    print("\n" + "="*60)
    print("  外参标定结果")
    print("="*60)
    print(f"  R (RealSense->Tianmou):\n{R}")
    print(f"  T: {T_vec}")
    print(f"  rvec (Rodrigues): {rvec}")
    print(f"\n  T的模: {np.linalg.norm(T_vec):.2f} mm")
    print(f"  极线误差: {epipolar_err_mean:.4f} +/- {epipolar_err_std:.4f} px")


if __name__ == "__main__":
    main()
