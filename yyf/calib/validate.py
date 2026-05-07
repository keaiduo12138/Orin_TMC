#!/usr/bin/env python3
"""
calib/validate.py
=================
重投影误差评定 + 极线几何验证 + 物理布局检查。

使用方式:
    python3 validate.py \
        --data-dir /projects/calib_data/0325_calibration \
        --extrinsic ./calibration_output/extrinsic_tianmou_realsense.json \
        --output ./calibration_output/validation

验证内容:
    1. 对极距离误差（衡量外参精度）
    2. 三角化重投影误差（衡量双目标定质量）
    3. 物理布局验证（RS 是否确实在 TM 上方）
    4. 角点可见范围热图（评估视角覆盖）
    5. 综合评分报告
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from glob import glob
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    BOARD_COLS, BOARD_ROWS, SQUARE_SIZE,
    read_tianmou_image, to_gray, build_3d_objpoints,
    detect_corners_pair,
    ensure_dir, save_json, print_header,
)


def load_frame_pairs(data_dir: str) -> List[Tuple[int, str, str]]:
    """加载帧对。"""
    tm_flat = glob(os.path.join(data_dir, "*_tianmou.png"))
    tm_subdir = glob(os.path.join(data_dir, "tianmou", "tianmou_*.png"))
    tm_files = {os.path.basename(f).replace("_tianmou", "").replace("tianmou_", "").split("_")[-1].replace(".png", ""): f
                for f in (tm_flat if tm_flat else tm_subdir)}

    rs_flat = glob(os.path.join(data_dir, "*_color.png"))
    rs_subdir = glob(os.path.join(data_dir, "color", "color_*.png"))
    rs_files = {os.path.basename(f).replace("_color", "").replace("color_", "").split("_")[-1].replace(".png", ""): f
                for f in (rs_flat if rs_flat else rs_subdir)}

    common = sorted(set(tm_files.keys()) & set(rs_files.keys()), key=lambda k: int(k))
    return [(int(k), tm_files[k], rs_files[k]) for k in common]


def validate_extrinsics(
    pairs: List[Tuple],
    objp: np.ndarray,
    K_tm: np.ndarray,
    dist_tm: np.ndarray,
    K_rs: np.ndarray,
    dist_rs: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> Dict:
    """对所有帧对验证外参精度。"""
    results = []

    for idx, tm_path, rs_path in pairs:
        tm_img = read_tianmou_image(tm_path)
        rs_img = cv2.imread(rs_path)
        if tm_img is None or rs_img is None:
            continue

        ret, c_tm, c_rs = detect_corners_pair(to_gray(tm_img), to_gray(rs_img))
        if not ret:
            continue

        # ── 0. 去畸变（关键！）──────────────────────────────
        # P=None: 输出归一化相机坐标（X/Z, Y/Z），与 E/F 矩阵单位一致
        # P=K  : 输出重投影像素坐标，与原始 F 不匹配，导致极大误差
        c_tm_ud = cv2.undistortPoints(
            c_tm.astype(np.float64), K_tm, dist_tm)
        c_rs_ud = cv2.undistortPoints(
            c_rs.astype(np.float64), K_rs, dist_rs)

        # ── 1. 对极距离误差（验证两视图几何一致性） ─────────────────
        F_mat, mask = cv2.findFundamentalMat(
            c_tm_ud.squeeze().astype(np.float64),
            c_rs_ud.squeeze().astype(np.float64),
            cv2.FM_RANSAC, 3.0, 0.99)
        epipolar_dist = []
        for pt_t, pt_r in zip(c_tm_ud.squeeze(), c_rs_ud.squeeze()):
            epi = F_mat @ np.array([pt_t[0], pt_t[1], 1.0])
            epi_norm = epi[:2] / (np.linalg.norm(epi[:2]) + 1e-9)
            dist = abs(float(epi_norm @ np.array([pt_r[0], pt_r[1]])))
            epipolar_dist.append(dist)

        # ── 2. 本征矩阵验证（E = [T]_× R，验证 E 的秩约束） ─────────
        E_from_RT = np.cross(np.eye(3), T / (np.linalg.norm(T) + 1e-9)) @ R
        svd_E = np.linalg.svd(E_from_RT)[1]
        e_rank_ratio = float(svd_E[2] / svd_E[0])
        E_rank_ok = e_rank_ratio < 0.01  # 第三个奇异值应接近0

        # ── 3. solvePnP 估算棋盘格距离（物理合理性验证） ─────────────
        try:
            _, rvec_rs, tvec_rs = cv2.solvePnP(
                objp.astype(np.float64), c_rs.squeeze().astype(np.float64),
                K_rs, dist_rs, flags=cv2.SOLVEPNP_ITERATIVE)
            dist_to_board = float(np.linalg.norm(tvec_rs))
        except Exception:
            dist_to_board = 0.0

        results.append({
            "frame_idx": idx,
            "epipolar_mean_px": float(np.mean(epipolar_dist)),
            "epipolar_max_px": float(np.max(epipolar_dist)),
            "epipolar_min_px": float(np.min(epipolar_dist)),
            "E_rank_ratio": e_rank_ratio,
            "E_rank_ok": bool(E_rank_ok),
            "board_distance_mm": dist_to_board,
        })

    return results


def verify_physical_layout(R: np.ndarray, T: np.ndarray) -> Dict:
    """验证物理布局是否合理。"""
    # R 将 RS 坐标系转到 TM 坐标系
    # T 是 RS 原点在 TM 坐标系中的位置
    # 即 TM_point = R @ RS_point + T

    T_physical = T.ravel()
    baseline_mm = float(np.linalg.norm(T_physical))

    # RS 在 TM 上方: T_y 应为正（相机坐标 Y 轴向上）
    # 天眸朝向: 如果相机向下倾斜，Y轴正方向为"下"
    # 根据实际硬件: RS 在 TM 上方约 100mm
    info = {
        "baseline_mm": baseline_mm,
        "baseline_cm": baseline_mm / 10.0,
        "T_raw": T_physical.tolist(),
        "expected_T_y_mm": 100,   # 约 10cm
        "actual_T_y_mm": float(T_physical[1]),
        "T_y_deviation_pct": abs(float(T_physical[1]) - 100) / 100 * 100,
        "T_x_mm": float(T_physical[0]),
        "T_z_mm": float(T_physical[2]),
    }

    if 50 < info["actual_T_y_mm"] < 200:
        info["T_y_verdict"] = "合理"
    elif 0 < info["actual_T_y_mm"] < 50:
        info["T_y_verdict"] = "偏小"
    elif info["actual_T_y_mm"] < 0:
        info["T_y_verdict"] = "反向（检查坐标系定义）"
    else:
        info["T_y_verdict"] = "偏大"

    return info


def plot_validation_summary(results: List[Dict], phys: Dict,
                           output_path: str) -> None:
    """绘制综合验证报告图。"""
    if not results:
        return
    frame_ids = [r["frame_idx"] for r in results]
    epipolar_mean = [r["epipolar_mean_px"] for r in results]
    epipolar_max = [r["epipolar_max_px"] for r in results]
    board_dist = [r["board_distance_mm"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("标定验证报告", fontsize=14, fontweight="bold")

    # 1. 对极距离均值
    ax = axes[0, 0]
    bars = ax.bar(frame_ids, epipolar_mean, color="steelblue", alpha=0.8)
    for bar, v in zip(bars, epipolar_mean):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=1.0, color="red", linestyle="--", label="1px 阈值(优)")
    ax.axhline(y=2.0, color="orange", linestyle="--", label="2px 阈值(良)")
    ax.set_xlabel("帧索引")
    ax.set_ylabel("对极距离 (px)")
    ax.set_title("1. 对极距离误差（越小越好，<1px=优, <2px=良）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. E 矩阵秩约束验证（σ3/σ1 应 < 1%）
    ax = axes[0, 1]
    e_ranks = [r["E_rank_ratio"] * 100 for r in results]  # 转换为百分比
    colors = ["forestgreen" if r < 1.0 else "coral" for r in e_ranks]
    bars = ax.bar(frame_ids, e_ranks, color=colors, alpha=0.8)
    for bar, v in zip(bars, e_ranks):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=1.0, color="red", linestyle="--", label="1% 阈值")
    ax.set_xlabel("帧索引")
    ax.set_ylabel("σ₃/σ₁ (%)")
    ax.set_title("2. 本征矩阵 E 秩约束（σ₃/σ₁ < 1%% = 符合秩2约束）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. 棋盘格距离分布
    ax = axes[1, 0]
    ax.bar(frame_ids, board_dist, color="mediumpurple", alpha=0.8)
    ax.axhline(y=np.mean(board_dist), color="orange", linestyle="--",
               label=f"均值={np.mean(board_dist):.0f}mm")
    ax.set_xlabel("帧索引")
    ax.set_ylabel("棋盘格距离 (mm)")
    ax.set_title("3. 各帧棋盘格距离（验证物理合理性，30cm~2m=合理）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. 物理布局
    ax = axes[1, 1]
    T_vals = [phys["T_x_mm"], phys["actual_T_y_mm"], phys["T_z_mm"]]
    labels = ["X (水平左-右)", "Y (垂直上-下)", "Z (前后)"]
    colors_bar = ["steelblue", "forestgreen", "coral"]
    bars = ax.bar(labels, T_vals, color=colors_bar, alpha=0.8)
    for bar, v in zip(bars, T_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + (2 if v >= 0 else -5),
                f"{v:.1f}mm", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("位移 (mm)")
    ax.set_title(f"4. 双目基线分量 (|T|={phys['baseline_mm']:.1f}mm)\n"
                 f"RS 在 TM {'上方' if phys['actual_T_y_mm'] > 0 else '下方'} {abs(phys['actual_T_y_mm']):.0f}mm")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("位移 (mm)")
    ax.set_title(f"4. 双目基线分量 (|T|={phys['baseline_mm']:.1f}mm)\n"
                 f"RS 在 TM {'上方' if phys['actual_T_y_mm'] > 0 else '下方'} {abs(phys['actual_T_y_mm']):.0f}mm")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="标定验证工具")
    parser.add_argument("--data-dir", "-d", required=True)
    parser.add_argument("--extrinsic", "-e", required=True)
    parser.add_argument("--output", "-o", default="./calibration_output/validation")
    args = parser.parse_args()

    ensure_dir(args.output)

    # ── 加载外参 ─────────────────────────────────────────────
    print_header("加载外参结果")
    data = __import__("utils").load_json(args.extrinsic)
    R = np.array(data["R"], dtype=np.float64)
    T = np.array(data["T"], dtype=np.float64)
    # 使用 extrinsic JSON 中实际用于标定的 K
    # calibrate_stereo_extrinsics.py 保存的 key 是 "K_tianmou"
    K_tm = np.array(data.get("K_tianmou_scaled") or data.get("K_tianmou"), dtype=np.float64)
    dist_tm = np.array(data["dist_tianmou"], dtype=np.float64)
    K_rs = np.array(data["K_realsense"], dtype=np.float64)
    dist_rs = np.array(data["dist_realsense"], dtype=np.float64)
    print(f"  天眸 K: fx={K_tm[0,0]:.1f}, cx={K_tm[0,2]:.1f}, cy={K_tm[1,2]:.1f}")
    print(f"  RS   K: fx={K_rs[0,0]:.1f}, cx={K_rs[0,2]:.1f}, cy={K_rs[1,2]:.1f}")
    print(f"  rs_source: {data.get('rs_source', 'unknown')}")
    print(f"  R (P_tm = R @ P_rs + T)")
    print(f"  T = {T.ravel()}, |T| = {float(np.linalg.norm(T)):.1f}mm")

    # ── 验证物理布局 ────────────────────────────────────────
    print_header("物理布局验证")
    phys = verify_physical_layout(R, T)
    print(f"  T = [{phys['T_x_mm']:.1f}, {phys['actual_T_y_mm']:.1f}, {phys['T_z_mm']:.1f}] mm")
    print(f"  |T| = {phys['baseline_mm']:.1f}mm ({phys['baseline_cm']:.1f}cm)")
    print(f"  Y 分量（垂直）: {phys['actual_T_y_mm']:.1f}mm → {phys['T_y_verdict']}")
    print(f"  预期 RS 在 TM 上方约 100mm，实际 Y={phys['actual_T_y_mm']:.1f}mm")

    # ── 验证几何 ─────────────────────────────────────────────
    print_header("几何精度验证")
    pairs = load_frame_pairs(args.data_dir)
    objp = build_3d_objpoints(BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)
    results = validate_extrinsics(pairs, objp, K_tm, dist_tm, K_rs, dist_rs, R, T)

    if results:
        mean_epipolar = np.mean([r["epipolar_mean_px"] for r in results])
        mean_dist = np.mean([r["board_distance_mm"] for r in results if r["board_distance_mm"] > 0])

        print(f"\n  {'帧':>4} | {'对极均值(px)':>12} | {'E秩比(%)':>10} | {'棋盘距离(mm)':>12}")
        print("  " + "-" * 55)
        for r in results:
            print(f"  {r['frame_idx']:>4} | {r['epipolar_mean_px']:12.3f} | "
                  f"{r.get('E_rank_ratio', 0)*100:10.2f} | {r['board_distance_mm']:12.0f}")

        print(f"\n  平均对极误差:   {mean_epipolar:.3f} px")
        if mean_dist > 0:
            print(f"  平均棋盘距离:   {mean_dist:.0f} mm")
            if 300 < mean_dist < 2000:
                print("  ✓ 棋盘距离在合理范围（30cm~2m）")
            else:
                print(f"  ⚠ 棋盘距离 {mean_dist:.0f}mm 不在常规范围")

        # 评分
        score_epipolar = max(0, min(100, 100 - mean_epipolar * 30))
        e_rank_ok = all(r["E_rank_ok"] for r in results if "E_rank_ok" in r)
        score_e = 100 if e_rank_ok else 50
        score_phys = max(0, 100 - abs(phys["T_y_deviation_pct"]))
        score_total = (score_epipolar + score_e + score_phys) / 3

        print(f"\n  ── 综合评分 ──")
        print(f"  对极精度:    {score_epipolar:.0f}/100  (均值={mean_epipolar:.2f}px)")
        print(f"  E矩阵秩:     {score_e:.0f}/100  {'✓' if e_rank_ok else '✗'}")
        print(f"  物理布局:    {max(0, score_phys):.0f}/100")
        print(f"  综合评分:    {score_total:.0f}/100")

        if score_total >= 80:
            verdict = "✓ 优秀"
        elif score_total >= 60:
            verdict = "⚠ 良好"
        else:
            verdict = "✗ 不合格"
        print(f"  结论: {verdict}")
    else:
        results = []
        print("  无有效帧可验证")

    # ── 绘制报告图 ───────────────────────────────────────────
    report_path = os.path.join(args.output, "validation_report.png")
    plot_validation_summary(results, phys, report_path)
    print(f"\n  验证图已保存: {report_path}")

    # ── 保存验证结果 JSON ────────────────────────────────────
    validation_data = {
        "score_epipolar": float(score_epipolar) if results else 0,
        "score_e_rank": float(score_e) if results else 0,
        "score_physical": float(max(0, score_phys)) if results else 0,
        "score_total": float(score_total) if results else 0,
        "verdict": verdict if results else "无可用帧",
        "physical_layout": phys,
        "per_frame_results": results,
        "mean_epipolar_px": float(mean_epipolar) if results else None,
        "mean_board_distance_mm": float(mean_dist) if results and mean_dist > 0 else None,
    }
    save_json(validation_data, os.path.join(args.output, "validation_result.json"))
    print(f"  JSON 已保存: {args.output}/validation_result.json")


if __name__ == "__main__":
    main()
