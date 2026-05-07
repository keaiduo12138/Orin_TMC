#!/usr/bin/env python3
"""
calib/run_calibration.py
========================
一键运行完整标定流程（分步骤，可选跳过中间步骤）。

使用方式:
    # 完整流程（全部重跑）
    python3 run_calibration.py \
        --data-dir /projects/calib_data/0325_calibration \
        --bag /projects/cxr_data/2026-3-24/17-26-26.bag \
        --output ./calibration_output

    # 跳过内参标定，只跑外参（已有内参结果时）
    python3 run_calibration.py \
        --data-dir /projects/calib_data/0325_calibration \
        --bag /projects/cxr_data/2026-3-24/17-26-26.bag \
        --skip-tm-intrinsic --skip-rs-compare \
        --output ./calibration_output

步骤:
    step1  天眸内参标定  → intrinsic_tianmou.json
    step2  RS 内参对比   → compare_realsense_intrinsics.json
    step3  双目外参标定  → extrinsic_tianmou_realsense.json
    step4  验证          → validation/validation_result.json
"""

import os
import sys
import argparse
import subprocess
import time

sys.path.insert(0, os.path.dirname(__file__))


def run_step(name: str, cmd: list, output_dir: str, skip: bool = False) -> dict:
    """运行单个标定步骤。"""
    print(f"\n{'#'*60}")
    print(f"#  {name}")
    print(f"{'#'*60}")
    t0 = time.time()
    if skip:
        print("  [跳过]")
        return {"skipped": True, "elapsed": 0}
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    success = result.returncode == 0
    print(f"\n  {'✓' if success else '✗'} {name} {'跳过' if skip else ('成功' if success else '失败')} "
          f"({elapsed:.1f}s)")
    return {"success": success, "elapsed": elapsed, "skipped": skip}


def main():
    parser = argparse.ArgumentParser(description="一键运行完整标定流程")
    parser.add_argument("--data-dir", "-d", required=True)
    parser.add_argument("--bag", "-b", default=None)
    parser.add_argument("--output", "-o", default="./calibration_output")
    parser.add_argument("--skip-tm-intrinsic", action="store_true",
                        help="跳过天眸内参标定（使用已有 intrinsic_tianmou.json）")
    parser.add_argument("--skip-rs-compare", action="store_true",
                        help="跳过 RS 内参对比（使用 factory 内参）")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过验证步骤")
    parser.add_argument("--tm-intrinsic", default=None,
                        help="指定天眸内参 JSON（默认: <output>/intrinsic_tianmou.json）")
    parser.add_argument("--rs-source", default="factory",
                        choices=["factory", "calibrated", "default"])
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--select-frames", action="store_true")
    args = parser.parse_args()

    od = args.output
    os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(od, "visualization"), exist_ok=True)

    # ── Step 1: 天眸内参 ─────────────────────────────────────
    step1 = run_step(
        "Step 1: 天眸内参标定",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "calibrate_tianmou_intrinsics.py"),
            "--data-dir", args.data_dir,
            "--output", od,
        ] + (["--select-frames"] if args.select_frames else []) +
           (["--frames"] + [str(f) for f in args.frames] if args.frames else []),
        od, args.skip_tm_intrinsic)

    # ── Step 2: RS 内参对比 ─────────────────────────────────
    step2 = run_step(
        "Step 2: RealSense 出厂 vs 实测内参对比",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "compare_realsense_intrinsics.py"),
            "--data-dir", args.data_dir,
            "--bag", args.bag or "",
            "--output", od,
        ] + (["--frames"] + [str(f) for f in args.frames] if args.frames else []),
        od, args.skip_rs_compare)

    # ── Step 3: 双目外参 ─────────────────────────────────────
    tm_intrinsic = args.tm_intrinsic or os.path.join(od, "intrinsic_tianmou.json")
    if args.skip_tm_intrinsic and not os.path.exists(tm_intrinsic):
        print(f"\n  [警告] 天眸内参文件 {tm_intrinsic} 不存在，改用 TM_K_REF")
        tm_intrinsic = "ref"

    step3 = run_step(
        "Step 3: 双目外参标定（TM ↔ RS）",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "calibrate_stereo_extrinsics.py"),
            "--data-dir", args.data_dir,
            "--tm-intrinsic", tm_intrinsic,
            "--rs-source", args.rs_source,
            "--output", od,
        ] + (["--select-frames"] if args.select_frames else []) +
           (["--frames"] + [str(f) for f in args.frames] if args.frames else []),
        od, False)

    # ── Step 4: 验证 ─────────────────────────────────────────
    step4 = run_step(
        "Step 4: 重投影误差评定 + 极线几何验证",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "validate.py"),
            "--data-dir", args.data_dir,
            "--extrinsic", os.path.join(od, "extrinsic_tianmou_realsense.json"),
            "--output", os.path.join(od, "validation"),
        ],
        od, args.skip_validation or not step3["success"])

    # ── 汇总报告 ───────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("#  标定完成汇总")
    print(f"{'#'*60}")
    total = sum(s.get("elapsed", 0) for s in [step1, step2, step3, step4])
    print(f"  总耗时: {total:.1f}s")
    print(f"\n  步骤状态:")
    for name, s in [("天眸内参", step1), ("RS内参对比", step2),
                    ("双目外参", step3), ("验证", step4)]:
        status = "跳过" if s["skipped"] else ("✓" if s["success"] else "✗")
        print(f"    {name}: {status}")
    print(f"\n  输出文件:")
    for f in ["intrinsic_tianmou.json",
              "compare_realsense_intrinsics.json",
              "extrinsic_tianmou_realsense.json"]:
        path = os.path.join(od, f)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"    {exists} {path}")
    val_path = os.path.join(od, "validation", "validation_result.json")
    print(f"    {'✓' if os.path.exists(val_path) else '✗'} {val_path}")


if __name__ == "__main__":
    main()
