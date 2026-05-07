#!/usr/bin/env python3
"""
calib/run_calibration.py
========================
一键运行完整标定流程（支持单数据集和多数据集联合标定）。

流程（4步，流水线式执行）：
    step0  RS Color 裁剪  →  color_cropped/  （仅当提供 --rs-raw-dirs 时自动执行）
    step1  天眸内参标定  →  intrinsic_tianmou.json
    step2  RS 内参标定   →  intrinsic_realsense_cropped.json
    step3  双目外参标定  →  extrinsic_tianmou_realsense.json
    step4  验证          →  validation/validation_result.json

使用方式:
    # ── 单数据集 ─────────────────────────────────────────────
    python3 run_calibration.py \
        --tm-data-dirs /projects/calib_data/dataset_A/tianmou/ \
        --rs-data-dirs /projects/calib_data/dataset_A/color_cropped/ \
        --rs-raw-dirs /projects/calib_data/dataset_A/color/ \
        -o ./calibration_output

    # ── 多数据集联合（同一安装、两段采集）────────────────────
    python3 run_calibration.py \
        --tm-data-dirs /projects/calib_data/A/tianmou/ /projects/calib_data/B/tianmou/ \
        --rs-data-dirs /projects/calib_data/A/color/   /projects/calib_data/B/color/ \
        --rs-raw-dirs /projects/calib_data/A/color/    /projects/calib_data/B/color/ \
        --sample-steps 5 8 \
        -o ./calibration_output

注意：
    - 天眸图像：提供预处理后的目录（640×320，已水平镜像）
    - RS 图像：默认需要先裁剪到 640×320（顶部80px + 底部80px）
      用 --rs-raw-dirs 提供原始 640×480 图像，脚本会自动裁剪
      用 --rs-data-dirs 直接提供已裁剪的 640×320 图像（跳过裁剪）
    - 多数据集时，所有目录数量必须一致
    - 外参标定默认使用 RS 裁剪后实测内参（intrinsic_realsense_cropped.json）
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
from glob import glob

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
    status = "跳过" if skip else ("成功" if success else "失败")
    print(f"\n  {'✓' if success else '✗'} {name} {status} ({elapsed:.1f}s)")
    return {"success": success, "elapsed": elapsed, "skipped": skip}


def crop_realsense_dirs(raw_dirs: list, cropped_dirs: list) -> bool:
    """对多个 RS 原始目录执行裁剪。"""
    print(f"\n{'='*60}")
    print(f"  Step 0: RS Color 裁剪（640×480 → 640×320）")
    print(f"{'='*60}")
    all_ok = True
    for raw_dir, cropped_dir in zip(raw_dirs, cropped_dirs):
        print(f"\n  裁剪: {raw_dir}")
        print(f"  → {cropped_dir}")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "crop_color_frames.py"),
            "--input", raw_dir,
            "--output", cropped_dir,
            "--crop-top", "80",
            "--crop-bottom", "80",
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ✗ 裁剪失败: {raw_dir}")
            all_ok = False
        else:
            print(f"  ✓ 裁剪完成")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="一键运行完整标定流程（支持多数据集联合）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 多数据集目录参数 ────────────────────────────────────
    grp = parser.add_argument_group("输入数据目录（必需）")
    grp.add_argument("--tm-data-dirs", "-t", nargs="+", required=True,
                    help="天眸图像目录列表（支持多数据集，空格分隔）。"
                         "天眸图像应已是预处理后的图像（640×320，已水平镜像）。")
    grp.add_argument("--rs-data-dirs", "-r", nargs="+", required=True,
                    help="RS Color 图像目录列表（支持多数据集）。"
                         "图像应为 640×320（已裁剪），用于外参标定。"
                         "若提供 --rs-raw-dirs，此处填裁剪后的目录。")
    grp.add_argument("--rs-raw-dirs", nargs="+", default=None,
                    help="RS Color 原始图像目录列表（640×480，多数据集）。"
                         "提供此参数时，脚本会自动裁剪到 640×320。"
                         "每个目录对应 --rs-data-dirs 中相同索引的目录（裁剪目标）。")

    # ── 抽样参数 ────────────────────────────────────────────
    grp2 = parser.add_argument_group("抽样控制")
    grp2.add_argument("--sample-steps", type=int, nargs="+", default=None,
                     help="每个数据集的抽样步长（默认各为5）。"
                          "例如 --sample-steps 5 8 表示数据集A每5帧取1、数据集B每8帧取1。")

    # ── 输出 ───────────────────────────────────────────────
    grp3 = parser.add_argument_group("输出控制")
    grp3.add_argument("--output", "-o", default="./calibration_output",
                     help="输出目录（默认 ./calibration_output）")
    grp3.add_argument("--skip-tm-intrinsic", action="store_true",
                     help="跳过天眸内参标定（使用已有 intrinsic_tianmou.json）")
    grp3.add_argument("--skip-rs-intrinsic", action="store_true",
                     help="跳过 RS 内参标定（使用已有 intrinsic_realsense_cropped.json）")
    grp3.add_argument("--skip-validation", action="store_true",
                     help="跳过验证步骤")
    grp3.add_argument("--tm-intrinsic", default=None,
                     help="指定天眸内参 JSON（默认: <output>/intrinsic_tianmou.json）")

    args = parser.parse_args()

    # ── 参数规范化 & 校验 ───────────────────────────────────
    tm_dirs = args.tm_data_dirs
    rs_dirs = args.rs_data_dirs
    n_datasets = len(tm_dirs)

    if len(rs_dirs) != n_datasets:
        print(f"错误: --tm-data-dirs({n_datasets}) 与 --rs-data-dirs({len(rs_dirs)}) 数量不一致")
        return

    if args.rs_raw_dirs is not None:
        if len(args.rs_raw_dirs) != n_datasets:
            print(f"错误: --rs-raw-dirs({len(args.rs_raw_dirs)}) 与 --tm-data-dirs({n_datasets}) 数量不一致")
            return
        raw_dirs = args.rs_raw_dirs
        cropped_dirs = rs_dirs  # 裁剪目标覆盖 rs_dirs
    else:
        raw_dirs = None
        cropped_dirs = None

    if args.sample_steps is None:
        sample_steps = [5] * n_datasets
    elif len(args.sample_steps) == 1:
        sample_steps = args.sample_steps * n_datasets
    elif len(args.sample_steps) != n_datasets:
        print(f"错误: --sample-steps 数量({len(args.sample_steps)}) 必须等于数据集数量({n_datasets})")
        return
    else:
        sample_steps = args.sample_steps

    # 构建 sample_steps 参数（空格分隔字符串）
    sample_steps_str = " ".join(str(s) for s in sample_steps)

    od = args.output
    os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(od, "visualization"), exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"#  标定配置")
    print(f"{'#'*60}")
    print(f"  输出目录: {od}")
    print(f"  数据集数量: {n_datasets}")
    print(f"  抽样步长: {sample_steps}")
    for i in range(n_datasets):
        raw_note = f"  (原始: {raw_dirs[i]})" if raw_dirs else ""
        print(f"  数据集{i+1}:")
        print(f"    天眸: {tm_dirs[i]}")
        print(f"    RS:   {rs_dirs[i]}{raw_note}")

    # ── Step 0: RS 裁剪（可选）──────────────────────────────
    if raw_dirs is not None:
        crop_ok = crop_realsense_dirs(raw_dirs, cropped_dirs)
        if not crop_ok:
            print("\n⚠ RS 裁剪失败，继续使用已有目录（若存在）...")

    # ── Step 1: 天眸内参 ─────────────────────────────────────
    tm_dirs_arg = " ".join(f'"{d}"' for d in tm_dirs)
    step1 = run_step(
        "Step 1: 天眸内参标定（多数据集联合）",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "calibrate_tianmou_intrinsics.py"),
            "--data-dirs",
        ] + tm_dirs + [
            "--sample-steps",
        ] + sample_steps +
        [
            "--output", od,
        ],
        od, args.skip_tm_intrinsic)

    # ── Step 2: RS 内参（裁剪后实测）─────────────────────────
    rs_dirs_arg = " ".join(f'"{d}"' for d in rs_dirs)
    step2 = run_step(
        "Step 2: RS 内参标定（裁剪后实测，多数据集联合）",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "calibrate_realsense_cropped.py"),
            "--data-dirs",
        ] + rs_dirs + [
            "--sample-steps",
        ] + sample_steps +
        [
            "--output", od,
        ],
        od, args.skip_rs_intrinsic)

    # ── Step 3: 双目外参 ─────────────────────────────────────
    tm_intrinsic = args.tm_intrinsic or os.path.join(od, "intrinsic_tianmou.json")
    if args.skip_tm_intrinsic and not os.path.exists(tm_intrinsic):
        print(f"\n  [警告] 天眸内参文件 {tm_intrinsic} 不存在，改用 TM_K_REF")
        tm_intrinsic = "ref"

    step3 = run_step(
        "Step 3: 双目外参标定（TM ↔ RS，多数据集联合）",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "calibrate_stereo_extrinsics.py"),
            "--tm-data-dirs",
        ] + tm_dirs + [
            "--rs-data-dirs",
        ] + rs_dirs + [
            "--sample-steps",
        ] + sample_steps +
        [
            "--tm-intrinsic", tm_intrinsic,
            "--rs-source", "calibrated",
            "--output", od,
        ],
        od, False)

    # ── Step 4: 验证 ─────────────────────────────────────────
    validate_extra_args = []
    if n_datasets > 1 or args.tm_data_dirs[0] != args.rs_data_dirs[0]:
        # 多数据集或分离目录模式：传分离目录参数
        validate_extra_args = ["--tm-data-dirs"] + tm_dirs + ["--rs-data-dirs"] + rs_dirs
    else:
        # 单数据集统一目录模式：传旧版参数
        validate_extra_args = ["--data-dir", tm_dirs[0]]

    step4 = run_step(
        f"Step 4: 重投影误差评定 + 极线几何验证（{n_datasets}个数据集）",
        [
            sys.executable, os.path.join(os.path.dirname(__file__), "validate.py"),
            "--extrinsic", os.path.join(od, "extrinsic_tianmou_realsense.json"),
            "--output", os.path.join(od, "validation"),
        ] + validate_extra_args,
        od, args.skip_validation or not step3["success"])

    # ── 汇总报告 ───────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("#  标定完成汇总")
    print(f"{'#'*60}")
    total = sum(s.get("elapsed", 0) for s in [step1, step2, step3, step4])
    print(f"  总耗时: {total:.1f}s")
    print(f"\n  步骤状态:")
    for name, s in [("天眸内参", step1), ("RS内参", step2),
                    ("双目外参", step3), ("验证", step4)]:
        status = "跳过" if s["skipped"] else ("✓ 成功" if s["success"] else "✗ 失败")
        print(f"    {name}: {status}")
    print(f"\n  输出文件:")
    for f in ["intrinsic_tianmou.json",
              "intrinsic_realsense_cropped.json",
              "extrinsic_tianmou_realsense.json"]:
        path = os.path.join(od, f)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"    {exists} {path}")
    val_path = os.path.join(od, "validation", "validation_result.json")
    print(f"    {'✓' if os.path.exists(val_path) else '✗'} {val_path}")


if __name__ == "__main__":
    main()
