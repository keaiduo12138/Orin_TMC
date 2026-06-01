#!/usr/bin/env python3
"""
batch_process.py — Process a single dataset through the full TianMouCV pipeline.

Usage:
    python batch_process.py <dataset_name>

Example:
    python batch_process.py 0530_extreme1

The script derives paths automatically:
    Source root  : /projects/<dataset_name>
    JSON file    : <base_workdir>/output_<dataset_name>.json
    Output dir   : /projects/calib_data/output_<dataset_name>
"""

import os
import sys
import subprocess
import argparse


def run_step(cmd, step_name, cwd=None):
    """Execute a shell command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"CMD : {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Step '{step_name}' failed with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)
    print(f"[OK] Step '{step_name}' completed.")


def main():
    parser = argparse.ArgumentParser(description="Process a single dataset through the full pipeline.")
    parser.add_argument("dataset_name", help="Dataset folder name, e.g. 0530_extreme1")
    parser.add_argument("--workdir", default="/home/nvidia/Desktop/cxr_multi_sensor/yyf_new",
                        help="Working directory where post_process.py and export_aligned_frames.py live")
    parser.add_argument("--calib-script", default="calib/depth_0408.py",
                        help="Relative path to the calibration script")
    parser.add_argument("--calib-output-root", default="/projects/calib_data",
                        help="Root directory for calibrated output")
    args = parser.parse_args()

    dataset = args.dataset_name

    # Derive paths from dataset name
    source_root = f"/projects/{dataset}"
    json_file = f"output_{dataset}.json"
    json_path = os.path.join(args.workdir, json_file)
    output_dir = f"{args.calib_output_root}/output_{dataset}"

    print(f"\n{'#'*60}")
    print(f"# Processing dataset: {dataset}")
    print(f"#   Source root : {source_root}")
    print(f"#   JSON file   : {json_path}")
    print(f"#   Output dir  : {output_dir}")
    print(f"{'#'*60}")

    # ------------------------------------------------------------------
    # Step 1 — Create output directory (sudo mkdir + chmod)
    # ------------------------------------------------------------------
    subprocess.run(["sudo", "mkdir", "-p", output_dir], check=True)
    subprocess.run(["sudo", "chmod", "777", output_dir], check=True)
    print(f"[OK] Output directory ready: {output_dir}")

    # ------------------------------------------------------------------
    # Step 2 — Generate aligned index
    # ------------------------------------------------------------------
    run_step([
        "python3", "post_process.py",
        "--root", source_root,
        "-o", json_path,
    ], "Generate aligned index (post_process.py)", cwd=args.workdir)

    # ------------------------------------------------------------------
    # Step 3 — Export time-aligned frames
    # ------------------------------------------------------------------
    run_step([
        "python3", "export_aligned_frames.py",
        "--root", source_root,
        "--index", json_path,
        "--output", output_dir,
        "--use-raw-depth",
    ], "Export time-aligned frames (export_aligned_frames.py)", cwd=args.workdir)

    # ------------------------------------------------------------------
    # Step 4 — Export spatially calibrated frames
    # ------------------------------------------------------------------
    run_step([
        "python3", args.calib_script,
        "--base", output_dir,
        "--index", json_path,
    ], "Export calibrated frames (depth_0408.py)", cwd=args.workdir)

    print(f"\n{'='*60}")
    print(f"[DONE] Dataset '{dataset}' fully processed.")
    print(f"       Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
