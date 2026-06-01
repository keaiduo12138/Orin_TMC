#!/usr/bin/env python3
"""
run_batch.py — Parallel batch scheduler for TianMouCV processing.

Usage:
    conda activate tianmoucv
    python run_batch.py

This script:
  1. Reads dataset names from batch_datasets.yaml
  2. Distributes them across N parallel processes (N = CPU core count)
  3. Each process runs batch_process.py for one dataset independently
  4. Streams every line of output to the terminal in real-time
  5. Logs each dataset's output to logs/<dataset_name>.log
"""

import os
import sys
import yaml
import multiprocessing
import subprocess
import argparse
import threading
from datetime import datetime


DEFAULT_WORKDIR = "/home/nvidia/Desktop/cxr_multi_sensor/yyf_new"
DEFAULT_CONFIG = "batch_datasets.yaml"


def get_cpu_count():
    return multiprocessing.cpu_count()


def _stream_loop(proc, dataset_name, log_file, queue):
    """Read subprocess output line-by-line, push to queue + write log."""
    for raw in iter(proc.stdout.readline, ""):
        if not raw:
            break
        line = raw.rstrip("\n")
        queue.put((dataset_name, line))
        with open(log_file, "a") as lf:
            lf.write(line + "\n")


def _worker(dataset_name, workdir, calib_script, calib_output_root, log_dir, queue):
    """
    Per-process worker: runs batch_process.py for one dataset.
    Each call runs in its own process with its own subprocess.
    """
    log_file = os.path.join(log_dir, f"{dataset_name}.log")
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(workdir, "batch_process.py"),
        dataset_name,
        "--workdir", workdir,
        "--calib-script", calib_script,
        "--calib-output-root", calib_output_root,
    ]

    with open(log_file, "w") as f:
        f.write(f"[{datetime.now().isoformat()}] Starting: {' '.join(cmd)}\n")
        f.write(f"[{datetime.now().isoformat()}] Log: {log_file}\n")
        f.write("=" * 60 + "\n")

    proc = subprocess.Popen(
        cmd, cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    t = threading.Thread(target=_stream_loop, args=(proc, dataset_name, log_file, queue), daemon=True)
    t.start()
    proc.wait()
    t.join(timeout=2)

    exit_code = proc.returncode
    queue.put((dataset_name, f"[EXIT] code={exit_code}"))

    with open(log_file, "a") as lf:
        lf.write(f"[EXIT] code={exit_code}\n")


def main():
    parser = argparse.ArgumentParser(description="Run TianMouCV batch processing in parallel.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to batch_datasets.yaml")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR,
                        help="Working directory")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running anything")
    args = parser.parse_args()

    workdir = args.workdir
    with open(os.path.join(workdir, args.config)) as f:
        cfg = yaml.safe_load(f)

    datasets          = cfg.get("datasets", [])
    calib_script      = cfg.get("calib_script", "calib/depth_0408.py")
    calib_output_root = cfg.get("calib_output_root", "/projects/calib_data")
    log_dir           = os.path.join(workdir, "logs")

    if not datasets:
        print("[ERROR] No datasets found in batch_datasets.yaml")
        sys.exit(1)

    n = min(args.workers or get_cpu_count(), len(datasets))
    total = len(datasets)

    print(f"\n{'#'*60}")
    print(f"# TianMouCV Batch Processor")
    print(f"# Config    : {os.path.join(workdir, args.config)}")
    print(f"# Datasets  : {len(datasets)}")
    print(f"# Workers   : {n} / {get_cpu_count()} cores")
    print(f"# Workdir   : {workdir}")
    print(f"# Log dir   : {log_dir}")
    print(f"#{'#'*59}")
    for i, ds in enumerate(datasets, 1):
        print(f"  [{i:02d}] {ds}")
    print(f"{'#'*60}\n")

    if args.dry_run:
        print("[DRY RUN] Nothing executed."); return

    os.makedirs(log_dir, exist_ok=True)

    # ── Shared queue: workers push lines, main thread prints them ────
    queue = multiprocessing.Queue()

    # ── Print queue consumer (runs in main thread) ───────────────────
    done_count = 0
    done_lock  = threading.Lock()
    results    = []

    def consume_queue():
        nonlocal done_count
        while True:
            try:
                dataset_name, line = queue.get(timeout=0.3)
            except:
                if all_procs_done.value:
                    break
                continue

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"{ts} [{dataset_name}] {line}", flush=True)

            if line.startswith("[EXIT]"):
                code = int(line.split("=")[1])
                with done_lock:
                    done_count += 1
                    results.append((dataset_name, code))
                sep = "=" * 50
                if code == 0:
                    print(f"{sep} [{done_count}/{total}] {dataset_name}: OK\n", flush=True)
                else:
                    print(f"{sep} [{done_count}/{total}] {dataset_name}: FAILED (exit {code})\n", flush=True)

    # ── Sentinel: tells the consumer when all processes are done ──────
    all_procs_done = multiprocessing.Value("i", 0)

    consumer = threading.Thread(target=consume_queue, daemon=True)
    consumer.start()

    print(f"[INFO] Starting at {datetime.now().strftime('%H:%M:%S')}\n")
    print("# ── Live output ──\n")

    # ── Dispatch workers ─────────────────────────────────────────────
    processes = []
    for ds in datasets:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{ts} [{ds}] [START]", flush=True)
        p = multiprocessing.Process(
            target=_worker,
            args=(ds, workdir, calib_script, calib_output_root, log_dir, queue),
        )
        p.start()
        processes.append(p)

    # ── Wait for all workers to finish ────────────────────────────────
    for p in processes:
        p.join()

    # Signal consumer that everything is done
    all_procs_done.value = 1
    consumer.join(timeout=2)

    all_ok = all(code == 0 for _, code in results)
    print(f"\n{'='*60}# Final Results{'='*60}")
    for name, code in sorted(results):
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {name}: {status}")
    print(f"{'='*60}")
    if all_ok:
        print(f"[DONE] All {total} datasets succeeded.")
    else:
        print(f"[DONE] Some failed — check logs in {log_dir}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
