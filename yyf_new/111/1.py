#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tianmoucv.data import TianmoucDataReader

TIANMOU_DATA_PATH = "/projects/cxr_data/20260309_1914/"
CAMERA_IDX = 0     # 你现在用的 cone
N = 1
K = 5              # 打印前 K 帧做验证

def to_np(x):
    # 兼容 torch tensor / numpy
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def img_signature(img_np):
    """用低成本签名判断两帧是否几乎相同（不依赖可视化）"""
    a = img_np
    if a.ndim == 3:
        a = a[..., 0]  # 取一个通道即可
    a = a.astype(np.int32)
    return {
        "shape": tuple(img_np.shape),
        "dtype": str(img_np.dtype),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "sum_mod": int(a.sum() % 1000003),   # 防止太大
    }

def main():
    reader = TianmoucDataReader(
        TIANMOU_DATA_PATH,
        print_info=True,
        N=N,
        camera_idx=CAMERA_IDX
    )
    print(f"\nlen(reader) = {len(reader)}")
    print("=" * 90)

    prev = None
    for i in range(min(K, len(reader))):
        sample = reader[i]
        meta = sample.get("meta", {})

        Cts = meta.get("C_timestamp", None)  # 你同事说正确的是这里 [start, end]
        sys_ts = sample.get("sysTimeStamp", None)

        F0 = to_np(sample["F0"])
        F1 = to_np(sample["F1"])

        sig0 = img_signature(F0)
        sig1 = img_signature(F1)

        print(f"\nFrame i={i}")
        print(f"  keys: {list(sample.keys())}")
        print(f"  meta keys: {list(meta.keys())}")
        print(f"  sysTimeStamp: {sys_ts}")
        print(f"  meta['C_timestamp']: {Cts}")

        # 重要：检查时间戳单调性与间隔
        if isinstance(Cts, (list, tuple, np.ndarray)) and len(Cts) >= 2:
            start, end = int(Cts[0]), int(Cts[1])
            print(f"  C_timestamp start/end: {start} / {end}   (digits: {len(str(start))}/{len(str(end))})")
            print(f"  exposure_like(end-start): {end-start}")
            if prev is not None and isinstance(prev.get("Cts"), (list, tuple, np.ndarray)) and len(prev["Cts"]) >= 2:
                prev_start, prev_end = int(prev["Cts"][0]), int(prev["Cts"][1])
                print(f"  gap(start - prev_end): {start - prev_end}")
                print(f"  gap(start - prev_start): {start - prev_start}")

        # 重要：验证“F0 是否等于上一帧的 F1”
        if prev is not None:
            prevF1 = prev["F1"]
            diff = np.mean(np.abs(F0.astype(np.float32) - prevF1.astype(np.float32)))
            eq_like = diff < 1e-3
            print(f"  check F0 ~= prev(F1): mean_abs_diff={diff:.6f}  => {'YES' if eq_like else 'NO'}")

        print(f"  F0 sig: {sig0}")
        print(f"  F1 sig: {sig1}")

        prev = {"F1": F1, "Cts": Cts}

    print("\n" + "=" * 90)
    print("如何解读：")
    print("1) 如果多次出现 `F0 ~= prev(F1) => YES`，说明 F0 就是上一帧的 F1（典型双缓冲）。")
    print("2) 如果 meta['C_timestamp'] 单调递增且 gap 合理（比如接近帧间隔），它对应“当前帧”（通常是 F1）。")
    print("3) digits 若是 15 位且 boot_timestamp 是 16 位，优先怀疑单位差（如 10us）；但要以 gap/end-start 的量级判断。")

if __name__ == "__main__":
    main()