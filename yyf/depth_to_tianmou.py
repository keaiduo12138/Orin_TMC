#!/usr/bin/env python3
"""
depth_to_tianmou.py
====================
将 RealSense 16位原始深度图重投影到天眸图像视角。

核心流程：
1. 使用 pyrealsense2 从 bag 文件读取原始 16 位深度
2. 根据标定外参（extrinsic_tianmou_realsense.json）将深度点云变换到天眸坐标系
3. 投影到天眸图像平面，生成与天眸尺寸相同的深度图
4. 输出：tianmou 视角的深度图（16位）+ 可视化伪彩色图 + 融合验证图

使用方式：
    python3 depth_to_tianmou.py \
        --bag /projects/cxr_data/xxx.bag \
        --calib calibration_neican/extrinsic_tianmou_realsense.json \
        --tianmou-dir /projects/calib_data/0323_calibration_v1/tianmou/ \
        --index-json /projects/calib_data/0323_calibration_v1/index.json \
        --output /projects/calib_data/0323_calibration_v1/tianmou_depth

    # 快速单帧测试
    python3 depth_to_tianmou.py --bag /projects/cxr_data/xxx.bag --frame 50 --preview
"""

import os
import sys
import json
import argparse
import time
import glob
import numpy as np
import cv2

# 尝试导入 pyrealsense2
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("WARNING: pyrealsense2 未安装，无法从 bag 读取原始深度")

from typing import Tuple, Optional, Dict, List


# ============================================================
#  深度重投影器（复用 calib/depth_reprojector.py 的逻辑）
# ============================================================

class DepthReprojector:
    """
    将 RS 深度图映射到 TM 图像视角的投影器。
    核心变换（来自标定结果）：
        P_tianmou = R @ P_rs + T
        即 P_rs = R^T @ (P_tianmou - T)
    """

    def __init__(
        self,
        extrinsic_path: str,
        tm_size: Tuple[int, int] = (320, 640),
        rs_size: Tuple[int, int] = (480, 640),
    ):
        self.tm_h, self.tm_w = tm_size
        self.rs_h, self.rs_w = rs_size

        with open(extrinsic_path) as f:
            calib = json.load(f)

        self.K_tm = np.array(calib["K_tianmou"], dtype=np.float64)
        self.dist_tm = np.array(calib["dist_tianmou"], dtype=np.float64)
        self.K_rs = np.array(calib["K_realsense"], dtype=np.float64)
        self.dist_rs = np.array(calib["dist_realsense"], dtype=np.float64)
        self.R = np.array(calib["R"], dtype=np.float64)
        self.T = np.array(calib["T"], dtype=np.float64)

        self.R_inv = self.R.T
        print(f"  外参加载: R(3x3), T={self.T}")

    def reproject(self, rs_depth: np.ndarray) -> np.ndarray:
        """
        将 RS 16位深度图映射到 TM 图像视角。

        Args:
            rs_depth: RS 深度图，shape (H_rs, W_rs)，单位 mm，0 = 无效

        Returns:
            depth_tm: shape (H_tm, W_tm)，单位 mm，未命中区域为 0
        """
        return self._reproject_rs_to_tm(rs_depth)

    def _reproject_rs_to_tm(self, rs_depth: np.ndarray) -> np.ndarray:
        """对 RS 深度图每个有效点投影到 TM 视角。"""
        rs_depth_f = rs_depth.astype(np.float32)

        mask = (rs_depth > 0)
        ys, xs = np.where(mask)
        ds = rs_depth_f[mask]

        if len(ds) == 0:
            return np.zeros((self.tm_h, self.tm_w), dtype=np.float32)

        # 步骤1: RS 像素 -> RS 相机 3D
        X_rs = (xs - self.K_rs[0, 2]) / self.K_rs[0, 0] * ds
        Y_rs = (ys - self.K_rs[1, 2]) / self.K_rs[1, 1] * ds
        Z_rs = ds.copy()

        # 步骤2: 变换到 TM 坐标系
        R_f = self.R.astype(np.float32)
        T_f = self.T.astype(np.float32)
        P_rs = np.stack([X_rs, Y_rs, Z_rs], axis=0)
        P_tm = R_f @ P_rs + T_f.reshape(3, 1)

        # 过滤: 去掉 TM 相机后方的点
        valid = P_tm[2] > 0
        P_tm = P_tm[:, valid]
        xs, ys, ds = xs[valid], ys[valid], ds[valid]

        if len(ds) == 0:
            return np.zeros((self.tm_h, self.tm_w), dtype=np.float32)

        # 步骤3: 投影到 TM 像素
        with np.errstate(divide="ignore", invalid="ignore"):
            u_tm = self.K_tm[0, 0] * P_tm[0] / P_tm[2] + self.K_tm[0, 2]
            v_tm = self.K_tm[1, 1] * P_tm[1] / P_tm[2] + self.K_tm[1, 2]

        # 步骤4: 截取到 TM 图像范围
        in_w = (u_tm >= 0) & (u_tm < self.tm_w)
        in_h = (v_tm >= 0) & (v_tm < self.tm_h)
        in_bounds = in_w & in_h

        u_tm = u_tm[in_bounds].astype(np.int32)
        v_tm = v_tm[in_bounds].astype(np.int32)
        ds_proj = ds[in_bounds]

        # 步骤5: 填入最近深度（多对一时取最近）
        depth_tm = np.zeros((self.tm_h, self.tm_w), dtype=np.float32)
        order = np.lexsort((ds_proj, u_tm * self.tm_h + v_tm))
        u_sorted = u_tm[order]
        v_sorted = v_tm[order]
        d_sorted = ds_proj[order]
        keys = u_sorted * self.tm_h + v_sorted
        starts = np.concatenate([[0], np.where(np.diff(keys) != 0)[0] + 1])
        depth_tm[v_sorted[starts], u_sorted[starts]] = d_sorted[starts]

        return depth_tm

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        """深度图转伪彩色（蓝=近，红=远），使用动态范围。"""
        valid_mask = depth > 0
        if not np.any(valid_mask):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        valid_depth = depth[valid_mask].astype(np.float32)
        # 动态范围：使用 2%-98% 分位数
        d_min = max(0, np.percentile(valid_depth, 2))
        d_max = np.percentile(valid_depth, 98)
        if d_max <= d_min:
            d_max = d_min + 1000

        norm = np.clip((depth.astype(np.float32) - d_min) / (d_max - d_min), 0, 1)
        norm = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color[depth == 0] = [0, 0, 0]
        return color

    def blend(self, tm_image: np.ndarray, depth_tm: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """TM 图像 + 深度伪彩色叠加图。"""
        depth_color = self.colorize(depth_tm)
        # 确保 tm_image 是 BGR
        if tm_image.ndim == 2:
            tm_image = cv2.cvtColor(tm_image, cv2.COLOR_GRAY2BGR)
        elif tm_image.shape[2] == 3 and tm_image.dtype == np.uint8:
            pass
        # 直接叠加：只在有深度的地方覆盖颜色
        result = tm_image.copy()
        mask = np.any(depth_color > 0, axis=2)
        result[mask] = cv2.addWeighted(tm_image, alpha, depth_color, 1 - alpha, 0)[mask]
        return result


# ============================================================
#  Bag 读取：从 bag 提取与天眸帧对齐的 16 位原始深度
# ============================================================

def read_bag_depth_frames(
    bag_path: str,
    index_mapping: List[Dict],
    tianmou_timestamps: List[int],
) -> Dict[int, np.ndarray]:
    """
    从 bag 文件读取 16 位原始深度帧。

    Args:
        bag_path: bag 文件路径
        index_mapping: 对齐索引列表，每项包含 tianmou_idx, depth_counter 等
        tianmou_timestamps: 天眸时间戳列表（微秒）

    Returns:
        {frame_idx: depth_array}，frame_idx 与 tianmou 帧索引对应
    """
    if not HAS_REALSENSE:
        print("ERROR: pyrealsense2 未安装")
        return {}

    print(f"打开 bag 文件: {bag_path}")
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat=False)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    depth_frames = {}
    frame_count = 0
    last_report = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                break

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            frame_ts_us = depth_frame.get_timestamp() * 1000  # ms -> us

            # 在 tianmou_timestamps 中找最近帧
            best_idx = 0
            best_diff = float('inf')
            for i, tm_ts in enumerate(tianmou_timestamps):
                diff = abs(frame_ts_us - tm_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            # 阈值：30ms 内的匹配才接受
            if best_diff < 30000:
                if best_idx not in depth_frames:
                    depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
                    depth_frames[best_idx] = depth_data

            frame_count += 1
            now = time.time()
            if now - last_report > 5:
                print(f"  已读取 {frame_count} 帧，匹配 {len(depth_frames)} 帧")
                last_report = now

    except Exception as e:
        print(f"  读取结束: {e}")
    finally:
        pipeline.stop()

    print(f"  Bag 读取完成: 共 {frame_count} 帧，找到 {len(depth_frames)} 个匹配")
    return depth_frames


def read_existing_depth_images(depth_dir: str, index_mapping: List[Dict]) -> Dict[int, np.ndarray]:
    """
    从已有目录读取深度图（处理伪彩色图或尝试获取原始深度）。

    对于伪彩色图（JET格式），会尝试反向估算深度值。
    """
    depth_images = {}

    # 尝试从 realsense_raw/depth 找原始深度
    raw_depth_dir = os.path.join(os.path.dirname(depth_dir.rstrip('/')), "realsense_raw/depth")
    if os.path.exists(raw_depth_dir):
        print(f"  尝试从 {raw_depth_dir} 读取原始深度...")
        raw_files = glob.glob(os.path.join(raw_depth_dir, "*.png"))
        if raw_files:
            # 检查是否是 16 位
            test_file = raw_files[0]
            img_test = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
            if img_test is not None and img_test.dtype == np.uint16:
                print(f"  找到 {len(raw_files)} 个 16 位原始深度文件")
                # 按文件名排序，建立索引
                raw_files_sorted = sorted(raw_files)
                for i, fpath in enumerate(raw_files_sorted):
                    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        depth_images[i] = img
                return depth_images
            else:
                print(f"  警告: 原始深度也是伪彩色格式 ({img_test.dtype if img_test is not None else 'None'})，将尝试反向估算深度")

    # 尝试从 depth_raw 目录读取原始深度（与 extract_raw_depth.py 输出对应）
    raw_depth_alt = os.path.join(os.path.dirname(depth_dir.rstrip('/')), "depth_raw")
    if os.path.exists(raw_depth_alt):
        print(f"  尝试从 {raw_depth_alt} 读取原始深度...")
        raw_files = glob.glob(os.path.join(raw_depth_alt, "*.png"))
        if raw_files:
            test_file = raw_files[0]
            img_test = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
            if img_test is not None and img_test.dtype == np.uint16:
                print(f"  找到 {len(raw_files)} 个 16 位原始深度文件")
                raw_files_sorted = sorted(raw_files)
                for i, fpath in enumerate(raw_files_sorted):
                    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        depth_images[i] = img
                return depth_images
            else:
                print(f"  警告: depth_raw 也是伪彩色格式，将尝试反向估算深度")

    # 尝试反向JET映射
    print(f"  从 {depth_dir} 读取伪彩色深度图...")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.png")))
    for i, fpath in enumerate(depth_files):
        # 读取为灰度（伪彩色图每个通道相同）
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 反向 JET 映射: 0->远(红色), 128->中(绿色), 255->近(蓝色)
            # 假设原始深度范围 200mm - 3000mm
            depth_est = 3000 - (img.astype(np.float32) / 255.0) * 2800
            depth_images[i] = depth_est.astype(np.uint16)
            if i == 0:
                print(f"    伪彩色反向估算: 假设深度范围 200~3000mm")

    print(f"  读取了 {len(depth_images)} 帧伪彩色深度")
    return depth_images


# ============================================================
#  主处理流程
# ============================================================

def process_depth_to_tianmou(
    bag_path: str,
    calib_path: str,
    tianmou_dir: str,
    index_json_path: str,
    output_dir: str,
    start_frame: int = 0,
    end_frame: int = -1,
):
    """主处理函数"""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_tm"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "color_tm"), exist_ok=True)

    # 1. 加载标定参数
    print("=" * 60)
    print("Step 1: 加载标定参数")
    with open(calib_path) as f:
        calib = json.load(f)
    print(f"  Tianmou 内参: fx={calib['K_tianmou'][0][0]:.1f}, fy={calib['K_tianmou'][1][1]:.1f}")
    print(f"  RS 基线: {calib.get('baseline_mm', 'N/A')} mm")

    # 2. 初始化重投影器
    reproj = DepthReprojector(calib_path, tm_size=(320, 640), rs_size=(480, 640))

    # 3. 加载对齐索引
    print("\nStep 2: 加载对齐索引")
    if os.path.exists(index_json_path):
        with open(index_json_path) as f:
            index_data = json.load(f)
        frames = index_data.get("frames", [])
        print(f"  索引文件: {len(frames)} 帧")
    else:
        print(f"  警告: 索引文件不存在，将按序号匹配")
        frames = None

    # 4. 读取天眸图像列表
    print("\nStep 3: 读取天眸图像列表")
    tm_files = sorted(glob.glob(os.path.join(tianmou_dir, "*.png")) + glob.glob(os.path.join(tianmou_dir, "*.jpg")))
    print(f"  天眸图像: {len(tm_files)} 帧")

    # 5. 读取深度数据
    print("\nStep 4: 读取深度数据")
    depth_dir = os.path.join(os.path.dirname(tianmou_dir.rstrip('/')), "depth")
    if os.path.exists(depth_dir):
        depth_images = read_existing_depth_images(depth_dir, frames)
    elif os.path.exists(bag_path):
        # 从 bag 读取
        tianmou_timestamps = []
        if frames:
            tianmou_timestamps = [f.get("tianmou_ts", 0) for f in frames]
        else:
            tianmou_timestamps = list(range(len(tm_files)))
        depth_images = read_bag_depth_frames(bag_path, frames, tianmou_timestamps)
    else:
        print("ERROR: 未找到深度数据")
        return

    # 6. 逐帧处理
    print("\nStep 5: 逐帧重投影")
    end_idx = end_frame if end_frame > 0 else len(tm_files)

    # 建立 tianmou_idx 到 depth_counter 的映射
    tianmou_to_depth = {}
    if frames:
        for f in frames:
            tm_idx = f.get("tianmou_idx", 0)
            depth_counter = f.get("depth_counter", tm_idx)
            tianmou_to_depth[tm_idx] = depth_counter

    for i in range(start_frame, min(end_idx, len(tm_files))):
        # 读取天眸图像
        tm_img = cv2.imread(tm_files[i])
        if tm_img is None:
            continue

        # 获取对应的深度 counter（用于从 depth_raw 读取）
        if frames and i < len(frames):
            depth_counter = frames[i].get("depth_counter", i)
        else:
            depth_counter = i

        # 使用 depth_counter 从 depth_images 查找
        if depth_counter not in depth_images:
            # 尝试找最近的深度帧
            if depth_images:
                nearest = min(depth_images.keys(), key=lambda k: abs(k - depth_counter))
                depth_counter = nearest
            else:
                continue

        rs_depth = depth_images[depth_counter]
        if rs_depth is None:
            continue

        # 重投影
        depth_tm = reproj.reproject(rs_depth)

        # 保存
        out_depth_path = os.path.join(output_dir, "depth_tm", f"depth_tm_{i:04d}.png")
        cv2.imwrite(out_depth_path, depth_tm.astype(np.uint16))

        # 保存可视化
        depth_color = reproj.colorize(depth_tm)
        out_color_path = os.path.join(output_dir, "color_tm", f"depth_tm_{i:04d}.png")
        cv2.imwrite(out_color_path, depth_color)

        # 保存叠加图
        blend_img = reproj.blend(tm_img, depth_tm, alpha=0.5)
        out_blend_path = os.path.join(output_dir, "blend", f"blend_{i:04d}.png")
        os.makedirs(os.path.join(output_dir, "blend"), exist_ok=True)
        cv2.imwrite(out_blend_path, blend_img)

        if i % 50 == 0:
            print(f"  处理进度: {i}/{min(end_idx, len(tm_files))}")

    print(f"\n处理完成! 结果保存在: {output_dir}")
    return output_dir


# ============================================================
#  命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RS 深度图 → TM 视角映射")
    parser.add_argument("--bag", "-b", required=True, help="bag 文件路径")
    parser.add_argument("--calib", "-c", required=True, help="标定外参 JSON 文件")
    parser.add_argument("--tianmou-dir", "-t", required=True, help="天眸图像目录")
    parser.add_argument("--index-json", "-i", default=None, help="对齐索引 JSON 文件")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--start", "-s", type=int, default=0, help="起始帧")
    parser.add_argument("--end", "-e", type=int, default=-1, help="结束帧(-1=全部)")
    parser.add_argument("--preview", "-p", action="store_true", help="单帧预览模式")

    args = parser.parse_args()

    process_depth_to_tianmou(
        bag_path=args.bag,
        calib_path=args.calib,
        tianmou_dir=args.tianmou_dir,
        index_json_path=args.index_json or "",
        output_dir=args.output,
        start_frame=args.start,
        end_frame=args.end,
    )
