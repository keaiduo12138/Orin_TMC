"""
calib/depth_reprojector.py
==========================
将 RealSense 深度图映射到天眸图像视角，生成 TM 尺寸的伪深度图。

核心变换（来自标定结果 extrinsic_tianmou_realsense.json）：
    P_tianmou = R @ P_rs + T
即：
    P_rs = R^T @ (P_tianmou - T)

使用方式：
    from depth_reprojector import DepthReprojector

    reproj = DepthReprojector(
        extrinsic_path="calibration_neican/extrinsic_tianmou_realsense.json",
        tm_size=(320, 640),   # H, W
        rs_size=(480, 640),
    )

    # 实时帧
    depth_tm_view = reproj.reproject(rs_depth, method='fast')

    # 或精细版本（适合离线处理）
    depth_tm_view = reproj.reproject(rs_depth, method='accurate')
"""

import json
import cv2
import numpy as np
from typing import Tuple, Literal


class DepthReprojector:
    """
    将 RS 深度图映射到 TM 图像视角的投影器。

    Attributes:
        K_tm, dist_tm : TM 内参 (3x3, 1x5)
        K_rs, dist_rs : RS 内参 (3x3, 1x5)
        R, T           : 外参 R (3x3), T (3,) — P_tm = R @ P_rs + T
        tm_h, tm_w     : TM 图像尺寸
        rs_h, rs_w     : RS 深度图尺寸
    """

    def __init__(
        self,
        extrinsic_path: str,
        tm_size: Tuple[int, int] = (320, 640),
        rs_size: Tuple[int, int] = (480, 640),
    ):
        self.tm_h, self.tm_w = tm_size
        self.rs_h, self.rs_w = rs_size

        # 加载标定结果
        with open(extrinsic_path) as f:
            calib = json.load(f)

        self.K_tm = np.array(calib["K_tianmou"], dtype=np.float64)
        self.dist_tm = np.array(calib["dist_tianmou"], dtype=np.float64)
        self.K_rs = np.array(calib["K_realsense"], dtype=np.float64)
        self.dist_rs = np.array(calib["dist_realsense"], dtype=np.float64)
        self.R = np.array(calib["R"], dtype=np.float64)
        self.T = np.array(calib["T"], dtype=np.float64)

        # 预计算
        self.R_inv = self.R.T          # 正交矩阵，逆 = 转置
        self._build_grid()

    # ------------------------------------------------------------------ #
    #  内部：预计算网格加速
    # ------------------------------------------------------------------ #
    def _build_grid(self):
        """预计算 TM 像素归一化坐标网格 (H*W, 3)。"""
        u = np.arange(self.tm_w, dtype=np.float64)
        v = np.arange(self.tm_h, dtype=np.float64)
        vv, uu = np.meshgrid(v, u, indexing="ij")
        self.tm_grid = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)

        # 预计算 RS 像素坐标网格（用于双线性插值）
        rs_u = np.arange(self.rs_w, dtype=np.float32)
        rs_v = np.arange(self.rs_h, dtype=np.float32)
        rs_vv, rs_uu = np.meshgrid(rs_v, rs_u, indexing="ij")
        self.rs_grid = np.stack([rs_uu, rs_vv], axis=-1).astype(np.float32)  # (H_rs, W_rs, 2)

    # ------------------------------------------------------------------ #
    #  核心投影
    # ------------------------------------------------------------------ #
    def reproject(
        self,
        rs_depth: np.ndarray,
        method: Literal["fast", "accurate"] = "fast",
    ) -> np.ndarray:
        """
        将 RS 深度图映射到 TM 图像视角。

        Args:
            rs_depth: RS 深度图，shape (H_rs, W_rs)，单位 mm，0 = 无效
            method:
                'fast'    — 从 RS 深度图出发，直接投影到 TM（推荐实时用）
                'accurate'— 从 TM 像素出发，ray march 查 RS 深度（更精确但慢）

        Returns:
            depth_tm: shape (H_tm, W_tm)，单位 mm，未命中区域为 0
        """
        if method == "fast":
            return self._reproject_rs_to_tm(rs_depth)
        else:
            return self._reproject_ray_march(rs_depth)

    # ------------------------------------------------------------------ #
    #  方法1: 从 RS 出发投影（高效，适合实时）
    # ------------------------------------------------------------------ #
    def _reproject_rs_to_tm(self, rs_depth: np.ndarray) -> np.ndarray:
        """
        对 RS 深度图中每个有效点：
          1. 反投影到 RS 相机坐标 P_rs = [X, Y, Z]
          2. 变换到 TM 坐标系   P_tm = R @ P_rs + T
          3. 投影到 TM 像素      (u_tm, v_tm)
          4. 填入输出深度图的最近邻位置
        """
        rs_depth_f = rs_depth.astype(np.float32)

        # 找出 RS 深度图中的有效点
        mask = (rs_depth > 0)
        ys, xs = np.where(mask)
        ds = rs_depth_f[mask]          # (N,)

        if len(ds) == 0:
            return np.zeros((self.tm_h, self.tm_w), dtype=np.float32)

        # ---------- 步骤1: RS 像素 -> RS 相机 3D ----------
        X_rs = (xs - self.K_rs[0, 2]) / self.K_rs[0, 0] * ds   # (N,)
        Y_rs = (ys - self.K_rs[1, 2]) / self.K_rs[1, 1] * ds   # (N,)
        Z_rs = ds.copy()                                        # (N,)

        # ---------- 步骤2: 变换到 TM 坐标系 ----------
        R_f = self.R.astype(np.float32)
        T_f = self.T.astype(np.float32)
        P_rs = np.stack([X_rs, Y_rs, Z_rs], axis=0)            # (3, N)
        P_tm = R_f @ P_rs + T_f.reshape(3, 1)                  # (3, N)

        # ---------- 过滤: 去掉 TM 相机后方的点 ----------
        valid = P_tm[2] > 0
        P_tm = P_tm[:, valid]
        xs, ys, ds = xs[valid], ys[valid], ds[valid]

        if len(ds) == 0:
            return np.zeros((self.tm_h, self.tm_w), dtype=np.float32)

        # ---------- 步骤3: 投影到 TM 像素 ----------
        with np.errstate(divide="ignore", invalid="ignore"):
            u_tm = self.K_tm[0, 0] * P_tm[0] / P_tm[2] + self.K_tm[0, 2]
            v_tm = self.K_tm[1, 1] * P_tm[1] / P_tm[2] + self.K_tm[1, 2]

        # ---------- 步骤4: 截取到 TM 图像范围 ----------
        in_w = (u_tm >= 0) & (u_tm < self.tm_w)
        in_h = (v_tm >= 0) & (v_tm < self.tm_h)
        in_bounds = in_w & in_h

        u_tm = u_tm[in_bounds].astype(np.int32)
        v_tm = v_tm[in_bounds].astype(np.int32)
        ds_proj = ds[in_bounds]

        # ---------- 步骤5: 原地填入最近深度（向量化+ scatter） ----------
        depth_tm = np.zeros((self.tm_h, self.tm_w), dtype=np.float32)
        # 处理重复投影（多对一）：用 np_lexsort 保证索引稳定，取每个 TM 像素对应的最近 RS 深度
        order = np.lexsort((ds_proj, u_tm * self.tm_h + v_tm))
        u_sorted = u_tm[order]
        v_sorted = v_tm[order]
        d_sorted = ds_proj[order]
        # 找每段起点
        keys = u_sorted * self.tm_h + v_sorted
        starts = np.concatenate([[0], np.where(np.diff(keys) != 0)[0] + 1])
        # 每段第一个点就是最近邻（ds_proj 已按深度从小到大排序）
        depth_tm[v_sorted[starts], u_sorted[starts]] = d_sorted[starts]

        return depth_tm

    # ------------------------------------------------------------------ #
    #  方法2: 从 TM 出发 ray march（精细，适合离线验证）
    # ------------------------------------------------------------------ #
    def _reproject_ray_march(self, rs_depth: np.ndarray) -> np.ndarray:
        """
        对 TM 图像每个像素，发出一条射线，在 RS 深度图上搜索命中点。
        """
        depth_tm = np.zeros((self.tm_h, self.tm_w), dtype=np.float32)
        rs_depth_f = rs_depth.astype(np.float32)

        # 预计算 RS 坐标变换系数（减少循环内重复计算）
        fx_rs = self.K_rs[0, 0]
        fy_rs = self.K_rs[1, 1]
        cx_rs = self.K_rs[0, 2]
        cy_rs = self.K_rs[1, 2]

        R_inv_f = self.R.T.astype(np.float32)
        T_f = self.T.astype(np.float32)

        # TM 内参
        fx_tm = self.K_tm[0, 0]
        fy_tm = self.K_tm[1, 1]
        cx_tm = self.K_tm[0, 2]
        cy_tm = self.K_tm[1, 2]

        for v in range(self.tm_h):
            for u in range(self.tm_w):
                # TM 像素 -> 归一化相机坐标
                xn = (u - cx_tm) / fx_tm
                yn = (v - cy_tm) / fy_tm

                # 在 TM 相机坐标系中的射线方向
                d_tm = np.array([xn, yn, 1.0], dtype=np.float32)
                # 变换到 RS 坐标系下的射线方向
                d_rs = R_inv_f @ d_tm
                norm = np.linalg.norm(d_rs)
                d_rs /= norm

                # Ray march: 从近处往远处扫
                d_min, d_max = 50.0, 6000.0
                step = max(1.0, (d_max - d_min) / 512)

                prev_rs_depth = 0.0
                for d in np.arange(d_min, d_max, step):
                    # 假设 3D 点（RS 坐标系）
                    Px = d * d_rs[0]
                    Py = d * d_rs[1]
                    Pz = d * d_rs[2]

                    # 投影到 RS 像素
                    if abs(Pz) < 1e-6:
                        continue
                    u_rs = fx_rs * Px / Pz + cx_rs
                    v_rs = fy_rs * Py / Pz + cy_rs

                    ui, vi = int(u_rs), int(v_rs)
                    if 0 <= ui < self.rs_w and 0 <= vi < self.rs_h:
                        actual_depth = rs_depth_f[vi, ui]
                        if actual_depth > 0:
                            # 检测命中（射线穿过了实际深度表面）
                            if d >= actual_depth - step:
                                depth_tm[v, u] = actual_depth
                                break
                            prev_rs_depth = actual_depth

        return depth_tm

    # ------------------------------------------------------------------ #
    #  辅助: 彩色深度图可视化
    # ------------------------------------------------------------------ #
    def colorize(
        self,
        depth: np.ndarray,
        min_d: float = 200,
        max_d: float = 3000,
    ) -> np.ndarray:
        """
        将深度图转为伪彩色图（蓝=近，红=远）。

        Args:
            depth: 深度图 (H, W)，单位 mm
            min_d, max_d: 色彩映射范围
        """
        norm = np.clip((depth.astype(np.float32) - min_d) / (max_d - min_d), 0, 1)
        norm = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        color[depth == 0] = [0, 0, 0]
        return color

    # ------------------------------------------------------------------ #
    #  辅助: 深度图滤波（填补空洞）
    # ------------------------------------------------------------------ #
    def fill_holes(
        self,
        depth: np.ndarray,
        kernel_size: int = 5,
        max_hole_depth: float = 1000.0,
    ) -> np.ndarray:
        """
        用最近邻填充深度图中的小空洞。

        Args:
            depth: 输入深度图
            kernel_size: 卷积核大小（越大填得越多）
            max_hole_depth: 允许填补的最大深度差
        """
        mask = (depth > 0).astype(np.uint8)
        # 膨胀：扩大有效区域
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel)

        # 用膨胀区域填充空洞
        dilated_f = dilated.astype(np.float32) * depth
        count = cv2.dilate(mask.astype(np.float32), kernel)

        filled = np.zeros_like(depth)
        nonzero = count > 0
        filled[nonzero] = dilated_f[nonzero] / count[nonzero]
        filled[depth > 0] = depth[depth > 0]  # 保留原始值

        return filled

    # ------------------------------------------------------------------ #
    #  辅助: 叠加 TM 图像 + 深度伪彩色的融合图
    # ------------------------------------------------------------------ #
    def blend(
        self,
        tm_image: np.ndarray,
        rs_depth: np.ndarray,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        生成 TM 图像与 RS 深度彩色化的叠加图，便于直观验证投影质量。
        """
        depth_color = self.colorize(rs_depth)
        return cv2.addWeighted(tm_image, alpha, depth_color, 1 - alpha, 0)

    # ------------------------------------------------------------------ #
    #  批量处理: 保存一组帧的映射结果（用于离线调试）
    # ------------------------------------------------------------------ #
    def reproject_batch(
        self,
        rs_depth_dir: str,
        output_dir: str,
        tm_image_dir: str = None,
        save_overlay: bool = False,
    ):
        """
        批量处理目录下所有深度图。

        Args:
            rs_depth_dir : 含 RS 深度图的文件夹
            output_dir   : 输出目录（含 TM 视角深度图）
            tm_image_dir : 可选，TM 图像目录（有则同时保存叠加图）
            save_overlay : 是否保存融合验证图
        """
        import os
        from glob import glob

        os.makedirs(output_dir, exist_ok=True)
        files = sorted(glob(os.path.join(rs_depth_dir, "*depth*")))

        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            rs_d = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if rs_d is None:
                continue

            depth_tm = self.reproject(rs_d, method="fast")

            out_path = os.path.join(output_dir, f"{name}_in_tm.png")
            cv2.imwrite(out_path, depth_tm.astype(np.uint16))

            if save_overlay and tm_image_dir:
                tm_img_path = os.path.join(tm_image_dir, os.path.basename(f).replace("depth", "color"))
                if os.path.exists(tm_img_path):
                    tm_img = cv2.imread(tm_img_path)
                    if tm_img is not None:
                        blend_img = self.blend(tm_img, depth_tm)
                        cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), blend_img)

        print(f"批量处理完成: {len(files)} 帧 -> {output_dir}")


# ====================================================================== #
#  命令行入口
# ====================================================================== #
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="RS 深度图 → TM 视角映射")
    parser.add_argument("--extrinsic", "-e", required=True,
                        help="extrinsic_tianmou_realsense.json 路径")
    parser.add_argument("--depth", "-d", required=True,
                        help="RS 深度图路径（或目录，批量模式）")
    parser.add_argument("--tm-image", "-t", default=None,
                        help="TM 图像路径（用于叠加验证）")
    parser.add_argument("--output", "-o", default="./depth_in_tm",
                        help="输出目录")
    parser.add_argument("--method", "-m", default="fast",
                        choices=["fast", "accurate"])
    parser.add_argument("--tm-size", nargs=2, type=int, default=[320, 640],
                        help="TM 图像尺寸 H W（默认 320 640）")
    parser.add_argument("--rs-size", nargs=2, type=int, default=[480, 640],
                        help="RS 深度图尺寸 H W（默认 480 640）")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式（depth 参数为目录）")
    parser.add_argument("--save-overlay", action="store_true",
                        help="保存融合叠加验证图")
    args = parser.parse_args()

    reproj = DepthReprojector(
        extrinsic_path=args.extrinsic,
        tm_size=tuple(args.tm_size),
        rs_size=tuple(args.rs_size),
    )

    os.makedirs(args.output, exist_ok=True)

    if args.batch:
        reproj.reproject_batch(
            args.depth, args.output,
            tm_image_dir=args.tm_image,
            save_overlay=args.save_overlay,
        )
    else:
        rs_depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if rs_depth is None:
            print(f"无法读取: {args.depth}")
            sys.exit(1)

        depth_tm = reproj.reproject(rs_depth, method=args.method)

        # 保存
        out_path = os.path.join(args.output, "depth_in_tm.png")
        cv2.imwrite(out_path, depth_tm.astype(np.uint16))
        print(f"已保存: {out_path}")

        # 叠加验证
        if args.tm_image:
            tm_img = cv2.imread(args.tm_image)
            if tm_img is not None:
                overlay = reproj.blend(tm_img, depth_tm)
                cv2.imwrite(os.path.join(args.output, "overlay.png"), overlay)
                print(f"叠加图已保存: {os.path.join(args.output, 'overlay.png')}")

        # 可视化
        color = reproj.colorize(depth_tm)
        cv2.imwrite(os.path.join(args.output, "depth_color.png"), color)
        print(f"伪彩色深度图已保存: {os.path.join(args.output, 'depth_color.png')}")
