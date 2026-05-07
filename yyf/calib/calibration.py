import pyrealsense2 as rs
import numpy as np

# 初始化并启动相机
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色流（必须启用，否则拿不到内参）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动 pipeline
profile = pipeline.start(config)

try:
    # 获取彩色流内参
    color_profile = profile.get_stream(rs.stream.color)
    video_profile = color_profile.as_video_stream_profile()
    intrinsics = video_profile.get_intrinsics()

    # 打印内参信息（你要的输出在这里）
    print("="*60)
    print("📷  RealSense 彩色相机内参信息")
    print("="*60)
    print(f"图像宽度:      {intrinsics.width}")
    print(f"图像高度:      {intrinsics.height}")
    print(f"fx (焦距x):    {intrinsics.fx:.6f}")
    print(f"fy (焦距y):    {intrinsics.fy:.6f}")
    print(f"ppx (主点x):   {intrinsics.ppx:.6f}")
    print(f"ppy (主点y):   {intrinsics.ppy:.6f}")
    print(f"畸变模型:      {intrinsics.model}")
    print(f"畸变系数 [k1,k2,p1,p2,k3]: {intrinsics.coeffs}")
    print("="*60)

    # 转换为内参矩阵 K 和畸变系数（和你原来格式一致）
    K_rs = np.array([
        [intrinsics.fx, 0,             intrinsics.ppx],
        [0,             intrinsics.fy, intrinsics.ppy],
        [0,             0,             1]
    ], dtype=np.float32)

    dist_rs = np.array(intrinsics.coeffs, dtype=np.float32)

    # 打印矩阵格式（方便你直接复制）
    print("\n📦 内参矩阵 K:")
    print(K_rs)
    print("\n📦 畸变系数 dist:")
    print(dist_rs)
    print("="*60)

finally:
    # 安全停止相机
    pipeline.stop()