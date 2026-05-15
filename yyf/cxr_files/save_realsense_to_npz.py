import pyrealsense2 as rs
import numpy as np
from pathlib import Path
import time
import threading

bag_path = Path("/home/xiangru/Multi_sensor_sync/data/2025-12-22/14-13-15.bag")
save_path = bag_path.with_suffix(".npz")

# 准备容器
color_frames = []
depth_raw_frames = []
depth_m_frames = []
timestamps_ms = []

# 互斥锁，防止多线程写入冲突（虽然 Python GIL 会处理，但为了安全）
lock = threading.Lock()
finished = threading.Event()

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(str(bag_path), repeat_playback=False)

# 全局变量存储 depth_scale
depth_scale = 0.0

def frame_callback(frame):
    global depth_scale
    try:
        # frame 是一个复合帧 (frameset)
        fs = frame.as_frameset()
        color = fs.get_color_frame()
        depth = fs.get_depth_frame()
        
        if not color or not depth:
            return

        # 获取数据 (注意：callback是在后台线程运行，要尽快拷贝数据)
        # 这里直接转 numpy 会有拷贝开销，但通常比 I/O 快
        c_np = np.asanyarray(color.get_data())
        d_raw = np.asanyarray(depth.get_data())
        
        # 简单计算，如果需要更高性能可以放到最后统一算
        if depth_scale > 0:
            d_m = d_raw * depth_scale
        else:
            d_m = d_raw * 0.001 # 默认兜底

        with lock:
            color_frames.append(c_np)
            depth_raw_frames.append(d_raw)
            depth_m_frames.append(d_m)
            timestamps_ms.append(fs.get_timestamp())
            
            # 打印进度
            if len(timestamps_ms) % 100 == 0:
                print(f"\r已读取: {len(timestamps_ms)} 帧", end="")

    except Exception as e:
        print(f"Error in callback: {e}")

# 启动 Pipeline
# 注意：start 传入 callback，就会自动进入回调模式
profile = pipeline.start(config, frame_callback)

# 获取设备并设置非实时模式
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False) # 关键：全速读取

# 获取深度比例
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("开始读取...")

# 等待回放结束
# 比较 tricky 的是 SDK 不会自动通知结束，我们需要检测状态或者轮询
try:
    while True:
        # 获取当前播放位置
        # pos = playback.get_position() 
        # duration = playback.get_duration()
        
        # 另一种判断结束的方法：检查是否还有新帧进来
        # 或者直接捕获流停止的信号（SDK在这里支持得一般）
        
        # 简单方法：轮询等待，如果长时间没有新帧增加，就认为结束
        current_len = len(timestamps_ms)
        time.sleep(1) # 等1秒
        if len(timestamps_ms) == current_len and current_len > 0:
            # 连续 1 秒没有新帧，且已经读到过数据，认为结束
            print("\n检测到数据停止更新，停止读取。")
            break
        elif len(timestamps_ms) == 0:
             # 还没开始读到数据，继续等
             pass

except KeyboardInterrupt:
    print("用户中断")

finally:
    pipeline.stop()

print(f"\n总共读取: {len(color_frames)} 帧")

if not color_frames:
    print("警告：未读取到任何帧！")
else:
    np.savez_compressed(
        save_path,
        color=np.stack(color_frames),
        depth_raw=np.stack(depth_raw_frames),
        depth_m=np.stack(depth_m_frames),
        timestamp_ms=np.array(timestamps_ms),
        depth_scale=np.array([depth_scale]),
    )
    print(f"已保存到 {save_path}")
