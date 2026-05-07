#!/usr/bin/env python3
"""
RealSense D455 Genlock模式录制脚本
- 将相机设置为Genlock/Slave模式，接收外部同步信号
- 录制深度和彩色流到bag文件
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
from datetime import datetime


class D455GenlockRecorder:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.device = None
        self.is_recording = False
        
    def find_d455_device(self):
        """查找D455设备"""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("未找到任何RealSense设备")
        
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"发现设备: {name}, 序列号: {serial}")
            
            if "D455" in name:
                print(f"已选择D455设备: {serial}")
                return dev, serial
        
        # 如果没有找到D455，使用第一个设备
        dev = devices[0]
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"警告: 未找到D455，使用设备: {name}, 序列号: {serial}")
        return dev, serial
    
    def set_genlock_mode(self, device, sync_mode=4):
        """
        设置同步模式
        inter_cam_sync_mode值:
          0 = Default (无同步，自由运行)
          1 = Master (主设备，发送同步信号)
          2 = Slave (从设备，接收同步信号)
          3 = Full Slave (完全从属模式)
          4 = Genlock (接收外部genlock信号)
        """
        depth_sensor = device.first_depth_sensor()
        
        mode_names = {
            0: "Default (自由运行)",
            1: "Master",
            2: "Slave",
            3: "Full Slave",
            4: "Genlock"
        }
        
        # 检查是否支持inter_cam_sync_mode
        if depth_sensor.supports(rs.option.inter_cam_sync_mode):
            try:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
                print(f"已设置为{mode_names.get(sync_mode, sync_mode)}模式 (mode {sync_mode})")
            except Exception as e:
                print(f"设置同步模式失败: {e}")
            
            # 验证设置
            current_mode = depth_sensor.get_option(rs.option.inter_cam_sync_mode)
            print(f"当前同步模式: {int(current_mode)} ({mode_names.get(int(current_mode), '未知')})")
        else:
            print("警告: 此设备不支持inter_cam_sync_mode选项")
        
        return depth_sensor
    
    def configure_streams(self, serial_number, output_file):
        """配置流和录制"""
        self.config = rs.config()
        
        # 指定设备
        self.config.enable_device(serial_number)
        
        # 配置深度流 - D455支持的分辨率
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        
        # 配置彩色流
        self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
        
        # 可选: 配置红外流
        # self.config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
        # self.config.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 30)
        
        # 启用录制到bag文件
        self.config.enable_record_to_file(output_file)
        print(f"将录制到: {output_file}")
        
    def start_recording(self, output_file=None, sync_mode=4):
        """开始录制"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"d455_recording_{timestamp}.bag"
        
        # 确保文件路径是绝对路径
        if not os.path.isabs(output_file):
            output_file = os.path.join(os.getcwd(), output_file)
        
        # 查找设备获取序列号
        device, serial = self.find_d455_device()
        
        # 重要：在pipeline启动之前设置同步模式
        self.set_genlock_mode(device, sync_mode)
        
        # 创建pipeline
        self.pipeline = rs.pipeline()
        
        # 配置流
        self.configure_streams(serial, output_file)
        
        # 启动pipeline
        print("启动录制...")
        profile = self.pipeline.start(self.config)
        
        # 确认同步设置
        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        if depth_sensor.supports(rs.option.inter_cam_sync_mode):
            mode = depth_sensor.get_option(rs.option.inter_cam_sync_mode)
            mode_names = {0: "Default", 1: "Master", 2: "Slave", 3: "Full Slave", 4: "Genlock"}
            print(f"确认同步模式: {int(mode)} ({mode_names.get(int(mode), '未知')})")
        
        self.is_recording = True
        self.device = dev
        
        return profile
    
    def stop_recording(self):
        """停止录制"""
        if self.pipeline:
            self.pipeline.stop()
            self.is_recording = False
            print("录制已停止")
    
    def record_with_preview(self, output_file=None, show_preview=True, sync_mode=4):
        """带预览的录制（按 'q' 停止）"""
        profile = self.start_recording(output_file, sync_mode)
        
        frame_count = 0
        print("\n录制中... 按 'q' 键停止\n")
        
        try:
            while True:
                # 等待帧
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                frame_count += 1
                
                # 获取时间戳
                depth_ts = depth_frame.get_timestamp()
                color_ts = color_frame.get_timestamp()
                frame_number = depth_frame.get_frame_number()
                
                # 每30帧打印一次状态
                if frame_count % 30 == 0:
                    print(f"帧数: {frame_count}, 帧号: {frame_number}, "
                          f"深度时间戳: {depth_ts:.3f}ms, 彩色时间戳: {color_ts:.3f}ms")
                
                if show_preview:
                    # 转换为numpy数组用于显示
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # 将深度图转换为彩色图用于可视化
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    
                    # 水平拼接图像
                    images = np.hstack((color_image, depth_colormap))
                    
                    # 显示帧信息
                    cv2.putText(images, f"Frame: {frame_count}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('D455 Genlock Recording (Press Q to stop)', images)
                    
                    # 检查退出键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                else:
                    # 无预览模式，检查键盘输入
                    pass
                    
        except KeyboardInterrupt:
            print("\n接收到中断信号")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            self.stop_recording()
            if show_preview:
                cv2.destroyAllWindows()
            print(f"总共录制了 {frame_count} 帧")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RealSense D455 Genlock模式录制')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出bag文件路径 (默认: d455_genlock_时间戳.bag)')
    parser.add_argument('--no-preview', action='store_true',
                        help='禁用预览窗口')
    parser.add_argument('--width', type=int, default=848,
                        help='图像宽度 (默认: 848)')
    parser.add_argument('--height', type=int, default=480,
                        help='图像高度 (默认: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='帧率 (默认: 30)')
    parser.add_argument('--sync-mode', type=int, default=4,
                        help='同步模式: 0=默认, 1=Master, 2=Slave, 4=Genlock (默认: 4)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RealSense D455 Genlock模式录制器")
    print("=" * 50)
    # 获取版本号（不同版本的pyrealsense2获取方式不同）
    try:
        version = rs.__version__
    except AttributeError:
        try:
            version = rs.pyrealsense2.__version__
        except AttributeError:
            version = "未知"
    print(f"pyrealsense2版本: {version}")
    print()
    
    recorder = D455GenlockRecorder()
    recorder.record_with_preview(
        output_file=args.output,
        show_preview=not args.no_preview,
        sync_mode=args.sync_mode
    )


if __name__ == "__main__":
    main()
