#!/usr/bin/env python3
import Jetson.GPIO as GPIO
import time
import sys

# 使用 BCM 编号模式
GPIO.setmode(GPIO.BCM)

# 使用 gpio-474 (PY.04)
output_pin = 474

try:
    # 设置 GPIO 为输出模式
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
    
    print(f"Jetson AGX Orin - 在 GPIO {output_pin} (PY.04) 输出 30Hz 信号")
    print("频率: 30Hz, 周期: 33.33ms, 占空比: 50%")
    print("按 Ctrl+C 停止")
    
    # 30Hz 参数
    frequency = 30  # Hz
    period = 1.0 / frequency  # 33.333ms
    half_period = period / 2  # 16.667ms
    
    counter = 0
    # 生成精确的 30Hz 方波
    while True:
        GPIO.output(output_pin, GPIO.HIGH)
        time.sleep(half_period)
        GPIO.output(output_pin, GPIO.LOW)
        time.sleep(half_period)
        
        counter += 1
        if counter % 100 == 0:  # 每100周期打印一次状态
            print(f"运行中... 已生成 {counter} 个周期")
        
except KeyboardInterrupt:
    print("\n✓ 程序被用户中断")
except Exception as e:
    print(f"✗ 发生错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 清理 GPIO 设置
    GPIO.cleanup()
    print("✓ GPIO 资源已释放")
