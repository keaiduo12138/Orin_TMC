#!/usr/bin/env python3
"""
GPIO 控制 30Hz 触发信号 + 1.8V 电源控制整合脚本

功能：
- 输出 30Hz 触发信号到 GPIO (用于同步事件相机)
- 可选：控制 1.8V 外部电源的 EN 引脚

硬件连接：
- 30Hz 触发信号: Pin 13 (gpio456) 或其他 GPIO
- 1.8V 电源 EN: Pin 15 (gpio433) 或通过 config 配置

使用示例：
  # 只启动 30Hz 触发，不控制 1.8V 电源
  sudo python3 gpio_30hz_agx.py

  # 启动 30Hz 触发 + 开启 1.8V 电源
  sudo python3 gpio_30hz_agx.py --enable-power-1v8

  # 只控制 1.8V 电源，不启动触发
  sudo python3 gpio_30hz_agx.py --only-power

  # 自定义 GPIO 引脚
  sudo python3 gpio_30hz_agx.py --trigger-gpio 32 --power-gpio 27
"""

import Jetson.GPIO as GPIO
import time
import sys
import argparse
import signal
import subprocess

# 默认引脚配置 (基于 Jetson AGX Orin 40pin header)
# 注意：这里使用 sysfs GPIO 编号，新版内核路径为 Pad 名称（power_1v8_ctrl.py 会自动处理）
# Pin 13 -> sysfs gpio456, Pad 名称 PR.00  ← 30Hz 触发信号
# Pin 15 -> sysfs gpio433, Pad 名称 PN.01  ← 1.8V 电源 EN
# 两个功能必须使用不同引脚，不可复用
DEFAULT_TRIGGER_GPIO = 456   # Pin 13, Pad PR.00（30Hz 触发信号输出到事件相机）
DEFAULT_POWER_GPIO = 433     # Pin 15, Pad PN.01（1.8V LDO EN 控制）

# 30Hz 参数
TRIGGER_FREQUENCY = 30  # Hz
TRIGGER_PERIOD = 1.0 / TRIGGER_FREQUENCY  # 33.333ms
TRIGGER_HALF_PERIOD = TRIGGER_PERIOD / 2  # 16.667ms

# 全局标志
running = True

def signal_handler(sig, frame):
    global running
    print("\n[INFO] 接收到中断信号，正在退出...")
    running = False


def control_power_1v8(command, gpio_num):
    """
    控制 1.8V 电源的 EN 引脚
    command: "on", "off", "status"
    """
    script_path = "/home/nvidia/Desktop/cxr_multi_sensor/yyf/power_1v8_ctrl.py"
    
    cmd = ["sudo", "python3", script_path, command, "--gpio", str(gpio_num)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"[POWER] {command} 执行成功")
            if result.stdout:
                print(f"[POWER] {result.stdout.strip()}")
            return True
        else:
            print(f"[POWER] {command} 执行失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"[POWER] 执行出错: {e}")
        return False


def setup_gpio(trigger_gpio, power_gpio, enable_trigger, enable_power):
    """初始化 GPIO 引脚"""
    GPIO.setmode(GPIO.BCM)
    
    if enable_trigger:
        GPIO.setup(trigger_gpio, GPIO.OUT, initial=GPIO.LOW)
        print(f"[GPIO] 触发信号引脚: GPIO {trigger_gpio} (Pin 13)")
    
    if enable_power:
        GPIO.setup(power_gpio, GPIO.OUT, initial=GPIO.LOW)
        print(f"[GPIO] 1.8V 电源 EN 引脚: GPIO {power_gpio} (Pin 15)")


def run_trigger_loop(trigger_gpio):
    """运行 30Hz 触发信号循环"""
    counter = 0
    print(f"[TRIGGER] 开始输出 30Hz 触发信号 (GPIO {trigger_gpio})")
    
    while running:
        GPIO.output(trigger_gpio, GPIO.HIGH)
        time.sleep(TRIGGER_HALF_PERIOD)
        GPIO.output(trigger_gpio, GPIO.LOW)
        time.sleep(TRIGGER_HALF_PERIOD)
        
        counter += 1
        if counter % 100 == 0:
            print(f"[TRIGGER] 运行中... 已生成 {counter} 个周期")


def show_status(trigger_gpio, power_gpio):
    """显示当前 GPIO 状态"""
    print("\n========== GPIO 状态 ==========")
    print(f"触发信号 GPIO: {trigger_gpio}")
    print(f"1.8V 电源 EN GPIO: {power_gpio}")
    
    # 读取当前值
    try:
        trigger_val = GPIO.input(trigger_gpio)
        print(f"触发信号当前: {'HIGH' if trigger_val else 'LOW'}")
    except:
        print("触发信号: 未初始化")
    
    try:
        power_val = GPIO.input(power_gpio)
        print(f"1.8V 电源 EN 当前: {'HIGH' if power_val else 'LOW'}")
    except:
        print("1.8V 电源 EN: 未初始化")
    
    print("================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="GPIO 控制: 30Hz 触发信号 + 1.8V 电源控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 启动 30Hz 触发
  %(prog)s --enable-power-1v8       # 启动 30Hz 触发 + 开启 1.8V 电源
  %(prog)s --only-power             # 只开启 1.8V 电源
  %(prog)s --power-off              # 关闭 1.8V 电源
  %(prog)s --status                 # 查看 GPIO 状态
  %(prog)s --trigger-gpio 32 --power-gpio 27  # 自定义引脚
        """
    )
    
    parser.add_argument("--trigger-gpio", type=int, default=DEFAULT_TRIGGER_GPIO,
                        help=f"触发信号 GPIO 编号 (默认: {DEFAULT_TRIGGER_GPIO})")
    parser.add_argument("--power-gpio", type=int, default=DEFAULT_POWER_GPIO,
                        help=f"1.8V 电源 EN GPIO 编号 (默认: {DEFAULT_POWER_GPIO})")
    parser.add_argument("--enable-power-1v8", action="store_true",
                        help="启动时同时开启 1.8V 电源")
    parser.add_argument("--only-power", action="store_true",
                        help="只控制 1.8V 电源，不启动触发信号")
    parser.add_argument("--power-off", action="store_true",
                        help="只关闭 1.8V 电源")
    parser.add_argument("--status", action="store_true",
                        help="查看 GPIO 状态")
    
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 状态查询模式
    if args.status:
        GPIO.setmode(GPIO.BCM)
        show_status(args.trigger_gpio, args.power_gpio)
        GPIO.cleanup()
        return 0
    
    # 单独控制电源模式
    if args.power_off:
        print(f"[INFO] 关闭 1.8V 电源 (GPIO {args.power_gpio})")
        control_power_1v8("off", args.power_gpio)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(args.power_gpio, GPIO.OUT, initial=GPIO.LOW)
        GPIO.cleanup()
        return 0
    
    # 初始化 GPIO
    enable_trigger = not args.only_power and not args.power_off
    enable_power = args.enable_power_1v8 or args.only_power
    
    setup_gpio(args.trigger_gpio, args.power_gpio, enable_trigger, enable_power)
    
    # 启动时开启 1.8V 电源
    if enable_power:
        print(f"[INFO] 启动时开启 1.8V 电源 (GPIO {args.power_gpio})")
        control_power_1v8("on", args.power_gpio)
        time.sleep(0.5)  # 等待电源稳定
    
    # 显示状态
    show_status(args.trigger_gpio, args.power_gpio)
    
    # 启动触发信号循环
    if enable_trigger:
        try:
            run_trigger_loop(args.trigger_gpio)
        except Exception as e:
            print(f"[ERROR] 触发循环异常: {e}")
    else:
        # 只控制电源时，等待中断
        print("[INFO] 只控制电源模式，按 Ctrl+C 退出")
        while running:
            time.sleep(0.1)
    
    # 退出时清理
    print("[INFO] 清理 GPIO...")
    GPIO.cleanup()
    
    # 退出时关闭 1.8V 电源
    if enable_power:
        print(f"[INFO] 退出时关闭 1.8V 电源")
        control_power_1v8("off", args.power_gpio)
    
    print("[INFO] 程序结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
