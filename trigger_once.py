import Jetson.GPIO as GPIO
import time

# 配置
PIN = 13  # 物理引脚 13 (Board 模式)
PULSE_WIDTH_US = 100  # 脉宽 100 微秒

def main():
    # 禁用警告
    GPIO.setwarnings(False)
    
    # 使用物理引脚编号 (BOARD模式)
    GPIO.setmode(GPIO.BOARD)
    
    # 设置引脚为输出，初始为低电平
    GPIO.setup(PIN, GPIO.OUT, initial=GPIO.LOW)
    
    print(f"准备在物理引脚 {PIN} 上产生一次脉冲...")
    print(f"目标脉宽: {PULSE_WIDTH_US} 微秒")

    try:
        # 产生脉冲
        # 拉高
        GPIO.output(PIN, GPIO.HIGH)
        
        # 保持高电平 100 微秒
        # 注意: Python 的 time.sleep 在微秒级可能不够极其精确，但在非实时系统上通常足够作为触发信号
        time.sleep(PULSE_WIDTH_US / 1e6)
        
        # 拉低
        GPIO.output(PIN, GPIO.LOW)
        
        print("脉冲发送完成")
        
    finally:
        # 清理 GPIO 状态
        GPIO.cleanup()

if __name__ == "__main__":
    main()

