import Jetson.GPIO as GPIO
import time

# 配置
# 物理引脚 13
PWM_PIN = 13
FREQUENCY_HZ = 30
PULSE_WIDTH_US = 100

def main():
    # 禁用警告
    GPIO.setwarnings(False)
    
    # 使用物理引脚编号 (BOARD模式)
    GPIO.setmode(GPIO.BOARD)
    
    # 设置引脚为输出
    GPIO.setup(PWM_PIN, GPIO.OUT, initial=GPIO.LOW)
    
    # 计算占空比
    # 周期 (秒)
    period_sec = 1.0 / FREQUENCY_HZ
    # 脉宽 (秒)
    pulse_width_sec = PULSE_WIDTH_US / 1e6
    # 占空比 (%) = (脉宽 / 周期) * 100
    duty_cycle = (pulse_width_sec / period_sec) * 100.0
    
    print(f"PWM 配置:")
    print(f"  设备: AGX Orin")
    print(f"  引脚: Board {PWM_PIN}")
    print(f"  频率: {FREQUENCY_HZ} Hz")
    print(f"  脉宽: {PULSE_WIDTH_US} us")
    print(f"  占空比: {duty_cycle:.4f}%")
    
    try:
        # 初始化 PWM
        pwm = GPIO.PWM(PWM_PIN, FREQUENCY_HZ)
        pwm.start(duty_cycle)
        
        print("PWM 波发生中... 按 Ctrl+C 停止")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n停止 PWM")
    finally:
        # 清理
        if 'pwm' in locals():
            pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
