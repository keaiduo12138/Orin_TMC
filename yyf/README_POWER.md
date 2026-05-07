# 1.8V 电源控制方案

## 硬件方案

### 硬件连接

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Jetson AGX Orin (40-pin Header)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Pin 1  ────────────►  3.3V  ────────────►  LDO VIN              │
│  Pin 6  ────────────►  GND   ────────────►  LDO GND              │
│                                                                     │
│  Pin 13 (GPIO456) ──►  30Hz 触发信号 ────►  事件相机 Trigger IN  │
│                                                                     │
│  Pin 15 (GPIO433) ──►  1.8V 电源 EN  ────►  LDO EN               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   外部 LDO      │
                    │  (如 RT9080)    │
                    │                 │
                    │  VIN = 3.3V    │
                    │  VOUT = 1.8V   │
                    │  EN = GPIO433  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   特殊芯片      │
                    │   供电 1.8V    │
                    └─────────────────┘
```

### 推荐 LDO 芯片

| 芯片型号 | 输入电压 | 输出电压 | EN 逻辑 | 特点 |
|---------|---------|---------|---------|------|
| RT9080 | 2.5-5.5V | 1.8V/3.3V 可调 | 高电平使能 | 便宜，常用 |
| AP2112K-1.8 | 2.5-5.5V | 1.8V 固定 | 高电平使能 | 低噪声 |
| MIC5301-1.8 | 2.3-5.5V | 1.8V 固定 | 高电平使能 | 超低压差 |

---

## 软件使用说明

### 1. 独立控制 1.8V 电源

```bash
# 开启 1.8V 电源
sudo python3 yyf/power_1v8_ctrl.py on --gpio 433

# 关闭 1.8V 电源
sudo python3 yyf/power_1v8_ctrl.py off --gpio 433

# 查看状态
sudo python3 yyf/power_1v8_ctrl.py status --gpio 433

# 发送脉冲 (用于复位等)
sudo python3 yyf/power_1v8_ctrl.py pulse --gpio 433 --duration-ms 100
```

### 2. 30Hz 触发 + 1.8V 电源整合脚本

```bash
# 只启动 30Hz 触发，不控制 1.8V 电源
sudo python3 yyf/gpio_30hz_agx.py

# 启动 30Hz 触发 + 开启 1.8V 电源
sudo python3 yyf/gpio_30hz_agx.py --enable-power-1v8

# 只开启 1.8V 电源，不启动触发
sudo python3 yyf/gpio_30hz_agx.py --only-power

# 关闭 1.8V 电源
sudo python3 yyf/gpio_30hz_agx.py --power-off

# 查看 GPIO 状态
sudo python3 yyf/gpio_30hz_agx.py --status

# 自定义 GPIO 引脚
sudo python3 yyf/gpio_30hz_agx.py --trigger-gpio 32 --power-gpio 27
```

### 3. C++ 主程序集成

修改 `yyf/config.yaml`:

```yaml
# 启用 1.8V 电源控制
enable_power_1v8: 1

# GPIO 编号 (sysfs GPIO number)
# Pin 13 -> gpio456
# Pin 15 -> gpio433
power_1v8_gpio: 433
```

运行主程序时，会自动：
1. 启动前开启 1.8V 电源
2. 录制/播放结束后关闭 1.8V 电源

---

## GPIO 引脚映射

| 物理引脚 | GPIO 编号 (BCM) | 功能 | 备注 |
|---------|----------------|------|------|
| Pin 13  | GPIO456 (gpio456) | 30Hz 触发信号 | 默认 |
| Pin 15  | GPIO433 (gpio433) | 1.8V 电源 EN | 默认 |
| Pin 16  | GPIO357 (gpio357) | 可选 | |
| Pin 18  | GPIO391 (gpio391) | 可选 (PWM) | |

---

## 注意事项

1. **权限**：GPIO 控制需要 root 权限，请使用 `sudo`

2. **电平匹配**：
   - Orin GPIO 输出 3.3V
   - LDO EN 通常接受 3.3V (确认你的 LDO 规格)
   - 如果特殊芯片通信接口是 1.8V，需要电平转换

3. **开机状态**：
   - 部分 GPIO 有默认上拉/下拉
   - 建议电路设计有下拉电阻确保安全

4. **使用顺序**：
   - 先给特殊芯片供电 (1.8V)
   - 再启动触发信号
   - 录制结束后先停止触发，再断电
