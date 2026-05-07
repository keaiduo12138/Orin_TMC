#!/usr/bin/env python3
"""
通过 Jetson GPIO 控制外部 1.8V 稳压电路 EN 引脚。

硬件假设：
- Orin Pin 1  (3.3V) -> LDO VIN
- Orin Pin 6  (GND)  -> LDO GND
- Orin Pin 15 (Pad: PN.01, sysfs gpio433) -> LDO EN (高电平使能)

说明：
- 本脚本不会把 GPIO "变成 1.8V 输出"。
- GPIO 仅作为开关信号控制外部 LDO EN，1.8V 由外部稳压芯片提供。
- 新版 JetPack 内核使用 Pad 名称（如 PN.01）而非数字（gpio433）作为 sysfs 目录名。
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


# Pin 15 对应 sysfs gpio433，Pad 名称为 PN.01
DEFAULT_GPIO = 433
GPIO_BASE = Path("/sys/class/gpio")

# sysfs GPIO 编号 -> Pad 名称映射（Jetson AGX Orin）
# 来源：Jetson.GPIO 库 JETSON_ORIN_PIN_DEFS
# 格式：chip_gpio + base(348 for tegra234-gpio, 316 for tegra234-gpio-aon)
_SYSFS_TO_PAD = {
    454: "PQ.06",   # Pin 7
    460: "PR.04",   # Pin 11
    398: "PH.07",   # Pin 12
    456: "PR.00",   # Pin 13  ← 30Hz 触发信号
    433: "PN.01",   # Pin 15  ← LDO EN 默认引脚
    325: "PAA.06",  # Pin 16
    391: "PI.02",   # Pin 18
    483: "PZ.05",   # Pin 19
    482: "PZ.04",   # Pin 21
    444: "PP.04",   # Pin 22
    481: "PZ.03",   # Pin 23
    484: "PZ.06",   # Pin 24
    485: "PZ.07",   # Pin 26
    317: "PAA.01",  # Pin 29
    316: "PAA.00",  # Pin 31
    324: "PAA.05",  # Pin 32
    318: "PAA.02",  # Pin 33
    401: "PH.03",   # Pin 35
    461: "PR.05",   # Pin 36
    319: "PAA.03",  # Pin 37
    400: "PH.02",   # Pin 38
    399: "PH.01",   # Pin 40
}


def get_pad_name(gpio_num: int) -> Optional[str]:
    """
    返回 sysfs GPIO 编号对应的 Pad 名称。
    优先查静态表，失败则尝试从 Jetson.GPIO 库动态查询。
    """
    if gpio_num in _SYSFS_TO_PAD:
        return _SYSFS_TO_PAD[gpio_num]

    try:
        from Jetson.GPIO import gpio_pin_data
        defs = gpio_pin_data.JETSON_ORIN_PIN_DEFS
        for entry in defs:
            chip_gpio, pad_name, chip_label = entry[0], entry[1], entry[2]
            base = 348 if chip_label == "tegra234-gpio" else 316
            if base + chip_gpio == gpio_num:
                return pad_name
    except Exception:
        pass

    return None


def gpio_path(gpio_num: int) -> Path:
    """
    返回该 GPIO 在 sysfs 中的实际目录路径。
    新版内核用 Pad 名称（PR.00）而非数字（gpio456）命名目录，两者都尝试。
    """
    numeric = GPIO_BASE / f"gpio{gpio_num}"
    if numeric.exists():
        return numeric

    pad = get_pad_name(gpio_num)
    if pad:
        named = GPIO_BASE / pad
        if named.exists():
            return named

    # 都不存在时返回 numeric，后续操作会给出清晰报错
    return numeric


def write_text(path: Path, text: str):
    path.write_text(text)


def ensure_exported(gpio_num: int):
    if gpio_path(gpio_num).exists():
        return
    # 尝试通过 export 导出
    try:
        write_text(GPIO_BASE / "export", str(gpio_num))
        time.sleep(0.1)  # 等待内核创建目录
    except OSError:
        pass

    if not gpio_path(gpio_num).exists():
        pad = get_pad_name(gpio_num)
        hint = f"（Pad 名称: {pad}）" if pad else ""
        raise FileNotFoundError(
            f"GPIO {gpio_num}{hint} export 后仍不存在，"
            f"请检查 GPIO 编号是否正确。\n"
            f"  可用路径: {list(GPIO_BASE.iterdir())}"
        )


def ensure_output_mode(gpio_num: int):
    direction = gpio_path(gpio_num) / "direction"
    if direction.read_text().strip() != "out":
        write_text(direction, "out")


def set_value(gpio_num: int, value: int):
    value_path = gpio_path(gpio_num) / "value"
    write_text(value_path, "1" if value else "0")


def get_value(gpio_num: int) -> int:
    value_path = gpio_path(gpio_num) / "value"
    return int(value_path.read_text().strip())


def unexport(gpio_num: int):
    gp = gpio_path(gpio_num)
    if gp.exists():
        try:
            write_text(GPIO_BASE / "unexport", str(gpio_num))
        except OSError:
            pass


def require_root_if_needed():
    if os.geteuid() != 0:
        print("[WARN] 建议使用 sudo 运行，否则可能无权限写 /sys/class/gpio")


def _gpio_info(gpio_num: int) -> str:
    pad = get_pad_name(gpio_num)
    path = gpio_path(gpio_num)
    pad_str = f", Pad={pad}" if pad else ""
    return f"GPIO{gpio_num}{pad_str}, sysfs路径={path}"


def cmd_on(gpio_num: int):
    ensure_exported(gpio_num)
    ensure_output_mode(gpio_num)
    set_value(gpio_num, 1)
    print(f"[{_gpio_info(gpio_num)}] = 1 -> EN 高电平，外部 1.8V 电源应开启")


def cmd_off(gpio_num: int):
    ensure_exported(gpio_num)
    ensure_output_mode(gpio_num)
    set_value(gpio_num, 0)
    print(f"[{_gpio_info(gpio_num)}] = 0 -> EN 低电平，外部 1.8V 电源应关闭")


def cmd_status(gpio_num: int):
    gp = gpio_path(gpio_num)
    if not gp.exists():
        print(f"GPIO{gpio_num} 尚未 export，路径不存在: {gp}")
        return 1
    v = get_value(gpio_num)
    print(f"[{_gpio_info(gpio_num)}] 当前值: {v}")
    if v == 1:
        print("状态推断: 外部 LDO EN=高，1.8V 供电开启")
    else:
        print("状态推断: 外部 LDO EN=低，1.8V 供电关闭")
    return 0


def cmd_pulse(gpio_num: int, duration_ms: int):
    ensure_exported(gpio_num)
    ensure_output_mode(gpio_num)
    set_value(gpio_num, 1)
    print(f"[{_gpio_info(gpio_num)}] = 1，保持 {duration_ms} ms")
    time.sleep(duration_ms / 1000.0)
    set_value(gpio_num, 0)
    print(f"[{_gpio_info(gpio_num)}] = 0，脉冲结束")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="控制外部 1.8V 稳压电路 EN 引脚")
    p.add_argument("command", choices=["on", "off", "status", "pulse", "release"],
                   help="控制命令")
    p.add_argument("--gpio", type=int, default=DEFAULT_GPIO,
                   help=f"sysfs GPIO 编号（默认 {DEFAULT_GPIO}，对应 Pin15/Pad PN.01）")
    p.add_argument("--duration-ms", type=int, default=100,
                   help="pulse 命令高电平持续时间 (ms)")
    return p


def main() -> int:
    require_root_if_needed()
    args = build_parser().parse_args()

    try:
        if args.command == "on":
            cmd_on(args.gpio)
            return 0
        if args.command == "off":
            cmd_off(args.gpio)
            return 0
        if args.command == "status":
            return cmd_status(args.gpio)
        if args.command == "pulse":
            cmd_pulse(args.gpio, args.duration_ms)
            return 0
        if args.command == "release":
            unexport(args.gpio)
            print(f"GPIO{args.gpio} 已 unexport")
            return 0
        print("未知命令")
        return 2
    except PermissionError:
        print("[ERR] 权限不足，请使用 sudo")
        return 13
    except FileNotFoundError as exc:
        print(f"[ERR] 系统路径不存在: {exc}")
        return 2
    except Exception as exc:
        print(f"[ERR] 执行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
