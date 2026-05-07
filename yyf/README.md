# 多传感器同步

## 1.8V 供电控制（GPIO 控制外部 LDO 使能）

在 `yyf/power_1v8_ctrl.py` 中新增了通过 GPIO 控制外部 1.8V 稳压电路 EN 引脚的脚本。

硬件连接（你给出的方案）：
- `Orin Pin 1 (3.3V)` -> 外部 LDO `VIN`
- `Orin Pin 6 (GND)` -> 外部 LDO `GND`
- `Orin Pin 13 (GPIO, sysfs 456)` -> 外部 LDO `EN`

终端使用示例：

```bash
cd /home/nvidia/Desktop/cxr_multi_sensor/yyf
sudo python3 power_1v8_ctrl.py on
sudo python3 power_1v8_ctrl.py status
sudo python3 power_1v8_ctrl.py off
```

可用命令：
- `on`：拉高 EN，开启外部 1.8V 供电
- `off`：拉低 EN，关闭外部 1.8V 供电
- `status`：读取当前 GPIO 电平并推断供电状态
- `pulse --duration-ms 100`：输出一个高电平脉冲后自动拉低
- `release`：unexport GPIO

> 注意：GPIO 不能直接输出 1.8V 电源轨，只能作为逻辑控制信号。真正 1.8V 由外部稳压芯片产生。

## 注意事项

- 在终端里自动激活conda会导致cmake查找conda底下的库而不是系统库

    已经通过命令`conda config --set auto_activate_base false`把自动激活关闭

- `librealsense`使用`2.55.1-0~realsense.3336`版本

    直接使用最新版本 (2.56.x) 会要求`fastcdr`和`fastrtps`的依赖导致[编译不通过](https://github.com/realsenseai/librealsense/issues/14158)

    安装命令：

    ```bash
    sudo apt install librealsense2=2.55.1-0~realsense.3336 librealsense2-dev=2.55.1-0~realsense.3336 librealsense2-utils=2.55.1-0~realsense.3336 librealsense2-gl=2.55.1-0~realsense.3336 librealsense2-udev-rules=2.55.1-0~realsense.3336 --allow-downgrades
    ```

- `dv-processing`要求`gcc`编译器版本至少为`13.0`

    `apt`下载

    ```bash
    sudo apt install gcc-13
    sudo apt install g++-13
    ```

    然后在`CMakeLists.txt`中手动指定

    ```cmake
    set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
    ```

    注意：这两行要放在`project`前

- `dv-processing`要求`boost`版本至少为`1.80`

    手动下载最新版本编译安装

    ```bash
    wget https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
    tar -xvf boost-1.89.0-cmake.tar.xz
    cd boost-1.89.0
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    ```
