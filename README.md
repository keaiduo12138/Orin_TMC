# 多传感器同步

## 注意事项

- 在终端里自动激活conda会导致cmake查找conda底下的库而不是系统库

    已经通过命令`conda config --set auto_activate_base false`把自动激活关闭

- `librealsense`使用`2.55.1-0~realsense.3336`版本
    直接使用最新版本 (2.56.x) 会要求`fastcdr`和`fastrtps`的依赖导致[编译不通过](https://github.com/realsenseai/librealsense/issues/14158)

    安装命令：

    ```bash
    sudo apt install librealsense2=2.55.1-0~realsense.3336 librealsense2-dev=2.55.1-0~realsense.3336 librealsense2-utils=2.55.1-0~realsense.3336 librealsense2-gl=2.55.1-0~realsense.3336 librealsense2-udev-rules=2.55.1-0~realsense.3336 --allow-downgrades
    ```
