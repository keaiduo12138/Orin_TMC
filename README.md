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
