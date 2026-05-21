cmake_minimum_required(VERSION 3.10)
project(simple_recorder)

set(CMAKE_CXX_STANDARD 17)

# 查找 RealSense 库
find_package(realsense2 REQUIRED)

# 编译 record_rs.cpp
add_executable(record_rs record_rs.cpp)
target_link_libraries(record_rs realsense2::realsense2)


