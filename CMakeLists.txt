cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")

project(multi-sensor-sync)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
find_package(dv-processing REQUIRED)

file(GLOB SOURCE_FILES "*.cpp")
add_executable(main ${SOURCE_FILES})
target_link_libraries(main
    realsense2::realsense2
    ${OpenCV_LIBS}
    yaml-cpp
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::ui
    dv::processing
)
