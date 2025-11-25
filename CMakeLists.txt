cmake_minimum_required(VERSION 3.10)
project(multi-sensor-sync)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB SOURCE_FILES "*.cpp")
add_executable(main ${SOURCE_FILES})
target_link_libraries(main
    realsense2::realsense2
    ${OpenCV_LIBS}
    yaml-cpp
)
