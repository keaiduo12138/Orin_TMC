#include <librealsense2/rs.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

// 获取当前日期字符串 YYYY-MM-DD
std::string today_date() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    std::stringstream ss;
    ss << std::put_time(&now_tm, "%Y-%m-%d");
    return ss.str();
}

// 获取当前时间字符串 HH-MM-SS
std::string today_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    std::stringstream ss;
    ss << std::put_time(&now_tm, "%H-%M-%S");
    return ss.str();
}

#include <csignal>

std::atomic<bool> stopped(false);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        stopped = true;
    }
}

int main(int argc, char* argv[]) {
    // 注册信号处理 (Ctrl+C)
    std::signal(SIGINT, signal_handler);

    try {
        // 1. 设置保存路径
        // 默认保存到 ~/Desktop/resource/YYYY-MM-DD/HH-MM-SS.bag
        std::string base_dir = "/projects/cxr_lyh/resource/";
        std::string date_dir = base_dir + today_date() + "/";
        
        // 创建目录
        std::filesystem::create_directories(date_dir);
        
        std::string bag_filename = date_dir + today_time() + ".bag";
        std::cout << "准备录制到文件: " << bag_filename << std::endl;

        // 2. 初始化 RealSense Pipeline
        rs2::pipeline pipe;
        rs2::config cfg;

        // 3. 启用录制到文件
        // 这行代码是关键，它指示 pipeline 将所有数据写入 bag 文件
        cfg.enable_record_to_file(bag_filename);

        // 4. 配置数据流 (分辨率和帧率)
        // 使用 848x480 @ 90fps (深度) 和 640x360 @ 90fps (彩色)
        // 可以根据需要修改这些参数
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 360, RS2_FORMAT_Z16, 90);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 90);

        // 5. 开始录制
        std::cout << "正在搜索 RealSense 设备..." << std::endl;
        rs2::pipeline_profile profile = pipe.start(cfg);
        
        // 获取设备信息
        rs2::device dev = profile.get_device();
        std::cout << "已连接设备: " << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
        std::cout << "开始录制... 按 Ctrl+C 停止" << std::endl;

        // 6. 循环保持程序运行
        // pipeline.start(cfg) 之后，后台就已经在自动录制了
        
        auto start_time = std::chrono::steady_clock::now();
        while (!stopped) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            
            std::cout << "\r已录制: " << duration << " 秒" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\n正在停止录制并保存文件..." << std::endl;
        pipe.stop();
        std::cout << "录制完成，文件已保存。" << std::endl;

    } catch (const rs2::error& e) {
        std::cerr << "\nRealSense 错误调用: " << e.get_failed_function() 
            << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "\n其他错误: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


