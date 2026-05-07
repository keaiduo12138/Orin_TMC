#include "realsense.hpp"
#include <filesystem>
#include <chrono>
#include <exception>
#include <thread>


RealSense::RealSense(const RunMode run_mode, std::string file_dir, std::string file_without_suffix)
: run_mode(run_mode) {
    // 录制播放模式设置
    if (this->run_mode == RunMode::RECORD) {
        std::string dir_path = expand_user(file_dir) + today_date() + '/';
        std::filesystem::create_directories(dir_path);
        this->file_to_save = dir_path + today_time() + ".bag";
    } else if (this->run_mode == RunMode::PLAY) {
        this->file_to_play = expand_user(file_without_suffix) + ".bag";
    } else {
        printf("Run Mode 非法, 程序退出\n");
    }
}

cv::Mat rsColor2cvMat(const rs2::frame &color) {
    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();

    cv::Mat bgr_frame(
        cv::Size(w, h), CV_8UC3,
        (void*)color.get_data(), cv::Mat::AUTO_STEP
    );

    return bgr_frame;
}

cv::Mat rsDepth2cvMat(const rs2::frame &depth) {
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();

    return cv::Mat(
        cv::Size(w, h), CV_16UC1,
        (void*)depth.get_data(), cv::Mat::AUTO_STEP
    );
}

void RealSense::record() {
    if (this->open() != 1) {
        printf("RealSense 打开失败\n");
        return;
    }
    printf("RealSense 开始录制: %s\n", file_to_save.c_str());
    rs2::frameset fs;
    try {
        while (!should_stop()) {
            if (pipe.poll_for_frames(&fs)) {
                rs2::frame depth = fs.get_depth_frame();
                if (depth) {
                    static bool has_last = false;
                    static double last_ts = 0.0;
                    double ts = depth.get_timestamp(); // unit: ms
                    if (has_last) {
                        printf("dt = %.3f ms\n", ts - last_ts);
                    } else {
                        printf("first frame\n");
                        has_last = true;
                    }
                    last_ts = ts;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    } catch (const rs2::error &e) {
        if (!should_stop()) {
            printf("RealSense 录制异常: %s\n", e.what());
        }
    }
    try {
        pipe.stop();
    } catch (const rs2::error &e) {
        printf("RealSense pipe.stop 异常: %s\n", e.what());
    }

    if (!file_to_save.empty()) {
        try {
            if (std::filesystem::exists(file_to_save)) {
                const auto sz = std::filesystem::file_size(file_to_save);
                printf("RealSense 录制结束, 文件大小: %zu bytes\n", static_cast<size_t>(sz));
            } else {
                printf("RealSense 录制结束, 未找到文件: %s\n", file_to_save.c_str());
            }
        } catch (const std::exception &e) {
            printf("RealSense 获取文件大小失败: %s\n", e.what());
        }
    }
}

void RealSense::play() {
    if (this->open() != 1) {
        printf("RealSense 打开失败\n");
        return;
    }
    printf("RealSense 开始播放: %s\n", file_to_play.c_str());
    rs2::frameset fs;
    while (!should_stop()) {
        if (pipe.poll_for_frames(&fs)) {
            // rs2::frame color = fs.get_color_frame();
            rs2::frame depth = fs.get_depth_frame();
            
            // cv::Mat cv_color = rsColor2cvMat(color);
            cv::Mat cv_depth = rsDepth2cvMat(depth);
            // cv::imshow("RealSense Color", cv_color);
            cv::imshow("RealSense Depth", cv_depth);
            cv::waitKey(1);
        }
    }
    try {
        pipe.stop();
    } catch (const rs2::error &e) {
        printf("RealSense pipe.stop 异常: %s\n", e.what());
    }
}

// 配置并启动相机
int RealSense::open() {
    if (run_mode == RunMode::RECORD) {
        // 设备检测
        rs2::device_list devices = this->ctx.query_devices();
        if (devices.size() < 1) {
            printf("RealSense 未连接, 退出\n");
            return -1;
        } else if (devices.size() > 1) {
            printf("有多个 RealSense, 退出\n");
            return -1;
        }

        // 设置 Genlock Mode
        rs2::sensor cam_sensor = devices[0].query_sensors().at(0);
        if (cam_sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE)) {
            cam_sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, 4);
            printf("设置 RealSense 为 Genlock Mode\n");
        } else {
            printf("此 RealSense 不支持 Genlock Mode\n");
        }

        // 启用内部原始时间戳
        if (cam_sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
            cam_sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0);
            printf("启用传感器原始 timestamp\n");
        } else {
            printf("无法启用传感器原始 timestamp\n");
        }

        cfg.enable_record_to_file(file_to_save);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);

        pipe = rs2::pipeline(this->ctx);
        pipe.start(cfg);
        return 1;
    } else if (this->run_mode == RunMode::PLAY) {
        if (!file_to_play.empty()) {
            try {
                if (std::filesystem::exists(file_to_play)) {
                    const auto sz = std::filesystem::file_size(file_to_play);
                    printf("RealSense 将要打开文件: %s (%zu bytes)\n", file_to_play.c_str(), static_cast<size_t>(sz));
                } else {
                    printf("RealSense 将要打开文件: %s (不存在)\n", file_to_play.c_str());
                }
            } catch (const std::exception &e) {
                printf("RealSense 获取播放文件大小失败: %s\n", e.what());
            }
        }

        cfg.enable_device_from_file(file_to_play);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);

        pipe = rs2::pipeline(this->ctx);
        try {
            pipe.start(cfg);
        } catch (const rs2::error &e) {
            printf("RealSense 播放打开失败: %s\n", e.what());
            return -1;
        }
        return 1;
    } else {
        printf("Run Mode 非法, 程序退出\n");
        return -1;
    }
}
