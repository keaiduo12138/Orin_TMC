#include "realsense.hpp"
#include <filesystem>


RealSense::RealSense(const RunMode run_mode, std::string file_dir, std::string file_without_suffix)
: run_mode(run_mode) {
    // 展开路径
    file_dir = expand_user(file_dir);
    file_without_suffix = expand_user(file_without_suffix);

    rs2::device_list devices = this->ctx.query_devices();
    if (devices.size() < 1) {
        printf("realsense未连接, 退出\n");
        return;
    } else if (devices.size() > 1) {
        printf("有多个realsense, 退出\n");
        return;
    }

    if (this->run_mode == RunMode::RECORD) {
        std::string dir_path = file_dir + today_date() + '/';
        std::filesystem::create_directories(dir_path);
        this->file_to_save = dir_path + today_time() + ".bag";
        this->camera = rs2::recorder(file_to_save, devices[0]);
    } else if (this->run_mode == RunMode::PLAY) {
        this->file_to_play = file_without_suffix + ".bag";
        this->player = ctx.load_device(file_to_play);
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
    this->open();
    printf("realsense 开始录制\n");
    while (true) {
        pipe.wait_for_frames(60000);
    }
    pipe.stop();
}

void RealSense::play() {
    this->open();
    printf("realsense 开始播放\n");
    rs2::frameset fs;
    while (true) {
        if (pipe.poll_for_frames(&fs)) {
            rs2::frame color = fs.get_color_frame();
            rs2::frame depth = fs.get_depth_frame();
            
            cv::Mat cv_color = rsColor2cvMat(color);
            cv::Mat cv_depth = rsDepth2cvMat(depth);
            cv::imshow("RealSense Color", cv_color);
            cv::imshow("RealSense Depth", cv_depth);
            cv::waitKey(1);
        }
    }
    pipe.stop();
}

void RealSense::open() {
    if (run_mode == RunMode::RECORD) {
        if (!camera.has_value()) {
            printf("camera 未初始化, 无法打开设备\n");
            return;
        }

        // 设置 Genlock Mode
        rs2::sensor cam_sensor = camera->query_sensors().at(0);
        if (cam_sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE)) {
            cam_sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, 4);
            printf("设置 realsense 为 Genlock Mode\n");
        } else {
            printf("此 realsense 不支持 Genlock Mode\n");
        }

        // 启用内部原始时间戳
        if (cam_sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
            cam_sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0);
            printf("启用传感器原始 timestamp\n");
        } else {
            printf("无法启用传感器原始 timestamp\n");
        }

        cfg.enable_record_to_file(file_to_save);
        cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 90);

        pipe = rs2::pipeline(this->ctx);
        pipe.start(cfg);
    } else if (this->run_mode == RunMode::PLAY) {
        cfg.enable_device_from_file(file_to_play);
        cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 90);

        pipe = rs2::pipeline(this->ctx);
        pipe.start(cfg);
    } else {
        printf("Run Mode 非法, 程序退出\n");
    }
}
