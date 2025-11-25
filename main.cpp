#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <librealsense2/rs.hpp>

int read_mode(std::string config_path) {
    YAML::Node node = YAML::LoadFile(config_path);
    return node["sync_mode"].as<int>();
}

int read_fps(std::string config_path) {
    YAML::Node node = YAML::LoadFile(config_path);
    return node["fps"].as<int>();
}

cv::Mat rsDepth2cvMat(const rs2::frame &depth) {
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();

    return cv::Mat(
        cv::Size(w, h), CV_16UC1,
        (void*)depth.get_data(), cv::Mat::AUTO_STEP
    );
}

cv::Mat rsColor2cvMat(const rs2::frame &color) {
    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();

    cv::Mat rgb_frame(
        cv::Size(w, h), CV_8UC3,
        (void*)color.get_data(), cv::Mat::AUTO_STEP
    );

    // 这里会发生一次内存拷贝
    cv::Mat bgr_frame;
    cv::cvtColor(rgb_frame, bgr_frame, cv::COLOR_RGB2BGR);

    return bgr_frame;
}

void openRS() {
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
        std::cerr << "没有相机" << std::endl;
        return;
    }

    // 只有一个D455
    rs2::device d455 = devices[0];
    printf("设备信息: %s\n", d455.get_info(RS2_CAMERA_INFO_NAME));

    /*
    enum inter_cam_sync_mode
    {
        INTERCAM_SYNC_DEFAULT = 0,
        INTERCAM_SYNC_MASTER = 1,
        INTERCAM_SYNC_SLAVE = 2,
        INTERCAM_SYNC_FULL_SLAVE = 3,
        INTERCAM_SYNC_MAX = 260  // 4-258 are for Genlock with burst count of 1-255 frames for each trigger.
                                    // 259 for Sending two frame - First with laser ON, and the other with laser OFF.
                                    // 260 for Sending two frame - First with laser OFF, and the other with laser ON.
    };
    */

    // https://github.com/IntelRealSense/librealsense/issues/14131
    // full slave模式是为D415相机设计的，即使这样D415相机也不能很好的工作
    // 因此官方并不推荐使用full slave

    // genlock模式的官方文档已没有在线链接，但可以下载离线pdf
    // https://github.com/user-attachments/files/21602818/External.Synchronization.pdf

    int sync_mode = read_mode("../config.yaml");
    rs2::sensor d455_sensor = d455.query_sensors()[0];
    if (d455_sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE)) {
        d455_sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, sync_mode);
        printf("设置D455为: %s\n",
            sync_mode == 0 ? "default mode" :
            sync_mode == 1 ? "master mode" :
            sync_mode == 2 ? "slave mode" :
            sync_mode == 3 ? "full slave mode" :
            "genlock mode"
        );
    } else {
        std::cerr << "该相机不支持硬件同步" << std::endl;
        return;
    }

    int fps = read_fps("../config.yaml");
    rs2::pipeline pipe(ctx);
    rs2::config cfg;
    cfg.enable_device(d455.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, fps);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, fps);

    pipe.start(cfg);

    rs2::frameset fs;
    // 打印帧率
    // int frame_cnt = 0;
    // auto last_time = std::chrono::high_resolution_clock::now();
    while(1) {
      if (pipe.poll_for_frames(&fs)) {
        // frame_cnt++;
        rs2::frame color = fs.get_color_frame();
        rs2::frame depth = fs.get_depth_frame();
        
        // auto now = std::chrono::high_resolution_clock::now();
        // double elapsed_seconds = std::chrono::duration<double>(now - last_time).count();
        
        // if (elapsed_seconds >= 1.0) {
        //     printf("fps: %.2f\n", frame_cnt / elapsed_seconds);
        //     frame_cnt = 0;
        //     last_time = now;
        // }
        cv::Mat cv_color = rsColor2cvMat(color);
        cv::Mat cv_depth = rsDepth2cvMat(depth);
        cv::imshow("color", cv_color);
        cv::imshow("depth", cv_depth);
        cv::waitKey(1);
      }
    }
}

int main() {
    openRS();
    return 0;
}
