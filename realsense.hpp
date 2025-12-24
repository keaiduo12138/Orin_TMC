#pragma once

#include <optional>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include "utils.hpp"


class RealSense {

public:
    RealSense(const RunMode run_mode, std::string file_dir, std::string file_without_suffix);

    ~RealSense() = default;

    void record();
    void play();

private:
    int open();

    rs2::context ctx;
    rs2::config cfg;
    rs2::pipeline pipe;
    std::optional<rs2::recorder> camera;
    std::optional<rs2::playback> player;

    std::string file_to_save;
    std::string file_to_play;

    int run_mode = 0;

};
