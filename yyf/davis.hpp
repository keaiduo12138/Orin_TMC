#pragma once

#include <optional>
#include <opencv2/opencv.hpp>

#include <dv-processing/io/camera/camera_input_base.hpp>
#include <dv-processing/io/camera/davis.hpp>
#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/io/mono_camera_writer.hpp>
#include <dv-processing/core/core.hpp>
#include <dv-processing/data/generate.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>

#include "utils.hpp"


class Davis {

public:
    Davis(const RunMode run_mode, std::string file_dir, std::string file_without_suffix);

    ~Davis() = default;

    void record();
    void play();

private:
    int open();

    cv::Size resolution;
    dv::io::camera::SyncCameraPtr camera;
    std::optional<dv::io::MonoCameraRecording> reader;
    std::optional<dv::io::MonoCameraWriter> writer;

    std::string file_to_save;
    std::string file_to_play;

    int run_mode = 0;

};

