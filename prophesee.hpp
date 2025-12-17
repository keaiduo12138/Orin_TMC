#pragma once

#include <thread>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/flip_x_algorithm.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/driver/ext_trigger.h>
#include <metavision/hal/facilities/i_trigger_in.h>

#include "utils.hpp"


class Prophesee {

public:
    Prophesee() = default;
    Prophesee(const RunMode run_mode, std::string file_dir, std::string file_without_suffix);

    ~Prophesee();

    void record();
    void play();

private:
    void open();
    void close();

    Metavision::Camera cam;

    std::string file_to_save;
    std::string file_to_play;

    int run_mode = 0;

};
