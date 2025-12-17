#include <iostream>
#include "realsense.hpp"
#include "prophesee.hpp"


int main() {
    YAML::Node config = YAML::LoadFile("../config.yaml");
    RunMode run_mode = static_cast<RunMode>(config["run_mode"].as<int>());
    std::string file_dir = config["file_dir"].as<std::string>();
    std::string file_without_suffix = config["file_without_suffix"].as<std::string>();

    RealSense d455 = RealSense(run_mode, file_dir, file_without_suffix);
    Prophesee ekv4 = Prophesee(run_mode, file_dir, file_without_suffix);

    if (run_mode == RunMode::RECORD) {
        std::jthread t1([&]() { d455.record(); });
        std::jthread t2([&]() { ekv4.record(); });
    } else if (run_mode == RunMode::PLAY) {
        std::jthread t1([&]() { d455.play(); });
        std::jthread t2([&]() { ekv4.play(); });
    } else {
        printf("错误! Run Mode 未定义\n");
        return -1;
    }


    return 0;
}
