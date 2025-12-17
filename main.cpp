#include "realsense.hpp"
#include "prophesee.hpp"


int main() {
    YAML::Node config = YAML::LoadFile("../config.yaml");
    RunMode run_mode = static_cast<RunMode>(config["run_mode"].as<int>());
    std::string file_dir = config["file_dir"].as<std::string>();
    std::string file_without_suffix = config["file_without_suffix"].as<std::string>();

    int enable_realsense = config["enable_realsense"].as<int>();
    int enable_prophesee = config["enable_prophesee"].as<int>();

    RealSense d455 = RealSense(run_mode, file_dir, file_without_suffix);
    Prophesee ekv4 = Prophesee(run_mode, file_dir, file_without_suffix);

    if (run_mode == RunMode::RECORD) {
        if (enable_realsense) std::jthread t1([&]() { d455.record(); });
        if (enable_prophesee) std::jthread t2([&]() { ekv4.record(); });
    } else if (run_mode == RunMode::PLAY) {
        if (enable_realsense) std::jthread t1([&]() { d455.play(); });
        if (enable_prophesee) std::jthread t2([&]() { ekv4.play(); });
    } else {
        printf("错误! Run Mode 未定义\n");
        return -1;
    }


    return 0;
}
