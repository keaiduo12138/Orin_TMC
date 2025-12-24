#include "realsense.hpp"
#include "prophesee.hpp"
#include "davis.hpp"
#include <vector>


int main() {
    YAML::Node config = YAML::LoadFile("../config.yaml");
    RunMode run_mode = static_cast<RunMode>(config["run_mode"].as<int>());
    std::string file_dir = config["file_dir"].as<std::string>();
    std::string file_without_suffix = config["file_without_suffix"].as<std::string>();

    int enable_realsense = config["enable_realsense"].as<int>();
    int enable_prophesee = config["enable_prophesee"].as<int>();
    int enable_davis = config["enable_davis"].as<int>();

    std::unique_ptr<RealSense> d455;
    std::unique_ptr<Prophesee> ekv4;
    std::unique_ptr<Davis> d346;

    if (enable_realsense) d455 = std::make_unique<RealSense>(run_mode, file_dir, file_without_suffix);
    if (enable_prophesee) ekv4 = std::make_unique<Prophesee>(run_mode, file_dir, file_without_suffix);
    if (enable_davis) d346 = std::make_unique<Davis>(run_mode, file_dir, file_without_suffix);

    std::vector<std::jthread> threads;

    if (run_mode == RunMode::RECORD) {
        if (enable_realsense) threads.emplace_back([&]() { d455->record(); });
        if (enable_prophesee) threads.emplace_back([&]() { ekv4->record(); });
        if (enable_davis) threads.emplace_back([&]() { d346->record(); });
    } else if (run_mode == RunMode::PLAY) {
        if (enable_realsense) threads.emplace_back([&]() { d455->play(); });
        if (enable_prophesee) threads.emplace_back([&]() { ekv4->play(); });
        if (enable_davis) threads.emplace_back([&]() { d346->play(); });
    } else {
        printf("错误! Run Mode 未定义\n");
        return -1;
    }
    return 0;
}
