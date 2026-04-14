+#include "realsense.hpp"
#include "prophesee.hpp"
#include "davis.hpp"
#include <csignal>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {
void handle_signal(int) {
    request_stop();
}

std::string exec_stdout(const std::string &cmd) {
    std::string out;
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        out += buf;
    }
    pclose(pipe);
    return out;
}

long long parse_timestamp_us(const std::string &s) {
    const std::string key = "Timestamp in microseconds:";
    auto pos = s.find(key);
    if (pos == std::string::npos) return -1;
    pos += key.size();
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    long long v = 0;
    bool any = false;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        any = true;
        v = v * 10 + (s[pos] - '0');
        pos++;
    }
    return any ? v : -1;
}
}

int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

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

    // 启动即轮询读取外部 timestamp，记录首个非零值
    threads.emplace_back([&]() {
        const std::string cmd = "/home/nvidia/Desktop/cxr_multi_sensor/test_timestamp/read_timestamp";

        const std::string out_dir = expand_user(file_dir) + today_date() + "/";
        try {
            std::filesystem::create_directories(out_dir);
        } catch (...) {
            // best-effort
        }
        const std::string out_path = out_dir + "boot_timestamp.csv";

        const auto t0 = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(240);

        while (!should_stop()) {
            if (std::chrono::steady_clock::now() - t0 > timeout) {
                std::ofstream f(out_path, std::ios::out);
                f << "status,timestamp_us\n";
                f << "timeout,0\n";
                f.close();
                printf("boot timestamp timeout -> %s\n", out_path.c_str());
                return;
            }

            const std::string out = exec_stdout(cmd);
            const long long ts_us = parse_timestamp_us(out);
            if (ts_us > 0) {
                std::ofstream f(out_path, std::ios::out);
                f << "status,timestamp_us\n";
                f << "ok," << ts_us << "\n";
                f.close();
                printf("Saved boot timestamp: %lld us -> %s\n", ts_us, out_path.c_str());
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

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
