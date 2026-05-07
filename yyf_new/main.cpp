#include "realsense.hpp"
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
#include <cstdlib>

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

// 1.8V 电源控制相关配置
const std::string POWER_CTRL_SCRIPT = "/home/nvidia/Desktop/cxr_multi_sensor/yyf/power_1v8_ctrl.py";
const int POWER_GPIO = 433;  // Pin 15 (GPIO433, Pad PN.01)

void power_1v8_on() {
    printf("[POWER] 开启 1.8V 电源 (GPIO %d)...\n", POWER_GPIO);
    std::string cmd = "sudo python3 " + POWER_CTRL_SCRIPT + " on --gpio " + std::to_string(POWER_GPIO);
    int ret = system(cmd.c_str());
    if (ret == 0) {
        printf("[POWER] 1.8V 电源已开启\n");
    } else {
        printf("[POWER] 警告: 1.8V 电源开启失败 (返回码: %d)\n", ret);
    }
}

void power_1v8_off() {
    printf("[POWER] 关闭 1.8V 电源 (GPIO %d)...\n", POWER_GPIO);
    std::string cmd = "sudo python3 " + POWER_CTRL_SCRIPT + " off --gpio " + std::to_string(POWER_GPIO);
    int ret = system(cmd.c_str());
    if (ret == 0) {
        printf("[POWER] 1.8V 电源已关闭\n");
    } else {
        printf("[POWER] 警告: 1.8V 电源关闭失败 (返回码: %d)\n", ret);
    }
}

void power_1v8_pulse(int duration_ms = 100) {
    printf("[POWER] 1.8V 电源脉冲 (%d ms)...\n", duration_ms);
    std::string cmd = "sudo python3 " + POWER_CTRL_SCRIPT + " pulse --gpio " + 
                      std::to_string(POWER_GPIO) + " --duration-ms " + std::to_string(duration_ms);
    system(cmd.c_str());
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
    
    // 新增：1.8V 电源控制配置
    int enable_power_1v8 = config["enable_power_1v8"].as<int>();
    int power_1v8_gpio = config["power_1v8_gpio"].as<int>(433);

    std::unique_ptr<RealSense> d435;
    std::unique_ptr<Prophesee> ekv4;
    std::unique_ptr<Davis> d346;

    if (enable_realsense) d435 = std::make_unique<RealSense>(run_mode, file_dir, file_without_suffix);
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

    // ========== 1.8V 电源控制 ==========
    // 注意：如果启用了 1.8V 电源控制，在录制/播放开始前开启
    if (enable_power_1v8 && (run_mode == RunMode::RECORD || run_mode == RunMode::PLAY)) {
        power_1v8_on();
        
        // 等待电源稳定 (可选)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (run_mode == RunMode::RECORD) {
        if (enable_realsense) threads.emplace_back([&]() { d435->record(); });
        if (enable_prophesee) threads.emplace_back([&]() { ekv4->record(); });
        if (enable_davis) threads.emplace_back([&]() { d346->record(); });
    } else if (run_mode == RunMode::PLAY) {
        if (enable_realsense) threads.emplace_back([&]() { d435->play(); });
        if (enable_prophesee) threads.emplace_back([&]() { ekv4->play(); });
        if (enable_davis) threads.emplace_back([&]() { d346->play(); });
    } else {
        printf("错误! Run Mode 未定义\n");
        return -1;
    }
    
    // 等待所有线程结束
    threads.clear();
    
    // ========== 关闭 1.8V 电源 ==========
    if (enable_power_1v8) {
        power_1v8_off();
    }
    
    return 0;
}
