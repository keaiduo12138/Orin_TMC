#include "prophesee.hpp"
#include <filesystem>


Prophesee::Prophesee(const RunMode run_mode, std::string file_dir, std::string file_without_suffix)
: run_mode(run_mode) {
    // 展开路径
    file_dir = expand_user(file_dir);
    file_without_suffix = expand_user(file_without_suffix);

    if (run_mode == RunMode::RECORD) {
        std::string dir_path = file_dir + today_date() + '/';
        std::filesystem::create_directories(dir_path);
        this->file_to_save = dir_path + today_time() + ".raw";
    } else if (run_mode == RunMode::PLAY) {
        this->file_to_play = file_without_suffix + ".raw";
    } else {
        printf("Run Mode 非法, 程序退出\n");
    }
}

Prophesee::~Prophesee() {
    this->close();
}

void Prophesee::record() {
    if (this->open() != 1) {
        printf("Prophesee 打开失败\n");
        return;
    }
    cam.ext_trigger().add_callback([&](
        const Metavision::EventExtTrigger *ev_begin,
        const Metavision::EventExtTrigger *ev_end) {
            for (const auto *ev = ev_begin; ev != ev_end; ++ev) {
                    printf("Prophesee 检测到信号 | 时间戳: %lld\n", ev->t);
                }
            }
        );

    while (cam.is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

void Prophesee::play() {
    if (this->open() != 1) {
        printf("Prophesee 打开失败\n");
        return;
    }
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(
        cam.geometry().width(),
        cam.geometry().height(),
        33333,
        30
    );
    cam.ext_trigger().add_callback([&](
        const Metavision::EventExtTrigger *begin,
        const Metavision::EventExtTrigger *end) {
            for (const auto *ev = begin; ev != end; ++ev) {
                printf("Prophesee 检测到信号 | 时间戳: %lld\n", ev->t);
            }
        }
    );
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        frame_gen.process_events(begin, end);
    });
    Metavision::Window window(
        "Prophesee Event",
        cam.geometry().width(),
        cam.geometry().height(),
        Metavision::BaseWindow::RenderMode::BGR
    );
    frame_gen.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        window.show(frame);
    });
    while (cam.is_running()) {
        static constexpr std::int64_t kSleepPeriodMs = 33;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }
}

int Prophesee::open() {
    if (run_mode == RunMode::RECORD) {
        Metavision::DeviceConfig device_config;
        device_config.set_format("EVT30");
        this->cam = Metavision::Camera::from_first_available(device_config);
        cam.get_device().get_facility<Metavision::I_TriggerIn>()->enable(Metavision::I_TriggerIn::Channel::Main);
        cam.start_recording(this->file_to_save);
        cam.start();
        return 1;
    } else if (run_mode == RunMode::PLAY) {
        this->cam = Metavision::Camera::from_file(file_to_play);
        cam.start();
        return 1;
    } else {
        printf("Run Mode 非法, 程序退出\n");
        return -1;
    }
}

void Prophesee::close() {
    if (run_mode == RunMode::RECORD) {
        cam.stop_recording(this->file_to_save);
    }
    cam.stop();
}
