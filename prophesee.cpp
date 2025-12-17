#include "prophesee.hpp"
#include <filesystem>


Prophesee::Prophesee(std::string file_dir, std::string file_to_play, int record_or_play) {
    if (record_or_play == 0) {
        std::string dir_path = file_dir + today_date() + '/';
        std::filesystem::create_directories(dir_path);
        this->file_to_save = dir_path + today_time() + ".raw";
    } else {
        this->file_to_save = file_dir;
    }
    this->file_to_play = file_to_play;
    this->record_or_play = record_or_play;
}

Prophesee::~Prophesee() {
    this->close();
}

void Prophesee::record() {
    this->open();
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
    this->open();
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(
        cam.geometry().width(),
        cam.geometry().height(),
        20000,
        50
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
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }
}

void Prophesee::open() {
    if (record_or_play == 0) {
        this->cam = Metavision::Camera::from_first_available();
        cam.get_device().get_facility<Metavision::I_TriggerIn>()->enable(Metavision::I_TriggerIn::Channel::Main);
        cam.start_recording(this->file_to_save);
        cam.start();
    } else if (record_or_play == 1) {
        this->cam = Metavision::Camera::from_file(file_to_play);
        cam.start();
    }
}

void Prophesee::close() {
    if (record_or_play == 0) {
        cam.stop_recording(this->file_to_save);
    }
    cam.stop();
}
