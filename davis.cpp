#include "davis.hpp"
#include <filesystem>


Davis::Davis(const RunMode run_mode, std::string file_dir, std::string file_without_suffix)
: run_mode(run_mode) {
    file_dir = expand_user(file_dir);
    file_without_suffix = expand_user(file_without_suffix);

    if (run_mode == RunMode::RECORD) {
        std::string dir_path = file_dir + today_date() + '/';
        std::filesystem::create_directories(dir_path);
        this->file_to_save = dir_path + today_time() + ".aedat4";
    } else if (run_mode == RunMode::PLAY) {
        this->file_to_play = file_without_suffix + ".aedat4";
    } else {
        printf("Run Mode 非法, 程序退出\n");
    }
}

void Davis::record() {
    if (this->open() != 1) {
        printf("Davis 打开失败\n");
        return;
    }
    printf("Davis 开始录制: %s\n", file_to_save.c_str());

    auto davis = dynamic_cast<dv::io::camera::DAVIS*>(camera.get());
    if (!davis) {
        printf("无法转换为 DAVIS 类型\n");
        return;
    }

    while (davis->isRunning()) {
        if (const auto &events = davis->getNextEventBatch(); events.has_value()) {
            writer->writeEvents(*events);
        }

        if (const auto &frame = davis->getNextFrame(); frame.has_value()) {
            writer->writeFrame(*frame);
        }

        if (const auto &imu = davis->getNextImuBatch(); imu.has_value()) {
            writer->writeImuPacket(*imu);
        }

        if (const auto &triggers = davis->getNextTriggerBatch(); triggers.has_value()) {
            writer->writeTriggerPacket(*triggers);

            // debug 打印外触发信息
            for (const auto &trigger : *triggers) {
                if (trigger.type == dv::TriggerType::EXTERNAL_SIGNAL_RISING_EDGE) {
                    printf("Davis 检测到信号 | 时间戳: %ld\n", trigger.timestamp);
                }
            }
        }
    }
}

void Davis::play() {
    if (this->open() != 1) {
        printf("Davis 文件打开失败\n");
        return;
    }
    printf("Davis 开始播放: %s\n", file_to_play.c_str());

    std::optional<int64_t> lastTimestamp = std::nullopt;

    dv::visualization::EventVisualizer visualizer(*reader->getEventResolution());
    visualizer.setBackgroundColor(dv::visualization::colors::darkGray);
    visualizer.setPositiveColor(dv::visualization::colors::white);
    visualizer.setNegativeColor(dv::visualization::colors::iniBlue);

    dv::EventStreamSlicer slicer;

    slicer.doEveryTimeInterval(std::chrono::milliseconds(33), [&visualizer](const dv::EventStore &events) {
        cv::Mat accu = visualizer.generateImage(events);
        cv::imshow("Davis Depth", accu);
    });

    int64_t frame_timestamp = 0;

    while (reader->isRunning()) {
        // 读取帧
        if (const auto frame = reader->getNextFrame(); frame.has_value()) {
            cv::imshow("Davis RGB", frame->image);
            frame_timestamp = frame->timestamp;
        }

        // 读取并处理事件，直到时间戳赶上当前帧
        // 一个 getNextEventBatch() 没有33ms的数据，所以会慢于rgb
        while (auto events = reader->getNextEventBatch()) {
            if (events->isEmpty()) {
                break;
            }
            
            slicer.accept(*events);
            
            if (events->getHighestTime() >= frame_timestamp) {
                break;
            }
        }

        // 读取触发信号
        // if (const auto triggers = reader->getNextTriggerBatch(); triggers.has_value()) {
        //     for (const auto &trigger : *triggers) {
        //         if (trigger.type == dv::TriggerType::EXTERNAL_SIGNAL_RISING_EDGE) {
        //             printf("Davis 检测到信号 | 时间戳: %ld\n", trigger.timestamp);
        //         }
        //     }
        // }

        cv::waitKey(33);
    }

    cv::destroyAllWindows();
}

int Davis::open() {
    if (run_mode == RunMode::RECORD) {
        auto cameras = dv::io::camera::discover();
        if (cameras.empty()) {
            printf("没发现 Davis 相机, 退出\n");
            return -1;
        } else if (cameras.size() > 1) {
            printf("发现多个 Davis 相机, 退出\n");
            return -1;
        }

        camera = dv::io::camera::openSync(cameras[0]);
        auto davis = dynamic_cast<dv::io::camera::DAVIS*>(camera.get());
        if (!davis) {
            printf("无法转换为 DAVIS 类型\n");
            return -1;
        }

        // 设置自动曝光
        davis->setAutoExposure(true);

        // 检测上升沿触发
        davis->setDetectorRisingEdges(true);
        davis->setDetectorFallingEdges(false);
        davis->setDetectorRunning(true);

        // 创建写入器
        auto resolution = davis->getEventResolution();
        if (!resolution.has_value()) {
            printf("无法获取 Davis 分辨率\n");
            return -1;
        }
        const auto config = dv::io::MonoCameraWriter::DAVISConfig("davis", *resolution);
        writer.emplace(file_to_save, config);

        printf("Davis 相机已打开\n");

        return 1;
    } else if (run_mode == RunMode::PLAY) {
        reader.emplace(file_to_play);
        printf("Davis 文件已打开\n");
        return 1;
    } else {
        printf("Run Mode 非法, 程序退出\n");
        return -1;
    }
}
