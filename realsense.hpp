#include <optional>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include "utils.hpp"


class RealSense {

public:
    RealSense(std::string file_dir, std::string file_to_play, const int record_or_play);

    ~RealSense() = default;

    void record();
    void play();

private:
    void open();

    rs2::context ctx;
    rs2::config cfg;
    rs2::pipeline pipe;
    std::optional<rs2::recorder> camera;
    std::optional<rs2::playback> player;

    std::string file_to_save;
    std::string file_to_play;

    int record_or_play = -1;

};
