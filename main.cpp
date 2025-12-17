#include <iostream>
#include "realsense.hpp"
#include "prophesee.hpp"


int main() {
    YAML::Node config = YAML::LoadFile("../config.yaml");
    std::string file_dir = config["prophesee"]["file_dir"].as<std::string>();
    std::string file_to_play = config["prophesee"]["file_to_play"].as<std::string>();
    int record_or_play = config["prophesee"]["record_or_play"].as<int>();

    Prophesee ekv4 = Prophesee(file_dir, file_to_play, record_or_play);

    std::string rs_file_dir = config["realsense"]["file_dir"].as<std::string>();
    std::string rs_file_to_play = config["realsense"]["file_to_play"].as<std::string>();
    int rs_record_or_play = config["realsense"]["record_or_play"].as<int>();

    RealSense d455 = RealSense(rs_file_dir, rs_file_to_play, rs_record_or_play);

    std::jthread t1([&]() {
        if (record_or_play == 0) {
            ekv4.record();
        } else if (record_or_play == 1) {
            ekv4.play();
        } else {
            std::cerr << "record_or_play 变量错误" << record_or_play << std::endl;
            return;
        }
    });    

    std::jthread t2([&]() {
        if (rs_record_or_play == 0) {
            d455.record();
        } else if (rs_record_or_play == 1) {
            d455.play();
        } else {
            std::cerr << "record_or_play 变量错误" << record_or_play << std::endl;
            return;
        }
    });


    return 0;
}
