#pragma once

#include <string>

enum RunMode {
    UNDEFINED = 0,
    RECORD = 1,
    PLAY = 2
};

std::string today_date();

std::string today_time();

std::string expand_user(std::string path);
