#include "utils.hpp"


std::time_t now_time = std::time(0);
std::tm *ltm = std::localtime(&now_time);

std::string today_date() {
    return std::to_string(ltm->tm_year + 1900) + "-" + std::to_string(ltm->tm_mon + 1) + "-" + std::to_string(ltm->tm_mday);
}

std::string today_time() {
    return std::to_string(ltm->tm_hour) + "-" + std::to_string(ltm->tm_min) + "-" + std::to_string(ltm->tm_sec);
}