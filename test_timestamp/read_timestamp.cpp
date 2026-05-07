#include <cstdio>
#include <cstdint>
#include <vector>
#include <unistd.h>
#include <cstring>

#include "cyusb.h"
#include "usb/inc/new_dataReceiver.h"

static const unsigned int kEndpointOut = 0x04;
static const unsigned int kEndpointCfgIn = 0x88;
static const unsigned int kTsReadHead = (0x99u << 24);
static const unsigned int kTsReadTail = (0xa0u << 24);

bool read_unix_timestamp(cyusb_handle* handle, uint64_t& outTimestamp10us) {
    if (handle == nullptr) return false;

    // Trigger device to place timestamp data on CFG IN endpoint
    std::vector<unsigned int> read_cmd;
    read_cmd.push_back(0);
    read_cmd.push_back(0);
    read_cmd.push_back(kTsReadHead);
    read_cmd.push_back(0x10c);
    read_cmd.push_back(kTsReadTail);
    read_cmd.push_back(0);
    read_cmd.push_back(0);

    long len_cmd = static_cast<long>(read_cmd.size() * 4);
    int transferred = 0;
    int r = cyusb_bulk_transfer(handle, kEndpointOut,
                                reinterpret_cast<unsigned char*>(read_cmd.data()),
                                len_cmd, &transferred, 5000);
    if (r != 0) return false;

    usleep(1000);

    const int buf_len = 2048;
    unsigned char buf[buf_len];
    transferred = 0;
    r = cyusb_bulk_transfer(handle, kEndpointCfgIn, buf, buf_len, &transferred, 5000);
    if (r != 0 || transferred <= 8) return false;

    uint32_t hi = 0;
    uint32_t lo = 0;
    std::memcpy(&hi, buf + 0, sizeof(uint32_t));
    std::memcpy(&lo, buf + 4, sizeof(uint32_t));

    outTimestamp10us = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    return true;
}

int main() {
    int r = cyusb_open();
    if (r < 0) {
        std::printf("cyusb_open failed: %d\n", r);
        return 1;
    }
    if (r == 0) {
        std::printf("No Cypress USB device found.\n");
        return 1;
    }

    cyusb_handle* h = cyusb_gethandle(0);
    if (!h) {
        std::printf("No USB handle available.\n");
        cyusb_close();
        return 1;
    }

    uint64_t ts10us = 0;
    bool ok = read_unix_timestamp(h, ts10us);
    if (!ok) {
        std::printf("Read timestamp failed.\n");
        cyusb_close();
        return 2;
    }

    uint64_t us = ts10us * 10ull;
    uint32_t hi = static_cast<uint32_t>(ts10us >> 32);
    uint32_t lo = static_cast<uint32_t>(ts10us & 0xffffffffu);
    std::printf("Timestamp(10us units): 0x%08x%08x (%llu)\n", hi, lo, (unsigned long long)ts10us);
    std::printf("Timestamp in microseconds: %llu\n", (unsigned long long)us);

    cyusb_close();
    return 0;
}


