#include <cstdio>
#include <cstdint>
#include <vector>
#include <unistd.h>
#include <cstring>

#include "cyusb.h"
#include "usb/inc/new_dataReceiver.h"

static const unsigned int kEndpointOut = 0x04;
static const unsigned int kEndpointCfgIn = 0x88;
// Use your FPGA timestamp protocol markers
static const unsigned int kTsReadHead = (0x99u << 24);
static const unsigned int kTsReadTail = (0xa0u << 24);

static void drain_in_endpoint(cyusb_handle* handle, unsigned char endpoint, int max_bytes, int per_try_timeout_ms) {
    if (handle == nullptr) return;
    std::vector<unsigned char> tmp(max_bytes);
    int transferred = 0;
    // Drain until timeout or no data
    for (;;) {
        int r = cyusb_bulk_transfer(handle, endpoint, tmp.data(), (int)tmp.size(), &transferred, per_try_timeout_ms);
        if (r != 0 || transferred <= 0) break;
        // Keep draining if device had stale data
    }
}

bool read_unix_timestamp(cyusb_handle* handle, uint64_t& outTimestamp10us) {
    if (handle == nullptr) return false;

    // Clear any stale data in IN endpoint to avoid mixing previous run
    drain_in_endpoint(handle, (unsigned char)kEndpointCfgIn, 2048, 50);

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
    if (r != 0 || transferred != len_cmd) {
        std::printf("bulk OUT failed: r=%d, transferred=%d/%ld\n", r, transferred, len_cmd);
        return false;
    }

    // Give the device more time to prepare the reply
    usleep(100000);

    const int buf_len = 2048;
    unsigned char buf[buf_len];

    // Retry several times with longer timeout; FPGA may need extra time
    int tries = 5;
    uint32_t hi = 0;
    uint32_t lo = 0;
    for (int t = 0; t < tries; ++t) {
        transferred = 0;
        r = cyusb_bulk_transfer(handle, kEndpointCfgIn, buf, buf_len, &transferred, 10000);
        if (r == 0 && transferred > 8) {
            std::memcpy(&hi, buf + 0, sizeof(uint32_t));
            std::memcpy(&lo, buf + 4, sizeof(uint32_t));
            if (hi != 0 || lo != 0) {
                outTimestamp10us = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
                return true;
            }
            // Zero data: re-trigger once more before next retry
            usleep(50000);
            int tx = 0;
            (void)cyusb_bulk_transfer(handle, kEndpointOut,
                                      reinterpret_cast<unsigned char*>(read_cmd.data()),
                                      len_cmd, &tx, 5000);
            usleep(100000);
        } else {
            // Read failed, short backoff and retry
            usleep(50000);
        }
    }

    std::printf("bulk IN failed: r=%d, transferred=%d\n", r, transferred);
    return false;
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

    // Ensure device configuration/interface are ready
    int cfg = 0;
    if (cyusb_get_configuration(h, &cfg) != 0) {
        std::printf("get_configuration failed\n");
    }
    if (cfg <= 0) {
        int rr = cyusb_set_configuration(h, 1);
        if (rr != 0) {
            std::printf("set_configuration(1) failed: %d\n", rr);
        }
    }

    int kact = cyusb_kernel_driver_active(h, 0);
    if (kact == 1) {
        int rr = cyusb_detach_kernel_driver(h, 0);
        if (rr != 0) {
            std::printf("detach_kernel_driver(0) failed: %d\n", rr);
        }
    }

    int rr = cyusb_claim_interface(h, 0);
    if (rr != 0) {
        std::printf("claim_interface(0) failed: %d\n", rr);
        cyusb_close();
        return 1;
    }

    // Optional: set alt setting 0 (ignore failure if device has none)
    (void)cyusb_set_interface_alt_setting(h, 0, 0);

    // Clear potential halts on endpoints
    (void)cyusb_clear_halt(h, (unsigned char)kEndpointOut);
    (void)cyusb_clear_halt(h, (unsigned char)kEndpointCfgIn);

    uint64_t ts10us = 0;
    bool ok = read_unix_timestamp(h, ts10us);
    if (!ok) {
        std::printf("Read timestamp failed.\n");
        cyusb_release_interface(h, 0);
        cyusb_close();
        return 2;
    }

    uint64_t us = ts10us * 10ull;
    uint32_t hi_print = static_cast<uint32_t>(ts10us >> 32);
    uint32_t lo_print = static_cast<uint32_t>(ts10us & 0xffffffffu);
    std::printf("Timestamp(10us units): 0x%08x%08x (%llu)\n", hi_print, lo_print, (unsigned long long)ts10us);
    std::printf("Timestamp in microseconds: %llu\n", (unsigned long long)us);

    cyusb_release_interface(h, 0);
    cyusb_close();
    return 0;
}