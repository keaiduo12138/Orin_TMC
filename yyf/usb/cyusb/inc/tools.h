#ifndef _USB_TOOLS_H
#define _USB_TOOLS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
/*
 * Exposure relative register address, useful in runtime config
 */
#define CT_INT0_ADDR 0x0117
#define CT_INT1_ADDR 0x0118
#define CT_INT2_ADDR 0x0119
#define C_GAIN_FINE_DAC0_ADDR 0x011D
#define C_GAIN_FINE_DAC1_ADDR 0x011E
#define C_GAIN_COARSE_DAC0_ADDR 0x011F
#define C_GAIN_COARSE_DAC1_ADDR 0x0120
#define R_GAIN_COARSE_MODE 0
#define R_GAIN_FINE_MODE 1
#define RT_INT0_ADDR 0x051A
#define RT_INT1_ADDR 0x051B
#define RT_INT2_ADDR 0x051C
#define RT_DELAY0_ADDR 0x0529
#define RT_DELAY1_ADDR 0x052A
#define RT_DELAY2_ADDR 0x052B
#define R_GAIN_FINE_DAC_0_ADDR 0x0547	
#define R_GAIN_FINE_DAC_1_ADDR 0x0548	
#define R_GAIN_COARSE_DAC_0_ADDR 0x054C
#define R_GAIN_COARSE_DAC_1_ADDR 0x054D
#define R_DAC_TH0_ADDR 0x0549	
#define R_DAC_TH1_ADDR 0x054A	
#define R_FILTERMODE_ADDR 0x0562
#define C_US_TO_CNT 50 // 1us = 50 * 20ns(clock cnt)
#define R_US_TO_CNT 100 // 1us = 100 * 10ns(clock cnt)
/*
 * Rod Exposure default values, in us
 */
//8 bit default value for none-integration registers
#define R_8BIT_TG0 0.6
#define R_8BIT_TG1 0.6
#define R_8BIT_TG2RST 0.4
#define R_8BIT_RST2TG 0.2
#define R_8BIT_IB2RST 0.2
#define R_8BIT_RST2GSSR 0.2
#define R_8BIT_GSSR2TG 0.2
#define R_8BIT_TG2GSSS 0.2
#define R_8BIT_GSSS2RST 0.4
#define R_8BIT_RST2IB 0.6

//4 bit default value for none-integration registers
#define R_4BIT_TG0 0.6
#define R_4BIT_TG1 0.6
#define R_4BIT_TG2RST 0.4
#define R_4BIT_RST2TG 0.2
#define R_4BIT_IB2RST 0.2
#define R_4BIT_RST2GSSR 0.2
#define R_4BIT_GSSR2TG 0.2
#define R_4BIT_TG2GSSS 0.2
#define R_4BIT_GSSS2RST 0.4
#define R_4BIT_RST2IB 0.6

//2 bit default value for none-integration registers
#define R_2BIT_TG0 0.5
#define R_2BIT_TG1 0.5
#define R_2BIT_TG2RST 0.4
#define R_2BIT_RST2TG 0.2
#define R_2BIT_IB2RST 0.2
#define R_2BIT_RST2GSSR 0.2
#define R_2BIT_GSSR2TG 0.2
#define R_2BIT_TG2GSSS 0.2
#define R_2BIT_GSSS2RST 0.4
#define R_2BIT_RST2IB 0.6

// Max and Min value for different interface and adc precision
#define R_MAX_8BIT_LVDS (660.0 - 0.1)
#define R_MIN_8BIT_LVDS (660.0 - 6.0)
#define R_MAX_4BIT_LVDS (330.0 - 0.1)
#define R_MIN_4BIT_LVDS (330.0 - 6.0)
#define R_MAX_2BIT_LVDS (100.0 - 0.1)
#define R_MIN_2BIT_LVDS (100.0 - 3.4)

#define R_MAX_8BIT_PARA (1320.0 - 0.1)
#define R_MIN_8BIT_PARA (1320.0 - 12.0)
#define R_MAX_4BIT_PARA (660.0 - 0.1)
#define R_MIN_4BIT_PARA (660.0 - 12.0)
#define R_MAX_2BIT_PARA (300.0 - 0.1)
#define R_MIN_2BIT_PARA (300.0 - 10.2)

#define R_PARA_SEL 0
#define R_LVDS_SEL 1

std::vector<std::vector<uint16_t>> parse_csv_cgf(const char* filepath, bool print);
std::string Trim(std::string& str);
// i2c 
int cone_exptime_preproc(float exp_time, uint8_t* ct_int0, uint8_t* ct_int1, uint8_t* ct_int2);
int rod_exptime_preproc(float exp_time, int adc_bit_prec, int interface,
						uint8_t* rt_int0, uint8_t* rt_int1, uint8_t* rt_int2, 
						uint8_t* rt_delay0, uint8_t* rt_delay1,uint8_t* rt_delay2);
int cone_anagain_preproc(float ana_gain, uint8_t* gain_coarse_0, uint8_t* gain_coarse_1, uint8_t* gain_fine_0, uint8_t* gain_fine_1);
int rod_gain_preproc(float ana_gain, uint8_t* gain_coarse_0, uint8_t* gain_coarse_1, uint8_t* gain_fine_0, uint8_t* gain_fine_1, uint8_t mode);
#endif
