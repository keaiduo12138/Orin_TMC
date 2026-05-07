#ifndef dataReceiver_H
#define dataReceiver_H

#include <stdio.h>
#include "libusb-1.0/libusb.h"
#include "bufferPool.h"
#include "blockQueue.h"
#include "../inc/cyusb.h"
#include <thread>
#include <memory>


namespace lynxicam {

    typedef unsigned short uint16;
    typedef unsigned int uint32;

    const uint32_t headerMask = 0xFF00000F;
    const uint32_t dataMask = 0xFE000000;
    const uint32_t conesHeaderFlag = 0xFA800001;
    const uint32_t rodHeaderFlag = 0xED000000;

    /********* SYS configure  *********/
    const unsigned int SYS_STOP = 0x0;
    const unsigned int SYS_RST = 0x1;
    const unsigned int CONE_EN = 0x10;
    const unsigned int PW_ON_GPIO = 0x200;
    const unsigned int FPGA_MODE_CFG_HEAD = (0xdd << 24);
    const unsigned int FPGA_MODE_CFG_TAIL = (0x96 << 24);
    //add by taoyi!
    const unsigned int FPGA_MODE_CFG_READ_HEAD = (0xee << 24); //
    const unsigned int FPGA_MODE_CFG_READ_TAIL = (0x5a << 24);
    const unsigned int I2C_MODE_CFG_HEAD = (0xff << 24); //
    const unsigned int I2C_MODE_CFG_TAIL = (0xa5 << 24); //
    const unsigned int I2C_MODE_CFG_READ_HEAD = (0xee << 24); //
    const unsigned int I2C_MODE_CFG_READ_TAIL = (0x5a << 24); //

    const unsigned int UNIX_SYS_DATA_HEAD = (0xbb << 24); //
    const unsigned int UNIX_SYS_DATA_TAIL = (0xc3 << 24); //

    const unsigned int AUTO_IIC_MODE_START = (0x88 << 24);
    const unsigned int AUTO_IIC_MODE_STOP = (0x11 << 24);

   
    const unsigned int METAINFO_DATA_START = (0xaa << 24); //
    const unsigned int METAINFO_DATA_TAIL = (0xd7 << 24); // 

using namespace lynxicam;
#define QueueSize 64





    class dataReceiver
    {

    private:
        


        bool unInit();
        bool download(unsigned int adc_bit_prec, unsigned int out_interface);

        unsigned char frame_buffer[1024 * 1024 * 10];

        long _frame_cnt;

        int count_rod = 0, count_cones = 0;

        uint16 _width;
        uint16 _height;
        uint32 _wh;
        uint32 _depth;
        uint32 _tot_bytes;

        unsigned char* buffer;
        unsigned char** pbuf;
        struct libusb_transfer** transfers;

        struct libusb_transfer** transfers2;
        
        unsigned char** databuffers;
        
        long maxPktLength;
        long packetPerXfer;
        std::mutex m_mux;

        bool _flag_stop;

        unsigned int m_adc_bit_prec;
        unsigned int m_out_interface;
        bool needInitDev = false;

        //pthread_cond_t eventCond;   // 条件变量
        //pthread_mutex_t eventMutex; // 互斥锁
        //bool eventOccurred;         // 事件标志


        std::thread m_thread;

        bool m_flag_stop;
        pthread_mutex_t m_mutex;
        pthread_cond_t m_cond;
        int m_curXfer;

        std::thread m_threads[QueueSize];
        std::thread m_m_threads[QueueSize];
        pthread_mutex_t m_mutexes[QueueSize];
        pthread_cond_t m_conds[QueueSize];

        int cam_idx = 0;




    public:
        explicit dataReceiver();
        ~dataReceiver();

        static void xfer_callback (struct libusb_transfer *transfer);

        int processReceivedData(struct libusb_transfer *transfer);

        long getFrameCount() const;

        void IndependentTransfer(int i);
        void m_thread_update(int i);


        bool open(cyusb_handle * cyUsbDev, int cam_idx);

        // Variables storing the user provided application configuration.
        static unsigned int endpoint ;	// Endpoint to be tested
        static unsigned int reqsize  ;	// Request size in number of packets
        static unsigned int queuedepth;	// Number of requests to queue
        static unsigned int duration;	// Duration of the test in seconds

        static unsigned char		eptype;			// Type of endpoint (transfer type)
        static unsigned int		pktsize;		// Maximum packet size for the endpoint

        static unsigned int            success_count;	// Number of successful transfers
        static unsigned int            failure_count;	// Number of failed transfers
        static unsigned int 		transfer_size;	// Size of data transfers performed so far
        static unsigned int		transfer_index;	// Write index into the transfer_size array
        static volatile bool		stop_transfers;	// Request to stop data transfers
        static volatile int		rqts_in_flight;	// Number of transfers that are in progress
        static volatile int		all_rqts_in_flight[64];

        static volatile int		rqts_in_flight_2;	// Number of transfers that are in progress
        static volatile int		all_rqts_in_flight_2[64];
        
        static struct timeval		start_ts;		// Data transfer start time stamp.
        static struct timeval		end_ts;			// Data transfer stop time stamp.

        void start();
        void stop();
        void RecvThread();
        bool initDev(int cam_idx);

        void RecvCallback(struct libusb_transfer* transfer);
       // static void StaticRecvCallback(struct libusb_transfer* transfer);

        void DataXferThread(int index);

        //std::shared_ptr<BufferPool> pConesDataPool;
        //std::shared_ptr<BufferPool> pRodDataPool;
        std::shared_ptr<BufferPool> pConesDataPool = std::make_shared<BufferPool>(maxPktLength, 512);
        std::shared_ptr<BufferPool> pRodDataPool = std::make_shared<BufferPool>(maxPktLength, 2048);

        //int conesFrameSize = (32 * 16 + 320 * 320 * 16) / 8;
        //std::shared_ptr<BufferPool> pConesDataPool = std::make_shared<BufferPool>(conesFrameSize, 254);


        //int RodFrameSize = ((32 * 16 + 320 * 320 * 16) / 8) * m_rodFrameCount;
        //int RodFrameSize = (32 * 16 + 320 * 320 * 16) / 8*100;

        //std::shared_ptr<BufferPool> pRodDataPool = std::make_shared<BufferPool>(RodFrameSize, 258);


        BlockQueue<void*> conesDataQueue;


        BlockQueue<void*> rodDataQueue;
        bool downloadFpgaMode(unsigned int adc_bit_prec, unsigned int out_interface);
        bool  download_unix_sys_data();
        bool download_metainfo_data(unsigned int cone_exp, unsigned int rod_exp, unsigned int rod_delay);
        bool download_auto_iic_start(unsigned int cone_exp_h_reg,unsigned int  cone_gain_h_reg,unsigned int cone_exp_l_reg,
                        unsigned int  cone_gain_l_reg, unsigned int   rod_exp_reg_h, unsigned int rod_delay_reg_h,
                        unsigned int rod_exp_reg_l,unsigned int  rod_delay_reg_l);
        bool download_auto_iic_stop();

        cyusb_handle* m_cyUsbDev = NULL;

        libusb_endpoint_descriptor* DataInEpt;
        libusb_endpoint_descriptor* DataOutEpt;
        libusb_endpoint_descriptor* CFGDataInEpt;

          // add by taoyi for I2C
        bool cfg_bulk_write_lvl(std::vector<unsigned int> iic_cfg_list );
        bool cfg_bulk_read_lvl(std::vector<unsigned int> cfg_read_addr_list,std::vector<unsigned int>& cfg_read_data );


    };

}
#endif // dataReceiver_H
