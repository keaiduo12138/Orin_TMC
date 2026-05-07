#ifndef camera_H
#define camera_H
#include "../inc/cyusb.h"
#include "new_dataReceiver.h"
#include "lyn_cam.h"
namespace lynxicam{
    struct Frame
    {
        int length = 0;
        void* data = nullptr;
    };


    class CameraHandle
    {
    private:
        /* data */
        
        
        //CCyUSBDevice* cyUsbDev;

        //libusb_device_handle* cyUsbDev;

        dataReceiver* dataRecv=new dataReceiver();

        std::shared_ptr<BufferPool> m_pConesFramePool;
        std::shared_ptr<BufferPool> m_pRodFramePool;

        BlockQueue<lynFrame_t>  m_conesFrameQueue;
        BlockQueue<lynFrame_t>  m_rodFrameQueue;
        BlockQueue<lynFrame_t> m_frameQueue;

        int idx=0;
        bool _DISABLE_SYNC=true;

        std::thread* conesThread = nullptr;
        std::thread* rodThread= nullptr;
        std::thread* frameThread= nullptr;

        bool _flag_stop = false;

        bool takeFrame();

        void ProcessConesThread();
        void ProcessRodThread();
        void ProcessFrameThread();



    public:
        CameraHandle(/* args */);
        ~CameraHandle();


        cyusb_handle *cyUsbDev=NULL;
        bool disableSyncTran(bool enable);

        uint32_t m_rodFrameCount = 1;


        libusb_device_descriptor deviceDesc;
        libusb_config_descriptor *configDesc;
        libusb_interface_descriptor *interfaceDesc;
        libusb_endpoint_descriptor *endpointDesc;
        libusb_ss_endpoint_companion_descriptor *companionDesc;





        int  if_numsettings;
        bool found_ep = false;
        unsigned char		eptype;			// Type of endpoint (transfer type)
        unsigned int		pktsize;		// Maximum packet size for the endpoint

        unsigned int endpoint   = 0x86;

        struct libusb_transfer **transfers = NULL;		// List of transfer structures.
        unsigned char **databuffers = NULL;			// List of data buffers.

        struct timeval t1, t2;					// Timestamps used for test duration control






        int rStatus=0;


        //void ProcessConesThread();
        //void ProcessRodThread();
        //void ProcessFrameThread();

        //先保留接口，看后续是否会有需求变更
        bool getConesFrame(void** imagebuffer);
        bool putConesFrame(void* imagebuffer);

        bool getRodFrame(void** imagebuffer, int& length);
        bool putRodFrame(void* imagebuffer);
        bool setRodFrameCount(uint32 frameCount);

        bool getFrame(lynFrame_t* frame);
        bool putFrame(lynFrame_t* frame);

        bool start();
        int GetDeviceCount();
        bool init(int idx);

        
        bool configFPGAMode(unsigned int adc_bit_prec, unsigned int out_interface);
        bool sendMetaInfo(unsigned int cone_exp, unsigned int rod_exp, unsigned int rod_delay);
        bool stop();
        bool uninit();
        // bool close_all();
        // taoyi add for iic
       // bool configIIC(std::vector<unsigned int> iic_cfg_list);
        bool write_IIC_simple(std::vector<unsigned int> iic_write_list);
        bool read_IIC_simple(std::vector<unsigned int> address_list,std::vector<unsigned int>& read_back);

        bool  configAutoIIC_stop();
        bool  configAutoIIC_start(unsigned int cone_exp_h_reg,unsigned int  cone_gain_h_reg,unsigned int cone_exp_l_reg,
                                  unsigned int  cone_gain_l_reg, unsigned int   rod_exp_reg_h, unsigned int rod_delay_reg_h,
                                  unsigned int rod_exp_reg_l,unsigned int  rod_delay_reg_l);

    };

}

#endif

