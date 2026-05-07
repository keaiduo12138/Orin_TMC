#ifndef _P_BUFFERPOOL
#define _P_BUFFERPOOL
/**
 * @file bufferpool.h
 * @author SDK_TEAM
 * @brief
 * @version 0.1
 * @date 2022-11-09
 *
 * Copyright:
 * © 2018 北京灵汐科技有限公司 版权所有。
 * 注意：以下内容均为北京灵汐科技有限公司原创，未经本公司允许，不得转载，否则将视为侵权；对于不遵守此声明或者其他违法使用以下内容者，本公司依法保留追究权。
 * © 2018 Lynxi Technologies Co., Ltd. All rights reserved.
 * NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
 *
 */

#include <cassert>
#include <condition_variable>
#include <iostream>

#ifdef _COMPILED_ON_MAC_OSX
#include <sys/malloc.h>
#define quick_exit exit
#else
#include <malloc.h>
#endif

#include <mutex>
#include <queue>

using namespace std::chrono;

class BufferPool
{
public:
    BufferPool(uint32_t elementSize, uint32_t elementCount)
    {
        m_elementSize = elementSize;
        m_elementCount = elementCount;
        for (uint32_t i = 0; i < m_elementCount; i++) {
            void* buffer = nullptr;

            buffer = malloc(m_elementSize);

            if (buffer == nullptr) {
                std::cout<<"Buffer is null, exit"<<std::endl;
                //quick_exit(-1);
            }
            //std::cout << "init" << buffer << std::endl;
            m_queue.push(buffer);
        }
    }
    ~BufferPool()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_condVar.wait_for(lock, std::chrono::seconds(1),
                [this] { return m_queue.size() == m_elementCount; })) {
               std::cout
                    << "[Warning]: m_queue is not full, some element may lose somewhere outside."
                    << m_queue.size()  << " ele " << m_elementCount << std::endl;
            }
        }

        void* buffer;
        std::unique_lock<std::mutex> lock(m_mutex);
        uint32_t size = m_queue.size();
        for (uint32_t i = 0; i < size; i++) {
            buffer = m_queue.front();
            free(buffer);
            m_queue.pop();
        }
    }
    void Push(void* buffer)
    {
        if (Full()) {
            printf("Pool is full, but buffer still push.\n");
            //std::cout << "Pool is full, but buffer still push.";
            //quick_exit(-1);
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(buffer);
        //std::cout << "buffer push " << buffer << std::endl;
        m_condVar.notify_one();
    }
    void* Pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_condVar.wait_for(lock, std::chrono::seconds(60),
            [this] { return !m_queue.empty(); })) {
            std::cout << "[Error]: m_queue is empty, pop null element."  << m_elementCount << std::endl;
            return nullptr;
        }
        
        void* buffer = m_queue.front();
        m_queue.pop();
        //std::cout << "buffer pop " << buffer << std::endl;
        return buffer;
    }


        void* Pop_plus(int* end_flag)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_condVar.wait_for(lock, std::chrono::seconds(60),
            [this] { return !m_queue.empty(); })) {
            std::cout << "[Error]: m_queue is empty, pop null element."  << m_elementCount << std::endl;
            return nullptr;
        }
        
        void* buffer = m_queue.front();
        m_queue.pop();
        //std::cout << "buffer pop " << buffer << std::endl;
         *end_flag = 1;
        return buffer;
    }



    bool Full()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        //std::cout << "m_elementCount" << m_elementCount << std::endl;
        return (m_queue.size() == m_elementCount);
    }

    uint32_t Size()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

protected:
    std::mutex m_mutex;
    std::condition_variable m_condVar;
    std::queue<void*> m_queue;

    uint32_t m_elementSize;
    uint32_t m_elementCount;
};  // BufferPool

#endif
