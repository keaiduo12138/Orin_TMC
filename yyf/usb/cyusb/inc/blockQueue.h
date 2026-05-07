#ifndef H_BLOCKQUEUE_
#define H_BLOCKQUEUE_
/**
 * @file blockQueue.hpp
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


#include <assert.h>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

 /********************************************************************************
 类名 : BlockQueue
 Description: 阻塞队列模板类，用于生产者消费者场景
 Example: BlockQueue<T> oQueue()
 *******************************************************************************/
template <typename T>
class BlockQueue
{
public:
    // make class non-copyable
    BlockQueue(const BlockQueue<T>&) = delete;
    BlockQueue& operator=(const BlockQueue<T>&) = delete;

    /**
     * @brief Construct a new Block Queue< T> object
     *
     * @param  maxSize 队列大小
     */
    explicit BlockQueue<T>(size_t maxSize = SIZE_MAX) : m_nMaxSize(maxSize), m_bQuit(false) {}

    /**
     * @brief 向队列里添加对象，可设置超时时间
     *
     * @param[in] x 添加的对象
     * @param[in] timeout 超时时间
     * @return true 成功; false 失败
     */
    bool put(const T& x, int timeout = 60)
    {
        if (m_bQuit) {
            return false;
        }

        std::unique_lock<std::mutex> locker(m_oMutex);
        m_oNotFullCV.wait_for(locker, std::chrono::microseconds(timeout),
            [this]() { return (m_queue.size() < m_nMaxSize) || m_bQuit; });

        if (m_bQuit || m_queue.size() == m_nMaxSize) {
            return false;
        }

        m_queue.push(x);
        //std::cout << "size" << m_queue.size() << std:: endl;
        m_oNotEmptyCV.notify_one();
        return true;
    }


bool put_plus(const T& x, int* end_flag, int timeout = 60)
{
    if (m_bQuit) {
        return false;
    }

    std::unique_lock<std::mutex> locker(m_oMutex);
    m_oNotFullCV.wait_for(locker, std::chrono::microseconds(timeout),
        [this]() { return (m_queue.size() < m_nMaxSize) || m_bQuit; });

    if (m_bQuit || m_queue.size() == m_nMaxSize) {
        return false;
    }

    m_queue.push(x);
    //std::cout << "size" << m_queue.size() << std::endl;
    m_oNotEmptyCV.notify_one();

    // 执行其他功能

    // 设置 end_flag 为 1，表示函数执行结束
    *end_flag = 1;

    return true;
}

    /**
     * @brief 从队列中获取一个对象，可设置超时时间，默认为 1 S
     *
     * @param  outRes 获取的对象
     * @param  timeout 超时时间
     * @return true 成功
     * @return false 失败
     */
    //bool take(T& outRes, int timeout = 1000)
    bool take(T& outRes, int timeout = 60)
    {
        if (m_bQuit) {
            return false;
        }
        std::unique_lock<std::mutex> locker(m_oMutex);
        m_oNotEmptyCV.wait_for(locker, std::chrono::microseconds(timeout),
            [this]() { return !m_queue.empty() || m_bQuit; });
        //printf("enter getframe in blockQueue.hpp 98  m_bQuit || m_queue.empty()=%d\n",m_bQuit || m_queue.empty());
        if (m_bQuit || m_queue.empty()) {
            return false;
        }
        outRes = m_queue.front();
        m_queue.pop();
        m_oNotFullCV.notify_one();
        return true;
    }

    /**
     * @brief 判断队列是否为空
     *
     * @return true 队列为空
     * @return false 队列不为空
     */
    bool empty() const
    {
        std::unique_lock<std::mutex> locker(m_oMutex);
        return m_queue.empty();
    }

    /**
     * @brief 清空队列
     *
     * @return true 清空成功
     */
    bool clear()
    {
        std::unique_lock<std::mutex> locker(m_oMutex);
        std::queue<T> emptyQueue;
        std::swap(m_queue, emptyQueue);
        return true;
    }

    /**
     * @brief 获取队列大小
     *
     * @return size_t 当前队列的大小
     */
    size_t size() const
    {
        std::unique_lock<std::mutex> locker(m_oMutex);
        return m_queue.size();
    }

    /**
     * @brief 获取队列最大容量
     *
     * @return size_t 队列最大容量
     */
    size_t maxSize() const
    {
        return m_nMaxSize;
    }

    /**
     * @brief 退出，以便阻塞的take和put能够返回
     *
     */
    void quit()
    {
        m_bQuit = true;
        m_oNotEmptyCV.notify_all();
        m_oNotFullCV.notify_all();
    }

    mutable std::mutex m_oMutex;
    std::condition_variable m_oNotEmptyCV;

private:
    //mutable std::mutex m_oMutex;
    //std::condition_variable m_oNotEmptyCV;
    std::condition_variable m_oNotFullCV;
    size_t m_nMaxSize;
    std::queue<T> m_queue;
    bool m_bQuit;
};

#endif
