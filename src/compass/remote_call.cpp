#include "compass/remote_call.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <netinet/tcp.h>


PythonRequestManager::PythonRequestManager() 
    :  persistentSocket_(-1), connected_(false),serverHost_("127.0.0.1"), serverPort_(18888), cacheLimit_(100000) {
}

PythonRequestManager::~PythonRequestManager() {
    disconnect();
}

void PythonRequestManager::disconnect() {
    std::lock_guard<std::mutex> lock(socketMutex_);
    if (persistentSocket_ >= 0) {
        shutdown(persistentSocket_, SHUT_RDWR);
        close(persistentSocket_);
        persistentSocket_ = -1;
        connected_ = false;
    }
}

bool PythonRequestManager::ensureConnected() {
    
    if (connected_ && persistentSocket_ >= 0) {
        // 快速检查连接是否还活着
        // MSG_PEEK: 查看数据但不移除
        // MSG_DONTWAIT: 非阻塞
        char buf;
        ssize_t n = recv(persistentSocket_, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
        
        if (n == 0) {
            // 连接已关闭
            std::cout << "[Warning] Connection closed by server" << std::endl;
            close(persistentSocket_);
            persistentSocket_ = -1;
            connected_ = false;
        } else if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 没有数据，但连接正常
                return true;
            } else {
                // 连接出错
                std::cerr << "[Error] Connection error: " << strerror(errno) << std::endl;
                close(persistentSocket_);
                persistentSocket_ = -1;
                connected_ = false;
            }
        } else {
            // n > 0，有数据（不应该出现，因为我们使用了请求-响应模式）
            std::cerr << "[Warning] Unexpected data on socket" << std::endl;
            return true;  // 暂时认为连接正常
        }
    }
    
    // 需要重新连接
    std::cout << "[Info] Establishing new connection..." << std::endl;
    persistentSocket_ = connectToServer();
    if (persistentSocket_ < 0) {
        std::cerr << "[Error] Failed to connect to server" << std::endl;
        connected_ = false;
        return false;
    }
    
    connected_ = true;
    std::cout << "[Info] Connected successfully" << std::endl;
    return true;
}

PythonRequestManager& PythonRequestManager::getInstance() {
    static PythonRequestManager instance;
    return instance;
}

std::string PythonRequestManager::computeHash(const std::string& params) {
    return params;
}

std::string PythonRequestManager::request(const std::string& params) {
    std::string key = computeHash(params);
    
    // ===== 第1步：读取缓存（共享锁，允许并发读） =====
    {
        std::shared_lock<std::shared_mutex> lock(cacheMutex_);
        auto cacheIt = cache_.find(key);  // ✅ 使用明确的变量名
        if (cacheIt != cache_.end()) {
            return cacheIt->second;
        }
    }
    
    // ===== 第2步：检查正在处理的请求（先用共享锁查询） =====
    std::shared_ptr<std::promise<std::string>> myPromise;
    std::future<std::string> myFuture;
    bool isFirstRequest = false;
    
    {
        // 先用共享锁检查是否存在
        std::shared_lock<std::shared_mutex> readLock(requestMutex_);
        auto pendingIt = pendingRequests_.find(key);  // ✅ 使用明确的变量名
        
        if (pendingIt != pendingRequests_.end()) {
            // ===== 情况A：已有相同请求，升级为独占锁添加等待者 =====
            readLock.unlock();  // 释放共享锁
            
            std::unique_lock<std::shared_mutex> writeLock(requestMutex_);
            
            // 双重检查
            pendingIt = pendingRequests_.find(key);  // ✅ 重新查找
            if (pendingIt != pendingRequests_.end()) {
                myPromise = std::make_shared<std::promise<std::string>>();
                myFuture = myPromise->get_future();
                pendingIt->second->waiters.push_back(myPromise);
            } else {
                // 在获取写锁期间，请求已完成，重新查缓存
                writeLock.unlock();
                
                std::shared_lock<std::shared_mutex> cacheLock(cacheMutex_);
                auto cacheIt = cache_.find(key);  // ✅ 明确是查缓存
                if (cacheIt != cache_.end()) {
                    return cacheIt->second;
                }
                // 否则继续作为首发者
                isFirstRequest = true;
            }
        } else {
            // ===== 情况B：我是第一个，升级为独占锁创建请求 =====
            readLock.unlock();
            
            std::unique_lock<std::shared_mutex> writeLock(requestMutex_);
            
            // 双重检查
            pendingIt = pendingRequests_.find(key);  // ✅ 重新查找
            if (pendingIt == pendingRequests_.end()) {
                isFirstRequest = true;
                auto pendingReq = std::make_shared<PendingRequest>();
                myFuture = pendingReq->promise.get_future();
                pendingRequests_[key] = pendingReq;
            } else {
                // 在获取写锁期间，有其他线程创建了请求
                myPromise = std::make_shared<std::promise<std::string>>();
                myFuture = myPromise->get_future();
                pendingIt->second->waiters.push_back(myPromise);
            }
        }
    }
    
    std::string result;
    
    if (isFirstRequest) {
        // ===== 执行实际的Python调用 =====
        try {
            result = executePythonRequest(params);
            
            // ===== 存入缓存（独占锁） =====
            {
                std::unique_lock<std::shared_mutex> lock(cacheMutex_);
                cache_[key] = result;
                cacheOrder_.push(key);
                
                // LRU淘汰
                while (cache_.size() > cacheLimit_) {
                    std::string oldKey = cacheOrder_.front();
                    cacheOrder_.pop();
                    cache_.erase(oldKey);
                }
            }
            
            // ===== 通知所有等待的线程（独占锁） =====
            {
                std::unique_lock<std::shared_mutex> lock(requestMutex_);
                auto pendingIt = pendingRequests_.find(key);  // ✅ 明确是查pending
                if (pendingIt != pendingRequests_.end()) {
                    pendingIt->second->promise.set_value(result);
                    for (auto& waiter : pendingIt->second->waiters) {
                        waiter->set_value(result);
                    }
                    pendingRequests_.erase(pendingIt);
                }
            }
            
        } catch (const std::exception& e) {
            // 错误处理
            std::unique_lock<std::shared_mutex> lock(requestMutex_);
            auto pendingIt = pendingRequests_.find(key);  // ✅ 明确变量名
            if (pendingIt != pendingRequests_.end()) {
                try {
                    pendingIt->second->promise.set_exception(std::current_exception());
                } catch (...) {}
                
                for (auto& waiter : pendingIt->second->waiters) {
                    try {
                        waiter->set_exception(std::current_exception());
                    } catch (...) {}
                }
                pendingRequests_.erase(pendingIt);
            }
            throw;
        }
    } else {
        // ===== 等待首发者的结果 =====
        result = myFuture.get();
    }
    
    return result;
}

std::string PythonRequestManager::executePythonRequest(const std::string& params) {
    std::lock_guard<std::mutex> lock(socketMutex_);

    if (!ensureConnected()) {
        throw std::runtime_error("Failed to establish connection to Python server");
    }
    
    try {
        // 发送请求
        if (!sendData(params)) {
            disconnect();
            throw std::runtime_error("Failed to send data to Python server");
        }
        
        // 接收响应
        std::string response = receiveData();
        
        if (response.empty()) {
            disconnect();
            throw std::runtime_error("Received empty response from Python server");
        }
        return response;
        
    } catch (...) {
        disconnect();
        throw;
    }
}

int PythonRequestManager::connectToServer() {
   int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return -1;
    }
    
    // 设置超时
    // struct timeval timeout;
    // timeout.tv_sec = 60;  // 长连接可以设置更长的超时
    // timeout.tv_usec = 0;
    // setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    // setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    // 设置 TCP_NODELAY
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
    // 设置 SO_KEEPALIVE，保持连接活跃
    flag = 1;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
    
    // 连接
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverPort_);
    
    if (inet_pton(AF_INET, serverHost_.c_str(), &serverAddr.sin_addr) <= 0) {
        close(sock);
        return -1;
    }
    
    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "[Error] Connection failed: " << strerror(errno) << std::endl;
        close(sock);
        return -1;
    }
    
    return sock;
}

bool PythonRequestManager::sendData(const std::string& data) {
    if (data.empty()) {
        std::cerr << "[Error] Attempting to send empty data" << std::endl;
        return false;
    }
    // ===== 第1步：发送长度 =====
    uint32_t length = htonl(data.length());
    size_t totalSent = 0;
    
    while (totalSent < sizeof(length)) {
        ssize_t sent = send(persistentSocket_, 
                           reinterpret_cast<const char*>(&length) + totalSent,
                           sizeof(length) - totalSent, 
                           0);
        
        if (sent < 0) {
            std::cerr << "[Error] send length failed: " << strerror(errno) << std::endl;
            return false;
        }
        
        totalSent += sent;
    }
    
    // ===== 第2步：发送实际数据 =====
    totalSent = 0;
    while (totalSent < data.length()) {
        ssize_t sent = send(persistentSocket_, 
                           data.c_str() + totalSent, 
                           data.length() - totalSent, 
                           0);
        
        if (sent < 0) {
            std::cerr << "[Error] send data failed: " << strerror(errno) << std::endl;
            return false;
        }
        
        totalSent += sent;
    }
    return true;
}

std::string PythonRequestManager::receiveData() {
    // ===== 第1步：接收4字节的长度字段 =====
    uint32_t length;
    size_t totalReceived = 0;
    
    // 确保收到完整的4字节
    while (totalReceived < sizeof(length)) {
        ssize_t received = recv(persistentSocket_, 
                               reinterpret_cast<char*>(&length) + totalReceived,
                               sizeof(length) - totalReceived, 
                               0);
        
        if (received < 0) {
            std::cerr << "[Error] recv length failed: " << strerror(errno) << std::endl;
            return "";
        }
        if (received == 0) {
            std::cerr << "[Error] Connection closed while receiving length" << std::endl;
            return "";
        }
        
        totalReceived += received;
    }
    
    length = ntohl(length);  // 网络字节序转本地字节序
    
    // 检查长度合理性（防止异常数据）
    if (length == 0) {
        std::cerr << "[Error] Received zero length" << std::endl;
        return "";
    }
    if (length > 10 * 1024 * 1024) {  // 假设最大10MB
        std::cerr << "[Error] Received unreasonable length: " << length << std::endl;
        return "";
    }
    
    // ===== 第2步：接收实际数据 =====
    std::string data;
    data.resize(length);
    totalReceived = 0;
    
    while (totalReceived < length) {
        ssize_t received = recv(persistentSocket_, 
                               &data[totalReceived], 
                               length - totalReceived, 
                               0);
        
        if (received < 0) {
            std::cerr << "[Error] recv data failed: " << strerror(errno) << std::endl;
            return "";
        }
        if (received == 0) {
            std::cerr << "[Error] Connection closed while receiving data. "
                     << "Received " << totalReceived << "/" << length << " bytes" 
                     << std::endl;
            return "";
        }
        
        totalReceived += received;
    }
    
    return data;
}

void PythonRequestManager::clearCache() {
    std::unique_lock<std::shared_mutex> lock(cacheMutex_);  // ✅ 独占锁
    cache_.clear();
    while (!cacheOrder_.empty()) {
        cacheOrder_.pop();
    }
}

void PythonRequestManager::setCacheLimit(size_t limit) {
    cacheLimit_ = limit;
}