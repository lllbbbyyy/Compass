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
        // Quick check if connection is still alive
        // MSG_PEEK: Peek data without removing
        // MSG_DONTWAIT: Non-blocking
        char buf;
        ssize_t n = recv(persistentSocket_, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
        
        if (n == 0) {
            // Connection closed
            std::cout << "[Warning] Connection closed by server" << std::endl;
            close(persistentSocket_);
            persistentSocket_ = -1;
            connected_ = false;
        } else if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No data but connection is fine
                return true;
            } else {
                // Connection error
                std::cerr << "[Error] Connection error: " << strerror(errno) << std::endl;
                close(persistentSocket_);
                persistentSocket_ = -1;
                connected_ = false;
            }
        } else {
            // n > 0, data exists (shouldn't happen in request-response mode)
            std::cerr << "[Warning] Unexpected data on socket" << std::endl;
            return true;  // Temporarily consider connection OK
        }
    }
    
    // Need to reconnect
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
    
    // ===== Step 1: Read cache (shared lock, allows concurrent reads) =====
    {
        std::shared_lock<std::shared_mutex> lock(cacheMutex_);
        auto cacheIt = cache_.find(key);  // Using clear variable names
        if (cacheIt != cache_.end()) {
            return cacheIt->second;
        }
    }
    
    // ===== Step 2: Check pending requests (first with shared lock) =====
    std::shared_ptr<std::promise<std::string>> myPromise;
    std::future<std::string> myFuture;
    bool isFirstRequest = false;
    
    {
        // First check with shared lock
        std::shared_lock<std::shared_mutex> readLock(requestMutex_);
        auto pendingIt = pendingRequests_.find(key);  // Clear variable naming
        
        if (pendingIt != pendingRequests_.end()) {
            // ===== Case A: Same request exists, upgrade to exclusive lock to add waiter =====
            readLock.unlock();  // Release shared lock
            
            std::unique_lock<std::shared_mutex> writeLock(requestMutex_);
            
            // Double check
            pendingIt = pendingRequests_.find(key);  // Recheck
            if (pendingIt != pendingRequests_.end()) {
                myPromise = std::make_shared<std::promise<std::string>>();
                myFuture = myPromise->get_future();
                pendingIt->second->waiters.push_back(myPromise);
            } else {
                // Request completed while acquiring write lock, check cache again
                writeLock.unlock();
                
                std::shared_lock<std::shared_mutex> cacheLock(cacheMutex_);
                auto cacheIt = cache_.find(key);  // Explicit cache check
                if (cacheIt != cache_.end()) {
                    return cacheIt->second;
                }
                // Otherwise continue as first requester
                isFirstRequest = true;
            }
        } else {
            // ===== Case B: I'm the first, upgrade to exclusive lock to create request =====
            readLock.unlock();
            
            std::unique_lock<std::shared_mutex> writeLock(requestMutex_);
            
            // Double check
            pendingIt = pendingRequests_.find(key);  // Recheck
            if (pendingIt == pendingRequests_.end()) {
                isFirstRequest = true;
                auto pendingReq = std::make_shared<PendingRequest>();
                myFuture = pendingReq->promise.get_future();
                pendingRequests_[key] = pendingReq;
            } else {
                // Another thread created request while acquiring write lock
                myPromise = std::make_shared<std::promise<std::string>>();
                myFuture = myPromise->get_future();
                pendingIt->second->waiters.push_back(myPromise);
            }
        }
    }
    
    std::string result;
    
    if (isFirstRequest) {
        // ===== call actual Python =====
        try {
            result = executePythonRequest(params);
            
            // ===== save cache unique lock =====
            {
                std::unique_lock<std::shared_mutex> lock(cacheMutex_);
                cache_[key] = result;
                cacheOrder_.push(key);
                
                // LRU
                while (cache_.size() > cacheLimit_) {
                    std::string oldKey = cacheOrder_.front();
                    cacheOrder_.pop();
                    cache_.erase(oldKey);
                }
            }
            // ===== Notify all waiting threads (exclusive lock) =====
            {
                std::unique_lock<std::shared_mutex> lock(requestMutex_);
                auto pendingIt = pendingRequests_.find(key);  //  is pending
                if (pendingIt != pendingRequests_.end()) {
                    pendingIt->second->promise.set_value(result);
                    for (auto& waiter : pendingIt->second->waiters) {
                        waiter->set_value(result);
                    }
                    pendingRequests_.erase(pendingIt);
                }
            }
            
        } catch (const std::exception& e) {
            // error process
            std::unique_lock<std::shared_mutex> lock(requestMutex_);
            auto pendingIt = pendingRequests_.find(key);  
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
        // ===== wait fir first res =====
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
        // Send request
        if (!sendData(params)) {
            disconnect();
            throw std::runtime_error("Failed to send data to Python server");
        }
        
        // Receive response
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
    
    // Set timeout
    // struct timeval timeout;
    // timeout.tv_sec = 60;  // Long connection can set longer timeout
    // timeout.tv_usec = 0;
    // setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    // setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    // Set TCP_NODELAY
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    
    // Set SO_KEEPALIVE to keep connection alive
    flag = 1;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
    
    // Connect
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
    // ===== Step 1: Send length =====
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
    
    // ===== Step 2: Send actual data =====
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
    // Step 1: Receive 4-byte length field
    uint32_t length;
    size_t totalReceived = 0;
    
    // Ensure complete 4 bytes are received
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
    
    length = ntohl(length);  // Convert network byte order to host byte order
    
    // Validate length (prevent abnormal data)
    if (length == 0) {
        std::cerr << "[Error] Received zero length" << std::endl;
        return "";
    }
    if (length > 10 * 1024 * 1024) {  // Assume maximum 10MB
        std::cerr << "[Error] Received unreasonable length: " << length << std::endl;
        return "";
    }
    
    // Step 2: Receive actual data
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
    std::unique_lock<std::shared_mutex> lock(cacheMutex_);  // Exclusive lock
    cache_.clear();
    while (!cacheOrder_.empty()) {
        cacheOrder_.pop();
    }
}

void PythonRequestManager::setCacheLimit(size_t limit) {
    cacheLimit_ = limit;
}