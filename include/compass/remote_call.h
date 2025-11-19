#ifndef PYTHON_REQUEST_MANAGER_H
#define PYTHON_REQUEST_MANAGER_H

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <thread>
#include <functional>
#include <shared_mutex>

class PythonRequestManager {
public:
    static PythonRequestManager& getInstance();
    
    // 禁止拷贝和赋值
    PythonRequestManager(const PythonRequestManager&) = delete;
    PythonRequestManager& operator=(const PythonRequestManager&) = delete;
    
    // 发起请求，返回结果
    std::string request(const std::string& params);
    
    // 清除缓存
    void clearCache();
    
    // 设置缓存大小限制
    void setCacheLimit(size_t limit);
    void disconnect();  // 新增：断开连接
    
    ~PythonRequestManager();

private:
    PythonRequestManager();
    
    // 计算参数的哈希值作为key
    std::string computeHash(const std::string& params);
    
    // 实际执行python调用
    std::string executePythonRequest(const std::string& params);
    
    // 长连接相关
    bool ensureConnected();
    int connectToServer();
    bool sendData(const std::string& data);  // 不再需要传入sock
    std::string receiveData();                // 不再需要传入sock

private:
    // 持久化连接
    int persistentSocket_;
    bool connected_;
    std::mutex socketMutex_;  // 保护socket的锁
    // 互斥锁
    std::shared_mutex cacheMutex_;
    std::shared_mutex requestMutex_;

    struct PendingRequest {
        std::promise<std::string> promise;
        std::vector<std::shared_ptr<std::promise<std::string>>> waiters;
    };
    
    // 缓存：key -> result
    std::unordered_map<std::string, std::string> cache_;
    
    // 正在处理的请求：key -> PendingRequest
    std::unordered_map<std::string, std::shared_ptr<PendingRequest>> pendingRequests_;
    
    // LRU缓存顺序维护
    std::queue<std::string> cacheOrder_;
    
    // 配置
    std::string serverHost_;
    int serverPort_;
    size_t cacheLimit_;
};

#endif // PYTHON_REQUEST_MANAGER_H