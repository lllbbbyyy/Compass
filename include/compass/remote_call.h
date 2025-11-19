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
    
    // Disable copying and assignment
    PythonRequestManager(const PythonRequestManager&) = delete;
    PythonRequestManager& operator=(const PythonRequestManager&) = delete;
    
    // Make request and return result
    std::string request(const std::string& params);
    
    // Clear cache
    void clearCache();
    
    // Set cache size limit
    void setCacheLimit(size_t limit);
    void disconnect();  // New: disconnect
    
    ~PythonRequestManager();

private:
    PythonRequestManager();
    
    // Compute hash of parameters as key
    std::string computeHash(const std::string& params);
    
    // Actually execute python call
    std::string executePythonRequest(const std::string& params);
    
    // Persistent connection related
    bool ensureConnected();
    int connectToServer();
    bool sendData(const std::string& data);  // No need to pass sock
    std::string receiveData();               // No need to pass sock

private:
    // Persistent connection
    int persistentSocket_;
    bool connected_;
    std::mutex socketMutex_;  // Lock for protecting socket
    // Mutexes
    std::shared_mutex cacheMutex_;
    std::shared_mutex requestMutex_;

    struct PendingRequest {
        std::promise<std::string> promise;
        std::vector<std::shared_ptr<std::promise<std::string>>> waiters;
    };
    
    // Cache: key -> result
    std::unordered_map<std::string, std::string> cache_;
    
    // Requests in progress: key -> PendingRequest
    std::unordered_map<std::string, std::shared_ptr<PendingRequest>> pendingRequests_;
    
    // LRU cache order maintenance
    std::queue<std::string> cacheOrder_;
    
    // Configuration
    std::string serverHost_;
    int serverPort_;
    size_t cacheLimit_;
};

#endif // PYTHON_REQUEST_MANAGER_H