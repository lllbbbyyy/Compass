#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <vector>
#include <atomic>
#include <functional>
#include <chrono>

class AutoScalingThreadPool {
public:
    AutoScalingThreadPool();
    ~AutoScalingThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;

private:
    void worker();

    std::mutex queueMutex;
    std::condition_variable condition;
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> threads;

    std::atomic<bool> stop;
    std::atomic<int> idleThreads;
    const int maxThreads = std::thread::hardware_concurrency(); // 可扩展上限
};

AutoScalingThreadPool::AutoScalingThreadPool() : stop(false), idleThreads(0) {}

AutoScalingThreadPool::~AutoScalingThreadPool() {
    stop = true;
    condition.notify_all();
    for (auto& t : threads)
        if (t.joinable())
            t.join();
}

void AutoScalingThreadPool::worker() {
    while (!stop) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (!condition.wait_for(lock, std::chrono::seconds(2), [this]() { return stop || !tasks.empty(); })) {
                // 超时未获取任务 → 退出线程
                return;
            }

            if (stop && tasks.empty()) return;

            task = std::move(tasks.front());
            tasks.pop();
        }

        idleThreads--;
        task();
        idleThreads++;
    }
}

template<class F, class... Args>
auto AutoScalingThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<return_type> res = task->get_future();

    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace([task]() { (*task)(); });
    }

    condition.notify_one();

    // 动态创建线程（如果没有空闲线程，且未超最大线程数）
    if (idleThreads <= 0 && threads.size() < maxThreads) {
        threads.emplace_back([this]() { this->worker(); });
        idleThreads++;
    }

    return res;
}

int main() {
    AutoScalingThreadPool pool;

    std::vector<std::future<int>> results;
    for (int i = 0; i < 20; ++i) {
        results.push_back(pool.enqueue([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return i * i;
        }));
    }

    for (auto& f : results)
        std::cout << f.get() << " ";
    std::cout << std::endl;
}