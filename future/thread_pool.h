#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace axono {
namespace core {

class ThreadPool {
public:
    static ThreadPool& getInstance() {
        static ThreadPool instance;
        return instance;
    }
    
    // 提交任务到线程池
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args);
    
    // 等待所有任务完成
    void wait_all();
    
    // 设置线程池大小
    void resize(size_t num_threads);
    
    // 获取线程池大小
    size_t size() const { return threads_.size(); }
    
private:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    void worker_thread();
    
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
};

// 并行计算辅助函数
template<typename Index, typename Func>
void parallel_for(Index start, Index end, Index step, Func&& f);

template<typename Index, typename Func>
void parallel_for(Index start, Index end, Func&& f) {
    parallel_for(start, end, 1, std::forward<Func>(f));
}

// 实现模板函数
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) {
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    auto future = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Submit to stopped ThreadPool");
        }
        
        active_tasks_++;
        tasks_.emplace([task, this]() {
            (*task)();
            active_tasks_--;
        });
    }
    
    condition_.notify_one();
    return future;
}

template<typename Index, typename Func>
void parallel_for(Index start, Index end, Index step, Func&& f) {
    auto& pool = ThreadPool::getInstance();
    
    const Index num_elements = (end - start + step - 1) / step;
    const Index block_size = std::max(Index(1), num_elements / (4 * pool.size()));
    
    std::vector<std::future<void>> futures;
    
    for (Index i = start; i < end; i += block_size * step) {
        Index local_end = std::min(i + block_size * step, end);
        futures.emplace_back(pool.submit([i, local_end, step, &f]() {
            for (Index j = i; j < local_end; j += step) {
                f(j);
            }
        }));
    }
    
    for (auto& future : futures) {
        future.get();
    }
}

}  // namespace core
}  // namespace axono
