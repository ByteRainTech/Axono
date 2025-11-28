#include "axono/core/thread_pool.h"

namespace axono {
namespace core {

ThreadPool::ThreadPool(size_t num_threads) { resize(num_threads); }

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  condition_.notify_all();
  for (auto &thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void ThreadPool::wait_all() {
  while (active_tasks_ > 0) {
    std::this_thread::yield();
  }
}

void ThreadPool::resize(size_t num_threads) {
  // 停止所有现有线程
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }

  condition_.notify_all();
  for (auto &thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  // 重置状态
  threads_.clear();
  stop_ = false;

  // 创建新线程
  threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back(&ThreadPool::worker_thread, this);
  }
}

void ThreadPool::worker_thread() {
  while (true) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(queue_mutex_);

      condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

      if (stop_ && tasks_.empty()) {
        return;
      }

      task = std::move(tasks_.front());
      tasks_.pop();
    }

    task();
  }
}

} // namespace core
} // namespace axono
