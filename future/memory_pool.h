#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace axono {
namespace core {

class MemoryPool {
public:
    static MemoryPool& getInstance() {
        static MemoryPool instance;
        return instance;
    }

    void* allocate(size_t size);
    void deallocate(void* ptr);
    void clear();

private:
    MemoryPool() = default;
    ~MemoryPool();
    
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks_;
    std::unordered_map<void*, size_t> ptr_to_block_;
    
    static constexpr size_t MIN_BLOCK_SIZE = 256;
    static constexpr size_t MAX_POOL_SIZE = 1024 * 1024 * 1024; // 1GB
};

}  // namespace core
}  // namespace axono
