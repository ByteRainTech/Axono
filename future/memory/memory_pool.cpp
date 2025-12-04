#include "axono/core/memory/memory_pool.h"

#include <algorithm>
#include <stdexcept>

namespace axono {
namespace core {

void *MemoryPool::allocate(size_t size) {
  // Round up size to nearest multiple of MIN_BLOCK_SIZE
  size = ((size + MIN_BLOCK_SIZE - 1) / MIN_BLOCK_SIZE) * MIN_BLOCK_SIZE;

  // Try to find an existing block
  for (size_t i = 0; i < blocks_.size(); ++i) {
    if (!blocks_[i].in_use && blocks_[i].size >= size) {
      blocks_[i].in_use = true;
      ptr_to_block_[blocks_[i].ptr] = i;
      return blocks_[i].ptr;
    }
  }

  // Need to allocate new block
  void *ptr = std::malloc(size);
  if (!ptr) {
    throw std::bad_alloc();
  }

  Block block{ptr, size, true};
  blocks_.push_back(block);
  ptr_to_block_[ptr] = blocks_.size() - 1;

  return ptr;
}

void MemoryPool::deallocate(void *ptr) {
  if (!ptr) return;

  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    std::free(ptr);
    return;
  }

  size_t index = it->second;
  blocks_[index].in_use = false;

  // Coalesce adjacent free blocks
  for (size_t i = 0; i < blocks_.size(); ++i) {
    if (i == index || blocks_[i].in_use) continue;

    char *ptr1 = static_cast<char *>(blocks_[i].ptr);
    char *ptr2 = static_cast<char *>(blocks_[index].ptr);

    if (ptr1 + blocks_[i].size == ptr2 || ptr2 + blocks_[index].size == ptr1) {
      // Merge blocks
      void *new_ptr = std::min(ptr1, ptr2);
      size_t new_size = blocks_[i].size + blocks_[index].size;

      ptr_to_block_.erase(blocks_[i].ptr);
      ptr_to_block_.erase(blocks_[index].ptr);

      Block new_block{new_ptr, new_size, false};
      blocks_[i] = new_block;
      blocks_.erase(blocks_.begin() + index);

      ptr_to_block_[new_ptr] = i;
      break;
    }
  }
}

void MemoryPool::clear() {
  for (const auto &block : blocks_) {
    std::free(block.ptr);
  }
  blocks_.clear();
  ptr_to_block_.clear();
}

MemoryPool::~MemoryPool() { clear(); }

}  // namespace core
}  // namespace axono
