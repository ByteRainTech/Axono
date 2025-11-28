#pragma once

#include "axono/core/lazy_tensor.h"
#include <memory>
#include <vector>

namespace axono {
namespace core {

// 定义算子融合模式
enum class FusionPattern {
  NONE,
  ADD_RELU,        // 加法+ReLU融合
  MATMUL_ADD,      // 矩阵乘法+加法融合
  MATMUL_ADD_RELU, // 矩阵乘法+加法+ReLU融合
  CONV_ADD,        // 卷积+加法融合
  CONV_RELU,       // 卷积+ReLU融合
  CONV_ADD_RELU    // 卷积+加法+ReLU融合
};

class FusedOp : public LazyOp {
public:
  FusedOp(std::vector<std::shared_ptr<LazyOp>> ops, FusionPattern pattern)
      : ops_(std::move(ops)), pattern_(pattern) {}

  void execute() override;
  bool can_fuse() const override { return false; }

private:
  std::vector<std::shared_ptr<LazyOp>> ops_;
  FusionPattern pattern_;
};

class OpFuser {
public:
  static OpFuser &getInstance() {
    static OpFuser instance;
    return instance;
  }

  // 尝试融合一系列操作
  std::vector<std::shared_ptr<LazyOp>>
  fuse(std::vector<std::shared_ptr<LazyOp>> ops);

private:
  OpFuser() = default;

  // 检查是否可以应用特定的融合模式
  bool can_apply_pattern(const std::vector<std::shared_ptr<LazyOp>> &ops,
                         size_t start_idx, FusionPattern pattern);

  // 创建融合后的操作
  std::shared_ptr<FusedOp>
  create_fused_op(const std::vector<std::shared_ptr<LazyOp>> &ops,
                  size_t start_idx, size_t end_idx, FusionPattern pattern);
};

} // namespace core
} // namespace axono
