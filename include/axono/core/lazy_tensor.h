#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "axono/core/tensor.h"

namespace axono {
namespace core {

class LazyOp {
public:
    virtual ~LazyOp() = default;
    virtual void execute() = 0;
    virtual bool can_fuse() const = 0;
};

class LazyTensor {
public:
    LazyTensor(const Tensor& tensor);
    
    void add_op(std::shared_ptr<LazyOp> op);
    void evaluate();
    bool needs_evaluation() const;
    
    Tensor& get_tensor() { return tensor_; }
    const Tensor& get_tensor() const { return tensor_; }

private:
    Tensor tensor_;
    std::vector<std::shared_ptr<LazyOp>> pending_ops_;
    bool needs_eval_ = false;
};

// 基本算子的Lazy实现
class LazyAdd : public LazyOp {
public:
    LazyAdd(LazyTensor& a, LazyTensor& b, LazyTensor& out)
        : a_(a), b_(b), out_(out) {}

    void execute() override;
    bool can_fuse() const override { return true; }

private:
    LazyTensor& a_;
    LazyTensor& b_;
    LazyTensor& out_;
};

class LazyMatMul : public LazyOp {
public:
    LazyMatMul(LazyTensor& a, LazyTensor& b, LazyTensor& out)
        : a_(a), b_(b), out_(out) {}

    void execute() override;
    bool can_fuse() const override { return false; }

private:
    LazyTensor& a_;
    LazyTensor& b_;
    LazyTensor& out_;
};

}  // namespace core
}  // namespace axono