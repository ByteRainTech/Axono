#include "axono/core/lazy_tensor.h"
#include "axono/core/operators/add.h"
#include "axono/core/operators/matmul.h"

namespace axono {
namespace core {

LazyTensor::LazyTensor(const Tensor& tensor) : tensor_(tensor) {}

void LazyTensor::add_op(std::shared_ptr<LazyOp> op) {
    pending_ops_.push_back(op);
    needs_eval_ = true;
}

void LazyTensor::evaluate() {
    if (!needs_eval_) return;

    // Execute all pending operations
    for (auto& op : pending_ops_) {
        op->execute();
    }
    
    pending_ops_.clear();
    needs_eval_ = false;
}

bool LazyTensor::needs_evaluation() const {
    return needs_eval_;
}

void LazyAdd::execute() {
    if (a_.needs_evaluation()) a_.evaluate();
    if (b_.needs_evaluation()) b_.evaluate();
    
    operators::add(a_.get_tensor(), b_.get_tensor(), out_.get_tensor());
}

void LazyMatMul::execute() {
    if (a_.needs_evaluation()) a_.evaluate();
    if (b_.needs_evaluation()) b_.evaluate();
    
    operators::matmul(a_.get_tensor(), b_.get_tensor(), out_.get_tensor());
}

}  // namespace core
}  // namespace axono
