#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {
namespace ops {

core::Status DispatchRelu(const core::Tensor &input, core::Tensor &output);
core::Status DispatchReluInplace(core::Tensor &tensor);
core::Status Relu(const core::Context &ctx, const core::Tensor &input,
                  core::Tensor &output);
core::Status ReluInplace(const core::Context &ctx, core::Tensor &tensor);

}  // namespace ops
}  // namespace cpu
}  // namespace compute
}  // namespace axono
