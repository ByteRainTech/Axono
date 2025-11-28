#include "axono/compute/cuda/operators.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cuda {
namespace operators {

AXONO_EXPORT core::Status Add(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b,
           core::Tensor &result);

AXONO_EXPORT core::Status AddScalar(const core::Context &ctx, const core::Tensor &a,
                                    void *scalar, size_t scalar_size, core::Tensor &result);

}
}
}
}
