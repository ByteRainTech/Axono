#include "axono/compute/cuda/operators/randn.h"
#include "axono/core/cuda/tensor/kernel.h"
#include "axono/core/types.h"
#include <curand_kernel.h>
#include <random>

namespace axono {
namespace compute {
namespace cuda {
namespace operators {

// CUDA 核函数：生成正态分布随机数
template <typename T>
__global__ void RandnKernel(T* data, size_t num_elements, float mean, float stddev, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // 生成标准正态分布（均值0，标准差1），再转换为目标分布
    float val = curand_normal(&state);
    data[idx] = static_cast<T>(mean + val * stddev);
}

// 分派函数：根据数据类型调用对应核函数
template <typename T>
core::Status DispatchRandn(const core::Context& ctx, core::Tensor& out, float mean, float stddev) {
    size_t num_elements = out.num_elements();
    if (num_elements == 0) return core::Status::OK;

    // 生成随机种子（使用 std::random_device）
    std::random_device rd;  // 现在可正确识别
    unsigned int seed = rd();

    // 启动核函数
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    RandnKernel<T><<<grid_size, block_size>>>(
        out.data<T>(), num_elements, mean, stddev, seed
    );

    return core::Status::OK;
}

// 对外接口实现
core::Status Randn(const core::Context& ctx, core::Tensor& out, float mean, float stddev) {
    if (out.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
        switch (out.dtype()) {
            case core::DataType::FLOAT32:
                return DispatchRandn<float>(ctx, out, mean, stddev);
            case core::DataType::FLOAT64:
                return DispatchRandn<double>(ctx, out, mean, stddev);
            default:
                return core::Status::UNSUPPORTED_TYPE;
        }
#else
        return core::Status::DEVICE_ERROR;  // 修复：使用正确的枚举值
#endif
    } else {
        return core::Status::DEVICE_ERROR;  // 修复：使用正确的枚举值
    }
}

// 显式实例化模板（避免链接错误）
template core::Status DispatchRandn<float>(const core::Context&, core::Tensor&, float, float);
template core::Status DispatchRandn<double>(const core::Context&, core::Tensor&, float, float);

}  // namespace operators
}  // namespace cuda
}  // namespace compute
}  // namespace axono
