// Axono/src/compute/cpu/operators/add.cpp
#include <cstring>

#include "axono/compute/cpu/operators/add.h"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

template <typename T>
AXONO_FORCE_INLINE void AddBroadcastKernel(const T *a, const T *b, T *out,
                                           size_t M, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t k = 0; k < K; ++k) {
      out[m * K + k] = a[m * K + k] + b[k];  // b 琚箍鎾�
    }
  }
}

// 閫愬厓绱犲姞娉曞唴鏍�
template <typename T>
AXONO_FORCE_INLINE void AddKernel(const T *a, const T *b, T *result,
                                  size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = a[i] + b[i];
  }
}

// 鏍囬噺鍔犳硶鍐呮牳
template <typename T>
AXONO_FORCE_INLINE void AddScalarKernel(const T *a, T scalar, T *result,
                                        size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = a[i] + scalar;
  }
}

// 绫诲瀷鍒嗘淳鐨勫姞娉�
AXONO_FORCE_INLINE core::Status DispatchAdd(const core::Tensor &a,
                                            const core::Tensor &b,
                                            core::Tensor &result) {
  // 1. dtype 蹇呴』涓€鑷�
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  auto a_shape = a.shape();
  auto b_shape = b.shape();

  /* 鎯呭舰 1锛氬畬鍏ㄥ悓褰㈢姸 鈫� 閫愬厓绱� */
  if (a.IsSameShape(b) && a.IsSameShape(result)) {
    const size_t num = a.num_elements();
    switch (a.dtype()) {
      case core::DataType::FLOAT32:
        AddKernel(a.data<float>(), b.data<float>(), result.data<float>(), num);
        return core::Status::OK;
      case core::DataType::FLOAT64:
        AddKernel(a.data<double>(), b.data<double>(), result.data<double>(),
                  num);
        return core::Status::OK;
      case core::DataType::INT32:
        AddKernel(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(),
                  num);
        return core::Status::OK;
      case core::DataType::INT64:
        AddKernel(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(),
                  num);
        return core::Status::OK;
      default:
        return core::Status::UNSUPPORTED_TYPE;
    }
  }

  /* 鎯呭舰 2锛歔M,K] + [K] 骞挎挱 */
  if (a_shape.size() == 2 && b_shape.size() == 1 && result.shape() == a_shape &&
      a_shape[1] == b_shape[0]) {
    const size_t M = a_shape[0];
    const size_t K = a_shape[1];
    switch (a.dtype()) {
      case core::DataType::FLOAT32:
        AddBroadcastKernel(a.data<float>(), b.data<float>(),
                           result.data<float>(), M, K);
        return core::Status::OK;
      case core::DataType::FLOAT64:
        AddBroadcastKernel(a.data<double>(), b.data<double>(),
                           result.data<double>(), M, K);
        return core::Status::OK;
      case core::DataType::INT32:
        AddBroadcastKernel(a.data<int32_t>(), b.data<int32_t>(),
                           result.data<int32_t>(), M, K);
        return core::Status::OK;
      case core::DataType::INT64:
        AddBroadcastKernel(a.data<int64_t>(), b.data<int64_t>(),
                           result.data<int64_t>(), M, K);
        return core::Status::OK;
      default:
        return core::Status::UNSUPPORTED_TYPE;
    }
  }

  return core::Status::SHAPE_MISMATCH;
}

// 绫诲瀷鍒嗘淳鐨勬爣閲忓姞娉�
AXONO_FORCE_INLINE core::Status DispatchAddScalar(const core::Tensor &a,
                                                  void *scalar,
                                                  size_t scalar_size,
                                                  core::Tensor &result) {
  auto num_elements = a.num_elements();

  // 妫€鏌ュ舰鐘朵竴鑷存€�
  if (!a.IsSameShape(result)) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 妫€鏌ユ暟鎹被鍨嬩竴鑷存€�
  if (a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 鏍规嵁鏁版嵁绫诲瀷閫夋嫨鍐呮牳
  switch (a.dtype()) {
    case core::DataType::FLOAT32: {
      float scalar_value = 0.0f;
      if (scalar_size >= sizeof(float)) {
        memcpy(&scalar_value, scalar, sizeof(float));
      }
      AddScalarKernel(a.data<float>(), scalar_value, result.data<float>(),
                      num_elements);
      break;
    }
    case core::DataType::FLOAT64: {
      double scalar_value = 0.0;
      if (scalar_size >= sizeof(double)) {
        memcpy(&scalar_value, scalar, sizeof(double));
      }
      AddScalarKernel(a.data<double>(), scalar_value, result.data<double>(),
                      num_elements);
      break;
    }
    case core::DataType::INT32: {
      int32_t scalar_value = 0;
      if (scalar_size >= sizeof(int32_t)) {
        memcpy(&scalar_value, scalar, sizeof(int32_t));
      }
      AddScalarKernel(a.data<int32_t>(), scalar_value, result.data<int32_t>(),
                      num_elements);
      break;
    }
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

core::Status Add(const core::Context &ctx, const core::Tensor &a,
                 const core::Tensor &b, core::Tensor &result) {
  (void)ctx;  // 鏆傛椂鏈娇鐢�

  // 妫€鏌ユ暟鎹被鍨嬩竴鑷存€�
  if (a.dtype() != b.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 璁剧疆缁撴灉寮犻噺鐨勫舰鐘�
  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  // 璁剧疆缁撴灉鐨勬暟鎹被鍨�
  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 璋冪敤鍐呮牳鎵ц鍔犳硶
  return DispatchAdd(a, b, result);
}

core::Status AddScalar(const core::Context &ctx, const core::Tensor &a,
                       void *scalar, size_t scalar_size, core::Tensor &result) {
  (void)ctx;  // 鏆傛椂鏈娇鐢�

  // 璁剧疆缁撴灉寮犻噺鐨勫舰鐘�
  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  // 璁剧疆缁撴灉鐨勬暟鎹被鍨�
  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 璋冪敤鍐呮牳鎵ц鏍囬噺鍔犳硶
  return DispatchAddScalar(a, scalar, scalar_size, result);
}

}  // namespace operators
}  // namespace cpu
}  // namespace compute
}  // namespace axono
