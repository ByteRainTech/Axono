"""
core.ops.Relu()
"""

from core import relu as relu_op, relu_ as relu_op_

from ..tensor import Tensor


def relu(a: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        raw_result = relu_op_(a._tensor)
    else:
        raw_result = relu_op(a._tensor)
    return Tensor.from_raw(raw_result)
