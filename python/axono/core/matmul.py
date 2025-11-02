"""
Axono Matmul
"""

from core import matmul as _matmul

from .tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    raw_result = _matmul(a._tensor, b._tensor)
    return Tensor.from_raw(raw_result)
