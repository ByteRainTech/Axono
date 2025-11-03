"""
Axono Add
"""

from core import add as _add

from ..tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    raw_result = _add(a._tensor, b._tensor)
    return Tensor.from_raw(raw_result)
