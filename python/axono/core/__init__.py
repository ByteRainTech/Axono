import os
import sys

library_path = os.path.dirname(os.path.dirname(__file__)) + "/library/"
sys.path.append(library_path)

from core import DataType, Status

from .tensor import Tensor
from .matmul import matmul

__all__ = ["DataType", "Status", "Tensor", "matmul"]
