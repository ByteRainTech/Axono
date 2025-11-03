"""
Axono Tensor - Python interface for Tensor class
"""

import ctypes

import numpy as np

from core import DataType, Status
from core import Tensor as _Tensor


class Tensor:
    """Python Tensor class wrapping C++ Tensor"""

    def __init__(
        self, dtype: DataType = DataType.FLOAT32, shape: list[int] | None = None
    ):
        """
        Initialize Tensor

        Args:
            dtype: Data type of tensor
            shape: Shape of tensor, if None creates empty tensor
        """
        if shape is None:
            self._tensor = _Tensor(dtype)
        else:
            self._tensor = _Tensor(dtype, shape)

    @classmethod
    def create(cls, dtype: DataType, shape: list[int]) -> "Tensor":
        """Create a new tensor"""
        return cls(dtype, shape)

    @classmethod
    def create_like(cls, other: "Tensor") -> "Tensor":
        """Create a tensor with same shape and dtype as another"""
        tensor = cls()
        tensor._tensor = _Tensor.create_like(other._tensor)
        return tensor

    @classmethod
    def from_raw(cls, raw_tensor):
        obj = cls.__new__(cls)
        obj._tensor = raw_tensor
        return obj

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Tensor":
        """Create tensor from numpy array - FIXED VERSION"""
        dtype_map = {
            np.int8: DataType.INT8,
            np.int16: DataType.INT16,
            np.int32: DataType.INT32,
            np.int64: DataType.INT64,
            np.float32: DataType.FLOAT32,
            np.float64: DataType.FLOAT64,
            np.bool_: DataType.BOOLEAN,
        }

        # Ensure we have a contiguous array
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        dtype = dtype_map.get(array.dtype.type, DataType.FLOAT32)
        tensor_obj = cls(dtype, list(array.shape))

        # Get the tensor data as a numpy array view and copy the input array data
        if array.dtype == np.int8:
            tensor_data = tensor_obj._tensor.data_int8()
        elif array.dtype == np.int16:
            tensor_data = tensor_obj._tensor.data_int16()
        elif array.dtype == np.int32:
            tensor_data = tensor_obj._tensor.data_int32()
        elif array.dtype == np.int64:
            tensor_data = tensor_obj._tensor.data_int64()
        elif array.dtype == np.float32:
            tensor_data = tensor_obj._tensor.data_float32()
        elif array.dtype == np.float64:
            tensor_data = tensor_obj._tensor.data_float64()
        elif array.dtype == np.bool_:
            tensor_data = tensor_obj._tensor.data_bool()
        else:
            # Default to float32 and convert the input array
            tensor_data = tensor_obj._tensor.data_float32()
            array = array.astype(np.float32)

        # Copy the numpy array data into the tensor's data
        tensor_data[:] = array

        return tensor_obj

    def __matmul__(self, other) -> "Tensor":
        from .operators import matmul

        return matmul(self, other)

    def to_numpy(self) -> np.ndarray:
        """Convert tensor to numpy array - FIXED VERSION"""

        if self.dtype == DataType.INT8:
            result = self._tensor.data_int8()
        elif self.dtype == DataType.INT16:
            result = self._tensor.data_int16()
        elif self.dtype == DataType.INT32:
            result = self._tensor.data_int32()
        elif self.dtype == DataType.INT64:
            result = self._tensor.data_int64()
        elif self.dtype == DataType.FLOAT32:
            result = self._tensor.data_float32()
        elif self.dtype == DataType.FLOAT64:
            result = self._tensor.data_float64()
        elif self.dtype == DataType.BOOLEAN:
            result = self._tensor.data_bool()
        else:
            raise ValueError(f"Unsupported dtype for numpy conversion: {self.dtype}")
        return result

    def reshape(self, new_shape: list[int]) -> "Tensor":
        """Reshape the tensor"""
        status = self._tensor.reshape(new_shape)
        if status != Status.OK:
            raise RuntimeError(f"Reshape failed with status: {status}")
        return self

    def resize(self, new_shape: list[int]) -> "Tensor":
        """Resize the tensor (may reallocate memory)"""
        status = self._tensor.resize(new_shape)
        if status != Status.OK:
            raise RuntimeError(f"Resize failed with status: {status}")
        return self

    def fill_zero(self) -> "Tensor":
        """Fill tensor with zeros"""
        status = self._tensor.fill_zero()
        if status != Status.OK:
            raise RuntimeError(f"Fill zero failed with status: {status}")
        return self

    def fill(self, value: int | float) -> "Tensor":
        """
        Fill tensor with the specified value.

        Parameters:
        -----------
        value : int or float
            The value to fill the tensor with

        Returns:
        --------
        Tensor
            self for method chaining

        Raises:
        -------
        ValueError
            If the value is incompatible with the tensor's data type
        RuntimeError
            If the fill operation fails
        """
        from ctypes import c_char_p, c_void_p, py_object, pythonapi

        dtype_map = {
            DataType.INT8: (np.int8, ctypes.c_int8),
            DataType.INT16: (np.int16, ctypes.c_int16),
            DataType.INT32: (np.int32, ctypes.c_int32),
            DataType.INT64: (np.int64, ctypes.c_int64),
            DataType.FLOAT32: (np.float32, ctypes.c_float),
            DataType.FLOAT64: (np.float64, ctypes.c_double),
            DataType.BOOLEAN: (np.bool_, ctypes.c_bool),
        }

        if self.dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype for fill: {self.dtype}")

        try:
            np_dtype, c_type = dtype_map[self.dtype]
            c_value = c_type(value)
            pythonapi.PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
            pythonapi.PyCapsule_New.restype = py_object

            capsule = pythonapi.PyCapsule_New(ctypes.addressof(c_value), None, None)

            status = self._tensor.fill(capsule, ctypes.sizeof(c_value))

            if status != Status.OK:
                raise RuntimeError(f"Fill failed with status: {status}")

            return self

        except OverflowError:
            raise ValueError(f"Value {value} is too large for dtype {self.dtype}")
        except Exception as e:
            raise RuntimeError(f"Fill operation failed: {e}")

    def copy_from(self, other: "Tensor") -> "Tensor":
        """Copy data from another tensor"""
        status = self._tensor.copy_from(other._tensor)
        if status != Status.OK:
            raise RuntimeError(f"Copy failed with status: {status}")
        return self

    def is_same_shape(self, other: "Tensor") -> bool:
        """Check if has same shape as another tensor"""
        return self._tensor.is_same_shape(other._tensor)

    @property
    def dtype(self) -> DataType:
        return self._tensor.dtype

    @property
    def shape(self) -> list[int]:
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def num_elements(self) -> int:
        return self._tensor.num_elements

    @property
    def num_bytes(self) -> int:
        return self._tensor.num_bytes

    def __repr__(self) -> str:
        return self._tensor.__repr__()

    def __str__(self) -> str:
        return self._tensor.__str__()

    @staticmethod
    def zeros(shape: list[int], dtype: DataType = DataType.FLOAT32) -> _Tensor:
        """Create a tensor filled with zeros"""
        tensor = Tensor(dtype, shape)
        tensor.fill_zero()
        return tensor

    @staticmethod
    def ones(shape: list[int], dtype: DataType = DataType.FLOAT32) -> _Tensor:
        """Create a tensor filled with ones"""
        tensor = Tensor(dtype, shape)
        tensor.fill(1)
        return tensor

    @staticmethod
    def full(
        shape: list[int], value: int | float, dtype: DataType = DataType.FLOAT32
    ) -> _Tensor:
        """Create a tensor filled with value"""
        tensor = Tensor(dtype, shape)
        tensor.fill(value)
        return tensor
