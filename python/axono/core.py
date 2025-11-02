"""
Axono core module - Python interface to C++ core library
"""

import numpy as np
import sys
import os

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .axono_core import (
        memory_copy,
        memory_copy_int8,
        memory_copy_int16, 
        memory_copy_int32,
        memory_copy_int64,
        memory_copy_float32,
        memory_copy_float64,
        Status,
        DataType
    )
except ImportError as e:
    raise ImportError(
        "Cannot import Axono core library. "
        "Please make sure the C++ extension is built correctly."
    ) from e


class AxonoContext:
    """Axono computation context"""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
    
    def __repr__(self):
        return f"AxonoContext(device_id={self.device_id})"


def copy_array(dst, src):
    """
    Copy data from source array to destination array.
    
    Parameters:
    -----------
    dst : numpy.ndarray
        Destination array
    src : numpy.ndarray  
        Source array
        
    Raises:
    -------
    ValueError
        If arrays have different shapes or incompatible types
    RuntimeError
        If memory copy fails
    """
    if dst.shape != src.shape:
        raise ValueError(f"Array shapes must match: {dst.shape} vs {src.shape}")
    
    if dst.dtype != src.dtype:
        raise ValueError(f"Array dtypes must match: {dst.dtype} vs {src.dtype}")
    
    # 根据数据类型选择相应的拷贝函数
    dtype_handlers = {
        np.int8: memory_copy_int8,
        np.int16: memory_copy_int16,
        np.int32: memory_copy_int32,
        np.int64: memory_copy_int64,
        np.float32: memory_copy_float32,
        np.float64: memory_copy_float64,
    }
    
    handler = dtype_handlers.get(dst.dtype.type, memory_copy)
    handler(dst, src)


def benchmark_memory_copy(size_mb=100, dtype=np.float32):
    """
    Benchmark memory copy performance.
    
    Parameters:
    -----------
    size_mb : int
        Size of arrays in megabytes
    dtype : numpy.dtype
        Data type of arrays
        
    Returns:
    --------
    dict
        Benchmark results including bandwidth
    """
    size_bytes = size_mb * 1024 * 1024
    element_size = np.dtype(dtype).itemsize
    num_elements = size_bytes // element_size
    
    # 创建测试数组
    src = np.random.rand(num_elements).astype(dtype)
    dst = np.empty_like(src)
    
    # 预热
    copy_array(dst, src)
    
    # 实际测试
    import time
    start_time = time.perf_counter()
    copy_array(dst, src)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    bandwidth = (size_bytes / (1024 * 1024)) / duration  # MB/s
    
    return {
        'size_mb': size_mb,
        'dtype': dtype,
        'duration_seconds': duration,
        'bandwidth_mb_s': bandwidth,
        'array_shape': src.shape
    }


def test_basic_functionality():
    """Test basic Axono functionality"""
    print("Testing Axono basic functionality...")
    
    # 测试不同数据类型的拷贝
    test_cases = [
        (np.int32, [1, 2, 3, 4, 5]),
        (np.float32, [1.1, 2.2, 3.3, 4.4, 5.5]),
        (np.float64, [1.1, 2.2, 3.3, 4.4, 5.5]),
        (np.int64, [100, 200, 300, 400, 500]),
    ]
    
    for dtype, data in test_cases:
        src = np.array(data, dtype=dtype)
        dst = np.empty_like(src)
        
        copy_array(dst, src)
        
        if np.array_equal(src, dst):
            print(f"✓ {dtype} copy test passed")
        else:
            print(f"✗ {dtype} copy test failed")
            print(f"  Source: {src}")
            print(f"  Destination: {dst}")
    
    # 测试大规模拷贝
    print("\nTesting large array copy...")
    large_src = np.random.rand(1000000).astype(np.float32)
    large_dst = np.empty_like(large_src)
    
    copy_array(large_dst, large_src)
    
    if np.allclose(large_src, large_dst):
        print("✓ Large array copy test passed")
    else:
        print("✗ Large array copy test failed")
    
    print("\nAll tests completed!")


# 导出公共API
__all__ = [
    'copy_array',
    'benchmark_memory_copy', 
    'test_basic_functionality',
    'AxonoContext',
    'Status',
    'DataType'
]
