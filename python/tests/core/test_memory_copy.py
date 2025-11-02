import sys

sys.path.append("../../")

from axono.core import memory_copy
import numpy as np
import unittest


class TestMemoryCopy(unittest.TestCase):
    def test_memory_copy(self):
        src = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        dst = np.zeros_like(src)
        print(f"Before: src={src}, dst={dst}")
        memory_copy(dst, src)
        print(f"After:  src={src}, dst={dst}")


if __name__ == "__main__":
    unittest.main()
