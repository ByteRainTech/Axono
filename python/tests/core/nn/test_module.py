# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from axono.core import Tensor
from axono.nn import Module

device = os.getenv("axono_default_device", "cpu")


class TestModule(unittest.TestCase):
    """nn.Module 测试"""
    def test_weight(self):
        """测试权重功能"""
        _Module = Module()
        data = Tensor(shape=[1], device=device)
        data.fill(1)
        _Module.add_weight("weight", data)
        # 测试填充
        _Module.parameters()["weight"].fill(2)
        self.assertEqual(_Module.parameters()["weight"].shape, [1])

if __name__ == "__main__":
    unittest.main(verbosity=2)
