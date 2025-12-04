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
"""
core.ops.Relu()
"""

from axonolib import relu as relu_op
from axonolib import relu_ as relu_op_

from ..tensor import Tensor


def relu(a: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        raw_result = relu_op_(a._tensor)
    else:
        raw_result = relu_op(a._tensor)
    return Tensor.from_raw(raw_result)
