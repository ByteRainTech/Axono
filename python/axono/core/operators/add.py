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
Axono Add
"""

from axonolib import add as _add

from ..tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    raw_result = _add(a._tensor, b._tensor)
    return Tensor.from_raw(raw_result)
