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
import importlib.util
import os
import sys
import unittest

sys.path.append("../")
sys.path.append("../../")

welcome = (
    """                                                                            
       db         8b        d8  ,ad8888ba,    888b      88    ,ad8888ba,    
      d88b         Y8,    ,8P  d8"'    `"8b   8888b     88   d8"'    `"8b   
     d8'`8b         `8b  d8'  d8'        `8b  88 `8b    88  d8'        `8b  
    d8'  `8b          Y88P    88          88  88  `8b   88  88          88  
   d8YaaaaY8b         d88b    88          88  88   `8b  88  88          88  
  d8"""
    """""8b      ,8P  Y8,  Y8,        ,8P  88    `8b 88  Y8,        ,8P  
 d8'        `8b    d8'    `8b  Y8a.    .a8P   88     `8888   Y8a.    .a8P   
d8'          `8b  8P        Y8  `"Y8888Y"'    88      `888    `"Y8888Y"'    
                                                                            
                                                                            """
)
print(welcome)
print("为您运行 单元测试。")
print("正发生")


def discover_tests_recursive(directory):
    """递归发现所有测试文件"""
    test_suite = unittest.TestSuite()

    for root, _dirs, files in os.walk(directory):
        _dirs[:] = [d for d in _dirs if d != ".ipynb_checkpoints"]
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                # 构建模块路径
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]

                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # 加载测试
                loader = unittest.TestLoader()
                tests = loader.loadTestsFromModule(module)
                test_suite.addTest(tests)

    return test_suite


def run_all_tests():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    suite = discover_tests_recursive(current_dir)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
