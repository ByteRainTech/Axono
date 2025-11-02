import unittest
import os
import importlib.util
import sys

def discover_tests_recursive(directory):
    """递归发现所有测试文件"""
    test_suite = unittest.TestSuite()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
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

if __name__ == '__main__':
    run_all_tests()
