import unittest
import os

def run_all_tests():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用discover方法递归发现所有测试文件
    loader = unittest.TestLoader()
    suite = loader.discover(current_dir, pattern='test_*.py')
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_all_tests()
