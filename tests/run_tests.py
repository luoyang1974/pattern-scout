import unittest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入测试模块
from tests.test_technical_indicators import TestTechnicalIndicators, TestTrendAnalyzer, TestPatternIndicators
from tests.test_data_connectors import TestCSVDataConnector, TestDataModels
from tests.test_pattern_detectors import (
    TestFlagDetector, TestPatternScanner, 
    TestTimeframeAdaptation, TestBackwardCompatibility
)
from tests.test_dataset_manager import TestDatasetManager, TestDatasetManagerEdgeCases
from tests.test_multi_timeframe import TestMultiTimeframe, TestTimeframeSpecificBehavior


def create_test_suite():
    """创建完整的测试套件"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 技术指标测试
    suite.addTest(loader.loadTestsFromTestCase(TestTechnicalIndicators))
    suite.addTest(loader.loadTestsFromTestCase(TestTrendAnalyzer))
    suite.addTest(loader.loadTestsFromTestCase(TestPatternIndicators))
    
    # 数据连接器测试
    suite.addTest(loader.loadTestsFromTestCase(TestCSVDataConnector))
    suite.addTest(loader.loadTestsFromTestCase(TestDataModels))
    
    # 形态检测器测试（统一架构）
    suite.addTest(loader.loadTestsFromTestCase(TestFlagDetector))
    suite.addTest(loader.loadTestsFromTestCase(TestPatternScanner))
    suite.addTest(loader.loadTestsFromTestCase(TestTimeframeAdaptation))
    suite.addTest(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    
    # 数据集管理器测试
    suite.addTest(loader.loadTestsFromTestCase(TestDatasetManager))
    suite.addTest(loader.loadTestsFromTestCase(TestDatasetManagerEdgeCases))
    
    # 多时间周期系统测试
    suite.addTest(loader.loadTestsFromTestCase(TestMultiTimeframe))
    suite.addTest(loader.loadTestsFromTestCase(TestTimeframeSpecificBehavior))
    
    return suite


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("PatternScout 测试套件")
    print("=" * 70)
    
    # 创建测试套件
    suite = create_test_suite()
    
    # 创建测试运行器
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    # 运行测试
    result = runner.run(suite)
    
    # 输出结果摘要
    print("\n" + "=" * 70)
    print("测试结果摘要")
    print("=" * 70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"总测试数: {total_tests}")
    print(f"成功: {successes}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    print(f"成功率: {(successes/total_tests)*100:.1f}%")
    
    if result.failures:
        print(f"\n失败的测试 ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n错误的测试 ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # 返回测试是否全部通过
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)