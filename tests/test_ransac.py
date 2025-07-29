"""
测试RANSAC趋势线拟合系统
"""
import unittest
import pandas as pd
import numpy as np

from src.patterns.base.ransac_trend_fitter import RANSACTrendLineFitter
from src.patterns.base.pattern_components import PatternComponents
from src.patterns.detectors.flag_detector import FlagDetector
from src.patterns.detectors.pennant_detector import PennantDetector


class TestRANSACTrendLineFitter(unittest.TestCase):
    """测试RANSAC趋势线拟合器"""
    
    def setUp(self):
        """设置测试环境"""
        self.fitter = RANSACTrendLineFitter(
            max_iterations=500,
            min_inliers_ratio=0.6,
            confidence=0.99
        )
        
        # 创建测试数据
        self.clean_data = self._create_clean_trend_data()
        self.noisy_data = self._create_noisy_trend_data()
        self.outlier_data = self._create_outlier_trend_data()
    
    def _create_clean_trend_data(self) -> pd.DataFrame:
        """创建干净的趋势数据"""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='15min')
        
        # 创建明显的上升趋势
        base_trend = np.linspace(100, 110, 20)
        prices = base_trend + np.random.normal(0, 0.1, 20)  # 很小的噪音
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST_CLEAN'] * 20,
            'open': prices - 0.1,
            'high': prices + 0.2,
            'low': prices - 0.2,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 20)
        })
    
    def _create_noisy_trend_data(self) -> pd.DataFrame:
        """创建有噪音的趋势数据"""
        dates = pd.date_range(start='2024-01-01', periods=25, freq='15min')
        
        # 创建带噪音的上升趋势
        base_trend = np.linspace(100, 115, 25)
        noise = np.random.normal(0, 1.0, 25)  # 较大的噪音
        prices = base_trend + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST_NOISY'] * 25,
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 25)
        })
    
    def _create_outlier_trend_data(self) -> pd.DataFrame:
        """创建包含异常值的趋势数据"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='15min')
        
        # 创建基础上升趋势
        base_trend = np.linspace(100, 120, 30)
        noise = np.random.normal(0, 0.5, 30)
        prices = base_trend + noise
        
        # 添加异常值
        outlier_indices = [5, 12, 18, 25]
        for idx in outlier_indices:
            if idx < len(prices):
                prices[idx] += np.random.choice([-8, 8])  # 极端异常值
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST_OUTLIER'] * 30,
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 30)
        })
    
    def test_clean_data_fitting(self):
        """测试干净数据的拟合效果"""
        point_indices = list(range(len(self.clean_data)))
        trend_line = self.fitter.fit_trend_line(self.clean_data, point_indices, 'close')
        
        self.assertIsNotNone(trend_line)
        self.assertGreater(trend_line.r_squared, 0.8)  # 干净数据应该有很高的R²
        self.assertGreater(trend_line.slope, 0)  # 上升趋势
        
        stats = self.fitter.get_fit_statistics()
        self.assertGreater(stats['inliers_ratio'], 0.8)  # 大部分点应该是内点
    
    def test_noisy_data_fitting(self):
        """测试噪音数据的拟合效果"""
        point_indices = list(range(len(self.noisy_data)))
        trend_line = self.fitter.fit_trend_line(self.noisy_data, point_indices, 'close')
        
        self.assertIsNotNone(trend_line)
        self.assertGreater(trend_line.slope, 0)  # 应该还能检测到上升趋势
        
        stats = self.fitter.get_fit_statistics()
        self.assertGreater(stats['inliers_ratio'], 0.5)  # 至少一半的点是内点
        self.assertLess(stats['iterations_used'], self.fitter.max_iterations)
    
    def test_outlier_robustness(self):
        """测试对异常值的鲁棒性"""
        point_indices = list(range(len(self.outlier_data)))
        
        # RANSAC拟合
        ransac_line = self.fitter.fit_trend_line(self.outlier_data, point_indices, 'close')
        
        # 传统OLS拟合（用于对比）
        prices = self.outlier_data['close'].values
        x = np.arange(len(prices))
        from scipy import stats
        slope_ols, intercept_ols, r_value_ols, _, _ = stats.linregress(x, prices)
        
        # RANSAC应该能更好地处理异常值
        self.assertIsNotNone(ransac_line)
        self.assertGreater(ransac_line.slope, 0)  # 检测到正确的上升趋势
        
        # RANSAC应该有更好的拟合质量（当有异常值时）
        if r_value_ols ** 2 < 0.7:  # 如果OLS因异常值拟合较差
            self.assertGreater(ransac_line.r_squared, r_value_ols ** 2)
        
        stats = self.fitter.get_fit_statistics()
        self.assertGreater(len(stats['outliers_indices']), 0)  # 应该检测到异常值
    
    def test_adaptive_threshold(self):
        """测试自适应阈值计算"""
        # 测试不同波动率数据的阈值自适应
        low_vol_data = self.clean_data  # 低波动率
        high_vol_data = self.noisy_data  # 高波动率
        
        # 获取两种数据的拟合统计
        self.fitter.fit_trend_line(low_vol_data, list(range(len(low_vol_data))), 'close')
        low_vol_stats = self.fitter.get_fit_statistics()
        
        self.fitter.fit_trend_line(high_vol_data, list(range(len(high_vol_data))), 'close')
        high_vol_stats = self.fitter.get_fit_statistics()
        
        # 高波动率数据应该使用更大的阈值
        self.assertGreater(high_vol_stats['threshold_used'], low_vol_stats['threshold_used'])
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        # 只有一个点
        result = self.fitter.fit_trend_line(self.clean_data[:1], [0], 'close')
        self.assertIsNone(result)
        
        # 两个点（最小需求）
        result = self.fitter.fit_trend_line(self.clean_data[:2], [0, 1], 'close')
        self.assertIsNotNone(result)
    
    def test_comparison_with_ols(self):
        """测试与OLS方法的比较功能"""
        point_indices = list(range(len(self.outlier_data)))
        comparison = self.fitter.compare_with_ols(self.outlier_data, point_indices, 'close')
        
        self.assertIn('ransac', comparison)
        self.assertIn('ols', comparison)
        self.assertIn('improvement', comparison)
        
        # 检查比较结果的结构
        self.assertIn('r_squared', comparison['ransac'])
        self.assertIn('outliers_count', comparison['ransac'])
        self.assertIn('inliers_ratio', comparison['ransac'])
        
        # 在有异常值的情况下，RANSAC应该检测到异常值
        self.assertGreater(comparison['ransac']['outliers_count'], 0)


class TestRANSACIntegration(unittest.TestCase):
    """测试RANSAC与检测器的集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.pattern_components = PatternComponents()
        self.flag_detector = FlagDetector()
        self.pennant_detector = PennantDetector()
        
        # 创建测试数据
        self.test_data = self._create_pattern_test_data()
    
    def _create_pattern_test_data(self) -> pd.DataFrame:
        """创建包含形态的测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        
        # 创建旗杆 + 旗面的价格走势
        np.random.seed(42)
        
        # 旗杆阶段（强烈上涨）
        flagpole_length = 20
        flagpole_prices = np.linspace(100, 108, flagpole_length)
        
        # 旗面阶段（水平整理）
        flag_length = 30
        flag_base = 107
        flag_prices = flag_base + np.random.normal(0, 0.3, flag_length)
        
        # 添加一些异常值到旗面中
        flag_prices[10] += 2.0  # 异常高点
        flag_prices[20] -= 1.5  # 异常低点
        
        # 后续走势
        continuation_length = 50
        continuation_prices = np.linspace(flag_base, 112, continuation_length)
        
        # 合成完整价格序列
        all_prices = np.concatenate([flagpole_prices, flag_prices, continuation_prices])
        
        # 生成OHLC数据
        opens = all_prices + np.random.normal(0, 0.1, len(all_prices))
        highs = all_prices + np.random.uniform(0.1, 0.5, len(all_prices))
        lows = all_prices - np.random.uniform(0.1, 0.5, len(all_prices))
        closes = all_prices
        
        # 确保OHLC逻辑
        for i in range(len(all_prices)):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        # 生成成交量
        volumes = np.random.uniform(1000000, 2000000, len(all_prices))
        volumes[:flagpole_length] *= 2.5  # 旗杆期间放量
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST_PATTERN'] * len(all_prices),
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def test_pattern_components_ransac_integration(self):
        """测试PatternComponents的RANSAC集成"""
        # 模拟旗面的摆动点
        swing_indices = [25, 30, 35, 40, 45]  # 旗面阶段的一些点
        
        # 使用RANSAC拟合
        trend_line_ransac = self.pattern_components.fit_trend_line_ransac(
            self.test_data, swing_indices, 'close', use_ransac=True
        )
        
        # 使用传统方法拟合
        trend_line_ols = self.pattern_components.fit_trend_line_ransac(
            self.test_data, swing_indices, 'close', use_ransac=False
        )
        
        self.assertIsNotNone(trend_line_ransac)
        self.assertIsNotNone(trend_line_ols)
        
        # 获取RANSAC统计信息
        stats = self.pattern_components.get_ransac_statistics()
        self.assertIn('inliers_ratio', stats)
        self.assertIn('outliers_indices', stats)
    
    def test_method_comparison(self):
        """测试拟合方法比较功能"""
        swing_indices = [25, 30, 35, 40, 45, 50]
        
        comparison = self.pattern_components.compare_fitting_methods(
            self.test_data, swing_indices, 'close'
        )
        
        self.assertIn('ransac', comparison)
        self.assertIn('ols', comparison)
        self.assertIn('improvement', comparison)
        
        # 检查改进指标
        improvement = comparison['improvement']
        self.assertIn('outliers_detected', improvement)
        self.assertIn('robustness_gain', improvement)
    
    def test_flag_detector_ransac_integration(self):
        """测试旗形检测器的RANSAC集成"""
        # 测试RANSAC开启和关闭的情况
        patterns_with_ransac = self.flag_detector.detect(self.test_data, '15m')
        
        # 创建禁用RANSAC的检测器
        config_no_ransac = self.flag_detector.config.copy()
        config_no_ransac['global']['enable_ransac_fitting'] = False
        flag_detector_no_ransac = FlagDetector(config_no_ransac)
        
        patterns_without_ransac = flag_detector_no_ransac.detect(self.test_data, '15m')
        
        # 两种方式都应该返回有效结果
        self.assertIsInstance(patterns_with_ransac, list)
        self.assertIsInstance(patterns_without_ransac, list)
    
    def test_pennant_detector_ransac_integration(self):
        """测试三角旗形检测器的RANSAC集成"""
        patterns = self.pennant_detector.detect(self.test_data, '15m')
        
        self.assertIsInstance(patterns, list)
        # 主要确保没有错误，具体的形态检测效果需要更复杂的测试数据
    
    def test_edge_cases(self):
        """测试边缘情况"""
        # 测试空数据
        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        result = self.pattern_components.fit_trend_line_ransac(empty_df, [], 'close')
        self.assertIsNone(result)
        
        # 测试单点数据
        single_point = self.test_data[:1]
        result = self.pattern_components.fit_trend_line_ransac(single_point, [0], 'close')
        self.assertIsNone(result)
        
        # 测试无效索引
        result = self.pattern_components.fit_trend_line_ransac(
            self.test_data, [999], 'close'  # 超出范围的索引
        )
        self.assertIsNone(result)


class TestRANSACPerformance(unittest.TestCase):
    """测试RANSAC性能"""
    
    def test_large_dataset_performance(self):
        """测试大数据集的性能"""
        # 创建较大的数据集
        dates = pd.date_range(start='2024-01-01', periods=500, freq='15min')
        
        # 基础趋势 + 噪音 + 异常值
        base_trend = np.linspace(100, 150, 500)
        noise = np.random.normal(0, 2.0, 500)
        prices = base_trend + noise
        
        # 添加随机异常值
        outlier_count = 50
        outlier_indices = np.random.choice(500, outlier_count, replace=False)
        for idx in outlier_indices:
            prices[idx] += np.random.choice([-10, 10])
        
        large_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST_LARGE'] * 500,
            'open': prices - 0.5,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 500)
        })
        
        # 测试性能
        import time
        
        fitter = RANSACTrendLineFitter(max_iterations=100)  # 限制迭代次数
        
        start_time = time.time()
        result = fitter.fit_trend_line(large_data, list(range(len(large_data))), 'close')
        end_time = time.time()
        
        # 应该在合理时间内完成
        self.assertLess(end_time - start_time, 5.0)  # 5秒内完成
        self.assertIsNotNone(result)
        
        stats = fitter.get_fit_statistics()
        self.assertGreater(stats['inliers_ratio'], 0.8)  # 应该正确识别大部分内点


if __name__ == '__main__':
    unittest.main()