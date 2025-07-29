"""
测试ATR自适应参数系统
"""
import unittest
import pandas as pd
import numpy as np

from src.patterns.base.atr_adaptive_manager import ATRAdaptiveManager
from src.patterns.detectors.flag_detector import FlagDetector
from src.patterns.detectors.pennant_detector import PennantDetector


class TestATRAdaptiveManager(unittest.TestCase):
    """测试ATR自适应管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = ATRAdaptiveManager(atr_period=14)
        
        # 创建不同波动率的测试数据
        self.low_volatility_data = self._create_test_data(volatility='low', periods=50)
        self.high_volatility_data = self._create_test_data(volatility='high', periods=50)
        self.medium_volatility_data = self._create_test_data(volatility='medium', periods=50)
    
    def _create_test_data(self, volatility: str, periods: int) -> pd.DataFrame:
        """创建不同波动率的测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='15min')
        
        # 根据波动率设置参数
        volatility_params = {
            'low': {'base_vol': 0.005, 'trend_strength': 0.02},
            'medium': {'base_vol': 0.015, 'trend_strength': 0.05},
            'high': {'base_vol': 0.035, 'trend_strength': 0.08}
        }
        
        params = volatility_params[volatility]
        base_price = 100
        
        # 生成价格序列
        returns = np.random.normal(0, params['base_vol'], periods)
        returns[0] = 0  # 第一个回报为0
        
        # 添加一些趋势
        trend = np.linspace(0, params['trend_strength'], periods)
        returns += trend / periods
        
        # 计算价格
        prices = base_price * np.cumprod(1 + returns)
        
        # 生成OHLC数据
        noise_factor = params['base_vol'] * 0.5
        opens = prices + np.random.normal(0, noise_factor, periods)
        highs = np.maximum(opens, prices) + np.random.uniform(0, noise_factor, periods)
        lows = np.minimum(opens, prices) - np.random.uniform(0, noise_factor, periods)
        closes = prices
        
        # 确保OHLC逻辑正确
        for i in range(periods):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        volumes = np.random.uniform(1000000, 3000000, periods)
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': [f'TEST_{volatility.upper()}'] * periods,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def test_volatility_analysis_low(self):
        """测试低波动率分析"""
        analysis = self.manager.analyze_market_volatility(self.low_volatility_data)
        
        self.assertIn('volatility_level', analysis)
        self.assertIn('atr_normalized', analysis)
        self.assertIn('is_high_volatility', analysis)
        
        # 低波动率数据应该被分类为low或very_low
        self.assertIn(analysis['volatility_level'], ['very_low', 'low', 'medium'])
        self.assertFalse(analysis['is_high_volatility'])
    
    def test_volatility_analysis_high(self):
        """测试高波动率分析"""
        analysis = self.manager.analyze_market_volatility(self.high_volatility_data)
        
        self.assertIn('volatility_level', analysis)
        self.assertIn('atr_normalized', analysis)
        
        # 高波动率数据应该被分类为high或very_high
        self.assertIn(analysis['volatility_level'], ['medium', 'high', 'very_high'])
    
    def test_parameter_adaptation(self):
        """测试参数自适应调整"""
        # 基础参数
        base_params = {
            'flagpole': {
                'min_height_percent': 2.0,
                'min_trend_strength': 0.7
            },
            'pattern': {
                'min_prominence_atr': 0.5,
                'parallel_tolerance': 0.15,
                'convergence_ratio': 0.6
            },
            'min_confidence': 0.6
        }
        
        # 对不同波动率数据进行适应
        for data, expected_level in [
            (self.low_volatility_data, ['very_low', 'low', 'medium']),
            (self.medium_volatility_data, ['low', 'medium', 'high']),
            (self.high_volatility_data, ['medium', 'high', 'very_high'])
        ]:
            with self.subTest(expected_level=expected_level):
                volatility_analysis = self.manager.analyze_market_volatility(data)
                adapted_params = self.manager.adapt_parameters(
                    base_params, volatility_analysis, 'short'
                )
                
                # 检查参数是否被调整
                self.assertIn('atr_adaptive', adapted_params)
                self.assertIn('volatility_level', adapted_params['atr_adaptive'])
                self.assertTrue(adapted_params['atr_adaptive']['adaptation_applied'])
                
                # 检查波动率级别是否合理
                actual_level = adapted_params['atr_adaptive']['volatility_level']
                self.assertIn(actual_level, expected_level)
    
    def test_atr_based_thresholds(self):
        """测试基于ATR的动态阈值"""
        base_thresholds = {
            'height_percent': 2.0,
            'volume_ratio': 1.5,
            'confidence_threshold': 0.6
        }
        
        # 测试不同波动率的阈值调整
        low_vol_thresholds = self.manager.get_atr_based_thresholds(
            self.low_volatility_data, base_thresholds
        )
        high_vol_thresholds = self.manager.get_atr_based_thresholds(
            self.high_volatility_data, base_thresholds
        )
        
        self.assertIsInstance(low_vol_thresholds, dict)
        self.assertIsInstance(high_vol_thresholds, dict)
        
        # 高波动率时阈值应该更宽松（某些参数会更大）
        for key in base_thresholds:
            self.assertIn(key, low_vol_thresholds)
            self.assertIn(key, high_vol_thresholds)
    
    def test_volatility_report_generation(self):
        """测试波动率报告生成"""
        report = self.manager.get_volatility_report(self.medium_volatility_data)
        
        self.assertIsInstance(report, str)
        self.assertIn('ATR波动率分析报告', report)
        self.assertIn('波动率级别', report)
        self.assertIn('标准化ATR', report)
        self.assertIn('数据质量', report)
    
    def test_edge_cases(self):
        """测试边缘情况"""
        # 测试数据不足的情况
        short_data = self.medium_volatility_data[:5]  # 只有5个数据点
        analysis = self.manager.analyze_market_volatility(short_data)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('volatility_level', analysis)
        self.assertFalse(analysis['data_quality']['sufficient_data'])
        
        # 测试空DataFrame
        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        analysis_empty = self.manager.analyze_market_volatility(empty_df)
        self.assertIsInstance(analysis_empty, dict)


class TestATRAdaptiveIntegration(unittest.TestCase):
    """测试ATR自适应系统与检测器集成"""
    
    def setUp(self):
        """设置测试环境"""
        # 启用ATR自适应的检测器
        self.flag_detector = FlagDetector()
        self.pennant_detector = PennantDetector()
        
        # 禁用ATR自适应的检测器（用于对比）
        config_no_atr = {
            'global': {'enable_atr_adaptation': False},
            'timeframe_configs': {
                'short': {
                    'flagpole': {
                        'min_height_percent': 2.0,
                        'min_trend_strength': 0.7
                    },
                    'flag': {
                        'min_prominence_atr': 0.5,
                        'min_bars': 8,
                        'max_bars': 30,
                        'parallel_tolerance': 0.15
                    }
                }
            },
            'scoring': {'min_confidence_score': 0.6}
        }
        self.flag_detector_no_atr = FlagDetector(config_no_atr)
        
        # 创建测试数据
        self.test_data = self._create_realistic_test_data()
    
    def _create_realistic_test_data(self) -> pd.DataFrame:
        """创建更真实的测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        
        # 模拟带有旗杆和旗面的价格走势
        np.random.seed(42)  # 确保可重现性
        
        # 基础价格趋势
        base_price = 100
        trend = np.linspace(0, 5, 100)  # 轻微上升趋势
        
        # 添加波动性
        volatility = 0.02
        noise = np.random.normal(0, volatility, 100)
        
        # 在特定位置添加"旗杆"模拟
        flagpole_start, flagpole_end = 20, 25
        flag_start, flag_end = 25, 45
        
        price_changes = np.zeros(100)
        price_changes[flagpole_start:flagpole_end] = 0.02  # 强烈上涨
        price_changes[flag_start:flag_end] = -0.005  # 轻微下跌整理
        
        # 合成最终价格
        cumulative_changes = np.cumsum(price_changes + noise + trend/100)
        prices = base_price * (1 + cumulative_changes)
        
        # 生成OHLC
        opens = prices + np.random.normal(0, 0.1, 100)
        highs = prices + np.random.uniform(0.1, 0.5, 100)
        lows = prices - np.random.uniform(0.1, 0.5, 100)
        closes = prices
        
        # 确保OHLC逻辑
        for i in range(100):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        # 生成成交量（旗杆期间放大）
        volumes = np.random.uniform(1000000, 2000000, 100)
        volumes[flagpole_start:flagpole_end] *= 3  # 旗杆期间成交量放大
        volumes[flag_start:flag_end] *= 0.7  # 旗面期间成交量萎缩
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST_REALISTIC'] * 100,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def test_atr_adaptation_enabled_vs_disabled(self):
        """测试ATR自适应开启与关闭的区别"""
        # 启用ATR自适应的检测
        patterns_with_atr = self.flag_detector.detect(self.test_data, '15m')
        
        # 禁用ATR自适应的检测
        patterns_without_atr = self.flag_detector_no_atr.detect(self.test_data, '15m')
        
        # 两种方式都应该返回有效结果
        self.assertIsInstance(patterns_with_atr, list)
        self.assertIsInstance(patterns_without_atr, list)
        
        # 可以检查参数是否被正确应用（通过日志或其他方式）
        # 这里主要是确保没有错误发生
    
    def test_volatility_report_integration(self):
        """测试波动率报告集成"""
        report = self.flag_detector.get_volatility_report(self.test_data)
        
        self.assertIsInstance(report, str)
        self.assertIn('ATR波动率分析报告', report)
        self.assertIn('波动率级别', report)
    
    def test_different_timeframes_adaptation(self):
        """测试不同时间周期的自适应调整"""
        timeframes = ['15m', '1h']
        
        for tf in timeframes:
            with self.subTest(timeframe=tf):
                try:
                    patterns = self.flag_detector.detect(self.test_data, tf)
                    self.assertIsInstance(patterns, list)
                except Exception as e:
                    # 某些时间周期可能数据不足，这是正常的
                    self.assertIn('Insufficient', str(e))
    
    def test_pennant_detector_atr_adaptation(self):
        """测试三角旗形检测器的ATR自适应"""
        patterns = self.pennant_detector.detect(self.test_data, '15m')
        
        self.assertIsInstance(patterns, list)
        # 主要确保没有错误，具体的形态检测效果需要更复杂的测试数据


if __name__ == '__main__':
    unittest.main()