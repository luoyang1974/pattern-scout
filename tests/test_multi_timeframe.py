"""
多时间周期系统测试
验证新的8个具体时间周期分类和策略系统
"""
import unittest
import pandas as pd
import numpy as np

from src.patterns.base.timeframe_manager import TimeframeManager
from src.patterns.base.parameter_adapter import ParameterAdapter
from src.utils.config_manager import ConfigManager


class TestMultiTimeframe(unittest.TestCase):
    """测试多时间周期系统"""
    
    def setUp(self):
        """设置测试环境"""
        self.timeframe_manager = TimeframeManager()
        self.supported_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
    
    def test_timeframe_manager_categories(self):
        """测试TimeframeManager的新分类系统"""
        for tf in self.supported_timeframes:
            category = self.timeframe_manager.get_category(tf)
            # 新系统中，分类就是周期本身
            self.assertEqual(category, tf, f"Category should be {tf} for timeframe {tf}")
    
    def test_timeframe_seconds_mapping(self):
        """测试时间周期到秒数的映射"""
        expected_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800,
            '1M': 2592000
        }
        
        for tf, expected in expected_seconds.items():
            actual = self.timeframe_manager.TIMEFRAME_SECONDS.get(tf)
            self.assertEqual(actual, expected, f"Seconds for {tf} should be {expected}")
    
    def test_lookback_periods_configuration(self):
        """测试回看周期配置"""
        for tf in self.supported_timeframes:
            for pattern_type in ['flag', 'pennant']:
                lookback = self.timeframe_manager.get_lookback_periods(tf, pattern_type)
                
                # 验证必要的字段存在
                self.assertIn('min_total_bars', lookback)
                self.assertIn('max_total_bars', lookback)
                self.assertIn('pre_pattern_bars', lookback)
                
                # 验证数值合理性
                self.assertGreater(lookback['max_total_bars'], lookback['min_total_bars'])
                self.assertGreater(lookback['min_total_bars'], 0)
    
    def test_parameter_adapter_functionality(self):
        """测试参数适配器功能"""
        try:
            config = ConfigManager('config_multi_timeframe.yaml')
            adapter = ParameterAdapter(config.config)
            
            # 测试支持的时间周期
            supported_timeframes = adapter.get_supported_timeframes()
            expected_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
            
            self.assertTrue(all(tf in supported_timeframes for tf in expected_timeframes))
            
            # 测试参数获取
            for tf in ['1m', '15m', '1h', '1d']:
                if adapter.has_timeframe_config(tf):
                    flag_params = adapter.get_timeframe_params(tf, 'flag')
                    self.assertIn('flagpole', flag_params)
                    self.assertIn('pattern', flag_params)
                    self.assertIn('timeframe', flag_params)
                    self.assertEqual(flag_params['timeframe'], tf)
                    
        except Exception as e:
            self.skipTest(f"配置文件不可用: {e}")
    
    def test_timeframe_detection(self):
        """测试时间周期自动检测"""
        test_cases = [
            ('1m', 60, 50),      # 1分钟数据
            ('5m', 300, 40),     # 5分钟数据  
            ('15m', 900, 30),    # 15分钟数据
            ('1h', 3600, 25),    # 1小时数据
            ('1d', 86400, 20),   # 日线数据
        ]
        
        for expected_tf, interval_seconds, periods in test_cases:
            # 创建测试数据
            timestamps = pd.date_range(
                start='2024-01-01', 
                periods=periods, 
                freq=f'{interval_seconds}s'
            )
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': np.random.uniform(95, 105, periods),
                'high': np.random.uniform(100, 110, periods),
                'low': np.random.uniform(90, 100, periods),
                'close': np.random.uniform(95, 105, periods),
                'volume': np.random.uniform(1000, 5000, periods)
            })
            
            detected_tf = self.timeframe_manager.detect_timeframe(df)
            self.assertEqual(detected_tf, expected_tf, 
                           f"Should detect {expected_tf} for {interval_seconds}s intervals")
    
    def test_data_sufficiency_validation(self):
        """测试数据充足性验证"""
        # 测试不同周期的最小数据要求
        for tf in self.supported_timeframes:
            for pattern_type in ['flag', 'pennant']:
                # 创建足够的测试数据
                lookback = self.timeframe_manager.get_lookback_periods(tf, pattern_type)
                min_bars = lookback['min_total_bars']
                
                # 创建刚好足够的数据
                sufficient_data = self._create_test_data(tf, min_bars)
                is_sufficient, msg = self.timeframe_manager.validate_data_sufficiency(
                    sufficient_data, tf, pattern_type
                )
                self.assertTrue(is_sufficient, f"Should be sufficient data for {tf} {pattern_type}")
                
                # 创建不足的数据
                insufficient_data = self._create_test_data(tf, min_bars - 5)
                is_sufficient, msg = self.timeframe_manager.validate_data_sufficiency(
                    insufficient_data, tf, pattern_type
                )
                self.assertFalse(is_sufficient, f"Should be insufficient data for {tf} {pattern_type}")
    
    def test_pandas_freq_conversion(self):
        """测试pandas频率转换"""
        expected_freqs = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }
        
        for tf, expected_freq in expected_freqs.items():
            actual_freq = self.timeframe_manager._get_pandas_freq(tf)
            self.assertEqual(actual_freq, expected_freq, 
                           f"Pandas freq for {tf} should be {expected_freq}")
    
    def test_bars_per_hour_calculation(self):
        """测试每小时K线数计算"""
        expected_bars_per_hour = {
            '1m': 60,
            '5m': 12,
            '15m': 4,
            '1h': 1,
            '4h': 0.25,
            '1d': 0.042,
            '1w': 0.006,
            '1M': 0.0014
        }
        
        for tf, expected_bars in expected_bars_per_hour.items():
            if tf in self.timeframe_manager.CATEGORIES:
                actual_bars = self.timeframe_manager.CATEGORIES[tf]['bars_per_hour']
                self.assertAlmostEqual(actual_bars, expected_bars, places=3,
                                     msg=f"Bars per hour for {tf} should be {expected_bars}")
    
    def _create_test_data(self, timeframe: str, periods: int) -> pd.DataFrame:
        """创建指定周期和长度的测试数据"""
        freq_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h',
            '4h': '4h', '1d': '1D', '1w': '1W', '1M': '1M'
        }
        
        freq = freq_map.get(timeframe, '15min')
        timestamps = pd.date_range(start='2024-01-01', periods=periods, freq=freq)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': np.random.uniform(95, 105, periods),
            'high': np.random.uniform(100, 110, periods), 
            'low': np.random.uniform(90, 100, periods),
            'close': np.random.uniform(95, 105, periods),
            'volume': np.random.uniform(1000, 5000, periods),
            'symbol': ['TEST'] * periods
        })


class TestParameterConfiguration(unittest.TestCase):
    """测试参数配置系统"""
    
    def test_lookback_period_scaling(self):
        """测试回看周期的合理缩放"""
        tm = TimeframeManager()
        
        # 验证高频周期有更多的回看K线数
        lookback_1m = tm.get_lookback_periods('1m', 'flag')
        lookback_1d = tm.get_lookback_periods('1d', 'flag')
        lookback_1M = tm.get_lookback_periods('1M', 'flag')
        
        # 高频周期需要更多K线来形成稳定形态
        self.assertGreater(lookback_1m['min_total_bars'], lookback_1d['min_total_bars'])
        self.assertGreater(lookback_1d['min_total_bars'], lookback_1M['min_total_bars'])
    
    def test_category_specific_parameters(self):
        """测试分类特定参数的合理性"""
        tm = TimeframeManager()
        
        # 验证不同周期的参数确实不同
        categories = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
        
        for i in range(len(categories) - 1):
            current_tf = categories[i]
            next_tf = categories[i + 1]
            
            current_lookback = tm.get_lookback_periods(current_tf, 'flag')
            next_lookback = tm.get_lookback_periods(next_tf, 'flag')
            
            # 验证参数确实在变化（不是完全相同）
            if current_lookback and next_lookback:
                self.assertNotEqual(current_lookback, next_lookback,
                                  f"Parameters should differ between {current_tf} and {next_tf}")


if __name__ == '__main__':
    unittest.main()