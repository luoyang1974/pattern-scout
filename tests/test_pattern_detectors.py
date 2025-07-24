"""
更新后的形态检测器测试用例
支持多周期架构和新的检测器接口
"""
import unittest
import pandas as pd
import numpy as np

from src.patterns.detectors import FlagDetector, PennantDetector, PatternScanner
from src.data.models.base_models import PatternType


class TestFlagDetector(unittest.TestCase):
    """测试旗形检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = FlagDetector()
        
        # 创建 15分钟数据（短周期测试）
        self.test_data_15m = self._create_flag_test_data('15m', 50)
        
        # 创建 1小时数据（短周期测试）
        self.test_data_1h = self._create_flag_test_data('1h', 30)
        
        # 创建 5分钟数据（超短周期测试）
        self.test_data_5m = self._create_flag_test_data('5m', 80)
    
    def _create_flag_test_data(self, timeframe: str, periods: int) -> pd.DataFrame:
        """创建旗形测试数据"""
        # 根据时间周期确定时间间隔
        freq_map = {'5m': '5min', '15m': '15min', '1h': '1h', '1d': '1D'}
        freq = freq_map.get(timeframe, '15min')
        
        dates = pd.date_range(start='2024-01-01', periods=periods, freq=freq)
        
        # 模拟旗杆 + 旗面形态
        flagpole_len = max(4, periods // 8)  # 动态旗杆长度
        flag_len = max(8, periods // 4)      # 动态旗面长度
        follow_len = periods - flagpole_len - flag_len
        
        # 旗杆：强烈上涨
        flagpole_prices = np.linspace(100, 110, flagpole_len)  # 10%上涨
        flagpole_volumes = np.random.uniform(2000000, 4000000, flagpole_len)
        
        # 旗面：向下倾斜的平行通道
        flag_high_start, flag_high_end = 110, 108
        flag_low_start, flag_low_end = 108, 106
        
        flag_highs = np.linspace(flag_high_start, flag_high_end, flag_len)
        flag_lows = np.linspace(flag_low_start, flag_low_end, flag_len)
        flag_prices = (flag_highs + flag_lows) / 2 + np.random.normal(0, 0.2, flag_len)
        flag_volumes = np.random.uniform(500000, 1200000, flag_len)
        
        # 后续走势
        follow_prices = 107 + np.random.normal(0, 0.8, follow_len)
        follow_volumes = np.random.uniform(1000000, 2000000, follow_len)
        
        all_prices = np.concatenate([flagpole_prices, flag_prices, follow_prices])
        all_volumes = np.concatenate([flagpole_volumes, flag_volumes, follow_volumes])
        
        # 生成OHLC数据
        opens = all_prices + np.random.uniform(-0.3, 0.3, periods)
        highs = all_prices + np.random.uniform(0.1, 0.6, periods)
        lows = all_prices - np.random.uniform(0.1, 0.6, periods)
        closes = all_prices
        
        # 确保OHLC逻辑正确
        for i in range(periods):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST'] * periods,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': all_volumes
        })
    
    def test_detect_flags_15m(self):
        """测试15分钟数据的旗形检测"""
        patterns = self.detector.detect(self.test_data_15m, '15m')
        
        self.assertIsInstance(patterns, list)
        
        # 验证形态属性（如果检测到）
        for pattern in patterns:
            self.assertEqual(pattern.symbol, 'TEST')
            self.assertEqual(pattern.pattern_type, PatternType.FLAG)
            self.assertIsNotNone(pattern.flagpole)
            self.assertGreater(pattern.confidence_score, 0)
    
    def test_detect_flags_auto_timeframe(self):
        """测试自动周期检测"""
        # 不指定时间周期，让检测器自动判断
        patterns = self.detector.detect(self.test_data_15m)
        
        self.assertIsInstance(patterns, list)
    
    def test_detect_flags_insufficient_data(self):
        """测试数据不足的情况"""
        insufficient_data = self.test_data_15m[:20]  # 只有20个数据点
        patterns = self.detector.detect(insufficient_data)
        
        # 数据不足应该返回空列表
        self.assertEqual(len(patterns), 0)
    
    def test_multi_timeframe_detection(self):
        """测试多周期检测"""
        timeframes = ['15m', '1h']
        results = self.detector.detect_multi_timeframe(self.test_data_15m, timeframes)
        
        self.assertIsInstance(results, dict)
        for tf in timeframes:
            self.assertIn(tf, results)
            self.assertIsInstance(results[tf], list)
    
    def test_custom_config(self):
        """测试自定义配置"""
        custom_config = {
            'timeframe_configs': {
                'short': {
                    'flagpole': {
                        'min_height_percent': 2.0,  # 更严格要求
                        'max_height_percent': 15.0
                    },
                    'scoring': {
                        'min_confidence_score': 0.7
                    }
                }
            }
        }
        
        custom_detector = FlagDetector(custom_config)
        patterns = custom_detector.detect(self.test_data_15m, '15m')
        
        self.assertIsInstance(patterns, list)


class TestPennantDetector(unittest.TestCase):
    """测试三角旗形检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = PennantDetector()
        
        # 创建测试数据
        self.test_data = self._create_pennant_test_data('15m', 40)
    
    def _create_pennant_test_data(self, timeframe: str, periods: int) -> pd.DataFrame:
        """创建三角旗形测试数据"""
        freq_map = {'5m': '5min', '15m': '15min', '1h': '1h', '1d': '1D'}
        freq = freq_map.get(timeframe, '15min')
        
        dates = pd.date_range(start='2024-01-01', periods=periods, freq=freq)
        
        flagpole_len = max(4, periods // 10)
        pennant_len = max(8, periods // 3)
        follow_len = periods - flagpole_len - pennant_len
        
        # 旗杆：快速上涨
        flagpole_prices = np.linspace(100, 108, flagpole_len)  # 8%上涨
        flagpole_volumes = np.random.uniform(2000000, 4000000, flagpole_len)
        
        # 三角旗面：收敛的三角形
        pennant_high_start, pennant_high_end = 108, 106.5
        pennant_low_start, pennant_low_end = 106, 106.8
        
        pennant_highs = np.linspace(pennant_high_start, pennant_high_end, pennant_len)
        pennant_lows = np.linspace(pennant_low_start, pennant_low_end, pennant_len)
        pennant_prices = (pennant_highs + pennant_lows) / 2 + np.random.normal(0, 0.1, pennant_len)
        pennant_volumes = np.random.uniform(400000, 800000, pennant_len)
        
        # 后续走势
        follow_prices = 107 + np.random.normal(0, 0.5, follow_len)
        follow_volumes = np.random.uniform(800000, 1500000, follow_len)
        
        all_prices = np.concatenate([flagpole_prices, pennant_prices, follow_prices])
        all_volumes = np.concatenate([flagpole_volumes, pennant_volumes, follow_volumes])
        
        # 生成OHLC数据
        opens = all_prices + np.random.uniform(-0.2, 0.2, periods)
        
        # 为三角旗形生成特殊的高低价
        highs = np.concatenate([
            flagpole_prices + np.random.uniform(0.1, 0.5, flagpole_len),
            pennant_highs + np.random.uniform(-0.1, 0.1, pennant_len),
            follow_prices + np.random.uniform(0.1, 0.4, follow_len)
        ])
        
        lows = np.concatenate([
            flagpole_prices - np.random.uniform(0.1, 0.5, flagpole_len),
            pennant_lows - np.random.uniform(-0.1, 0.1, pennant_len),
            follow_prices - np.random.uniform(0.1, 0.4, follow_len)
        ])
        
        closes = all_prices
        
        # 确保OHLC逻辑正确
        for i in range(periods):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST'] * periods,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': all_volumes
        })
    
    def test_detect_pennants_basic(self):
        """测试基本三角旗形检测"""
        patterns = self.detector.detect(self.test_data, '15m')
        
        self.assertIsInstance(patterns, list)
        
        # 验证形态属性（如果检测到）
        for pattern in patterns:
            self.assertEqual(pattern.symbol, 'TEST')
            self.assertEqual(pattern.pattern_type, PatternType.PENNANT)
            self.assertIsNotNone(pattern.flagpole)
            self.assertGreater(pattern.confidence_score, 0)
    
    def test_pennant_convergence_detection(self):
        """测试收敛检测"""
        patterns = self.detector.detect(self.test_data, '15m')
        
        # 如果检测到形态，验证收敛特性
        for pattern in patterns:
            # 应该有两条边界线
            self.assertEqual(len(pattern.pattern_boundaries), 2)
            
            # 验证收敛性（结束时价格范围应该小于开始时）
            upper_line, lower_line = pattern.pattern_boundaries
            start_range = abs(upper_line.start_price - lower_line.start_price)
            end_range = abs(upper_line.end_price - lower_line.end_price)
            
            if start_range > 0:
                convergence_ratio = 1 - (end_range / start_range)
                self.assertGreater(convergence_ratio, 0.3)  # 至少30%收敛
    
    def test_detect_pennants_insufficient_data(self):
        """测试数据不足的情况"""
        insufficient_data = self.test_data[:15]
        patterns = self.detector.detect(insufficient_data)
        
        self.assertEqual(len(patterns), 0)


class TestPatternScanner(unittest.TestCase):
    """测试统一的形态扫描器"""
    
    def setUp(self):
        """设置测试环境"""
        self.scanner = PatternScanner()
        
        # 创建包含多种形态的测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        
        # 简单的价格数据
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        
        opens = base_prices + np.random.uniform(-0.5, 0.5, 100)
        highs = base_prices + np.random.uniform(0.2, 1.0, 100)
        lows = base_prices - np.random.uniform(0.2, 1.0, 100)
        closes = base_prices
        volumes = np.random.uniform(1000000, 3000000, 100)
        
        # 确保OHLC逻辑正确
        for i in range(100):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['SCANNER_TEST'] * 100,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def test_scan_all_patterns(self):
        """测试扫描所有形态"""
        results = self.scanner.scan(self.test_data, timeframe='15m')
        
        self.assertIsInstance(results, dict)
        
        # 应该包含所有支持的形态类型
        expected_patterns = [PatternType.FLAG, PatternType.PENNANT]
        for pattern_type in expected_patterns:
            self.assertIn(pattern_type, results)
            self.assertIsInstance(results[pattern_type], list)
    
    def test_scan_specific_pattern(self):
        """测试扫描特定形态"""
        flag_patterns = self.scanner.scan_single_pattern(
            self.test_data, PatternType.FLAG, '15m'
        )
        
        self.assertIsInstance(flag_patterns, list)
        
        # 验证所有检测到的形态都是正确类型
        for pattern in flag_patterns:
            self.assertEqual(pattern.pattern_type, PatternType.FLAG)
    
    def test_multi_timeframe_scan(self):
        """测试多周期扫描"""
        timeframes = ['15m', '1h']
        results = self.scanner.scan_multi_timeframe(
            self.test_data, timeframes, [PatternType.FLAG]
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn(PatternType.FLAG, results)
        
        flag_results = results[PatternType.FLAG]
        for tf in timeframes:
            self.assertIn(tf, flag_results)
            self.assertIsInstance(flag_results[tf], list)
    
    def test_get_detector(self):
        """测试获取检测器实例"""
        flag_detector = self.scanner.get_detector(PatternType.FLAG)
        pennant_detector = self.scanner.get_detector(PatternType.PENNANT)
        
        self.assertIsInstance(flag_detector, FlagDetector)
        self.assertIsInstance(pennant_detector, PennantDetector)
        
        # 测试未知形态类型
        unknown_detector = self.scanner.get_detector('unknown_pattern')
        self.assertIsNone(unknown_detector)


class TestTimeframeAdaptation(unittest.TestCase):
    """测试周期自适应功能"""
    
    def test_ultra_short_timeframe(self):
        """测试超短周期处理"""
        # 创建5分钟数据
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
        prices = 100 + np.cumsum(np.random.normal(0, 0.05, 200))
        
        test_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['ULTRA_SHORT'] * 200,
            'open': prices,
            'high': prices + np.random.uniform(0, 0.2, 200),
            'low': prices - np.random.uniform(0, 0.2, 200),
            'close': prices,
            'volume': np.random.uniform(500000, 2000000, 200)
        })
        
        detector = FlagDetector()
        patterns = detector.detect(test_data, '5m')
        
        self.assertIsInstance(patterns, list)
    
    def test_medium_long_timeframe(self):
        """测试中长周期处理"""
        # 创建日线数据
        dates = pd.date_range(start='2024-01-01', periods=60, freq='1D')
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, 60))
        
        test_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['DAILY'] * 60,
            'open': prices,
            'high': prices + np.random.uniform(0.5, 2, 60),
            'low': prices - np.random.uniform(0.5, 2, 60),
            'close': prices,
            'volume': np.random.uniform(5000000, 20000000, 60)
        })
        
        detector = PennantDetector()
        patterns = detector.detect(test_data, '1d')
        
        self.assertIsInstance(patterns, list)


class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性"""
    
    def test_pennant_pattern_type(self):
        """测试 Pennant 形态类型"""
        # 确保 PENNANT 形态类型正确
        self.assertEqual(PatternType.PENNANT, "pennant")
        
        # 确保 PennantDetector 正常工作
        from src.patterns.detectors import PennantDetector
        detector = PennantDetector()
        self.assertIsInstance(detector, PennantDetector)
        self.assertEqual(detector.get_pattern_type(), PatternType.PENNANT)


if __name__ == '__main__':
    # 设置更详细的测试输出
    unittest.main(verbosity=2)