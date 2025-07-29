"""
测试算法优化功能
测试摆动点检测、增强成交量分析和改进的形态验证
"""
import unittest
import pandas as pd
import numpy as np

from src.patterns.base.pattern_components import PatternComponents
from src.patterns.detectors.flag_detector import FlagDetector
from src.patterns.detectors.pennant_detector import PennantDetector
from src.data.models.base_models import TrendLine


class TestSwingPointDetection(unittest.TestCase):
    """测试智能摆动点检测算法"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建带有明显摆动点的测试数据
        dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
        
        # 创建正弦波形态的价格数据，模拟摆动
        base_price = 100
        amplitude = 5
        noise = np.random.normal(0, 0.1, 50)
        
        prices = base_price + amplitude * np.sin(np.linspace(0, 4 * np.pi, 50)) + noise
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.uniform(-0.2, 0.2, 50),
            'high': prices + np.random.uniform(0.1, 0.5, 50),
            'low': prices - np.random.uniform(0.1, 0.5, 50),
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 50),
            'symbol': 'TEST'
        })
    
    def test_basic_swing_detection(self):
        """测试基础摆动点检测"""
        swing_highs, swing_lows = PatternComponents.find_swing_points(
            self.test_df, window=3
        )
        
        # 应该检测到多个摆动点
        self.assertGreater(len(swing_highs), 0)
        self.assertGreater(len(swing_lows), 0)
        
        # 摆动高点应该在高价位
        for idx in swing_highs:
            window_start = max(0, idx - 3)
            window_end = min(len(self.test_df), idx + 4)
            window_highs = self.test_df['high'].iloc[window_start:window_end]
            self.assertEqual(self.test_df['high'].iloc[idx], window_highs.max())
    
    def test_atr_filtering(self):
        """测试基于ATR的突出度过滤"""
        # 不使用ATR过滤
        swing_highs_no_filter, swing_lows_no_filter = PatternComponents.find_swing_points(
            self.test_df, window=3, min_prominence_atr_multiple=None
        )
        
        # 使用ATR过滤
        swing_highs_filtered, swing_lows_filtered = PatternComponents.find_swing_points(
            self.test_df, window=3, min_prominence_atr_multiple=0.5
        )
        
        # 过滤后的点应该更少
        self.assertLessEqual(len(swing_highs_filtered), len(swing_highs_no_filter))
        self.assertLessEqual(len(swing_lows_filtered), len(swing_lows_no_filter))
    
    def test_time_distance_filtering(self):
        """测试时间间隔过滤"""
        # 创建密集的摆动点
        indices = [1, 2, 3, 10, 11, 20, 21, 22, 30]
        
        # 使用简单过滤
        filtered = PatternComponents._filter_by_time_distance(indices, min_distance=3)
        
        # 检查过滤结果
        for i in range(1, len(filtered)):
            self.assertGreaterEqual(filtered[i] - filtered[i-1], 3)


class TestEnhancedVolumeAnalysis(unittest.TestCase):
    """测试增强的成交量分析功能"""
    
    def setUp(self):
        """设置测试数据"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='15min')
        
        # 创建递减的成交量数据
        base_volume = 2000000
        volume_trend = np.linspace(base_volume, base_volume * 0.5, 30)
        volume_noise = np.random.normal(0, 50000, 30)
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(99, 101, 30),
            'high': np.random.uniform(100, 102, 30),
            'low': np.random.uniform(98, 100, 30),
            'close': np.random.uniform(99, 101, 30),
            'volume': volume_trend + volume_noise,
            'symbol': 'TEST'
        })
    
    def test_volume_pattern_basic(self):
        """测试基础成交量模式分析"""
        result = PatternComponents.analyze_volume_pattern(
            self.test_df['volume'], pattern_type='decreasing'
        )
        
        self.assertEqual(result['trend'], 'decreasing')
        self.assertGreater(result['consistency'], 0.8)  # 应该有高一致性
        self.assertGreater(result['score'], 0.8)  # 匹配期望模式
    
    def test_volume_pattern_enhanced(self):
        """测试增强的成交量分析"""
        result = PatternComponents.analyze_volume_pattern_enhanced(
            self.test_df, 
            pattern_start=5,
            pattern_end=25
        )
        
        # 检查所有返回字段
        self.assertIn('trend', result)
        self.assertIn('trend_slope', result)
        self.assertIn('trend_r2', result)
        self.assertIn('contraction_ratio', result)
        self.assertIn('volatility', result)
        self.assertIn('spike_count', result)
        self.assertIn('liquidity_healthy', result)
        self.assertIn('health_score', result)
        
        # 健康度应该较高（因为是理想的递减模式）
        self.assertGreater(result['health_score'], 0.6)
    
    def test_volume_spike_detection(self):
        """测试成交量异常峰值检测"""
        # 添加一些峰值
        self.test_df.loc[10, 'volume'] = 5000000  # 异常高
        self.test_df.loc[20, 'volume'] = 4500000  # 异常高
        
        spikes = PatternComponents._detect_volume_spikes(self.test_df, threshold=2.0)
        
        # 应该检测到峰值
        self.assertIn(10, spikes)
        self.assertIn(20, spikes)


class TestImprovedFlagValidation(unittest.TestCase):
    """测试改进的旗形验证逻辑"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = FlagDetector()
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=37, freq='15min')
        
        # 创建旗杆
        flagpole_prices = np.linspace(100, 110, 10)
        
        # 创建旗面（向下倾斜的平行通道）
        flag_length = 20
        upper_line = np.linspace(110, 108, flag_length)
        lower_line = np.linspace(108, 106, flag_length)
        
        # 组合价格
        prices = np.concatenate([
            flagpole_prices,
            # 旗面价格在上下边界间震荡
            upper_line - np.sin(np.linspace(0, 2*np.pi, flag_length)) * 0.5,
            np.array([107, 108, 109, 108, 107, 108, 109])  # 结束价格
        ])
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.uniform(-0.1, 0.1, len(prices)),
            'high': prices + np.random.uniform(0.1, 0.3, len(prices)),
            'low': prices - np.random.uniform(0.1, 0.3, len(prices)),
            'close': prices,
            'volume': np.concatenate([
                np.random.uniform(2000000, 3000000, 10),  # 旗杆高成交量
                np.random.uniform(1000000, 1500000, 20),  # 旗面低成交量
                np.random.uniform(1200000, 1800000, 7)    # 恢复
            ]),
            'symbol': 'TEST'
        })
        
        # 创建测试用的趋势线
        self.upper_line = TrendLine(
            start_time=dates[10],
            end_time=dates[30],
            start_price=110,
            end_price=108,
            slope=-0.1,
            r_squared=0.95
        )
        
        self.lower_line = TrendLine(
            start_time=dates[10],
            end_time=dates[30],
            start_price=108,
            end_price=106,
            slope=-0.1,
            r_squared=0.95
        )
    
    def test_divergence_check(self):
        """测试通道发散检查"""
        # 平行通道（不发散）
        is_valid = self.detector._verify_no_divergence(
            self.upper_line, self.lower_line, self.test_df.iloc[10:30]
        )
        self.assertTrue(is_valid)
        
        # 创建发散的通道
        diverging_upper = TrendLine(
            start_time=self.upper_line.start_time,
            end_time=self.upper_line.end_time,
            start_price=110,
            end_price=112,  # 向上发散
            slope=0.1,
            r_squared=0.95
        )
        
        is_valid = self.detector._verify_no_divergence(
            diverging_upper, self.lower_line, self.test_df.iloc[10:30]
        )
        self.assertFalse(is_valid)
    
    def test_breakout_preparation(self):
        """测试突破准备度评估"""
        flag_data = self.test_df.iloc[10:30]
        
        readiness = self.detector._verify_breakout_preparation(
            flag_data, self.upper_line, self.lower_line, {'pattern': {}}
        )
        
        # 应该返回0-1之间的分数
        self.assertGreaterEqual(readiness, 0)
        self.assertLessEqual(readiness, 1)


class TestImprovedPennantValidation(unittest.TestCase):
    """测试改进的三角旗形验证逻辑"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = PennantDetector()
        
        # 创建对称三角形数据
        dates = pd.date_range(start='2024-01-01', periods=30, freq='15min')
        
        # 收敛的上下边界
        upper_boundary = np.linspace(110, 105, 30)
        lower_boundary = np.linspace(100, 105, 30)
        
        # 价格在边界间震荡
        prices = []
        for i in range(30):
            if i % 4 < 2:
                prices.append(upper_boundary[i] - 0.2)
            else:
                prices.append(lower_boundary[i] + 0.2)
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': upper_boundary + 0.1,
            'low': lower_boundary - 0.1,
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 30),
            'symbol': 'TEST'
        })
    
    def test_triangle_classification(self):
        """测试三角形分类"""
        # 对称三角形
        symmetric_upper = TrendLine(
            start_time=self.test_df['timestamp'].iloc[0],
            end_time=self.test_df['timestamp'].iloc[-1],
            start_price=110,
            end_price=105,
            slope=-0.2,
            r_squared=0.9
        )
        
        symmetric_lower = TrendLine(
            start_time=self.test_df['timestamp'].iloc[0],
            end_time=self.test_df['timestamp'].iloc[-1],
            start_price=100,
            end_price=105,
            slope=0.2,
            r_squared=0.9
        )
        
        triangle_type = self.detector._classify_triangle(
            symmetric_upper, symmetric_lower, self.test_df
        )
        self.assertEqual(triangle_type, 'symmetric')
        
        # 上升三角形（上边界水平）
        ascending_upper = TrendLine(
            start_time=self.test_df['timestamp'].iloc[0],
            end_time=self.test_df['timestamp'].iloc[-1],
            start_price=110,
            end_price=110,
            slope=0,
            r_squared=1.0
        )
        
        triangle_type = self.detector._classify_triangle(
            ascending_upper, symmetric_lower, self.test_df
        )
        self.assertEqual(triangle_type, 'ascending')
    
    def test_asymmetric_quality_calculation(self):
        """测试非对称三角形质量计算"""
        # 创建上升三角形
        horizontal_line = TrendLine(
            start_time=self.test_df['timestamp'].iloc[0],
            end_time=self.test_df['timestamp'].iloc[-1],
            start_price=110,
            end_price=110,
            slope=0,
            r_squared=1.0
        )
        
        trending_line = TrendLine(
            start_time=self.test_df['timestamp'].iloc[0],
            end_time=self.test_df['timestamp'].iloc[-1],
            start_price=100,
            end_price=108,
            slope=0.3,
            r_squared=0.85
        )
        
        quality = self.detector._calculate_asymmetric_quality(
            horizontal_line, trending_line, self.test_df, 'ascending'
        )
        
        # 质量分数应该在合理范围内
        self.assertGreater(quality, 0.5)
        self.assertLessEqual(quality, 1.0)


if __name__ == '__main__':
    unittest.main()