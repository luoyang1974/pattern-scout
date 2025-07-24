import unittest
import pandas as pd
import numpy as np

from src.patterns.indicators.technical_indicators import TechnicalIndicators, TrendAnalyzer, PatternIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """测试技术指标计算"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建模拟价格数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # 确保可重现
        
        # 生成有趋势的价格数据
        base_price = 100
        trend = np.linspace(0, 20, 100)  # 上升趋势
        noise = np.random.normal(0, 2, 100)  # 噪声
        prices = base_price + trend + noise
        
        # 生成高低价
        highs = prices + np.random.uniform(0.5, 3, 100)
        lows = prices - np.random.uniform(0.5, 3, 100)
        volumes = np.random.uniform(1000000, 5000000, 100)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': prices - np.random.uniform(-1, 1, 100),
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        self.prices = pd.Series(prices, index=dates)
        self.highs = pd.Series(highs, index=dates)
        self.lows = pd.Series(lows, index=dates)
        self.volumes = pd.Series(volumes, index=dates)
    
    def test_moving_average(self):
        """测试移动平均线计算"""
        ma_10 = TechnicalIndicators.moving_average(self.prices, 10)
        
        # 检查结果长度
        self.assertEqual(len(ma_10), len(self.prices))
        
        # 检查前9个值为NaN
        self.assertTrue(ma_10[:9].isna().all())
        
        # 检查第10个值是否正确
        expected_ma10 = self.prices[:10].mean()
        self.assertAlmostEqual(ma_10.iloc[9], expected_ma10, places=5)
    
    def test_exponential_moving_average(self):
        """测试指数移动平均线计算"""
        ema_12 = TechnicalIndicators.exponential_moving_average(self.prices, 12)
        
        # 检查结果长度
        self.assertEqual(len(ema_12), len(self.prices))
        
        # EMA应该没有NaN值（除了第一个可能）
        self.assertFalse(ema_12[1:].isna().any())
    
    def test_rsi(self):
        """测试RSI计算"""
        rsi = TechnicalIndicators.rsi(self.prices, 14)
        
        # 检查结果长度
        self.assertEqual(len(rsi), len(self.prices))
        
        # RSI值应该在0-100之间
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_macd(self):
        """测试MACD计算"""
        macd_line, macd_signal, macd_histogram = TechnicalIndicators.macd(self.prices)
        
        # 检查结果长度
        self.assertEqual(len(macd_line), len(self.prices))
        self.assertEqual(len(macd_signal), len(self.prices))
        self.assertEqual(len(macd_histogram), len(self.prices))
        
        # 检查MACD柱状图是否等于线与信号线的差
        valid_indices = ~(macd_line.isna() | macd_signal.isna())
        diff = macd_line[valid_indices] - macd_signal[valid_indices]
        histogram = macd_histogram[valid_indices]
        np.testing.assert_array_almost_equal(diff, histogram, decimal=5)
    
    def test_bollinger_bands(self):
        """测试布林带计算"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(self.prices, 20, 2)
        
        # 检查结果长度
        self.assertEqual(len(upper), len(self.prices))
        self.assertEqual(len(middle), len(self.prices))
        self.assertEqual(len(lower), len(self.prices))
        
        # 检查上轨总是大于中轨，中轨总是大于下轨
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        self.assertTrue((upper[valid_indices] > middle[valid_indices]).all())
        self.assertTrue((middle[valid_indices] > lower[valid_indices]).all())
    
    def test_atr(self):
        """测试平均真实波幅计算"""
        atr = TechnicalIndicators.atr(self.highs, self.lows, self.prices, 14)
        
        # 检查结果长度
        self.assertEqual(len(atr), len(self.prices))
        
        # ATR值应该为正
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())


class TestTrendAnalyzer(unittest.TestCase):
    """测试趋势分析器"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
    
    def test_detect_trend_direction_uptrend(self):
        """测试上升趋势检测"""
        # 创建明显的上升趋势
        uptrend_prices = pd.Series(np.linspace(100, 120, 20) + np.random.normal(0, 0.5, 20))
        
        direction = TrendAnalyzer.detect_trend_direction(uptrend_prices)
        self.assertEqual(direction, "uptrend")
    
    def test_detect_trend_direction_downtrend(self):
        """测试下降趋势检测"""
        # 创建明显的下降趋势
        downtrend_prices = pd.Series(np.linspace(120, 100, 20) + np.random.normal(0, 0.5, 20))
        
        direction = TrendAnalyzer.detect_trend_direction(downtrend_prices)
        self.assertEqual(direction, "downtrend")
    
    def test_detect_trend_direction_sideways(self):
        """测试横盘趋势检测"""
        # 创建横盘震荡
        sideways_prices = pd.Series(100 + np.random.normal(0, 2, 20))
        
        direction = TrendAnalyzer.detect_trend_direction(sideways_prices)
        self.assertEqual(direction, "sideways")
    
    def test_calculate_trend_strength(self):
        """测试趋势强度计算"""
        # 强趋势
        strong_trend = pd.Series(np.linspace(100, 120, 20))
        strength = TrendAnalyzer.calculate_trend_strength(strong_trend)
        self.assertGreater(strength, 0.9)
        
        # 弱趋势/无趋势
        weak_trend = pd.Series(100 + np.random.normal(0, 5, 20))
        strength = TrendAnalyzer.calculate_trend_strength(weak_trend)
        self.assertLess(strength, 0.7)


class TestPatternIndicators(unittest.TestCase):
    """测试形态指标"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        
        # 创建OHLC数据
        n_points = 50
        close_prices = 100 + np.random.normal(0, 2, n_points)
        high_prices = close_prices + np.random.uniform(0.5, 2, n_points)
        low_prices = close_prices - np.random.uniform(0.5, 2, n_points)
        open_prices = close_prices + np.random.uniform(-1, 1, n_points)
        volumes = np.random.uniform(1000000, 3000000, n_points)
        
        self.open_prices = pd.Series(open_prices)
        self.high_prices = pd.Series(high_prices)
        self.low_prices = pd.Series(low_prices)
        self.close_prices = pd.Series(close_prices)
        self.volumes = pd.Series(volumes)
    
    def test_calculate_price_range(self):
        """测试价格区间计算"""
        price_range = PatternIndicators.calculate_price_range(self.high_prices, self.low_prices)
        
        # 价格区间应该为正
        self.assertTrue((price_range > 0).all())
        
        # 检查计算正确性
        expected_range = (self.high_prices - self.low_prices) / self.low_prices * 100
        pd.testing.assert_series_equal(price_range, expected_range)
    
    def test_calculate_body_size(self):
        """测试实体大小计算"""
        body_size = PatternIndicators.calculate_body_size(self.open_prices, self.close_prices)
        
        # 实体大小应该为正
        self.assertTrue((body_size >= 0).all())
    
    def test_detect_high_volume_bars(self):
        """测试高成交量K线检测"""
        high_volume_bars = PatternIndicators.detect_high_volume_bars(self.volumes, window=10, threshold=1.5)
        
        # 结果应该是布尔值
        self.assertTrue(high_volume_bars.dtype == bool)
        
        # 应该有一些高成交量的K线
        self.assertTrue(high_volume_bars.sum() > 0)
    
    def test_calculate_volatility(self):
        """测试波动率计算"""
        volatility = PatternIndicators.calculate_volatility(self.close_prices, window=20)
        
        # 波动率应该为正
        valid_vol = volatility.dropna()
        self.assertTrue((valid_vol > 0).all())


if __name__ == '__main__':
    # 运行测试
    unittest.main()