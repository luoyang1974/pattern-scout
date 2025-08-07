"""
动态旗杆检测器测试
测试阶段1：基于动态阈值的旗杆检测
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.patterns.detectors.dynamic_flagpole_detector import DynamicFlagpoleDetector
from src.patterns.base.market_regime_detector import DualRegimeBaselineManager
from src.data.models.base_models import MarketRegime, Flagpole


class TestDynamicFlagpoleDetector(unittest.TestCase):
    """动态旗杆检测器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟的基线管理器
        self.baseline_manager = Mock(spec=DualRegimeBaselineManager)
        self.detector = DynamicFlagpoleDetector(self.baseline_manager)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建包含旗杆的测试数据"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
        np.random.seed(42)
        
        # 基础价格走势
        base_prices = np.linspace(100, 110, 200)
        noise = np.random.normal(0, 0.5, 200)
        
        # 创建明显的旗杆（急剧上升阶段）
        flagpole_start = 50
        flagpole_end = 65
        
        prices = base_prices + noise
        # 在旗杆区域创建急剧上升
        flagpole_growth = np.linspace(0, 15, flagpole_end - flagpole_start + 1)
        prices[flagpole_start:flagpole_end+1] += flagpole_growth
        
        # 创建成交量爆发（旗杆特征）
        volumes = np.random.randint(10000, 20000, 200)
        volumes[flagpole_start:flagpole_end+1] *= 3  # 旗杆期间成交量放大
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': volumes
        })
        
        # 创建另一个包含下降旗杆的数据集
        down_prices = 120 - base_prices + noise
        down_prices[flagpole_start:flagpole_end+1] -= flagpole_growth  # 下降旗杆
        
        self.test_df_down = pd.DataFrame({
            'timestamp': dates,
            'open': down_prices * 1.001,
            'high': down_prices * 1.005,
            'low': down_prices * 0.995,
            'close': down_prices,
            'volume': volumes  # 同样的成交量模式
        })
    
    def test_detect_flagpoles_up_trend(self):
        """测试上升旗杆检测"""
        # 配置基线管理器返回值
        mock_baseline = Mock()
        mock_baseline.get_regime_thresholds.return_value = {
            'min_height_threshold': 0.8,
            'trend_strength_threshold': 0.6,
            'volume_burst_threshold': 2.0
        }
        self.baseline_manager.get_current_baseline.return_value = mock_baseline
        
        flagpoles = self.detector.detect_flagpoles(
            self.test_df, 
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        # 验证检测到旗杆
        self.assertGreater(len(flagpoles), 0)
        
        # 验证第一个旗杆的基本属性
        first_flagpole = flagpoles[0]
        self.assertEqual(first_flagpole.direction, "up")
        self.assertGreater(first_flagpole.height_percent, 0)
        self.assertGreater(first_flagpole.volume_ratio, 1.0)
        self.assertIsNotNone(first_flagpole.slope_score)
    
    def test_detect_flagpoles_down_trend(self):
        """测试下降旗杆检测"""
        # 配置基线管理器
        mock_baseline = Mock()
        mock_baseline.get_regime_thresholds.return_value = {
            'min_height_threshold': 0.8,
            'trend_strength_threshold': 0.6,
            'volume_burst_threshold': 2.0
        }
        self.baseline_manager.get_current_baseline.return_value = mock_baseline
        
        flagpoles = self.detector.detect_flagpoles(
            self.test_df_down, 
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        # 验证检测到下降旗杆
        self.assertGreater(len(flagpoles), 0)
        
        first_flagpole = flagpoles[0]
        self.assertEqual(first_flagpole.direction, "down")
        self.assertGreater(first_flagpole.height_percent, 0)
    
    def test_slope_score_calculation(self):
        """测试斜率评分计算"""
        # 创建简单的线性上升数据
        simple_df = self.test_df.iloc[50:66].copy()  # 旗杆区域
        simple_df.reset_index(drop=True, inplace=True)
        
        slope_score = self.detector._calculate_slope_score(
            simple_df, 
            MarketRegime.HIGH_VOLATILITY
        )
        
        # 验证斜率评分
        self.assertIsInstance(slope_score, float)
        self.assertGreaterEqual(slope_score, 0.0)
        self.assertLessEqual(slope_score, 1.0)
        
        # 对于明显的上升趋势，斜率评分应该较高
        self.assertGreater(slope_score, 0.5)
    
    def test_volume_burst_detection(self):
        """测试成交量爆发检测"""
        flagpole_start = 50
        flagpole_end = 65
        
        volume_ratio = self.detector._calculate_volume_burst(
            self.test_df,
            flagpole_start,
            flagpole_end,
            baseline_window=20
        )
        
        # 验证成交量比率
        self.assertIsInstance(volume_ratio, float)
        self.assertGreater(volume_ratio, 1.0)  # 应该有成交量放大
        
        # 由于我们在创建数据时放大了3倍成交量，比率应该接近3
        self.assertGreater(volume_ratio, 2.5)
    
    def test_gap_detection(self):
        """测试跳空缺口检测"""
        # 创建有跳空的数据
        gap_df = self.test_df.copy()
        gap_index = 60
        
        # 制造向上跳空
        gap_df.loc[gap_index:, 'open'] += 5
        gap_df.loc[gap_index:, 'high'] += 5
        gap_df.loc[gap_index:, 'low'] += 5
        gap_df.loc[gap_index:, 'close'] += 5
        
        gap_info = self.detector._detect_gap(gap_df, gap_index)
        
        if gap_info:
            self.assertEqual(gap_info['direction'], 'up')
            self.assertGreater(gap_info['gap_size'], 0)
    
    def test_dynamic_threshold_adaptation(self):
        """测试动态阈值适应"""
        # 测试高波动环境阈值
        high_vol_thresholds = self.detector._get_regime_adjusted_thresholds(
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        # 测试低波动环境阈值
        low_vol_thresholds = self.detector._get_regime_adjusted_thresholds(
            MarketRegime.LOW_VOLATILITY,
            '1h'
        )
        
        # 验证阈值结构
        expected_keys = [
            'min_height_threshold', 'trend_strength_threshold', 
            'volume_burst_threshold', 'min_duration'
        ]
        
        for key in expected_keys:
            self.assertIn(key, high_vol_thresholds)
            self.assertIn(key, low_vol_thresholds)
        
        # 验证高波动环境通常需要更高的阈值
        self.assertGreaterEqual(
            high_vol_thresholds['min_height_threshold'],
            low_vol_thresholds['min_height_threshold']
        )
    
    def test_flagpole_validation(self):
        """测试旗杆验证逻辑"""
        # 创建有效的旗杆数据
        valid_flagpole_data = {
            'start_idx': 50,
            'end_idx': 65,
            'height_percent': 12.5,
            'slope_score': 0.85,
            'volume_ratio': 2.8,
            'direction': 'up'
        }
        
        is_valid = self.detector._validate_flagpole(
            valid_flagpole_data, 
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        self.assertTrue(is_valid)
        
        # 测试无效的旗杆（高度不够）
        invalid_flagpole_data = valid_flagpole_data.copy()
        invalid_flagpole_data['height_percent'] = 0.3  # 太低
        
        is_invalid = self.detector._validate_flagpole(
            invalid_flagpole_data,
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        self.assertFalse(is_invalid)
    
    def test_multiple_timeframes(self):
        """测试多时间周期适应"""
        timeframes = ['15m', '1h', '4h', '1d']
        
        mock_baseline = Mock()
        mock_baseline.get_regime_thresholds.return_value = {
            'min_height_threshold': 1.0,
            'trend_strength_threshold': 0.6,
            'volume_burst_threshold': 2.0
        }
        self.baseline_manager.get_current_baseline.return_value = mock_baseline
        
        for timeframe in timeframes:
            flagpoles = self.detector.detect_flagpoles(
                self.test_df,
                MarketRegime.HIGH_VOLATILITY,
                timeframe
            )
            
            # 每个时间周期都应该能正常运行
            self.assertIsInstance(flagpoles, list)
            
            # 验证时间周期影响检测参数
            thresholds = self.detector._get_regime_adjusted_thresholds(
                MarketRegime.HIGH_VOLATILITY,
                timeframe
            )
            self.assertIsInstance(thresholds, dict)
    
    def test_flagpole_creation(self):
        """测试旗杆对象创建"""
        # 模拟旗杆数据
        start_time = self.test_df.iloc[50]['timestamp']
        end_time = self.test_df.iloc[65]['timestamp']
        start_price = self.test_df.iloc[50]['close']
        end_price = self.test_df.iloc[65]['close']
        
        flagpole = self.detector._create_flagpole_record(
            start_idx=50,
            end_idx=65,
            start_time=start_time,
            end_time=end_time,
            start_price=start_price,
            end_price=end_price,
            height_percent=15.2,
            slope_score=0.85,
            volume_ratio=2.8,
            direction='up',
            gap_info=None
        )
        
        # 验证旗杆对象属性
        self.assertIsInstance(flagpole, Flagpole)
        self.assertEqual(flagpole.direction, 'up')
        self.assertEqual(flagpole.start_time, start_time)
        self.assertEqual(flagpole.end_time, end_time)
        self.assertEqual(flagpole.start_price, start_price)
        self.assertEqual(flagpole.end_price, end_price)
        self.assertEqual(flagpole.height_percent, 15.2)
        self.assertEqual(flagpole.volume_ratio, 2.8)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空数据
        empty_df = pd.DataFrame()
        flagpoles = self.detector.detect_flagpoles(
            empty_df, 
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        self.assertEqual(len(flagpoles), 0)
        
        # 测试数据量不足
        small_df = self.test_df.head(10)
        flagpoles = self.detector.detect_flagpoles(
            small_df,
            MarketRegime.HIGH_VOLATILITY, 
            '1h'
        )
        self.assertEqual(len(flagpoles), 0)
        
        # 测试无波动数据（平盘）
        flat_df = self.test_df.copy()
        flat_df['close'] = 100.0  # 所有价格相同
        flat_df['open'] = 100.0
        flat_df['high'] = 100.0
        flat_df['low'] = 100.0
        
        flagpoles = self.detector.detect_flagpoles(
            flat_df,
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        self.assertEqual(len(flagpoles), 0)


if __name__ == '__main__':
    unittest.main()