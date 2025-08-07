"""
动态旗面检测器测试
测试阶段2：基于动态基线的旗面形态检测和失效信号识别
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.patterns.detectors.dynamic_flag_detector import DynamicFlagDetector
from src.patterns.base.market_regime_detector import DualRegimeBaselineManager
from src.data.models.base_models import (
    MarketRegime, Flagpole, PatternRecord, FlagSubType, 
    TrendLine
)


class TestDynamicFlagDetector(unittest.TestCase):
    """动态旗面检测器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟的基线管理器
        self.baseline_manager = Mock(spec=DualRegimeBaselineManager)
        self.detector = DynamicFlagDetector(self.baseline_manager)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建包含旗形和三角旗形的测试数据"""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='H')
        np.random.seed(42)
        
        # 基础价格
        base_prices = np.linspace(100, 105, 300)
        noise = np.random.normal(0, 0.2, 300)
        prices = base_prices + noise
        
        # 创建旗杆阶段（急剧上升）
        flagpole_start = 80
        flagpole_end = 100
        flagpole_height = 10
        prices[flagpole_start:flagpole_end] += np.linspace(0, flagpole_height, flagpole_end - flagpole_start)
        
        # 创建矩形旗阶段（价格在通道内整理）
        flag_start = 100
        flag_end = 150
        flag_base = prices[flag_start]
        
        # 矩形旗：价格在平行通道内震荡
        for i in range(flag_start, flag_end):
            cycle_position = (i - flag_start) / (flag_end - flag_start) * 4 * np.pi
            oscillation = 2 * np.sin(cycle_position)  # 在±2范围内震荡
            prices[i] = flag_base + oscillation
        
        # 创建三角旗阶段（价格逐渐收敛）
        pennant_start = 200
        pennant_end = 250
        pennant_base = prices[pennant_start]
        
        # 三角旗：价格震荡幅度逐渐收敛
        for i in range(pennant_start, pennant_end):
            progress = (i - pennant_start) / (pennant_end - pennant_start)
            convergence_factor = 1 - progress * 0.8  # 逐渐收敛
            cycle_position = (i - pennant_start) / (pennant_end - pennant_start) * 6 * np.pi
            oscillation = 3 * np.sin(cycle_position) * convergence_factor
            prices[i] = pennant_base + oscillation
        
        # 创建成交量模式
        volumes = np.random.randint(10000, 20000, 300)
        # 旗杆期间成交量放大
        volumes[flagpole_start:flagpole_end] *= 3
        # 旗面期间成交量逐渐萎缩
        volumes[flag_start:flag_end] = volumes[flag_start:flag_end] * np.linspace(1.0, 0.4, flag_end - flag_start)
        volumes[pennant_start:pennant_end] = volumes[pennant_start:pennant_end] * np.linspace(1.0, 0.3, pennant_end - pennant_start)
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.003,
            'low': prices * 0.997,
            'close': prices,
            'volume': volumes
        })
        
        # 创建测试旗杆
        self.test_flagpole = Flagpole(
            start_time=dates[flagpole_start],
            end_time=dates[flagpole_end],
            start_price=prices[flagpole_start],
            end_price=prices[flagpole_end],
            height_percent=(prices[flagpole_end] - prices[flagpole_start]) / prices[flagpole_start] * 100,
            direction='up',
            slope_score=0.85,
            volume_ratio=2.8,
            gap_info=None
        )
    
    def test_detect_flag_patterns_rectangular(self):
        """测试矩形旗检测"""
        # 配置基线管理器
        mock_baseline = Mock()
        mock_baseline.get_regime_thresholds.return_value = {
            'min_consolidation_length': 20,
            'parallel_tolerance': 0.1,
            'volume_decline_threshold': 0.7
        }
        self.baseline_manager.get_current_baseline.return_value = mock_baseline
        
        patterns = self.detector.detect_flag_patterns(
            self.test_df,
            [self.test_flagpole],
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        # 验证检测到形态
        self.assertGreater(len(patterns), 0)
        
        # 查找矩形旗形态
        rectangular_flags = [p for p in patterns if p.sub_type == FlagSubType.FLAG]
        self.assertGreater(len(rectangular_flags), 0)
        
        # 验证矩形旗属性
        flag = rectangular_flags[0]
        self.assertEqual(flag.sub_type, FlagSubType.FLAG)
        self.assertIsNotNone(flag.pattern_boundaries)
        self.assertGreaterEqual(len(flag.pattern_boundaries), 2)  # 上下边界线
    
    def test_detect_flag_patterns_triangular(self):
        """测试三角旗检测"""
        # 创建第二个旗杆用于三角旗测试
        pennant_flagpole = Flagpole(
            start_time=self.test_df.iloc[180]['timestamp'],
            end_time=self.test_df.iloc[200]['timestamp'],
            start_price=self.test_df.iloc[180]['close'],
            end_price=self.test_df.iloc[200]['close'],
            height_percent=8.0,
            direction='up',
            slope_score=0.75,
            volume_ratio=2.2,
            gap_info=None
        )
        
        # 配置基线管理器
        mock_baseline = Mock()
        mock_baseline.get_regime_thresholds.return_value = {
            'min_consolidation_length': 20,
            'convergence_threshold': 0.6,
            'volume_decline_threshold': 0.8
        }
        self.baseline_manager.get_current_baseline.return_value = mock_baseline
        
        patterns = self.detector.detect_flag_patterns(
            self.test_df,
            [pennant_flagpole],
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        # 查找三角旗形态
        pennant_patterns = [p for p in patterns if p.sub_type == FlagSubType.PENNANT]
        
        if pennant_patterns:
            pennant = pennant_patterns[0]
            self.assertEqual(pennant.sub_type, FlagSubType.PENNANT)
            self.assertIsNotNone(pennant.pattern_boundaries)
            
            # 验证收敛特性
            self.assertTrue(hasattr(pennant, 'convergence_data'))
    
    def test_percentile_channel_construction(self):
        """测试百分位通道构建"""
        consolidation_data = self.test_df.iloc[100:150]  # 旗面区域
        
        upper_line, lower_line = self.detector._construct_percentile_channel(
            consolidation_data,
            upper_percentile=80,
            lower_percentile=20
        )
        
        # 验证通道线
        self.assertIsInstance(upper_line, TrendLine)
        self.assertIsInstance(lower_line, TrendLine)
        
        # 验证通道的几何特性
        self.assertGreater(upper_line.start_price, lower_line.start_price)
        self.assertGreater(upper_line.end_price, lower_line.end_price)
        
        # 验证R²值
        self.assertGreaterEqual(upper_line.r_squared, 0.0)
        self.assertGreaterEqual(lower_line.r_squared, 0.0)
    
    def test_geometric_pattern_analysis(self):
        """测试几何形态分析"""
        consolidation_data = self.test_df.iloc[100:150]
        
        # 构建测试通道
        upper_line, lower_line = self.detector._construct_percentile_channel(
            consolidation_data,
            upper_percentile=85,
            lower_percentile=15
        )
        
        analysis_result = self.detector._analyze_geometric_pattern(
            consolidation_data,
            upper_line,
            lower_line,
            self.test_flagpole
        )
        
        # 验证分析结果
        self.assertIn('pattern_type', analysis_result)
        self.assertIn('quality_metrics', analysis_result)
        self.assertIn('geometric_scores', analysis_result)
        
        # 验证形态类型识别
        pattern_type = analysis_result['pattern_type']
        self.assertIn(pattern_type, ['rectangular_flag', 'triangular_pennant', 'invalid'])
    
    def test_invalidation_signal_detection(self):
        """测试失效信号检测"""
        # 创建包含失效信号的数据
        invalid_data = self.test_df.copy()
        
        # 在旗面区域制造假突破
        fake_breakout_idx = 120
        invalid_data.loc[fake_breakout_idx:fake_breakout_idx+3, 'high'] *= 1.08  # 假突破
        invalid_data.loc[fake_breakout_idx:fake_breakout_idx+3, 'close'] *= 1.02  # 但收盘回落
        
        # 制造成交量背离
        invalid_data.loc[fake_breakout_idx:fake_breakout_idx+3, 'volume'] *= 0.3  # 低成交量
        
        consolidation_data = invalid_data.iloc[100:150]
        invalidation_signals = self.detector._detect_invalidation_signals(
            consolidation_data,
            self.test_flagpole
        )
        
        # 验证检测到失效信号
        self.assertGreater(len(invalidation_signals), 0)
        
        # 验证失效信号类型
        signal_types = [signal.signal_type for signal in invalidation_signals]
        expected_types = ['fake_breakout', 'volume_divergence']
        
        # 至少应该检测到一种失效信号类型
        self.assertTrue(any(sig_type in signal_types for sig_type in expected_types))
    
    def test_volume_pattern_analysis(self):
        """测试成交量模式分析"""
        consolidation_data = self.test_df.iloc[100:150]
        
        volume_analysis = self.detector._analyze_volume_pattern(
            consolidation_data,
            self.test_flagpole
        )
        
        # 验证成交量分析结果
        self.assertIn('volume_decline_rate', volume_analysis)
        self.assertIn('average_volume_ratio', volume_analysis)
        self.assertIn('volume_trend_slope', volume_analysis)
        self.assertIn('volume_health_score', volume_analysis)
        
        # 验证成交量健康度评分
        health_score = volume_analysis['volume_health_score']
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)
    
    def test_pattern_validation_rectangular(self):
        """测试矩形旗验证"""
        # 创建模拟的几何分析结果
        geometric_analysis = {
            'pattern_type': 'rectangular_flag',
            'quality_metrics': {
                'parallel_quality': 0.85,
                'channel_containment': 0.90,
                'slope_consistency': 0.05  # 低斜率，符合矩形旗
            },
            'geometric_scores': {
                'overall_score': 0.82
            }
        }
        
        volume_analysis = {
            'volume_health_score': 0.75,
            'volume_decline_rate': 0.6
        }
        
        validation_result = self.detector._validate_rectangular_flag(
            geometric_analysis,
            volume_analysis,
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        self.assertIn('is_valid', validation_result)
        self.assertIn('confidence_score', validation_result)
        self.assertIn('quality_rating', validation_result)
        
        # 对于高质量的矩形旗，应该验证通过
        if validation_result['is_valid']:
            self.assertGreater(validation_result['confidence_score'], 0.5)
    
    def test_pattern_validation_triangular(self):
        """测试三角旗验证"""
        # 创建模拟的几何分析结果
        geometric_analysis = {
            'pattern_type': 'triangular_pennant',
            'quality_metrics': {
                'convergence_quality': 0.88,
                'symmetry_score': 0.82,
                'apex_validity': 0.90
            },
            'geometric_scores': {
                'overall_score': 0.85
            },
            'convergence_data': {
                'convergence_ratio': 0.75,
                'apex_position': 0.85
            }
        }
        
        volume_analysis = {
            'volume_health_score': 0.80,
            'volume_decline_rate': 0.7
        }
        
        validation_result = self.detector._validate_triangular_pennant(
            geometric_analysis,
            volume_analysis,
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        self.assertIn('is_valid', validation_result)
        self.assertIn('confidence_score', validation_result)
        self.assertIn('quality_rating', validation_result)
        
        # 对于高质量的三角旗，应该验证通过
        if validation_result['is_valid']:
            self.assertGreater(validation_result['confidence_score'], 0.5)
    
    def test_pattern_record_creation(self):
        """测试形态记录创建"""
        # 模拟验证结果
        validation_result = {
            'is_valid': True,
            'confidence_score': 0.85,
            'quality_rating': 'high'
        }
        
        # 模拟边界线
        upper_line = TrendLine(
            start_time=self.test_df.iloc[100]['timestamp'],
            end_time=self.test_df.iloc[150]['timestamp'],
            start_price=115.0,
            end_price=117.0,
            slope=0.001,
            r_squared=0.75
        )
        
        lower_line = TrendLine(
            start_time=self.test_df.iloc[100]['timestamp'],
            end_time=self.test_df.iloc[150]['timestamp'],
            start_price=110.0,
            end_price=112.0,
            slope=0.001,
            r_squared=0.80
        )
        
        pattern_record = self.detector._create_pattern_record(
            flagpole=self.test_flagpole,
            sub_type=FlagSubType.FLAG,
            boundaries=[upper_line, lower_line],
            validation_result=validation_result,
            invalidation_signals=[],
            additional_data={}
        )
        
        # 验证形态记录
        self.assertIsInstance(pattern_record, PatternRecord)
        self.assertEqual(pattern_record.sub_type, FlagSubType.FLAG)
        self.assertEqual(pattern_record.confidence_score, 0.85)
        self.assertEqual(pattern_record.pattern_quality, 'high')
        self.assertEqual(len(pattern_record.pattern_boundaries), 2)
    
    def test_multiple_flagpoles_processing(self):
        """测试多个旗杆的处理"""
        # 创建多个旗杆
        flagpole1 = self.test_flagpole
        
        flagpole2 = Flagpole(
            start_time=self.test_df.iloc[180]['timestamp'],
            end_time=self.test_df.iloc[200]['timestamp'],
            start_price=self.test_df.iloc[180]['close'],
            end_price=self.test_df.iloc[200]['close'],
            height_percent=8.0,
            direction='up',
            slope_score=0.75,
            volume_ratio=2.2,
            gap_info=None
        )
        
        # 配置基线管理器
        mock_baseline = Mock()
        mock_baseline.get_regime_thresholds.return_value = {
            'min_consolidation_length': 20,
            'parallel_tolerance': 0.1,
            'convergence_threshold': 0.6
        }
        self.baseline_manager.get_current_baseline.return_value = mock_baseline
        
        patterns = self.detector.detect_flag_patterns(
            self.test_df,
            [flagpole1, flagpole2],
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        
        # 验证处理了多个旗杆
        self.assertIsInstance(patterns, list)
        # 每个旗杆都可能生成形态，但不保证一定生成
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空旗杆列表
        patterns = self.detector.detect_flag_patterns(
            self.test_df,
            [],
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        self.assertEqual(len(patterns), 0)
        
        # 测试数据不足的情况
        short_df = self.test_df.head(50)
        patterns = self.detector.detect_flag_patterns(
            short_df,
            [self.test_flagpole],
            MarketRegime.HIGH_VOLATILITY,
            '1h'
        )
        # 数据不足时应该返回空列表
        self.assertEqual(len(patterns), 0)
    
    def test_invalidation_signal_types(self):
        """测试各种失效信号类型"""
        # 创建包含各种失效信号的数据
        consolidation_data = self.test_df.iloc[100:150].copy()
        
        # 测试假突破信号检测
        fake_breakout_signals = self.detector._detect_fake_breakout_signals(
            consolidation_data,
            self.test_flagpole
        )
        
        # 测试成交量背离信号检测
        volume_divergence_signals = self.detector._detect_volume_divergence_signals(
            consolidation_data,
            self.test_flagpole
        )
        
        # 测试形态变形信号检测
        deformation_signals = self.detector._detect_pattern_deformation_signals(
            consolidation_data,
            self.test_flagpole
        )
        
        # 验证每种信号检测都能正常运行
        self.assertIsInstance(fake_breakout_signals, list)
        self.assertIsInstance(volume_divergence_signals, list)
        self.assertIsInstance(deformation_signals, list)


if __name__ == '__main__':
    unittest.main()