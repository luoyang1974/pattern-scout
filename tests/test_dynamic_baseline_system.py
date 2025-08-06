"""
动态基线系统测试
测试三层鲁棒统计保护和市场状态检测
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.patterns.base.robust_statistics import RobustStatistics
from src.patterns.base.market_regime_detector import SmartRegimeDetector, DualRegimeBaselineManager
from src.data.models.base_models import MarketRegime, DynamicBaseline


class TestThreeLayerRobustStatistics(unittest.TestCase):
    """三层鲁棒统计保护测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.robust_stats = RobustStatistics()
        
        # 创建测试数据
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 1000)
        outliers = np.array([200, 250, 300, 10, 5])  # 极端异常值
        
        # 混合正常数据和异常值
        positions = np.random.choice(range(len(normal_data)), size=len(outliers), replace=False)
        for i, pos in enumerate(positions):
            normal_data[pos] = outliers[i]
            
        self.test_data = normal_data
        self.clean_data = normal_data[~np.isin(range(len(normal_data)), positions)]
    
    def test_mad_filter(self):
        """测试MAD过滤器"""
        test_series = pd.Series(self.test_data)
        filtered_data = self.robust_stats.mad_filter(
            test_series, 
            threshold=3.0
        )
        
        # 验证异常值被正确识别
        self.assertLess(len(filtered_data), len(self.test_data))
        
        # 验证过滤后数据的统计特性更稳定
        original_std = np.std(self.test_data)
        filtered_std = np.std(filtered_data)
        self.assertLess(filtered_std, original_std)
        
    def test_winsorize(self):
        """测试Winsorize方法"""
        test_series = pd.Series(self.test_data)
        winsorized_data = self.robust_stats.winsorize(
            test_series, 
            limits=(0.05, 0.95)
        )
        
        # 验证数据长度保持不变
        self.assertEqual(len(winsorized_data), len(self.test_data))
        
        # 验证极端值被限制
        p5 = np.percentile(self.test_data, 5)
        p95 = np.percentile(self.test_data, 95)
        
        self.assertGreaterEqual(winsorized_data.min(), p5)
        self.assertLessEqual(winsorized_data.max(), p95)
    
    def test_robust_percentiles(self):
        """测试鲁棒分位数计算"""
        percentiles = [10, 25, 50, 75, 90]
        test_series = pd.Series(self.test_data)
        robust_percentiles = self.robust_stats.robust_percentiles(
            test_series, 
            percentiles=percentiles
        )
        
        # 验证返回正确数量的分位数
        self.assertEqual(len(robust_percentiles), len(percentiles))
        
        # 验证分位数的单调性
        values = list(robust_percentiles.values())
        for i in range(len(values) - 1):
            self.assertLessEqual(values[i], values[i + 1])
    
    def test_adaptive_threshold(self):
        """测试自适应阈值调整"""
        # 测试基线百分位数调整
        baseline_percentiles = {75: 100.0, 85: 105.0, 90: 110.0, 95: 115.0}
        
        # 创建当前数据
        current_data = pd.Series(np.random.normal(102, 8, 100))
        
        adjusted_percentiles = self.robust_stats.adaptive_threshold_adjustment(
            baseline_percentiles,
            current_data,
            adjustment_factor=0.1
        )
        
        # 验证返回正确的结构
        self.assertEqual(len(adjusted_percentiles), len(baseline_percentiles))
        
        # 验证调整后的值
        for p in baseline_percentiles:
            self.assertIn(p, adjusted_percentiles)
            self.assertIsInstance(adjusted_percentiles[p], float)


class TestSmartRegimeDetector(unittest.TestCase):
    """智能市场状态检测器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = SmartRegimeDetector(
            atr_period=20,
            high_vol_threshold=0.7,
            low_vol_threshold=0.3,
            stability_buffer=3
        )
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # 创建不同波动率阶段的数据
        np.random.seed(42)
        
        # 低波动阶段（前60天）
        low_vol_returns = np.random.normal(0.001, 0.01, 60)
        
        # 高波动阶段（中间80天）
        high_vol_returns = np.random.normal(0.002, 0.05, 80)
        
        # 回到低波动阶段（后60天）
        low_vol_returns_2 = np.random.normal(0.001, 0.015, 60)
        
        returns = np.concatenate([low_vol_returns, high_vol_returns, low_vol_returns_2])
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(10000, 100000, len(dates))
        })
    
    def test_regime_detection(self):
        """测试市场状态检测"""
        # 测试初始检测
        initial_regime = self.detector.update(self.test_df[:30])
        self.assertIn(initial_regime, [MarketRegime.LOW_VOLATILITY, MarketRegime.UNKNOWN])
        
        # 测试高波动检测
        high_vol_regime = self.detector.update(self.test_df[60:120])
        self.assertEqual(high_vol_regime, MarketRegime.HIGH_VOLATILITY)
        
        # 测试状态转换
        final_regime = self.detector.update(self.test_df[160:])
        self.assertEqual(final_regime, MarketRegime.LOW_VOLATILITY)
    
    def test_anti_oscillation_mechanism(self):
        """测试防震荡机制"""
        # 创建震荡数据
        oscillation_data = self.test_df.copy()
        
        # 逐步更新，观察状态稳定性
        regimes = []
        for i in range(30, len(oscillation_data), 10):
            regime = self.detector.update(oscillation_data[:i])
            regimes.append(regime)
        
        # 验证不会频繁震荡
        regime_changes = sum(1 for i in range(1, len(regimes)) 
                           if regimes[i] != regimes[i-1])
        self.assertLess(regime_changes, len(regimes) * 0.3)  # 状态变化不超过30%
    
    def test_regime_confidence(self):
        """测试状态置信度"""
        self.detector.update(self.test_df[:100])
        confidence = self.detector.get_regime_confidence()
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_market_snapshot_creation(self):
        """测试市场快照创建"""
        self.detector.update(self.test_df[:100])
        snapshot = self.detector.create_market_snapshot(self.test_df[:100])
        
        # 验证快照包含必要信息
        self.assertIsNotNone(snapshot.current_regime)
        self.assertIsNotNone(snapshot.volatility_percentile)
        self.assertIsNotNone(snapshot.regime_confidence)
        self.assertIsNotNone(snapshot.data_window_size)
    
    def test_volatility_metrics(self):
        """测试波动率指标"""
        self.detector.update(self.test_df)
        metrics = self.detector.get_volatility_metrics()
        
        expected_keys = [
            'current_volatility_percentile', 'atr_value', 'volatility_trend',
            'regime_stability_score', 'transition_probability'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)


class TestDualRegimeBaselineManager(unittest.TestCase):
    """双状态基线管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.baseline_manager = DualRegimeBaselineManager(history_window=100)
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 150)))
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(10000, 100000, len(dates))
        })
    
    def test_baseline_update(self):
        """测试基线更新"""
        # 设置高波动状态并更新基线
        self.baseline_manager.set_active_regime(MarketRegime.HIGH_VOLATILITY)
        
        baseline = self.baseline_manager.update_baseline(
            self.test_df, 
            MarketRegime.HIGH_VOLATILITY
        )
        
        self.assertIsInstance(baseline, DynamicBaseline)
        self.assertEqual(baseline.regime, MarketRegime.HIGH_VOLATILITY)
        self.assertIsNotNone(baseline.percentile_25)
        self.assertIsNotNone(baseline.percentile_75)
        
        # 验证基线已存储
        stored_baselines = self.baseline_manager.baselines[MarketRegime.HIGH_VOLATILITY]
        self.assertGreater(len(stored_baselines), 0)
    
    def test_dual_regime_separation(self):
        """测试双状态分离"""
        # 更新高波动基线
        self.baseline_manager.update_baseline(
            self.test_df[:75], 
            MarketRegime.HIGH_VOLATILITY
        )
        
        # 更新低波动基线
        self.baseline_manager.update_baseline(
            self.test_df[75:], 
            MarketRegime.LOW_VOLATILITY
        )
        
        # 验证两个状态的基线分别存储
        high_vol_baselines = self.baseline_manager.baselines[MarketRegime.HIGH_VOLATILITY]
        low_vol_baselines = self.baseline_manager.baselines[MarketRegime.LOW_VOLATILITY]
        
        self.assertGreater(len(high_vol_baselines), 0)
        self.assertGreater(len(low_vol_baselines), 0)
    
    def test_history_window_management(self):
        """测试历史窗口管理"""
        # 连续更新超过窗口大小的基线
        for i in range(120):
            subset_df = self.test_df[i:i+30]
            if len(subset_df) >= 20:  # 最小数据要求
                self.baseline_manager.update_baseline(
                    subset_df, 
                    MarketRegime.HIGH_VOLATILITY
                )
        
        # 验证历史窗口限制
        stored_baselines = self.baseline_manager.baselines[MarketRegime.HIGH_VOLATILITY]
        self.assertLessEqual(len(stored_baselines), 100)  # history_window
    
    def test_baseline_summary(self):
        """测试基线汇总"""
        # 更新一些基线数据
        self.baseline_manager.update_baseline(
            self.test_df[:75], 
            MarketRegime.HIGH_VOLATILITY
        )
        self.baseline_manager.update_baseline(
            self.test_df[75:], 
            MarketRegime.LOW_VOLATILITY
        )
        
        summary = self.baseline_manager.get_baseline_summary()
        
        self.assertIn('high_volatility_baselines', summary)
        self.assertIn('low_volatility_baselines', summary)
        self.assertIn('total_data_points', summary)
        self.assertIn('latest_update', summary)
    
    def test_baseline_export_import(self):
        """测试基线导出导入"""
        # 创建一些基线数据
        self.baseline_manager.update_baseline(
            self.test_df, 
            MarketRegime.HIGH_VOLATILITY
        )
        
        # 导出
        exported_data = self.baseline_manager.export_baselines()
        
        # 验证导出格式
        self.assertIn(MarketRegime.HIGH_VOLATILITY.value, exported_data)
        baseline_data = exported_data[MarketRegime.HIGH_VOLATILITY.value]
        self.assertIsInstance(baseline_data, list)
        self.assertGreater(len(baseline_data), 0)


if __name__ == '__main__':
    unittest.main()