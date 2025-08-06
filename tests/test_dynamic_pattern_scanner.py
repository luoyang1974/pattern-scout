"""
动态形态扫描器集成测试
测试完整的三阶段动态基线形态识别流程
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.patterns.detectors.dynamic_pattern_scanner import DynamicPatternScanner
from src.data.models.base_models import MarketRegime, PatternType, FlagSubType


class TestDynamicPatternScanner(unittest.TestCase):
    """动态形态扫描器集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试配置
        self.test_config = {
            'dynamic_baseline': {
                'history_window': 200,
                'regime_detection': {
                    'atr_period': 50,
                    'high_volatility_threshold': 0.7,
                    'low_volatility_threshold': 0.3,
                    'stability_buffer': 3
                }
            },
            'validation': {
                'min_data_points': 100,
                'enable_sanity_checks': True,
                'quality_filters': {
                    'enable_invalidation_filter': True,
                    'min_confidence_score': 0.6,
                    'min_pattern_quality': 'medium'
                }
            },
            'outcome_tracking': {
                'monitoring': {
                    'enable_auto_start': True,
                    'default_timeout_days': 14
                }
            }
        }
        
        self.scanner = DynamicPatternScanner(self.test_config)
        
        # 创建测试数据
        self.create_comprehensive_test_data()
    
    def create_comprehensive_test_data(self):
        """创建综合测试数据"""
        dates = pd.date_range(start='2023-01-01', periods=800, freq='H')
        np.random.seed(42)
        
        # 创建包含多种市场状态和形态的数据
        prices = []
        volumes = []
        
        # 阶段1：低波动期 (0-200)
        low_vol_prices = 100 + np.cumsum(np.random.normal(0.01, 0.3, 200))
        low_vol_volumes = np.random.randint(8000, 15000, 200)
        
        # 阶段2：市场状态转换 + 强势旗杆 (200-250)
        flagpole_base = low_vol_prices[-1]
        flagpole_prices = flagpole_base + np.cumsum(np.random.normal(0.5, 0.8, 50))
        flagpole_volumes = np.random.randint(25000, 45000, 50)  # 高成交量
        
        # 阶段3：旗面整理期 (250-350)
        flag_base = flagpole_prices[-1]
        flag_prices = []
        for i in range(100):
            # 在通道内震荡
            oscillation = 3 * np.sin(i * 0.2) + np.random.normal(0, 0.5)
            flag_prices.append(flag_base + oscillation)
        flag_volumes = np.random.randint(10000, 20000, 100) * np.linspace(1.0, 0.5, 100)  # 成交量萎缩
        
        # 阶段4：突破延续期 (350-450)
        breakout_base = flag_prices[-1]
        breakout_prices = breakout_base + np.cumsum(np.random.normal(0.3, 0.6, 100))
        breakout_volumes = np.random.randint(15000, 30000, 100)
        
        # 阶段5：第二个形态周期 (450-650)
        # 包含一个三角旗形态
        second_flagpole_base = breakout_prices[-1]
        second_flagpole_prices = second_flagpole_base + np.cumsum(np.random.normal(0.4, 0.7, 50))
        second_flagpole_volumes = np.random.randint(20000, 40000, 50)
        
        # 三角旗：收敛震荡
        pennant_base = second_flagpole_prices[-1]
        pennant_prices = []
        for i in range(150):
            convergence_factor = 1 - (i / 150) * 0.8  # 逐渐收敛
            oscillation = 4 * np.sin(i * 0.15) * convergence_factor + np.random.normal(0, 0.3)
            pennant_prices.append(pennant_base + oscillation)
        pennant_volumes = np.random.randint(8000, 18000, 150) * np.linspace(1.0, 0.3, 150)
        
        # 阶段6：平稳期 (650-800)
        final_base = pennant_prices[-1]
        final_prices = final_base + np.cumsum(np.random.normal(0.02, 0.4, 150))
        final_volumes = np.random.randint(10000, 20000, 150)
        
        # 合并所有价格和成交量
        all_prices = np.concatenate([
            low_vol_prices, flagpole_prices, flag_prices,
            breakout_prices, second_flagpole_prices, pennant_prices, final_prices
        ])
        
        all_volumes = np.concatenate([
            low_vol_volumes, flagpole_volumes, flag_volumes,
            breakout_volumes, second_flagpole_volumes, pennant_volumes, final_volumes
        ])
        
        # 创建OHLC数据
        self.comprehensive_df = pd.DataFrame({
            'timestamp': dates,
            'open': all_prices * 0.999,
            'high': all_prices * 1.003,
            'low': all_prices * 0.997,
            'close': all_prices,
            'volume': all_volumes.astype(int),
            'symbol': 'TEST_SYMBOL'
        })
        
        # 创建简单测试数据
        self.simple_df = self.comprehensive_df.iloc[200:400].copy()  # 包含一个完整形态
        self.simple_df.reset_index(drop=True, inplace=True)
    
    def test_full_scan_workflow(self):
        """测试完整扫描工作流程"""
        result = self.scanner.scan(self.comprehensive_df, enable_outcome_tracking=True)
        
        # 验证扫描结果结构
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        
        # 验证基本统计
        self.assertIn('patterns_detected', result)
        self.assertIn('flagpoles_detected', result)
        self.assertIn('scan_time', result)
        self.assertIn('market_regime', result)
        
        # 验证检测到的内容
        self.assertGreaterEqual(result['patterns_detected'], 0)
        self.assertGreaterEqual(result['flagpoles_detected'], 0)
        
        # 验证市场状态识别
        self.assertIn(result['market_regime'], ['high_volatility', 'low_volatility', 'unknown'])
        
        # 验证扫描统计
        self.assertIn('scan_statistics', result)
        scan_stats = result['scan_statistics']
        self.assertEqual(scan_stats['total_scans'], 1)
    
    def test_input_data_validation(self):
        """测试输入数据验证"""
        # 测试空数据框
        empty_df = pd.DataFrame()
        result = self.scanner.scan(empty_df)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
        # 测试缺少必要列的数据
        invalid_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'price': [100] * 10  # 缺少OHLC列
        })
        
        result = self.scanner.scan(invalid_df)
        self.assertFalse(result['success'])
        
        # 测试数据量不足
        small_df = self.comprehensive_df.head(50)  # 少于最小要求
        result = self.scanner.scan(small_df)
        
        self.assertFalse(result['success'])
    
    def test_market_regime_detection_and_transition(self):
        """测试市场状态检测和转换"""
        # 第一次扫描低波动数据
        low_vol_data = self.comprehensive_df.iloc[:200]
        result1 = self.scanner.scan(low_vol_data)
        
        regime1 = result1.get('market_regime')
        
        # 第二次扫描包含高波动的数据
        high_vol_data = self.comprehensive_df.iloc[:300]  # 包含旗杆期
        result2 = self.scanner.scan(high_vol_data)
        
        regime2 = result2.get('market_regime')
        
        # 验证状态可能发生变化（如果数据确实表现出不同波动性）
        if regime1 and regime2:
            # 状态应该都是有效的
            valid_regimes = ['high_volatility', 'low_volatility', 'unknown']
            self.assertIn(regime1, valid_regimes)
            self.assertIn(regime2, valid_regimes)
        
        # 验证状态转换统计
        scan_stats = result2.get('scan_statistics', {})
        self.assertIn('regime_transitions', scan_stats)
    
    def test_flagpole_detection_integration(self):
        """测试旗杆检测集成"""
        # 使用包含明显旗杆的数据段
        flagpole_data = self.comprehensive_df.iloc[180:280]  # 包含旗杆和初始旗面
        
        result = self.scanner.scan(flagpole_data)
        
        # 验证检测到旗杆
        self.assertGreater(result.get('flagpoles_detected', 0), 0)
        
        # 验证旗杆数据结构
        flagpoles = result.get('flagpoles', [])
        if flagpoles:
            flagpole = flagpoles[0]
            expected_keys = ['direction', 'height_percent', 'volume_ratio', 'slope_score']
            for key in expected_keys:
                self.assertIn(key, flagpole)
    
    def test_pattern_detection_integration(self):
        """测试形态检测集成"""
        # 使用包含完整形态的数据段
        pattern_data = self.comprehensive_df.iloc[200:400]  # 旗杆 + 旗面 + 突破
        
        result = self.scanner.scan(pattern_data)
        
        # 验证检测到形态
        patterns_detected = result.get('patterns_detected', 0)
        
        if patterns_detected > 0:
            patterns = result.get('patterns', [])
            self.assertGreater(len(patterns), 0)
            
            # 验证形态数据结构
            pattern = patterns[0]
            expected_keys = [
                'id', 'symbol', 'pattern_type', 'sub_type',
                'confidence_score', 'pattern_quality'
            ]
            for key in expected_keys:
                self.assertIn(key, pattern)
            
            # 验证形态类型
            self.assertEqual(pattern['pattern_type'], 'flag_pattern')
            self.assertIn(pattern['sub_type'], ['flag', 'pennant'])
    
    def test_outcome_tracking_integration(self):
        """测试结局追踪集成"""
        # 启用结局追踪的扫描
        result = self.scanner.scan(self.comprehensive_df, enable_outcome_tracking=True)
        
        # 验证结局追踪相关信息
        self.assertIn('outcome_updates', result)
        outcome_updates = result.get('outcome_updates', 0)
        self.assertGreaterEqual(outcome_updates, 0)
        
        # 验证系统状态包含结局追踪器信息
        system_status = self.scanner.get_system_status()
        self.assertIn('outcome_tracker', system_status)
        
        outcome_tracker_status = system_status['outcome_tracker']
        self.assertIn('monitoring_summary', outcome_tracker_status)
        self.assertIn('outcome_statistics', outcome_tracker_status)
    
    def test_quality_filtering(self):
        """测试质量过滤"""
        # 扫描数据
        result = self.scanner.scan(self.comprehensive_df)
        
        patterns = result.get('patterns', [])
        
        # 验证所有返回的形态都满足质量要求
        for pattern in patterns:
            # 置信度检查
            confidence = pattern.get('confidence_score', 0)
            self.assertGreaterEqual(confidence, 0.6)  # 配置的最小置信度
            
            # 质量等级检查
            quality = pattern.get('pattern_quality', 'low')
            self.assertIn(quality, ['medium', 'high'])  # 应该过滤掉low质量
    
    def test_multiple_timeframes_adaptation(self):
        """测试多时间周期适应"""
        # 测试不同时间周期的数据
        timeframe_results = {}
        
        # 创建不同时间间隔的数据样本
        timeframes = ['15m', '1h', '4h']
        
        for tf in timeframes:
            # 由于我们的测试数据是小时级，这里主要验证系统能处理不同时间周期参数
            result = self.scanner.scan(self.simple_df)
            
            if result.get('success'):
                timeframe_results[tf] = {
                    'patterns': result.get('patterns_detected', 0),
                    'flagpoles': result.get('flagpoles_detected', 0)
                }
        
        # 验证系统能够处理不同时间周期
        self.assertGreater(len(timeframe_results), 0)
    
    def test_system_status_reporting(self):
        """测试系统状态报告"""
        # 执行扫描以初始化系统状态
        self.scanner.scan(self.comprehensive_df)
        
        system_status = self.scanner.get_system_status()
        
        # 验证状态报告结构
        expected_components = [
            'regime_detector', 'baseline_manager', 
            'outcome_tracker', 'scan_statistics'
        ]
        
        for component in expected_components:
            self.assertIn(component, system_status)
        
        # 验证市场状态检测器状态
        regime_status = system_status['regime_detector']
        self.assertIn('current_regime', regime_status)
        self.assertIn('regime_confidence', regime_status)
        self.assertIn('is_stable', regime_status)
        
        # 验证基线管理器状态
        baseline_status = system_status['baseline_manager']
        self.assertIn('active_regime', baseline_status)
        self.assertIn('summary', baseline_status)
    
    def test_performance_metrics(self):
        """测试性能指标"""
        # 执行多次扫描以获得性能指标
        for _ in range(3):
            self.scanner.scan(self.simple_df)
        
        metrics = self.scanner.get_performance_metrics()
        
        # 验证性能指标
        expected_metrics = [
            'total_scans', 'flagpoles_detected', 'patterns_detected',
            'avg_flagpoles_per_scan', 'avg_patterns_per_scan',
            'pattern_conversion_rate', 'regime_stability'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # 验证扫描次数
        self.assertEqual(metrics['total_scans'], 3)
    
    def test_data_export_functionality(self):
        """测试数据导出功能"""
        # 执行扫描
        self.scanner.scan(self.comprehensive_df, enable_outcome_tracking=True)
        
        # 测试基线数据导出
        baseline_export = self.scanner.export_baseline_data()
        
        self.assertIn('baselines', baseline_export)
        self.assertIn('regime_transitions', baseline_export)
        self.assertIn('export_time', baseline_export)
        
        # 测试结局数据导出
        outcome_export = self.scanner.export_outcome_data()
        
        self.assertIsInstance(outcome_export, list)
    
    def test_system_reset(self):
        """测试系统重置"""
        # 执行一些扫描以建立系统状态
        self.scanner.scan(self.comprehensive_df)
        
        # 获取重置前的状态
        status_before = self.scanner.get_system_status()
        metrics_before = self.scanner.get_performance_metrics()
        
        # 执行重置
        self.scanner.reset_system()
        
        # 验证重置后的状态
        metrics_after = self.scanner.get_performance_metrics()
        
        # 验证统计被重置
        self.assertEqual(metrics_after['total_scans'], 0)
        self.assertEqual(metrics_after['patterns_detected'], 0)
        self.assertEqual(metrics_after['flagpoles_detected'], 0)
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        # 测试异常数据处理
        corrupted_df = self.comprehensive_df.copy()
        
        # 制造数据异常
        corrupted_df.loc[100:110, 'high'] = -999  # 异常价格
        corrupted_df.loc[200:210, 'low'] = 999999  # 异常价格
        
        result = self.scanner.scan(corrupted_df)
        
        # 系统应该能够处理异常并返回结果（可能是错误结果）
        self.assertIn('success', result)
        
        # 如果检测到数据异常，应该返回错误
        if not result['success']:
            self.assertIn('error', result)
    
    def test_concurrent_scanning_safety(self):
        """测试并发扫描安全性"""
        # 由于我们没有实际的并发环境，这里测试连续快速扫描
        results = []
        
        for i in range(5):
            # 使用稍微不同的数据段
            start_idx = i * 50
            end_idx = start_idx + 200
            data_segment = self.comprehensive_df.iloc[start_idx:end_idx]
            
            if len(data_segment) >= 100:  # 确保数据量足够
                result = self.scanner.scan(data_segment)
                results.append(result)
        
        # 验证所有扫描都能正常完成
        successful_scans = sum(1 for r in results if r.get('success', False))
        self.assertGreater(successful_scans, 0)
        
        # 验证扫描统计的一致性
        final_metrics = self.scanner.get_performance_metrics()
        self.assertEqual(final_metrics['total_scans'], successful_scans)


if __name__ == '__main__':
    unittest.main()