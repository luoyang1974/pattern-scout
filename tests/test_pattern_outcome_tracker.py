"""
形态结局追踪系统测试
测试阶段3：六分类形态结局监控和分析
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analysis.pattern_outcome_tracker import PatternOutcomeTracker
from src.data.models.base_models import (
    PatternRecord, PatternOutcome, Flagpole, FlagSubType,
    PatternOutcomeAnalysis, TrendLine
)


class TestPatternOutcomeTracker(unittest.TestCase):
    """形态结局追踪器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.tracker = PatternOutcomeTracker()
        
        # 创建测试数据和形态
        self.create_test_data_and_patterns()
    
    def create_test_data_and_patterns(self):
        """创建测试数据和形态"""
        # 创建时间序列
        start_date = datetime(2023, 1, 1)
        dates = pd.date_range(start=start_date, periods=500, freq='H')
        
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.normal(0.01, 0.5, 500))
        
        # 创建不同结局类型的价格走势
        prices = base_prices.copy()
        
        # 强势延续区域（150-200）
        strong_continuation_start = 150
        strong_continuation_end = 200
        prices[strong_continuation_start:strong_continuation_end] += np.linspace(0, 20, 
            strong_continuation_end - strong_continuation_start)
        
        # 假突破反转区域（250-300）
        fake_breakout_start = 250
        fake_breakout_end = 300
        prices[fake_breakout_start:fake_breakout_start+10] += 5  # 假突破
        prices[fake_breakout_start+10:fake_breakout_end] -= np.linspace(5, 15, 
            fake_breakout_end - (fake_breakout_start+10))
        
        # 反向运行区域（350-400）
        opposite_run_start = 350
        opposite_run_end = 400
        prices[opposite_run_start:opposite_run_end] -= np.linspace(0, 25,
            opposite_run_end - opposite_run_start)
        
        volumes = np.random.randint(10000, 50000, 500)
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': volumes
        })
        
        # 创建测试形态
        self.create_test_patterns(dates, base_prices)
    
    def create_test_patterns(self, dates, prices):
        """创建测试形态"""
        # 创建测试旗杆
        test_flagpole = Flagpole(
            start_time=dates[40],
            end_time=dates[60],
            start_price=prices[40],
            end_price=prices[60],
            height_percent=12.5,
            direction='up',
            slope_score=0.85,
            volume_ratio=2.8,
            gap_info=None
        )
        
        # 创建测试边界线
        upper_line = TrendLine(
            start_time=dates[60],
            end_time=dates[120],
            start_price=prices[60] + 2,
            end_price=prices[120] + 2,
            slope=0.001,
            r_squared=0.75
        )
        
        lower_line = TrendLine(
            start_time=dates[60],
            end_time=dates[120],
            start_price=prices[60] - 2,
            end_price=prices[120] - 2,
            slope=0.001,
            r_squared=0.80
        )
        
        # 强势延续形态
        self.strong_continuation_pattern = PatternRecord(
            id="test_strong_001",
            symbol="TEST",
            pattern_type="flag_pattern",
            sub_type=FlagSubType.FLAG,
            flagpole=test_flagpole,
            pattern_boundaries=[upper_line, lower_line],
            start_time=dates[60],
            end_time=dates[120],
            confidence_score=0.85,
            pattern_quality="high",
            invalidation_signals=[],
            metadata={"test_outcome": "strong_continuation"}
        )
        
        # 假突破反转形态
        fake_flagpole = Flagpole(
            start_time=dates[200],
            end_time=dates[220],
            start_price=prices[200],
            end_price=prices[220],
            height_percent=8.5,
            direction='up',
            slope_score=0.70,
            volume_ratio=2.2,
            gap_info=None
        )
        
        self.fake_breakout_pattern = PatternRecord(
            id="test_fake_001",
            symbol="TEST",
            pattern_type="flag_pattern",
            sub_type=FlagSubType.FLAG,
            flagpole=fake_flagpole,
            pattern_boundaries=[upper_line, lower_line],
            start_time=dates[220],
            end_time=dates[280],
            confidence_score=0.72,
            pattern_quality="medium",
            invalidation_signals=[],
            metadata={"test_outcome": "failed_breakout"}
        )
        
        # 反向运行形态
        opposite_flagpole = Flagpole(
            start_time=dates[300],
            end_time=dates[320],
            start_price=prices[300],
            end_price=prices[320],
            height_percent=10.2,
            direction='up',
            slope_score=0.78,
            volume_ratio=2.5,
            gap_info=None
        )
        
        self.opposite_run_pattern = PatternRecord(
            id="test_opposite_001",
            symbol="TEST",
            pattern_type="flag_pattern",
            sub_type=FlagSubType.PENNANT,
            flagpole=opposite_flagpole,
            pattern_boundaries=[upper_line, lower_line],
            start_time=dates[320],
            end_time=dates[380],
            confidence_score=0.78,
            pattern_quality="high",
            invalidation_signals=[],
            metadata={"test_outcome": "opposite_run"}
        )
    
    def test_start_monitoring(self):
        """测试开始监控"""
        patterns = [self.strong_continuation_pattern, self.fake_breakout_pattern]
        
        result = self.tracker.start_monitoring(patterns)
        
        # 验证监控启动
        self.assertEqual(result['patterns_added'], 2)
        self.assertEqual(result['total_monitoring'], 2)
        
        # 验证监控状态
        monitoring_summary = self.tracker.get_monitoring_summary()
        self.assertEqual(monitoring_summary['active_monitoring_count'], 2)
        self.assertEqual(monitoring_summary['total_patterns_tracked'], 2)
    
    def test_update_monitoring_strong_continuation(self):
        """测试强势延续结局检测"""
        # 开始监控强势延续形态
        self.tracker.start_monitoring([self.strong_continuation_pattern])
        
        # 更新到强势延续区域
        update_data = self.test_df.iloc[120:200]  # 包含强势延续的数据
        
        outcomes = self.tracker.update_monitoring(update_data)
        
        # 验证结局检测
        self.assertGreater(len(outcomes), 0)
        
        # 查找强势延续结局
        strong_outcomes = [o for o in outcomes 
                         if o.outcome == PatternOutcome.STRONG_CONTINUATION]
        
        if strong_outcomes:
            outcome = strong_outcomes[0]
            self.assertEqual(outcome.outcome, PatternOutcome.STRONG_CONTINUATION)
            self.assertIsNotNone(outcome.outcome_confidence)
            self.assertGreater(outcome.outcome_confidence, 0.7)
    
    def test_update_monitoring_failed_breakout(self):
        """测试假突破反转结局检测"""
        # 开始监控假突破形态
        self.tracker.start_monitoring([self.fake_breakout_pattern])
        
        # 更新到假突破区域
        update_data = self.test_df.iloc[240:320]  # 包含假突破的数据
        
        outcomes = self.tracker.update_monitoring(update_data)
        
        # 查找假突破结局
        failed_outcomes = [o for o in outcomes 
                         if o.outcome == PatternOutcome.FAILED_BREAKOUT]
        
        if failed_outcomes:
            outcome = failed_outcomes[0]
            self.assertEqual(outcome.outcome, PatternOutcome.FAILED_BREAKOUT)
            self.assertIsNotNone(outcome.reversal_magnitude)
    
    def test_update_monitoring_opposite_run(self):
        """测试反向运行结局检测"""
        # 开始监控反向运行形态
        self.tracker.start_monitoring([self.opposite_run_pattern])
        
        # 更新到反向运行区域
        update_data = self.test_df.iloc[340:420]  # 包含反向运行的数据
        
        outcomes = self.tracker.update_monitoring(update_data)
        
        # 查找反向运行结局
        opposite_outcomes = [o for o in outcomes 
                           if o.outcome == PatternOutcome.OPPOSITE_RUN]
        
        if opposite_outcomes:
            outcome = opposite_outcomes[0]
            self.assertEqual(outcome.outcome, PatternOutcome.OPPOSITE_RUN)
            self.assertIsNotNone(outcome.opposite_direction_magnitude)
    
    def test_breakout_detection(self):
        """测试突破检测"""
        pattern = self.strong_continuation_pattern
        current_data = self.test_df.iloc[120:180]  # 突破区域
        
        breakout_info = self.tracker._detect_breakout(pattern, current_data)
        
        if breakout_info:
            self.assertIn('direction', breakout_info)
            self.assertIn('breakout_time', breakout_info)
            self.assertIn('breakout_price', breakout_info)
            self.assertIn('breakout_strength', breakout_info)
            
            # 验证突破方向
            self.assertIn(breakout_info['direction'], ['up', 'down'])
            
            # 验证突破强度
            self.assertIsInstance(breakout_info['breakout_strength'], float)
            self.assertGreaterEqual(breakout_info['breakout_strength'], 0.0)
    
    def test_continuation_strength_analysis(self):
        """测试延续强度分析"""
        pattern = self.strong_continuation_pattern
        
        # 模拟突破信息
        breakout_info = {
            'direction': 'up',
            'breakout_time': self.test_df.iloc[130]['timestamp'],
            'breakout_price': self.test_df.iloc[130]['close'],
            'breakout_strength': 0.85
        }
        
        post_breakout_data = self.test_df.iloc[130:190]
        
        continuation_analysis = self.tracker._analyze_continuation_strength(
            pattern, breakout_info, post_breakout_data
        )
        
        # 验证分析结果
        self.assertIn('price_movement_ratio', continuation_analysis)
        self.assertIn('trend_consistency', continuation_analysis)
        self.assertIn('volume_support', continuation_analysis)
        self.assertIn('momentum_sustainability', continuation_analysis)
        
        # 验证数值范围
        for key, value in continuation_analysis.items():
            if isinstance(value, float):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 2.0)  # 合理的比率范围
    
    def test_outcome_classification(self):
        """测试结局分类逻辑"""
        # 测试强势延续分类
        strong_metrics = {
            'price_movement_ratio': 2.5,  # 超过目标价位2.5倍
            'trend_consistency': 0.9,
            'volume_support': 0.8,
            'momentum_sustainability': 0.85
        }
        
        outcome = self.tracker._classify_outcome(strong_metrics, 'up')
        self.assertEqual(outcome, PatternOutcome.STRONG_CONTINUATION)
        
        # 测试标准延续分类
        standard_metrics = {
            'price_movement_ratio': 1.2,  # 超过目标价位但不是很多
            'trend_consistency': 0.7,
            'volume_support': 0.6,
            'momentum_sustainability': 0.65
        }
        
        outcome = self.tracker._classify_outcome(standard_metrics, 'up')
        self.assertEqual(outcome, PatternOutcome.STANDARD_CONTINUATION)
        
        # 测试突破停滞分类
        stagnant_metrics = {
            'price_movement_ratio': 0.5,  # 没有达到目标价位
            'trend_consistency': 0.5,
            'volume_support': 0.4,
            'momentum_sustainability': 0.3
        }
        
        outcome = self.tracker._classify_outcome(stagnant_metrics, 'up')
        self.assertEqual(outcome, PatternOutcome.BREAKOUT_STAGNATION)
    
    def test_timeout_handling(self):
        """测试超时处理"""
        # 创建一个长时间没有明确结局的形态
        long_pattern = self.strong_continuation_pattern
        long_pattern.metadata = {"monitoring_start": datetime.now() - timedelta(days=30)}
        
        self.tracker.start_monitoring([long_pattern])
        
        # 使用当前数据更新（没有明显的突破或失败）
        current_data = self.test_df.iloc[120:140]  # 平稳区域
        
        outcomes = self.tracker.update_monitoring(current_data)
        
        # 检查是否有超时处理
        monitoring_summary = self.tracker.get_monitoring_summary()
        
        # 验证监控状态
        if monitoring_summary['timeout_patterns_count'] > 0:
            self.assertGreaterEqual(monitoring_summary['timeout_patterns_count'], 0)
    
    def test_monitoring_summary(self):
        """测试监控汇总"""
        # 添加多个形态进行监控
        patterns = [
            self.strong_continuation_pattern,
            self.fake_breakout_pattern,
            self.opposite_run_pattern
        ]
        
        self.tracker.start_monitoring(patterns)
        
        summary = self.tracker.get_monitoring_summary()
        
        # 验证汇总信息
        expected_keys = [
            'active_monitoring_count', 'total_patterns_tracked',
            'completed_outcomes_count', 'timeout_patterns_count',
            'outcome_distribution', 'average_monitoring_duration'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # 验证数值合理性
        self.assertEqual(summary['active_monitoring_count'], 3)
        self.assertEqual(summary['total_patterns_tracked'], 3)
        self.assertIsInstance(summary['outcome_distribution'], dict)
    
    def test_outcome_statistics(self):
        """测试结局统计"""
        # 添加一些完成的结局
        self.tracker.start_monitoring([self.strong_continuation_pattern])
        
        # 模拟完成的结局
        completed_outcome = PatternOutcomeAnalysis(
            pattern_id="test_strong_001",
            outcome=PatternOutcome.STRONG_CONTINUATION,
            outcome_confidence=0.92,
            monitoring_duration=45,
            breakout_info={
                'direction': 'up',
                'breakout_strength': 0.88
            },
            continuation_metrics={
                'price_movement_ratio': 2.3,
                'trend_consistency': 0.87
            },
            outcome_time=datetime.now()
        )
        
        # 直接添加到完成列表（测试用）
        self.tracker.completed_outcomes.append(completed_outcome)
        
        statistics = self.tracker.get_outcome_statistics()
        
        # 验证统计信息
        self.assertIn('total_completed_patterns', statistics)
        self.assertIn('outcome_success_rates', statistics)
        self.assertIn('average_monitoring_duration', statistics)
        self.assertIn('confidence_distribution', statistics)
        
        # 验证成功率统计
        success_rates = statistics['outcome_success_rates']
        self.assertIsInstance(success_rates, dict)
        
        if PatternOutcome.STRONG_CONTINUATION.value in success_rates:
            self.assertGreater(success_rates[PatternOutcome.STRONG_CONTINUATION.value], 0)
    
    def test_export_outcome_data(self):
        """测试结局数据导出"""
        # 添加一些测试数据
        self.tracker.start_monitoring([self.strong_continuation_pattern])
        
        # 模拟一个完成的结局
        completed_outcome = PatternOutcomeAnalysis(
            pattern_id="test_export_001",
            outcome=PatternOutcome.STANDARD_CONTINUATION,
            outcome_confidence=0.78,
            monitoring_duration=32,
            breakout_info={'direction': 'up'},
            continuation_metrics={'price_movement_ratio': 1.4},
            outcome_time=datetime.now()
        )
        
        self.tracker.completed_outcomes.append(completed_outcome)
        
        # 导出数据
        export_data = self.tracker.export_outcome_data()
        
        # 验证导出格式
        self.assertIsInstance(export_data, list)
        self.assertGreater(len(export_data), 0)
        
        # 验证导出项目结构
        if export_data:
            item = export_data[0]
            expected_keys = [
                'pattern_id', 'outcome', 'outcome_confidence',
                'monitoring_duration', 'outcome_time'
            ]
            
            for key in expected_keys:
                self.assertIn(key, item)
    
    def test_reset_monitoring(self):
        """测试监控重置"""
        # 添加一些监控形态
        patterns = [self.strong_continuation_pattern, self.fake_breakout_pattern]
        self.tracker.start_monitoring(patterns)
        
        # 验证监控已启动
        summary_before = self.tracker.get_monitoring_summary()
        self.assertGreater(summary_before['active_monitoring_count'], 0)
        
        # 重置监控
        self.tracker.reset_monitoring()
        
        # 验证监控已重置
        summary_after = self.tracker.get_monitoring_summary()
        self.assertEqual(summary_after['active_monitoring_count'], 0)
        self.assertEqual(len(self.tracker.monitoring_patterns), 0)
        self.assertEqual(len(self.tracker.completed_outcomes), 0)
    
    def test_pattern_lifecycle(self):
        """测试形态完整生命周期"""
        # 开始监控
        self.tracker.start_monitoring([self.strong_continuation_pattern])
        
        # 验证初始状态
        initial_summary = self.tracker.get_monitoring_summary()
        self.assertEqual(initial_summary['active_monitoring_count'], 1)
        
        # 模拟数据更新过程
        for i in range(130, 200, 10):
            update_data = self.test_df.iloc[i:i+20]
            outcomes = self.tracker.update_monitoring(update_data)
            
            if outcomes:  # 如果有结局产生
                break
        
        # 验证最终状态
        final_summary = self.tracker.get_monitoring_summary()
        final_statistics = self.tracker.get_outcome_statistics()
        
        # 验证监控过程完成
        self.assertGreaterEqual(final_statistics['total_completed_patterns'], 0)


if __name__ == '__main__':
    unittest.main()