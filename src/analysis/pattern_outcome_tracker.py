"""
形态结局追踪系统（阶段3）
实现6种结局分类和数据归档功能
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from src.data.models.base_models import (
    PatternRecord, PatternOutcome, PatternOutcomeAnalysis
)


class PatternOutcomeTracker:
    """
    形态结局追踪器
    监控已确认形态的后续走势并分类结局
    """
    
    def __init__(self):
        """初始化结局追踪器"""
        self.monitoring_patterns: Dict[str, PatternRecord] = {}
        self.completed_analyses: Dict[str, PatternOutcomeAnalysis] = {}
        
    def start_monitoring(self, pattern_records: List[PatternRecord]):
        """
        开始监控形态列表
        
        Args:
            pattern_records: 要监控的形态记录列表
        """
        for pattern in pattern_records:
            if pattern.id not in self.monitoring_patterns:
                pattern.is_monitoring = True
                self.monitoring_patterns[pattern.id] = pattern
                
                logger.info(f"Started monitoring pattern {pattern.id} ({pattern.sub_type})")
    
    def update_monitoring(self, df: pd.DataFrame) -> List[PatternOutcomeAnalysis]:
        """
        更新监控状态并分析结局
        
        Args:
            df: 最新的OHLCV数据
            
        Returns:
            已完成分析的结局列表
        """
        if not self.monitoring_patterns:
            return []
        
        current_time = df.iloc[-1]['timestamp']
        completed_analyses = []
        patterns_to_remove = []
        
        for pattern_id, pattern in self.monitoring_patterns.items():
            # 检查监控窗口是否超时
            if self._is_monitoring_timeout(pattern, current_time):
                # 超时分析
                analysis = self._analyze_timeout_outcome(pattern, df)
                completed_analyses.append(analysis)
                patterns_to_remove.append(pattern_id)
                
                logger.info(f"Pattern {pattern_id} monitoring timeout, "
                          f"outcome: {analysis.outcome.value}")
                continue
            
            # 分析当前状态
            analysis = self._analyze_current_outcome(pattern, df)
            
            if analysis and analysis.outcome != PatternOutcome.MONITORING:
                # 结局已确定
                completed_analyses.append(analysis)
                patterns_to_remove.append(pattern_id)
                
                logger.info(f"Pattern {pattern_id} outcome determined: {analysis.outcome.value}")
        
        # 移除已完成的监控
        for pattern_id in patterns_to_remove:
            pattern = self.monitoring_patterns.pop(pattern_id, None)
            if pattern:
                pattern.is_monitoring = False
        
        # 存储已完成的分析
        for analysis in completed_analyses:
            self.completed_analyses[analysis.pattern_id] = analysis
        
        return completed_analyses
    
    def _is_monitoring_timeout(self, pattern: PatternRecord, current_time: datetime) -> bool:
        """
        检查监控是否超时
        监控窗口 = Flag_Duration * 8 根K线
        """
        # 从形态确认时开始计算
        pattern_end_time = pattern.pattern_boundaries[-1].end_time
        
        # 计算超时时间（这里简化为8倍形态持续时间）
        timeout_duration = pattern.pattern_duration * 8
        
        # 假设每个K线15分钟（需要根据实际时间周期调整）
        timeout_time = pattern_end_time + timedelta(minutes=15 * timeout_duration)
        
        return current_time >= timeout_time
    
    def _analyze_current_outcome(self, pattern: PatternRecord, df: pd.DataFrame) -> Optional[PatternOutcomeAnalysis]:
        """
        分析当前形态结局
        
        Args:
            pattern: 形态记录
            df: 价格数据
            
        Returns:
            结局分析或None（如果仍在监控中）
        """
        # 获取关键水平定义
        key_levels = self._define_key_levels(pattern)
        
        # 获取监控期间的价格数据
        monitoring_data = self._get_monitoring_data(pattern, df)
        
        if monitoring_data.empty:
            return None
        
        # 分析价格走势
        outcome, analysis_details = self._classify_outcome(
            pattern, key_levels, monitoring_data
        )
        
        if outcome == PatternOutcome.MONITORING:
            return None  # 仍在监控中
        
        # 创建结局分析
        analysis = PatternOutcomeAnalysis(
            pattern_id=pattern.id,
            outcome=outcome,
            analysis_date=datetime.now(),
            monitoring_duration=len(monitoring_data),
            breakout_level=key_levels['breakout_level'],
            invalidation_level=key_levels['invalidation_level'],
            target_projection_1=key_levels['target_projection_1'],
            risk_distance=key_levels['risk_distance'],
            actual_high=monitoring_data['high'].max(),
            actual_low=monitoring_data['low'].min(),
            breakthrough_occurred=analysis_details['breakthrough_occurred'],
            breakthrough_direction=analysis_details['breakthrough_direction'],
            success_ratio=analysis_details.get('success_ratio'),
            holding_period=analysis_details.get('holding_period'),
            final_return=analysis_details.get('final_return')
        )
        
        return analysis
    
    def _analyze_timeout_outcome(self, pattern: PatternRecord, df: pd.DataFrame) -> PatternOutcomeAnalysis:
        """分析超时情况的结局"""
        key_levels = self._define_key_levels(pattern)
        monitoring_data = self._get_monitoring_data(pattern, df)
        
        if monitoring_data.empty:
            outcome = PatternOutcome.UNKNOWN
            analysis_details = {
                'breakthrough_occurred': False,
                'breakthrough_direction': None
            }
        else:
            # 检查是否发生了突破但没有明确结局
            current_price = monitoring_data.iloc[-1]['close']
            
            if (current_price > key_levels['breakout_level'] and pattern.flagpole.direction == 'up') or \
               (current_price < key_levels['breakout_level'] and pattern.flagpole.direction == 'down'):
                # 发生突破但停滞
                outcome = PatternOutcome.BREAKOUT_STAGNATION
                analysis_details = {
                    'breakthrough_occurred': True,
                    'breakthrough_direction': pattern.flagpole.direction
                }
            else:
                # 内部瓦解
                outcome = PatternOutcome.INTERNAL_COLLAPSE
                analysis_details = {
                    'breakthrough_occurred': False,
                    'breakthrough_direction': None
                }
        
        return PatternOutcomeAnalysis(
            pattern_id=pattern.id,
            outcome=outcome,
            analysis_date=datetime.now(),
            monitoring_duration=len(monitoring_data) if not monitoring_data.empty else 0,
            breakout_level=key_levels['breakout_level'],
            invalidation_level=key_levels['invalidation_level'],
            target_projection_1=key_levels['target_projection_1'],
            risk_distance=key_levels['risk_distance'],
            actual_high=monitoring_data['high'].max() if not monitoring_data.empty else np.nan,
            actual_low=monitoring_data['low'].min() if not monitoring_data.empty else np.nan,
            breakthrough_occurred=analysis_details['breakthrough_occurred'],
            breakthrough_direction=analysis_details['breakthrough_direction'],
            success_ratio=None,
            holding_period=len(monitoring_data) if not monitoring_data.empty else 0,
            final_return=None
        )
    
    def _define_key_levels(self, pattern: PatternRecord) -> Dict[str, float]:
        """
        定义关键水平
        
        Args:
            pattern: 形态记录
            
        Returns:
            关键水平字典
        """
        if not pattern.pattern_boundaries:
            raise ValueError("Pattern has no boundaries")
        
        upper_boundary = pattern.pattern_boundaries[0]
        lower_boundary = pattern.pattern_boundaries[-1] if len(pattern.pattern_boundaries) > 1 else upper_boundary
        
        # 突破监测位（通道上下轨）
        if pattern.flagpole.direction == 'up':
            breakout_level = upper_boundary.end_price
            invalidation_level = lower_boundary.end_price  # 旗面最低点
        else:
            breakout_level = lower_boundary.end_price
            invalidation_level = upper_boundary.end_price  # 旗面最高点
        
        # 目标映射一（一个旗杆等距）
        pole_amplitude = pattern.flagpole.height
        if pattern.flagpole.direction == 'up':
            target_projection_1 = breakout_level + pole_amplitude
        else:
            target_projection_1 = breakout_level - pole_amplitude
        
        # 风险映射距离
        risk_distance = abs(invalidation_level - breakout_level)
        
        return {
            'breakout_level': breakout_level,
            'invalidation_level': invalidation_level,
            'target_projection_1': target_projection_1,
            'risk_distance': risk_distance
        }
    
    def _get_monitoring_data(self, pattern: PatternRecord, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取监控期间的价格数据
        
        Args:
            pattern: 形态记录
            df: 价格数据
            
        Returns:
            监控期间的数据
        """
        # 从形态结束时间开始监控
        pattern_end_time = pattern.pattern_boundaries[-1].end_time
        
        # 获取监控期间的数据
        monitoring_data = df[df['timestamp'] > pattern_end_time].copy()
        
        return monitoring_data
    
    def _classify_outcome(self, pattern: PatternRecord, 
                         key_levels: Dict[str, float],
                         monitoring_data: pd.DataFrame) -> Tuple[PatternOutcome, Dict[str, Any]]:
        """
        根据价格走势分类结局
        
        Args:
            pattern: 形态记录
            key_levels: 关键水平
            monitoring_data: 监控数据
            
        Returns:
            (结局分类, 分析详情)
        """
        if monitoring_data.empty:
            return PatternOutcome.MONITORING, {'breakthrough_occurred': False, 'breakthrough_direction': None}
        
        breakout_level = key_levels['breakout_level']
        invalidation_level = key_levels['invalidation_level']
        target_projection_1 = key_levels['target_projection_1']
        risk_distance = key_levels['risk_distance']
        
        direction = pattern.flagpole.direction
        current_price = monitoring_data.iloc[-1]['close']
        high_price = monitoring_data['high'].max()
        low_price = monitoring_data['low'].min()
        
        # 检查是否触及失效位
        if direction == 'up':
            invalidation_hit = low_price <= invalidation_level
        else:
            invalidation_hit = high_price >= invalidation_level
        
        # 检查是否发生突破
        if direction == 'up':
            breakout_occurred = high_price > breakout_level
            opposite_breakout = low_price < breakout_level
        else:
            breakout_occurred = low_price < breakout_level
            opposite_breakout = high_price > breakout_level
        
        analysis_details = {
            'breakthrough_occurred': breakout_occurred,
            'breakthrough_direction': direction if breakout_occurred else None
        }
        
        # 分类逻辑
        if invalidation_hit and not breakout_occurred:
            # 5. 内部瓦解
            return PatternOutcome.INTERNAL_COLLAPSE, analysis_details
        
        elif opposite_breakout:
            # 检查反向运行距离
            if direction == 'up':
                opposite_distance = breakout_level - low_price
            else:
                opposite_distance = high_price - breakout_level
            
            if opposite_distance > risk_distance * 1.5:
                # 6. 反向运行
                analysis_details['breakthrough_direction'] = 'down' if direction == 'up' else 'up'
                return PatternOutcome.OPPOSITE_RUN, analysis_details
        
        elif breakout_occurred:
            # 检查突破后的表现
            if direction == 'up':
                max_extension = high_price - breakout_level
                target_distance = target_projection_1 - breakout_level
            else:
                max_extension = breakout_level - low_price
                target_distance = breakout_level - target_projection_1
            
            # 检查是否到达目标
            if max_extension >= target_distance:
                if not invalidation_hit:
                    # 1. 强势延续
                    analysis_details['success_ratio'] = 1.0
                    analysis_details['final_return'] = max_extension / risk_distance if risk_distance > 0 else np.nan
                    return PatternOutcome.STRONG_CONTINUATION, analysis_details
            
            # 检查是否超过风险距离的1.5倍
            elif max_extension > risk_distance * 1.5:
                if not invalidation_hit:
                    # 2. 标准延续
                    analysis_details['success_ratio'] = max_extension / target_distance if target_distance > 0 else np.nan
                    analysis_details['final_return'] = max_extension / risk_distance if risk_distance > 0 else np.nan
                    return PatternOutcome.STANDARD_CONTINUATION, analysis_details
            
            # 检查是否触及失效位
            elif invalidation_hit:
                # 4. 假突破反转
                return PatternOutcome.FAILED_BREAKOUT, analysis_details
            
            else:
                # 检查是否在窄幅震荡
                price_range = monitoring_data['high'].max() - monitoring_data['low'].min()
                if price_range < risk_distance * 0.5:
                    # 3. 突破停滞
                    return PatternOutcome.BREAKOUT_STAGNATION, analysis_details
        
        # 默认情况：仍在监控中
        return PatternOutcome.MONITORING, analysis_details
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            'monitoring_count': len(self.monitoring_patterns),
            'completed_count': len(self.completed_analyses),
            'monitoring_patterns': [
                {
                    'id': p.id,
                    'symbol': p.symbol,
                    'sub_type': p.sub_type,
                    'detection_date': p.detection_date.isoformat(),
                    'confidence_score': p.confidence_score
                }
                for p in self.monitoring_patterns.values()
            ]
        }
    
    def get_outcome_statistics(self) -> Dict[str, Any]:
        """获取结局统计信息"""
        if not self.completed_analyses:
            return {
                'total_analyzed': 0,
                'outcome_distribution': {},
                'success_rate': 0.0,
                'average_holding_period': 0.0
            }
        
        analyses = list(self.completed_analyses.values())
        total_count = len(analyses)
        
        # 结局分布统计
        outcome_counts = {}
        for analysis in analyses:
            outcome = analysis.outcome.value
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        outcome_distribution = {
            outcome: count / total_count
            for outcome, count in outcome_counts.items()
        }
        
        # 成功率计算（强势延续 + 标准延续）
        success_outcomes = [
            PatternOutcome.STRONG_CONTINUATION.value,
            PatternOutcome.STANDARD_CONTINUATION.value
        ]
        success_count = sum(outcome_counts.get(outcome, 0) for outcome in success_outcomes)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # 平均持有周期
        holding_periods = [a.monitoring_duration for a in analyses if a.monitoring_duration]
        average_holding_period = np.mean(holding_periods) if holding_periods else 0.0
        
        # 平均回报（仅成功案例）
        successful_analyses = [
            a for a in analyses 
            if a.outcome.value in success_outcomes and a.final_return is not None
        ]
        average_return = np.mean([a.final_return for a in successful_analyses]) if successful_analyses else 0.0
        
        return {
            'total_analyzed': total_count,
            'outcome_distribution': outcome_distribution,
            'success_rate': success_rate,
            'average_holding_period': average_holding_period,
            'average_return': average_return,
            'outcome_counts': outcome_counts
        }
    
    def export_outcome_data(self) -> List[Dict[str, Any]]:
        """导出结局数据"""
        return [analysis.to_dict() for analysis in self.completed_analyses.values()]
    
    def get_pattern_analysis(self, pattern_id: str) -> Optional[PatternOutcomeAnalysis]:
        """获取特定形态的分析结果"""
        return self.completed_analyses.get(pattern_id)
    
    def reset_monitoring(self):
        """重置监控状态"""
        # 停止所有监控
        for pattern in self.monitoring_patterns.values():
            pattern.is_monitoring = False
        
        self.monitoring_patterns.clear()
        logger.info("Pattern outcome monitoring reset")
    
    def clear_completed_analyses(self):
        """清空已完成的分析"""
        self.completed_analyses.clear()
        logger.info("Completed pattern analyses cleared")


class OutcomeAnalyzer:
    """结局分析器 - 提供高级分析功能"""
    
    def __init__(self, outcome_tracker: PatternOutcomeTracker):
        """初始化分析器"""
        self.outcome_tracker = outcome_tracker
    
    def analyze_pattern_performance(self, 
                                  filter_by: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析形态表现
        
        Args:
            filter_by: 过滤条件
            
        Returns:
            性能分析结果
        """
        analyses = list(self.outcome_tracker.completed_analyses.values())
        
        if filter_by:
            analyses = self._filter_analyses(analyses, filter_by)
        
        if not analyses:
            return {'message': 'No data available for analysis'}
        
        return {
            'total_patterns': len(analyses),
            'outcome_breakdown': self._analyze_outcome_breakdown(analyses),
            'success_metrics': self._calculate_success_metrics(analyses),
            'risk_return_profile': self._analyze_risk_return(analyses),
            'timing_analysis': self._analyze_timing_patterns(analyses)
        }
    
    def _filter_analyses(self, analyses: List[PatternOutcomeAnalysis], 
                        filter_by: Dict[str, Any]) -> List[PatternOutcomeAnalysis]:
        """根据条件过滤分析结果"""
        filtered = analyses
        
        # 可以根据需要添加过滤逻辑
        # 例如：按时间范围、结局类型、成功率等过滤
        
        return filtered
    
    def _analyze_outcome_breakdown(self, analyses: List[PatternOutcomeAnalysis]) -> Dict[str, Any]:
        """分析结局分布"""
        outcome_counts = {}
        for analysis in analyses:
            outcome = analysis.outcome.value
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        total = len(analyses)
        breakdown = {
            'counts': outcome_counts,
            'percentages': {k: v/total*100 for k, v in outcome_counts.items()}
        }
        
        return breakdown
    
    def _calculate_success_metrics(self, analyses: List[PatternOutcomeAnalysis]) -> Dict[str, float]:
        """计算成功指标"""
        success_outcomes = [
            PatternOutcome.STRONG_CONTINUATION.value,
            PatternOutcome.STANDARD_CONTINUATION.value
        ]
        
        total = len(analyses)
        successful = [a for a in analyses if a.outcome.value in success_outcomes]
        
        success_rate = len(successful) / total if total > 0 else 0.0
        
        # 平均成功回报
        avg_return = np.mean([a.final_return for a in successful if a.final_return is not None])
        avg_return = avg_return if not np.isnan(avg_return) else 0.0
        
        return {
            'success_rate': success_rate,
            'average_successful_return': avg_return,
            'strong_continuation_rate': len([a for a in successful if a.outcome == PatternOutcome.STRONG_CONTINUATION]) / total,
            'standard_continuation_rate': len([a for a in successful if a.outcome == PatternOutcome.STANDARD_CONTINUATION]) / total
        }
    
    def _analyze_risk_return(self, analyses: List[PatternOutcomeAnalysis]) -> Dict[str, float]:
        """分析风险回报特征"""
        returns = [a.final_return for a in analyses if a.final_return is not None]
        
        if not returns:
            return {'message': 'No return data available'}
        
        return {
            'average_return': np.mean(returns),
            'median_return': np.median(returns),
            'return_std': np.std(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        }
    
    def _analyze_timing_patterns(self, analyses: List[PatternOutcomeAnalysis]) -> Dict[str, float]:
        """分析时间模式"""
        holding_periods = [a.monitoring_duration for a in analyses if a.monitoring_duration > 0]
        
        if not holding_periods:
            return {'message': 'No timing data available'}
        
        return {
            'average_holding_period': np.mean(holding_periods),
            'median_holding_period': np.median(holding_periods),
            'max_holding_period': np.max(holding_periods),
            'min_holding_period': np.min(holding_periods)
        }