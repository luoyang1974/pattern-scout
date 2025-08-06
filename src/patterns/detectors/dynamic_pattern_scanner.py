"""
动态形态扫描器
整合动态基线系统的完整形态识别流程
阶段0->1->2->3->4->5的统一入口（六阶段）
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from src.data.models.base_models import (
    PatternRecord, Flagpole, MarketRegime, MarketSnapshot
)
from src.patterns.base.market_regime_detector import (
    SmartRegimeDetector, DualRegimeBaselineManager
)
from src.patterns.detectors.dynamic_flagpole_detector import DynamicFlagpoleDetector
from src.patterns.detectors.dynamic_flag_detector import DynamicFlagDetector
from src.analysis.pattern_outcome_analyzer import PatternOutcomeAnalyzer
from src.storage.pattern_data_exporter import PatternDataExporter
from src.visualization.pattern_chart_generator import PatternChartGenerator
from src.patterns.base.timeframe_manager import TimeframeManager


class DynamicPatternScanner:
    """
    动态形态扫描器
    实现完整的六阶段动态基线旗形识别与数据归档流程
    阶段0: 动态基线预计算 
    阶段1: 动态旗杆识别
    阶段2: 动态旗面识别  
    阶段3: 形态结局分析
    阶段4: 形态数据输出
    阶段5: 形态可视化及图表输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动态形态扫描器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化核心组件
        self._initialize_components()
        
        # 六阶段统计信息
        self.scan_statistics = {
            'total_scans': 0,
            'regime_transitions': 0,
            'flagpoles_detected': 0,
            'patterns_detected': 0,
            'outcomes_analyzed': 0,
            'data_exports': 0,
            'charts_generated': 0,
            'patterns_monitored': 0
        }
        
    def _initialize_components(self):
        """初始化各阶段组件"""
        # 阶段0：动态基线系统
        regime_config = self.config.get('dynamic_baseline', {}).get('regime_detection', {})
        self.regime_detector = SmartRegimeDetector(
            atr_period=regime_config.get('atr_period', 100),
            high_vol_threshold=regime_config.get('high_volatility_threshold', 0.65),
            low_vol_threshold=regime_config.get('low_volatility_threshold', 0.35),
            stability_buffer=regime_config.get('stability_buffer', 5),
            history_window=self.config.get('dynamic_baseline', {}).get('history_window', 500)
        )
        
        self.baseline_manager = DualRegimeBaselineManager(
            history_window=self.config.get('dynamic_baseline', {}).get('history_window', 500)
        )
        
        # 阶段1：动态旗杆检测器
        self.flagpole_detector = DynamicFlagpoleDetector(self.baseline_manager)
        
        # 阶段2：动态旗面检测器
        self.flag_detector = DynamicFlagDetector(self.baseline_manager)
        
        # 阶段3：结局分析器
        outcome_config = self.config.get('outcome_analysis', {})
        self.outcome_analyzer = PatternOutcomeAnalyzer(outcome_config)
        
        # 阶段4：数据输出器
        data_export_config = self.config.get('data_export', {})
        output_root = data_export_config.get('output_root', 'output/data')
        self.data_exporter = PatternDataExporter(output_root)
        
        # 阶段5：图表生成器
        chart_config = self.config.get('visualization', {})
        self.chart_generator = PatternChartGenerator(chart_config)
        
        # 辅助组件
        self.timeframe_manager = TimeframeManager()
        
        logger.info("动态形态扫描器已初始化 - 六阶段系统就绪")
    
    def scan(self, df: pd.DataFrame, 
             enable_outcome_analysis: bool = True,
             enable_data_export: bool = True,
             enable_chart_generation: bool = True) -> Dict[str, Any]:
        """
        执行完整的六阶段动态基线形态识别与数据归档扫描
        
        Args:
            df: OHLCV数据框
            enable_outcome_analysis: 是否启用结局分析（阶段3）
            enable_data_export: 是否启用数据输出（阶段4）
            enable_chart_generation: 是否启用图表生成（阶段5）
            
        Returns:
            扫描结果字典
        """
        self.scan_statistics['total_scans'] += 1
        scan_start_time = datetime.now()
        
        logger.info(f"Starting dynamic pattern scan on {len(df)} data points")
        
        # 数据验证
        if not self._validate_input_data(df):
            return self._create_error_result("Invalid input data")
        
        # 检测时间周期
        timeframe = self.timeframe_manager.detect_timeframe(df)
        logger.info(f"Detected timeframe: {timeframe}")
        
        try:
            # 阶段0：更新市场状态和动态基线
            current_regime = self._update_market_regime(df)
            
            # 创建市场快照
            market_snapshot = self.regime_detector.create_market_snapshot(df)
            
            # 阶段1：动态旗杆检测
            flagpoles = self._detect_flagpoles(df, current_regime, timeframe)
            
            # 阶段2：动态旗面检测
            patterns = self._detect_flag_patterns(df, flagpoles, current_regime, timeframe)
            
            # 阶段3：结局分析（可选）
            outcome_analyses = []
            if enable_outcome_analysis:
                outcome_analyses = self._analyze_pattern_outcomes(df, patterns)
            
            # 阶段4：数据输出（可选）
            data_export_results = {}
            if enable_data_export:
                data_export_results = self._export_pattern_data(patterns, market_snapshot, outcome_analyses)
            
            # 阶段5：图表生成（可选）
            chart_generation_results = {}
            if enable_chart_generation:
                chart_generation_results = self._generate_pattern_charts(df, patterns, outcome_analyses, market_snapshot)
            
            # 统计更新
            self.scan_statistics['flagpoles_detected'] += len(flagpoles)
            self.scan_statistics['patterns_detected'] += len(patterns)
            self.scan_statistics['outcomes_analyzed'] = len(outcome_analyses)
            self.scan_statistics['data_exports'] = len(data_export_results)
            self.scan_statistics['charts_generated'] = len(chart_generation_results)
            
            # 创建扫描结果
            scan_result = {
                'success': True,
                'scan_time': (datetime.now() - scan_start_time).total_seconds(),
                'timeframe': timeframe,
                'market_regime': current_regime.value,
                'market_snapshot': market_snapshot.to_dict(),
                'flagpoles_detected': len(flagpoles),
                'patterns_detected': len(patterns),
                'patterns': [p.to_dict() for p in patterns],
                'flagpoles': [fp.to_dict() for fp in flagpoles],
                'outcome_analyses': len(outcome_analyses),
                'outcome_analysis_results': [oa.to_dict() for oa in outcome_analyses],
                'data_export_summary': data_export_results,
                'chart_generation_summary': chart_generation_results,
                'regime_metrics': self.regime_detector.get_volatility_metrics(),
                'baseline_summary': self.baseline_manager.get_baseline_summary(),
                'scan_statistics': self.scan_statistics.copy()
            }
            
            logger.info(f"Scan completed: {len(patterns)} patterns detected in {scan_result['scan_time']:.2f}s")
            return scan_result
            
        except Exception as e:
            logger.error(f"Error during pattern scan: {str(e)}")
            return self._create_error_result(f"Scan error: {str(e)}")
    
    def _validate_input_data(self, df: pd.DataFrame) -> bool:
        """验证输入数据"""
        validation_config = self.config.get('validation', {})
        
        # 基本结构检查
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
        
        # 数据量检查
        min_data_points = validation_config.get('min_data_points', 100)
        if len(df) < min_data_points:
            logger.warning(f"Insufficient data points: {len(df)} < {min_data_points}")
            return False
        
        # 数据合理性检查
        if validation_config.get('enable_sanity_checks', True):
            if not self._perform_sanity_checks(df):
                return False
        
        return True
    
    def _perform_sanity_checks(self, df: pd.DataFrame) -> bool:
        """执行数据合理性检查"""
        try:
            # 价格合理性检查
            for _, row in df.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and
                       row['low'] <= row['close'] <= row['high']):
                    logger.error(f"Invalid OHLC data at {row['timestamp']}")
                    return False
            
            # 时间序列检查
            timestamps = pd.to_datetime(df['timestamp'])
            if not timestamps.is_monotonic_increasing:
                logger.error("Timestamps are not in ascending order")
                return False
            
            # 价格连续性检查（简单）
            price_changes = df['close'].pct_change().dropna()
            if (abs(price_changes) > 0.2).any():  # 20%单日变化预警
                logger.warning("Detected large price changes (>20%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Sanity check error: {str(e)}")
            return False
    
    def _update_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """更新市场状态和动态基线"""
        # 更新市场状态
        previous_regime = self.regime_detector.get_current_regime()
        current_regime = self.regime_detector.update(df)
        
        # 记录状态转换
        if previous_regime != current_regime and previous_regime != MarketRegime.UNKNOWN:
            self.scan_statistics['regime_transitions'] += 1
            logger.info(f"Market regime transition: {previous_regime.value} -> {current_regime.value}")
        
        # 更新基线管理器的活跃状态
        self.baseline_manager.set_active_regime(current_regime)
        
        return current_regime
    
    def _detect_flagpoles(self, df: pd.DataFrame, 
                         current_regime: MarketRegime,
                         timeframe: str) -> List[Flagpole]:
        """执行动态旗杆检测"""
        logger.debug(f"Starting flagpole detection in {current_regime.value} regime")
        
        flagpoles = self.flagpole_detector.detect_flagpoles(
            df, current_regime, timeframe
        )
        
        logger.info(f"Flagpole detection completed: {len(flagpoles)} flagpoles found")
        return flagpoles
    
    def _detect_flag_patterns(self, df: pd.DataFrame,
                            flagpoles: List[Flagpole],
                            current_regime: MarketRegime,
                            timeframe: str) -> List[PatternRecord]:
        """执行动态旗面检测"""
        if not flagpoles:
            logger.debug("No flagpoles available for pattern detection")
            return []
        
        logger.debug(f"Starting flag pattern detection for {len(flagpoles)} flagpoles")
        
        patterns = self.flag_detector.detect_flag_patterns(
            df, flagpoles, current_regime, timeframe
        )
        
        # 质量过滤
        filtered_patterns = self._apply_quality_filters(patterns)
        
        logger.info(f"Flag pattern detection completed: {len(filtered_patterns)} patterns found "
                   f"(filtered from {len(patterns)})")
        
        return filtered_patterns
    
    def _apply_quality_filters(self, patterns: List[PatternRecord]) -> List[PatternRecord]:
        """应用质量过滤器"""
        quality_config = self.config.get('validation', {}).get('quality_filters', {})
        
        if not quality_config.get('enable_invalidation_filter', True):
            return patterns
        
        filtered_patterns = []
        min_confidence = quality_config.get('min_confidence_score', 0.6)
        min_quality = quality_config.get('min_pattern_quality', 'medium')
        
        quality_levels = {'low': 0, 'medium': 1, 'high': 2}
        min_quality_level = quality_levels.get(min_quality, 1)
        
        for pattern in patterns:
            # 置信度过滤
            if pattern.confidence_score < min_confidence:
                continue
            
            # 质量等级过滤
            pattern_quality_level = quality_levels.get(pattern.pattern_quality, 0)
            if pattern_quality_level < min_quality_level:
                continue
            
            # 关键失效信号过滤
            if pattern.has_critical_invalidation_signals():
                logger.debug(f"Pattern {pattern.id} filtered due to critical invalidation signals")
                continue
            
            filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _analyze_pattern_outcomes(self, df: pd.DataFrame, 
                                patterns: List[PatternRecord]) -> List:
        """执行阶段3：形态结局分析"""
        if not patterns:
            return []
        
        logger.debug(f"开始阶段3：结局分析 - {len(patterns)}个形态")
        
        # 开始监控新形态
        self.outcome_analyzer.start_monitoring(patterns)
        
        # 更新现有监控并获取完成的分析
        completed_analyses = self.outcome_analyzer.update_monitoring(df)
        
        logger.info(f"阶段3完成：{len(completed_analyses)}个结局分析")
        return completed_analyses
    
    def _export_pattern_data(self, patterns: List[PatternRecord],
                           market_snapshot: Optional[MarketSnapshot],
                           outcome_analyses: List) -> Dict[str, Any]:
        """执行阶段4：形态数据输出"""
        if not patterns:
            return {'patterns_exported': 0, 'outcomes_exported': 0}
        
        logger.debug(f"开始阶段4：数据输出 - {len(patterns)}个形态")
        
        # 批量保存形态记录（包含市场快照）
        snapshots = [market_snapshot] * len(patterns) if market_snapshot else None
        pattern_export_result = self.data_exporter.batch_save_patterns(patterns, snapshots)
        
        # 批量保存结局分析
        outcome_export_result = {}
        if outcome_analyses:
            outcome_export_result = self.data_exporter.batch_save_outcomes(outcome_analyses)
        
        export_summary = {
            'patterns_exported': pattern_export_result.get('success', 0),
            'pattern_export_errors': pattern_export_result.get('errors', 0),
            'outcomes_exported': outcome_export_result.get('success', 0),
            'outcome_export_errors': outcome_export_result.get('errors', 0),
            'export_statistics': self.data_exporter.get_export_statistics()
        }
        
        logger.info(f"阶段4完成：{export_summary['patterns_exported']}个形态数据已导出")
        return export_summary
    
    def _generate_pattern_charts(self, df: pd.DataFrame,
                               patterns: List[PatternRecord],
                               outcome_analyses: List,
                               market_snapshot: Optional[MarketSnapshot]) -> Dict[str, Any]:
        """执行阶段5：形态可视化及图表输出"""
        if not patterns:
            return {'charts_generated': 0}
        
        logger.debug(f"开始阶段5：图表生成 - {len(patterns)}个形态")
        
        # 准备数据映射（这里简化处理，实际可能需要多个symbol的数据）
        df_dict = {}
        for pattern in patterns:
            df_dict[pattern.symbol] = df
        
        # 准备快照列表
        snapshots = [market_snapshot] * len(patterns) if market_snapshot else None
        
        # 批量生成图表
        chart_paths = self.chart_generator.batch_generate_charts(
            patterns, df_dict, outcome_analyses, snapshots
        )
        
        # 更新形态记录中的图表路径
        for pattern in patterns:
            if pattern.id in chart_paths:
                pattern.chart_path = chart_paths[pattern.id]
        
        chart_summary = {
            'charts_generated': len(chart_paths),
            'chart_paths': chart_paths,
            'summary_chart': chart_paths.get('summary', '')
        }
        
        logger.info(f"阶段5完成：{chart_summary['charts_generated']}个图表已生成")
        return chart_summary
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'success': False,
            'error': error_message,
            'patterns_detected': 0,
            'patterns': [],
            'flagpoles_detected': 0,
            'flagpoles': [],
            'scan_statistics': self.scan_statistics.copy()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'regime_detector': {
                'current_regime': self.regime_detector.get_current_regime().value,
                'regime_confidence': self.regime_detector.get_regime_confidence(),
                'is_stable': self.regime_detector.is_regime_stable(),
                'metrics': self.regime_detector.get_volatility_metrics()
            },
            'baseline_manager': {
                'active_regime': self.baseline_manager.active_regime.value,
                'summary': self.baseline_manager.get_baseline_summary()
            },
            'outcome_analyzer': {
                'monitoring_summary': self.outcome_analyzer.get_monitoring_summary(),
                'outcome_statistics': self.outcome_analyzer.get_outcome_statistics()
            },
            'data_exporter': {
                'export_statistics': self.data_exporter.get_export_statistics()
            },
            'scan_statistics': self.scan_statistics.copy()
        }
    
    def export_baseline_data(self) -> Dict[str, Any]:
        """导出基线数据"""
        return {
            'baselines': self.baseline_manager.export_baselines(),
            'regime_transitions': [
                {
                    'from': t.from_regime.value,
                    'to': t.to_regime.value,
                    'time': t.transition_time.isoformat(),
                    'confidence': t.confidence,
                    'trigger_value': t.trigger_value
                }
                for t in self.regime_detector.regime_transitions
            ],
            'export_time': datetime.now().isoformat()
        }
    
    def export_outcome_data(self) -> List[Dict[str, Any]]:
        """导出结局数据"""
        return self.outcome_analyzer.export_outcome_data()
    
    def export_comprehensive_dataset(self, output_path: Optional[str] = None) -> str:
        """导出综合数据集"""
        return self.data_exporter.export_comprehensive_dataset(output_path)
    
    def reset_system(self):
        """重置整个六阶段系统"""
        logger.info("重置动态形态扫描器 - 六阶段系统")
        
        # 重置各阶段组件
        # 阶段0: 动态基线系统
        self.regime_detector.reset()
        self.baseline_manager = DualRegimeBaselineManager(
            history_window=self.config.get('dynamic_baseline', {}).get('history_window', 500)
        )
        
        # 阶段3: 结局分析器
        self.outcome_analyzer.reset_monitoring()
        self.outcome_analyzer.clear_completed_analyses()
        
        # 阶段4: 数据输出器
        self.data_exporter.clear_cache()
        
        # 重置六阶段统计
        self.scan_statistics = {
            'total_scans': 0,
            'regime_transitions': 0,
            'flagpoles_detected': 0,
            'patterns_detected': 0,
            'outcomes_analyzed': 0,
            'data_exports': 0,
            'charts_generated': 0,
            'patterns_monitored': 0
        }
        
        logger.info("六阶段系统重置完成")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取六阶段系统性能指标"""
        stats = self.scan_statistics.copy()
        
        # 计算衍生指标
        if stats['total_scans'] > 0:
            stats['avg_flagpoles_per_scan'] = stats['flagpoles_detected'] / stats['total_scans']
            stats['avg_patterns_per_scan'] = stats['patterns_detected'] / stats['total_scans']
            stats['avg_outcomes_per_scan'] = stats['outcomes_analyzed'] / stats['total_scans']
            stats['avg_exports_per_scan'] = stats['data_exports'] / stats['total_scans']
            stats['avg_charts_per_scan'] = stats['charts_generated'] / stats['total_scans']
        else:
            stats['avg_flagpoles_per_scan'] = 0
            stats['avg_patterns_per_scan'] = 0
            stats['avg_outcomes_per_scan'] = 0
            stats['avg_exports_per_scan'] = 0
            stats['avg_charts_per_scan'] = 0
        
        # 转换率指标
        if stats['flagpoles_detected'] > 0:
            stats['pattern_conversion_rate'] = stats['patterns_detected'] / stats['flagpoles_detected']
        else:
            stats['pattern_conversion_rate'] = 0
        
        if stats['patterns_detected'] > 0:
            stats['outcome_analysis_rate'] = stats['outcomes_analyzed'] / stats['patterns_detected']
            stats['data_export_rate'] = stats['data_exports'] / stats['patterns_detected']  
            stats['chart_generation_rate'] = stats['charts_generated'] / stats['patterns_detected']
        else:
            stats['outcome_analysis_rate'] = 0
            stats['data_export_rate'] = 0
            stats['chart_generation_rate'] = 0
        
        # 系统健康指标
        stats['regime_stability'] = self.regime_detector.get_regime_confidence()
        stats['baseline_coverage'] = len(self.baseline_manager.baselines[MarketRegime.HIGH_VOLATILITY]) + \
                                   len(self.baseline_manager.baselines[MarketRegime.LOW_VOLATILITY])
        
        # 六阶段完整性指标
        stats['six_stage_completeness'] = {
            'stage_0_ready': self.regime_detector.is_regime_stable(),
            'stage_1_active': hasattr(self, 'flagpole_detector'),
            'stage_2_active': hasattr(self, 'flag_detector'),
            'stage_3_active': hasattr(self, 'outcome_analyzer'),
            'stage_4_active': hasattr(self, 'data_exporter'),
            'stage_5_active': hasattr(self, 'chart_generator')
        }
        
        return stats