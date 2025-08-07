"""
市场状态检测器（增强版）
基于ATR的波动状态机，支持智能状态切换和动态基线管理
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from src.data.models.base_models import (
    MarketRegime, IndicatorType, DynamicBaseline, MarketSnapshot
)
from src.patterns.indicators.technical_indicators import TechnicalIndicators
from src.patterns.base.robust_statistics import RobustStatistics


@dataclass
class RegimeTransition:
    """状态转换记录"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    confidence: float
    trigger_value: float
    

class RegimeDetector:
    """市场状态检测器（防震荡版）"""
    
    def __init__(self, 
                 atr_period: int = 100,
                 high_vol_threshold: float = 0.65,
                 low_vol_threshold: float = 0.35,
                 history_window: int = 500,
                 stability_buffer: int = 5):
        """
        初始化智能市场状态检测器
        
        Args:
            atr_period: ATR计算周期
            high_vol_threshold: 高波动阈值(百分位数)
            low_vol_threshold: 低波动阈值(百分位数)  
            history_window: 历史数据窗口大小
            stability_buffer: 稳定缓冲期（需要连续确认次数）
        """
        self.atr_period = atr_period
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.history_window = history_window
        self.stability_buffer = stability_buffer
        
        # ATR历史数据缓冲区
        self.atr_history: List[float] = []
        self.current_regime = MarketRegime.UNKNOWN
        
        # 防震荡机制
        self.pending_regime: Optional[MarketRegime] = None
        self.confirmation_count = 0
        self.regime_transitions: List[RegimeTransition] = []
        
        # 技术指标和统计工具
        self.tech_indicators = TechnicalIndicators()
        self.robust_stats = RobustStatistics()
        
        # 状态稳定性追踪
        self.regime_stability_score = 1.0
        self.last_transition_time: Optional[datetime] = None

    def update(self, df: pd.DataFrame) -> MarketRegime:
        """
        更新市场状态（智能版本）
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            当前市场状态
        """
        if len(df) < self.atr_period:
            logger.warning(f"Insufficient data for ATR calculation: {len(df)} < {self.atr_period}")
            return MarketRegime.UNKNOWN
        
        # 计算当前ATR
        current_time = df.iloc[-1]['timestamp']
        atr_values = self.tech_indicators.calculate_atr(df, period=self.atr_period)
        current_atr = atr_values.iloc[-1]
        
        # 更新ATR历史（使用稳健化方法）
        self.atr_history.append(current_atr)
        if len(self.atr_history) > self.history_window:
            self.atr_history = self.atr_history[-self.history_window:]
        
        # 需要足够的历史数据进行分析
        min_atr_history = min(50, max(20, len(df) // 5))  # 自适应最小ATR历史要求
        if len(self.atr_history) < min_atr_history:
            logger.debug(f"Insufficient ATR history: {len(self.atr_history)} < {min_atr_history}")
            logger.debug(f"Consider using more data for better regime detection")
            return MarketRegime.UNKNOWN
        
        # 使用稳健化方法计算阈值
        atr_series = pd.Series(self.atr_history)
        robust_percentiles = self.robust_stats.robust_percentiles(
            atr_series, 
            percentiles=[35, 65]  # 对应low_vol和high_vol阈值
        )
        
        high_vol_threshold_value = robust_percentiles.get(65, np.nan)
        low_vol_threshold_value = robust_percentiles.get(35, np.nan)
        
        if np.isnan(high_vol_threshold_value) or np.isnan(low_vol_threshold_value):
            logger.warning("Unable to calculate robust thresholds, using fallback method")
            high_vol_threshold_value = atr_series.quantile(self.high_vol_threshold)
            low_vol_threshold_value = atr_series.quantile(self.low_vol_threshold)
        
        # 判断潜在的状态
        potential_regime = self._determine_potential_regime(
            current_atr, high_vol_threshold_value, low_vol_threshold_value
        )
        
        # 智能状态切换逻辑
        previous_regime = self.current_regime
        self.current_regime = self._smart_regime_transition(
            potential_regime, current_time, current_atr
        )
        
        # 更新稳定性评分
        self._update_regime_stability()
        
        # 记录状态变化
        if previous_regime != MarketRegime.UNKNOWN and previous_regime != self.current_regime:
            self._record_regime_transition(
                previous_regime, self.current_regime, current_time, current_atr
            )
        
        # 详细日志记录
        atr_percentile = (atr_series <= current_atr).sum() / len(atr_series)
        logger.debug(f"Market regime: {self.current_regime.value}, ATR={current_atr:.6f}, "
                    f"percentile={atr_percentile:.2%}, stability={self.regime_stability_score:.3f}")
        
        return self.current_regime
    
    def _determine_potential_regime(self, current_atr: float, 
                                  high_threshold: float, 
                                  low_threshold: float) -> MarketRegime:
        """确定潜在的市场状态"""
        if current_atr >= high_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif current_atr <= low_threshold:
            return MarketRegime.LOW_VOLATILITY
        else:
            # 在中间区域，倾向于保持当前状态
            return self.current_regime if self.current_regime != MarketRegime.UNKNOWN else MarketRegime.LOW_VOLATILITY
    
    def _smart_regime_transition(self, potential_regime: MarketRegime,
                               current_time: datetime, current_atr: float) -> MarketRegime:
        """
        智能状态切换逻辑（防震荡）
        
        Args:
            potential_regime: 基于当前数据判断的潜在状态
            current_time: 当前时间
            current_atr: 当前ATR值
            
        Returns:
            经过智能判断后的实际状态
        """
        # 如果潜在状态与当前状态相同，重置确认计数
        if potential_regime == self.current_regime:
            self.pending_regime = None
            self.confirmation_count = 0
            return self.current_regime
        
        # 如果这是一个新的潜在状态变化
        if self.pending_regime != potential_regime:
            self.pending_regime = potential_regime
            self.confirmation_count = 1
            
            logger.debug(f"New potential regime: {potential_regime.value}, "
                        f"confirmation count: {self.confirmation_count}/{self.stability_buffer}")
            return self.current_regime
        
        # 如果是相同的潜在状态，增加确认计数
        self.confirmation_count += 1
        
        # 检查是否达到确认阈值
        if self.confirmation_count >= self.stability_buffer:
            # 额外的合理性检查
            if self._validate_regime_transition(self.current_regime, potential_regime, current_atr):
                logger.info(f"Regime transition confirmed: {self.current_regime.value} -> {potential_regime.value} "
                          f"after {self.confirmation_count} confirmations")
                
                # 重置状态
                self.pending_regime = None
                self.confirmation_count = 0
                self.last_transition_time = current_time
                
                return potential_regime
            else:
                # 转换无效，重置
                logger.warning(f"Regime transition validation failed: {self.current_regime.value} -> {potential_regime.value}")
                self.pending_regime = None
                self.confirmation_count = 0
                return self.current_regime
        
        logger.debug(f"Pending regime: {potential_regime.value}, "
                    f"confirmation count: {self.confirmation_count}/{self.stability_buffer}")
        
        return self.current_regime
    
    def _validate_regime_transition(self, from_regime: MarketRegime, 
                                   to_regime: MarketRegime, current_atr: float) -> bool:
        """
        验证状态转换的合理性
        
        Args:
            from_regime: 源状态
            to_regime: 目标状态
            current_atr: 当前ATR值
            
        Returns:
            转换是否合理
        """
        # 防止频繁转换（时间间隔检查）
        if (self.last_transition_time is not None and 
            len(self.atr_history) > 0):
            # 如果上次转换距离现在太近，拒绝转换
            recent_transitions = [t for t in self.regime_transitions[-10:] 
                                if (datetime.now() - t.transition_time).total_seconds() < 3600]  # 1小时内
            
            if len(recent_transitions) >= 3:  # 1小时内超过3次转换
                logger.warning(f"Too many recent regime transitions ({len(recent_transitions)}), blocking transition")
                return False
        
        # ATR值应该支持新状态
        atr_series = pd.Series(self.atr_history)
        current_percentile = (atr_series <= current_atr).sum() / len(atr_series)
        
        if to_regime == MarketRegime.HIGH_VOLATILITY:
            # 转向高波动状态时，ATR应该相对较高
            if current_percentile < 0.6:  # 低于60%百分位数
                logger.debug(f"ATR percentile {current_percentile:.2%} too low for high volatility regime")
                return False
        elif to_regime == MarketRegime.LOW_VOLATILITY:
            # 转向低波动状态时，ATR应该相对较低
            if current_percentile > 0.4:  # 高于40%百分位数
                logger.debug(f"ATR percentile {current_percentile:.2%} too high for low volatility regime")
                return False
        
        return True
    
    def _update_regime_stability(self):
        """更新状态稳定性评分"""
        if len(self.regime_transitions) == 0:
            self.regime_stability_score = 1.0
            return
        
        # 基于最近的转换频率计算稳定性
        recent_transitions = len([t for t in self.regime_transitions[-20:]])  # 最近20次转换
        stability_score = max(0.1, 1.0 - recent_transitions * 0.05)
        
        # 加入确认计数的影响
        if self.confirmation_count > 0:
            stability_score *= 0.8  # 正在等待确认时稳定性降低
        
        self.regime_stability_score = stability_score
    
    def _record_regime_transition(self, from_regime: MarketRegime, to_regime: MarketRegime,
                                transition_time: datetime, trigger_value: float):
        """记录状态转换"""
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_time=transition_time,
            confidence=self.regime_stability_score,
            trigger_value=trigger_value
        )
        
        self.regime_transitions.append(transition)
        
        # 保持转换历史在合理范围内
        if len(self.regime_transitions) > 100:
            self.regime_transitions = self.regime_transitions[-100:]
        
        logger.info(f"Regime transition recorded: {from_regime.value} -> {to_regime.value} "
                   f"at {transition_time}, ATR={trigger_value:.6f}")

    def get_current_regime(self) -> MarketRegime:
        """获取当前市场状态"""
        return self.current_regime
    
    def get_regime_confidence(self) -> float:
        """获取当前状态的置信度"""
        base_confidence = self.regime_stability_score
        
        # 如果正在等待确认，降低置信度
        if self.pending_regime is not None:
            confirmation_factor = 1.0 - (self.confirmation_count / self.stability_buffer) * 0.5
            base_confidence *= confirmation_factor
        
        return base_confidence

    def get_atr_percentile(self) -> float:
        """获取当前ATR在历史分布中的百分位数"""
        if len(self.atr_history) < 2:
            return 0.5
        
        atr_series = pd.Series(self.atr_history)
        current_atr = atr_series.iloc[-1]
        return (atr_series <= current_atr).sum() / len(atr_series)

    def create_market_snapshot(self, df: pd.DataFrame) -> MarketSnapshot:
        """
        创建市场环境快照
        
        Args:
            df: 市场数据
            
        Returns:
            市场快照对象
        """
        current_time = df.iloc[-1]['timestamp']
        
        # 计算近期波动率
        recent_returns = df['close'].pct_change().dropna()
        recent_volatility = recent_returns.iloc[-20:].std() * np.sqrt(252) if len(recent_returns) >= 20 else np.nan
        
        # 判断趋势环境
        if len(df) >= 20:
            sma_short = df['close'].rolling(5).mean().iloc[-1]
            sma_long = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_short > sma_long:
                trend_context = "bullish"
            elif current_price < sma_short < sma_long:
                trend_context = "bearish"
            else:
                trend_context = "sideways"
        else:
            trend_context = "unknown"
        
        # 判断成交量环境
        if 'volume' in df.columns and len(df) >= 20:
            recent_volume = df['volume'].iloc[-5:].mean()
            avg_volume = df['volume'].iloc[-20:].mean()
            
            if recent_volume > avg_volume * 1.5:
                volume_context = "high"
            elif recent_volume < avg_volume * 0.7:
                volume_context = "low"
            else:
                volume_context = "normal"
        else:
            volume_context = "unknown"
        
        return MarketSnapshot(
            timestamp=current_time,
            regime=self.current_regime,
            atr_percentile=self.get_atr_percentile(),
            recent_volatility=recent_volatility,
            trend_context=trend_context,
            volume_context=volume_context,
            baseline_stability=self.regime_stability_score
        )

    def get_volatility_metrics(self) -> Dict[str, Any]:
        """
        获取详细的波动性指标
        
        Returns:
            波动性指标字典
        """
        if len(self.atr_history) < 10:
            return {
                'current_atr': np.nan,
                'atr_percentile': 0.5,
                'regime': self.current_regime.value,
                'regime_confidence': 0.5,
                'regime_stability': 0.0,
                'pending_transition': None,
                'recent_transitions': 0
            }
        
        atr_series = pd.Series(self.atr_history)
        current_atr = atr_series.iloc[-1]
        atr_percentile = self.get_atr_percentile()
        
        # 最近转换统计
        recent_transitions = len([t for t in self.regime_transitions[-10:]])
        
        return {
            'current_atr': current_atr,
            'atr_percentile': atr_percentile,
            'regime': self.current_regime.value,
            'regime_confidence': self.get_regime_confidence(),
            'regime_stability': self.regime_stability_score,
            'pending_transition': self.pending_regime.value if self.pending_regime else None,
            'confirmation_progress': f"{self.confirmation_count}/{self.stability_buffer}",
            'recent_transitions': recent_transitions,
            'atr_mean': atr_series.mean(),
            'atr_std': atr_series.std(),
            'total_transitions': len(self.regime_transitions)
        }

    def is_regime_stable(self, min_stability: float = 0.8) -> bool:
        """
        判断当前市场状态是否稳定
        
        Args:
            min_stability: 最小稳定性阈值
            
        Returns:
            状态是否稳定
        """
        return (self.regime_stability_score >= min_stability and 
                self.pending_regime is None)

    def reset(self):
        """重置检测器状态"""
        self.atr_history.clear()
        self.current_regime = MarketRegime.UNKNOWN
        self.pending_regime = None
        self.confirmation_count = 0
        self.regime_transitions.clear()
        self.regime_stability_score = 1.0
        self.last_transition_time = None
        logger.info("Regime detector reset")


class BaselineManager:
    """基线管理器（增强版）"""
    
    def __init__(self, history_window: int = 500):
        """
        初始化基线管理器
        
        Args:
            history_window: 历史数据窗口大小
        """
        self.baselines: Dict[MarketRegime, Dict[IndicatorType, DynamicBaseline]] = {
            MarketRegime.HIGH_VOLATILITY: {},
            MarketRegime.LOW_VOLATILITY: {},
        }
        self.active_regime = MarketRegime.UNKNOWN
        self.history_window = history_window
        self.robust_stats = RobustStatistics()
        
        # 数据缓冲区（用于滚动计算）
        self.indicator_buffers: Dict[MarketRegime, Dict[IndicatorType, List[float]]] = {
            MarketRegime.HIGH_VOLATILITY: {},
            MarketRegime.LOW_VOLATILITY: {},
        }
    
    def update_baseline(self, regime: MarketRegime, indicator_type: IndicatorType, 
                       new_value: float, timestamp: datetime):
        """
        更新指定状态下的基线
        
        Args:
            regime: 市场状态
            indicator_type: 指标类型
            new_value: 新的指标值
            timestamp: 时间戳
        """
        if regime not in self.baselines:
            return
        
        # 初始化缓冲区
        if regime not in self.indicator_buffers:
            self.indicator_buffers[regime] = {}
        if indicator_type not in self.indicator_buffers[regime]:
            self.indicator_buffers[regime][indicator_type] = []
        
        # 添加新数据到缓冲区
        buffer = self.indicator_buffers[regime][indicator_type]
        buffer.append(new_value)
        
        # 维护窗口大小
        if len(buffer) > self.history_window:
            buffer = buffer[-self.history_window:]
            self.indicator_buffers[regime][indicator_type] = buffer
        
        # 需要足够的数据才能计算稳健基线
        if len(buffer) < 50:
            logger.debug(f"Insufficient data for {regime.value} {indicator_type.value}: {len(buffer)} < 50")
            return
        
        # 计算稳健化百分位数
        data_series = pd.Series(buffer)
        robust_percentiles = self.robust_stats.robust_percentiles(data_series)
        
        # 计算稳定性得分
        stability_score = self.robust_stats.calculate_stability_score(data_series)
        
        # 创建或更新基线
        baseline = DynamicBaseline(
            regime=regime,
            indicator_type=indicator_type,
            sample_size=len(buffer),
            percentiles=robust_percentiles,
            last_updated=timestamp,
            is_stable=stability_score > 0.7,
            raw_data_window=buffer[-100:] if len(buffer) >= 100 else buffer  # 保留最近100个点
        )
        
        self.baselines[regime][indicator_type] = baseline
        
        logger.debug(f"Updated {regime.value} baseline for {indicator_type.value}: "
                    f"P90={robust_percentiles.get(90, np.nan):.4f}, "
                    f"stability={stability_score:.3f}, sample_size={len(buffer)}")
    
    def get_threshold(self, regime: MarketRegime, indicator_type: IndicatorType,
                     percentile: int, fallback_value: Optional[float] = None) -> float:
        """
        获取指定状态和指标的阈值
        
        Args:
            regime: 市场状态
            indicator_type: 指标类型
            percentile: 百分位数
            fallback_value: 备用值
            
        Returns:
            阈值
        """
        # 首先尝试获取指定状态的基线
        if (regime in self.baselines and 
            indicator_type in self.baselines[regime]):
            
            baseline = self.baselines[regime][indicator_type]
            threshold = baseline.get_threshold(percentile)
            
            if threshold is not None and not np.isnan(threshold):
                return threshold
        
        # 如果当前状态没有数据，尝试使用另一个状态的数据
        for alt_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
            if alt_regime != regime and alt_regime in self.baselines:
                if indicator_type in self.baselines[alt_regime]:
                    baseline = self.baselines[alt_regime][indicator_type]
                    threshold = baseline.get_threshold(percentile)
                    
                    if threshold is not None and not np.isnan(threshold):
                        logger.warning(f"Using {alt_regime.value} baseline for {indicator_type.value} "
                                     f"P{percentile} in {regime.value} regime")
                        return threshold
        
        # 使用备用值或默认值
        if fallback_value is not None:
            logger.warning(f"Using fallback value {fallback_value} for {indicator_type.value} "
                         f"P{percentile} in {regime.value} regime")
            return fallback_value
        
        # 返回一个合理的默认值（调整为更适合实际数据的阈值）
        default_values = {
            IndicatorType.SLOPE_SCORE: {90: 0.5, 85: 0.3, 75: 0.2},  # 大幅降低斜率阈值
            IndicatorType.VOLUME_BURST: {90: 2.0, 85: 1.5, 75: 1.2},  # 适当降低量能阈值
            IndicatorType.RETRACE_DEPTH: {75: 0.4, 60: 0.3, 50: 0.25},
            IndicatorType.VOLUME_CONTRACTION: {60: 0.7, 50: 0.8, 40: 0.9},
        }
        
        if indicator_type in default_values and percentile in default_values[indicator_type]:
            default_val = default_values[indicator_type][percentile]
            logger.warning(f"Using default value {default_val} for {indicator_type.value} "
                         f"P{percentile} in {regime.value} regime")
            return default_val
        
        logger.error(f"No threshold available for {indicator_type.value} P{percentile} in {regime.value}")
        return np.nan
    
    def has_sufficient_data(self, regime: MarketRegime, indicator_type: IndicatorType,
                           min_samples: int = 50) -> bool:
        """
        检查指定状态是否有足够的基线数据
        
        Args:
            regime: 市场状态
            indicator_type: 指标类型
            min_samples: 最少样本数
            
        Returns:
            是否有足够数据
        """
        if (regime in self.baselines and 
            indicator_type in self.baselines[regime]):
            baseline = self.baselines[regime][indicator_type]
            return baseline.sample_size >= min_samples
        
        return False
    
    def get_baseline_stability(self, regime: MarketRegime, indicator_type: IndicatorType) -> float:
        """
        获取基线稳定性得分
        
        Args:
            regime: 市场状态
            indicator_type: 指标类型
            
        Returns:
            稳定性得分 (0-1)
        """
        if (regime in self.baselines and 
            indicator_type in self.baselines[regime]):
            baseline = self.baselines[regime][indicator_type]
            
            if baseline.raw_data_window:
                data_series = pd.Series(baseline.raw_data_window)
                return self.robust_stats.calculate_stability_score(data_series)
        
        return 0.5  # 默认中等稳定性
    
    def set_active_regime(self, regime: MarketRegime):
        """设置当前活跃的市场状态"""
        if self.active_regime != regime:
            logger.info(f"Active regime changed: {self.active_regime.value} -> {regime.value}")
            self.active_regime = regime
    
    def get_active_threshold(self, indicator_type: IndicatorType, percentile: int,
                           fallback_value: Optional[float] = None) -> float:
        """
        获取当前活跃状态下的阈值
        
        Args:
            indicator_type: 指标类型
            percentile: 百分位数
            fallback_value: 备用值
            
        Returns:
            阈值
        """
        return self.get_threshold(self.active_regime, indicator_type, percentile, fallback_value)
    
    def export_baselines(self) -> Dict[str, Any]:
        """导出所有基线数据"""
        export_data = {}
        for regime, indicators in self.baselines.items():
            export_data[regime.value] = {}
            for indicator_type, baseline in indicators.items():
                export_data[regime.value][indicator_type.value] = baseline.to_dict()
        
        return export_data
    
    def get_baseline_summary(self) -> Dict[str, Any]:
        """获取基线摘要统计"""
        summary = {
            'total_baselines': 0,
            'stable_baselines': 0,
            'regimes': {},
            'active_regime': self.active_regime.value
        }
        
        for regime, indicators in self.baselines.items():
            regime_info = {
                'indicators': len(indicators),
                'stable_indicators': 0,
                'avg_sample_size': 0,
                'avg_stability': 0
            }
            
            if indicators:
                sample_sizes = []
                stabilities = []
                
                for indicator_type, baseline in indicators.items():
                    summary['total_baselines'] += 1
                    if baseline.is_stable:
                        summary['stable_baselines'] += 1
                        regime_info['stable_indicators'] += 1
                    
                    sample_sizes.append(baseline.sample_size)
                    stability = self.get_baseline_stability(regime, indicator_type)
                    stabilities.append(stability)
                
                regime_info['avg_sample_size'] = np.mean(sample_sizes)
                regime_info['avg_stability'] = np.mean(stabilities)
            
            summary['regimes'][regime.value] = regime_info
        
        return summary