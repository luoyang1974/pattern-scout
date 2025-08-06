"""
基础数据模型定义
支持动态基线旗形识别算法和结局追踪系统
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any, Dict
from enum import Enum


class PatternType(Enum):
    """形态类型枚举"""
    FLAG_PATTERN = "flag_pattern"  # 统一的旗形检测（包含矩形旗和三角旗）


class FlagSubType(Enum):
    """旗形子类型枚举"""
    FLAG = "flag"        # 矩形旗（平行通道）
    PENNANT = "pennant"  # 三角旗（收敛三角形）


class PatternOutcome(Enum):
    """形态结局分类枚举"""
    STRONG_CONTINUATION = "strong_continuation"      # 强势延续
    STANDARD_CONTINUATION = "standard_continuation"  # 标准延续
    BREAKOUT_STAGNATION = "breakout_stagnation"     # 突破停滞
    FAILED_BREAKOUT = "failed_breakout"             # 假突破反转
    INTERNAL_COLLAPSE = "internal_collapse"         # 内部瓦解
    OPPOSITE_RUN = "opposite_run"                   # 反向运行
    MONITORING = "monitoring"                       # 监控中
    UNKNOWN = "unknown"                             # 未知


class MarketRegime(Enum):
    """市场波动状态枚举"""
    HIGH_VOLATILITY = "high_volatility"  # 高波动状态
    LOW_VOLATILITY = "low_volatility"    # 低波动状态
    UNKNOWN = "unknown"                   # 未知状态


class IndicatorType(Enum):
    """动态指标类型枚举"""
    SLOPE_SCORE = "slope_score"                    # 斜率分
    VOLUME_BURST = "volume_burst"                  # 量能爆发比
    RETRACE_DEPTH = "retrace_depth"                # 回撤深度比
    VOLUME_CONTRACTION = "volume_contraction"      # 量能收缩比
    VOLATILITY_DROP = "volatility_drop"            # 波动下降比
    CHANNEL_WIDTH = "channel_width"                # 通道宽度
    PARALLELISM = "parallelism"                    # 平行度
    CONVERGENCE = "convergence"                    # 收敛度


@dataclass
class PriceData:
    """OHLCV价格数据模型"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Invalid OHLC data: high/low values inconsistent")


@dataclass
class TrendLine:
    """趋势线模型"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    slope: float
    r_squared: float  # 拟合度
    
    @property
    def height(self) -> float:
        """趋势线高度"""
        return abs(self.end_price - self.start_price)
    
    @property
    def duration_hours(self) -> float:
        """趋势线持续时间（小时）"""
        return (self.end_time - self.start_time).total_seconds() / 3600


@dataclass
class Flagpole:
    """旗杆模型（增强版）"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    height: float           # 绝对高度
    height_percent: float   # 百分比高度
    direction: str          # 'up' or 'down'
    volume_ratio: float     # 相对平均成交量的比例
    bars_count: int         # K线数量
    
    # 新增：动态检测指标
    slope_score: float      # 动态斜率分
    volume_burst: float     # 动态量能爆发比
    impulse_bar_ratio: float # 高动量K线占比
    retracement_ratio: float # 内部回撤比例
    trend_strength: float   # 趋势强度（R²）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'start_price': self.start_price,
            'end_price': self.end_price,
            'height': self.height,
            'height_percent': self.height_percent,
            'direction': self.direction,
            'volume_ratio': self.volume_ratio,
            'bars_count': self.bars_count,
            'slope_score': self.slope_score,
            'volume_burst': self.volume_burst,
            'impulse_bar_ratio': self.impulse_bar_ratio,
            'retracement_ratio': self.retracement_ratio,
            'trend_strength': self.trend_strength
        }


@dataclass
class DynamicBaseline:
    """动态基线数据类"""
    regime: MarketRegime
    indicator_type: IndicatorType
    sample_size: int
    percentiles: Dict[int, float]  # P75, P85, P90, P95
    last_updated: datetime
    is_stable: bool = False
    raw_data_window: Optional[List[float]] = None  # 原始数据窗口
    
    def get_threshold(self, percentile: int) -> Optional[float]:
        """获取指定百分位数的阈值"""
        return self.percentiles.get(percentile)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime.value,
            'indicator_type': self.indicator_type.value,
            'sample_size': self.sample_size,
            'percentiles': self.percentiles,
            'last_updated': self.last_updated.isoformat(),
            'is_stable': self.is_stable
        }


@dataclass
class PatternOutcomeAnalysis:
    """形态结局分析数据类"""
    pattern_id: str
    outcome: PatternOutcome
    analysis_date: datetime
    monitoring_duration: int  # 监控的K线数量
    
    # 关键水平定义
    breakout_level: float
    invalidation_level: float
    target_projection_1: float
    risk_distance: float
    
    # 实际价格走势
    actual_high: float
    actual_low: float
    breakthrough_occurred: bool
    breakthrough_direction: Optional[str]  # 'up' or 'down'
    
    # 结局度量
    success_ratio: Optional[float]  # 成功比例
    holding_period: Optional[int]   # 持有周期
    final_return: Optional[float]   # 最终回报
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'outcome': self.outcome.value,
            'analysis_date': self.analysis_date.isoformat(),
            'monitoring_duration': self.monitoring_duration,
            'breakout_level': self.breakout_level,
            'invalidation_level': self.invalidation_level,
            'target_projection_1': self.target_projection_1,
            'risk_distance': self.risk_distance,
            'actual_high': self.actual_high,
            'actual_low': self.actual_low,
            'breakthrough_occurred': self.breakthrough_occurred,
            'breakthrough_direction': self.breakthrough_direction,
            'success_ratio': self.success_ratio,
            'holding_period': self.holding_period,
            'final_return': self.final_return
        }


@dataclass
class InvalidationSignal:
    """失效前置判别信号"""
    signal_type: str           # 'fake_breakout', 'volatility_decay', 'volume_divergence'
    trigger_time: datetime
    trigger_price: float
    severity: float           # 严重程度 0-1
    description: str
    is_critical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_type': self.signal_type,
            'trigger_time': self.trigger_time.isoformat(),
            'trigger_price': self.trigger_price,
            'severity': self.severity,
            'description': self.description,
            'is_critical': self.is_critical
        }


@dataclass
class PatternRecord:
    """形态记录数据类（完全重构版）"""
    id: str
    symbol: str
    pattern_type: str  # PatternType.value
    sub_type: str      # FlagSubType.value  
    detection_date: datetime
    flagpole: Flagpole
    pattern_boundaries: List[TrendLine]
    pattern_duration: int  # K线数量
    confidence_score: float
    pattern_quality: str  # 质量等级
    
    # 市场状态信息
    market_regime: MarketRegime
    used_baselines: Dict[str, float]  # 使用的动态基线阈值
    
    # 失效前置判别结果
    invalidation_signals: List[InvalidationSignal]
    breakout_readiness: float  # 突破准备度 0-1
    
    # 可选的额外信息
    timeframe: Optional[str] = None
    volume_pattern: Optional[Dict[str, Any]] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    
    # 结局追踪相关
    outcome_analysis: Optional[PatternOutcomeAnalysis] = None
    is_monitoring: bool = False
    chart_file_path: Optional[str] = None  # 图表文件路径
    
    # 形态环境快照
    environment_snapshot: Optional[Dict[str, Any]] = None  # 识别时的市场环境
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'pattern_type': self.pattern_type,
            'sub_type': self.sub_type,
            'detection_date': self.detection_date.isoformat(),
            'flagpole': self.flagpole.to_dict(),
            'pattern_boundaries': [
                {
                    'start_time': boundary.start_time.isoformat(),
                    'end_time': boundary.end_time.isoformat(),
                    'start_price': boundary.start_price,
                    'end_price': boundary.end_price,
                    'slope': boundary.slope,
                    'r_squared': boundary.r_squared
                } for boundary in self.pattern_boundaries
            ],
            'pattern_duration': self.pattern_duration,
            'confidence_score': self.confidence_score,
            'pattern_quality': self.pattern_quality,
            'market_regime': self.market_regime.value,
            'used_baselines': self.used_baselines,
            'invalidation_signals': [signal.to_dict() for signal in self.invalidation_signals],
            'breakout_readiness': self.breakout_readiness,
            'timeframe': self.timeframe,
            'volume_pattern': self.volume_pattern,
            'additional_metrics': self.additional_metrics,
            'outcome_analysis': self.outcome_analysis.to_dict() if self.outcome_analysis else None,
            'is_monitoring': self.is_monitoring,
            'chart_file_path': self.chart_file_path,
            'environment_snapshot': self.environment_snapshot
        }

    def add_invalidation_signal(self, signal: InvalidationSignal):
        """添加失效信号"""
        if self.invalidation_signals is None:
            self.invalidation_signals = []
        self.invalidation_signals.append(signal)
    
    def has_critical_invalidation_signals(self) -> bool:
        """检查是否有关键失效信号"""
        return any(signal.is_critical for signal in (self.invalidation_signals or []))
    
    def get_quality_classification(self) -> str:
        """获取质量分类"""
        if self.confidence_score >= 0.8:
            return "high"
        elif self.confidence_score >= 0.6:
            return "medium"
        else:
            return "low"


@dataclass 
class MarketSnapshot:
    """市场环境快照"""
    timestamp: datetime
    regime: MarketRegime
    atr_percentile: float
    recent_volatility: float
    trend_context: str  # 'bullish', 'bearish', 'sideways'
    volume_context: str  # 'high', 'normal', 'low'
    baseline_stability: float  # 基线稳定性
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'regime': self.regime.value,
            'atr_percentile': self.atr_percentile,
            'recent_volatility': self.recent_volatility,
            'trend_context': self.trend_context,
            'volume_context': self.volume_context,
            'baseline_stability': self.baseline_stability
        }


@dataclass
class BreakthroughResult:
    """突破分析结果"""
    pattern_id: str
    breakthrough_detected: bool
    breakthrough_time: Optional[datetime]
    breakthrough_price: Optional[float]
    breakthrough_direction: str  # 'up', 'down', 'none'
    breakthrough_strength: float  # 0-1
    volume_confirmation: bool
    price_target: Optional[float]
    success_probability: float
    failure_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'breakthrough_detected': self.breakthrough_detected,
            'breakthrough_time': self.breakthrough_time.isoformat() if self.breakthrough_time else None,
            'breakthrough_price': self.breakthrough_price,
            'breakthrough_direction': self.breakthrough_direction,
            'breakthrough_strength': self.breakthrough_strength,
            'volume_confirmation': self.volume_confirmation,
            'price_target': self.price_target,
            'success_probability': self.success_probability,
            'failure_risk': self.failure_risk
        }


@dataclass
class AnalysisResult:
    """综合分析结果"""
    symbol: str
    analysis_time: datetime
    patterns_analyzed: int
    breakthroughs_detected: int
    breakthrough_results: List[BreakthroughResult]
    market_context: MarketSnapshot
    success_rate: float
    risk_assessment: str  # 'low', 'medium', 'high'
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'analysis_time': self.analysis_time.isoformat(),
            'patterns_analyzed': self.patterns_analyzed,
            'breakthroughs_detected': self.breakthroughs_detected,
            'breakthrough_results': [br.to_dict() for br in self.breakthrough_results],
            'market_context': self.market_context.to_dict(),
            'success_rate': self.success_rate,
            'risk_assessment': self.risk_assessment,
            'recommendations': self.recommendations
        }