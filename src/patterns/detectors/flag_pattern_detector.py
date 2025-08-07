"""
旗面检测器（阶段2）
实现基于基线的旗面识别算法和失效前置判别
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from loguru import logger

from src.data.models.base_models import (
    Flagpole, TrendLine, PatternRecord, FlagSubType, PatternType,
    MarketRegime, IndicatorType, InvalidationSignal
)
from src.patterns.base.market_regime_detector import BaselineManager
from src.patterns.indicators.technical_indicators import TechnicalIndicators


class FlagPatternDetector:
    """
    旗面检测器
    实现基于基线的旗面识别和失效前置判别
    """
    
    def __init__(self, baseline_manager: BaselineManager):
        """
        初始化旗面检测器
        
        Args:
            baseline_manager: 基线管理器
        """
        self.baseline_manager = baseline_manager
        self.tech_indicators = TechnicalIndicators()
    
    def detect_flag_patterns(self, df: pd.DataFrame,
                           flagpoles: List[Flagpole],
                           current_regime: MarketRegime,
                           timeframe: str = "15m") -> List[PatternRecord]:
        """
        检测旗面形态
        
        Args:
            df: OHLCV数据
            flagpoles: 检测到的旗杆
            current_regime: 当前市场状态
            timeframe: 时间周期
            
        Returns:
            检测到的形态记录列表
        """
        if not flagpoles:
            logger.info("No flagpoles provided for flag pattern detection")
            return []
        
        logger.info(f"Starting dynamic flag pattern detection for {len(flagpoles)} flagpoles")
        
        detected_patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for flagpole in flagpoles:
            # 检测旗面形成超时规则
            timeout_bars = self._calculate_timeout_bars(flagpole, current_regime)
            
            # 寻找旗面
            pattern = self._detect_flag_after_pole(
                df, flagpole, current_regime, timeout_bars, symbol, timeframe
            )
            
            if pattern:
                detected_patterns.append(pattern)
                logger.info(f"Detected {pattern.sub_type} pattern: "
                          f"confidence={pattern.confidence_score:.3f}, "
                          f"duration={pattern.pattern_duration} bars")
        
        logger.info(f"Total flag patterns detected: {len(detected_patterns)}")
        return detected_patterns
    
    def _calculate_timeout_bars(self, flagpole: Flagpole, regime: MarketRegime) -> int:
        """
        计算旗面形成超时K线数
        根据市场状态动态调整超时规则
        """
        base_timeout = flagpole.bars_count * 5
        
        # 根据市场状态调整
        if regime == MarketRegime.HIGH_VOLATILITY:
            # 高波动时缩短超时时间
            return max(base_timeout // 2, flagpole.bars_count * 3)
        elif regime == MarketRegime.LOW_VOLATILITY:
            # 低波动时延长超时时间
            return flagpole.bars_count * 7
        
        return base_timeout
    
    def _detect_flag_after_pole(self, df: pd.DataFrame,
                               flagpole: Flagpole,
                               regime: MarketRegime,
                               timeout_bars: int,
                               symbol: str,
                               timeframe: str) -> Optional[PatternRecord]:
        """
        在旗杆后检测旗面形态
        
        Args:
            df: 数据框
            flagpole: 旗杆
            regime: 市场状态
            timeout_bars: 超时K线数
            symbol: 品种代码
            timeframe: 时间周期
            
        Returns:
            形态记录或None
        """
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        if flagpole_end_idx >= len(df) - 5:  # 需要足够的后续数据
            return None
        
        # 计算搜索范围
        search_start_idx = flagpole_end_idx + 1
        search_end_idx = min(len(df), search_start_idx + timeout_bars)
        
        # 动态K线范围
        min_flag_bars, max_flag_bars = self._get_flag_bars_range(timeframe, regime)
        max_flag_bars = min(max_flag_bars, search_end_idx - search_start_idx)
        
        if max_flag_bars < min_flag_bars:
            return None
        
        best_pattern = None
        best_score = 0
        
        # 尝试不同长度的旗面
        for flag_duration in range(min_flag_bars, max_flag_bars + 1):
            flag_start_idx = search_start_idx
            flag_end_idx = flag_start_idx + flag_duration
            
            if flag_end_idx >= len(df):
                continue
            
            flag_data = df.iloc[flag_start_idx:flag_end_idx + 1].copy()
            
            # 阶段2.2：盘整本质过滤
            if not self._passes_consolidation_filter(flag_data, flagpole, regime, df):
                continue
            
            # 阶段2.3：几何形态甄别
            pattern_result = self._analyze_geometric_pattern(
                flag_data, flagpole, regime, df, symbol, timeframe
            )
            
            if pattern_result and pattern_result.confidence_score > best_score:
                best_score = pattern_result.confidence_score
                best_pattern = pattern_result
        
        return best_pattern
    
    def _get_flag_bars_range(self, timeframe: str, regime: MarketRegime) -> Tuple[int, int]:
        """获取旗面K线数量范围"""
        base_ranges = {
            '1m': (15, 60),
            '5m': (12, 48),
            '15m': (8, 40),
            '1h': (6, 30),
            '4h': (4, 20),
            '1d': (3, 15),
        }
        
        min_bars, max_bars = base_ranges.get(timeframe, (8, 40))
        
        # 根据市场状态调整
        if regime == MarketRegime.HIGH_VOLATILITY:
            # 高波动时旗面更短
            return min_bars, int(max_bars * 0.8)
        elif regime == MarketRegime.LOW_VOLATILITY:
            # 低波动时旗面更长
            return min_bars, int(max_bars * 1.2)
        
        return min_bars, max_bars
    
    def _passes_consolidation_filter(self, flag_data: pd.DataFrame,
                                   flagpole: Flagpole,
                                   regime: MarketRegime,
                                   full_df: pd.DataFrame) -> bool:
        """
        盘整本质过滤器
        验证旗面是否符合"能量压缩"特征
        """
        # 获取动态阈值
        retrace_threshold = self.baseline_manager.get_threshold(
            regime, IndicatorType.RETRACE_DEPTH, 75, 0.4
        )
        volume_contraction_threshold = self.baseline_manager.get_threshold(
            regime, IndicatorType.VOLUME_CONTRACTION, 60, 0.7
        )
        volatility_threshold = self.baseline_manager.get_threshold(
            regime, IndicatorType.VOLATILITY_DROP, 75, 0.8
        )
        
        # 1. 回撤深度验证
        flag_range = flag_data['high'].max() - flag_data['low'].min()
        retrace_ratio = flag_range / flagpole.height if flagpole.height > 0 else 1.0
        
        if retrace_ratio > retrace_threshold:
            logger.debug(f"Retrace ratio {retrace_ratio:.3f} exceeds threshold {retrace_threshold:.3f}")
            return False
        
        # 2. 量能收缩验证
        if 'volume' in flag_data.columns:
            flag_avg_volume = flag_data['volume'].mean()
            pole_avg_volume = flagpole.volume_ratio  # 这里应该是实际音量值
            
            if pole_avg_volume > 0:
                contraction_ratio = flag_avg_volume / pole_avg_volume
                if contraction_ratio > volume_contraction_threshold:
                    logger.debug(f"Volume contraction ratio {contraction_ratio:.3f} "
                               f"exceeds threshold {volume_contraction_threshold:.3f}")
                    return False
        
        # 3. 波动下降验证
        flag_volatility = flag_data['close'].std()
        pole_volatility = self._calculate_pole_volatility(flagpole, full_df)
        
        if pole_volatility > 0:
            volatility_ratio = flag_volatility / pole_volatility
            if volatility_ratio > volatility_threshold:
                logger.debug(f"Volatility ratio {volatility_ratio:.3f} "
                           f"exceeds threshold {volatility_threshold:.3f}")
                return False
        
        # 4. 量能趋势验证（成交量必须下降或持平）
        if 'volume' in flag_data.columns and len(flag_data) >= 3:
            volume_trend = self._calculate_volume_trend(flag_data)
            if volume_trend > 0.05:  # 允许5%的上升
                logger.debug(f"Volume trend {volume_trend:.3f} is increasing")
                return False
        
        logger.debug("Passed consolidation filter")
        return True
    
    def _analyze_geometric_pattern(self, flag_data: pd.DataFrame,
                                 flagpole: Flagpole,
                                 regime: MarketRegime,
                                 full_df: pd.DataFrame,
                                 symbol: str,
                                 timeframe: str) -> Optional[PatternRecord]:
        """
        几何形态甄别与结构确认
        """
        # 构建分位通道
        channel_boundaries = self._build_percentile_channel(flag_data)
        if not channel_boundaries:
            return None
        
        upper_line, lower_line = channel_boundaries
        
        # 动态宽度检查
        if not self._validate_channel_width(upper_line, lower_line, flagpole, regime):
            return None
        
        # 形态分类
        pattern_type = self._classify_pattern_type(upper_line, lower_line, flag_data)
        if pattern_type == 'invalid':
            return None
        
        # 形态稳固性校验
        stability_score = self._validate_pattern_stability(flag_data, upper_line, lower_line)
        if stability_score < 0.6:
            logger.debug(f"Pattern stability {stability_score:.3f} below threshold")
            return None
        
        # 失效前置判别
        invalidation_signals = self._detect_invalidation_signals(
            flag_data, upper_line, lower_line, full_df
        )
        
        # 如果有关键失效信号，拒绝该形态
        critical_signals = [s for s in invalidation_signals if s.is_critical]
        if critical_signals:
            logger.debug(f"Critical invalidation signals detected: {len(critical_signals)}")
            return None
        
        # 计算置信度
        confidence_score = self._calculate_confidence_score(
            flag_data, flagpole, upper_line, lower_line, pattern_type, stability_score
        )
        
        # 创建形态记录
        pattern_record = self._create_pattern_record(
            symbol=symbol,
            flagpole=flagpole,
            boundaries=[upper_line, lower_line],
            duration=len(flag_data),
            confidence_score=confidence_score,
            sub_type=FlagSubType.FLAG if pattern_type == 'flag' else FlagSubType.PENNANT,
            regime=regime,
            invalidation_signals=invalidation_signals,
            timeframe=timeframe,
            additional_metrics={
                'stability_score': stability_score,
                'channel_type': pattern_type
            }
        )
        
        return pattern_record
    
    def _build_percentile_channel(self, flag_data: pd.DataFrame) -> Optional[Tuple[TrendLine, TrendLine]]:
        """
        构建分位通道（U_t 95%分位，L_t 5%分位）
        """
        if len(flag_data) < 5:
            return None
        
        # 计算分位数边界
        highs = flag_data['high']
        lows = flag_data['low']
        timestamps = flag_data['timestamp']
        
        # 使用滚动分位数构建动态边界
        window = max(3, len(flag_data) // 3)
        upper_percentile = highs.rolling(window=window, min_periods=2).quantile(0.95)
        lower_percentile = lows.rolling(window=window, min_periods=2).quantile(0.05)
        
        # 拟合趋势线
        x = np.arange(len(flag_data))
        
        # 上边界线性拟合
        upper_valid = ~upper_percentile.isna()
        if upper_valid.sum() < 2:
            return None
        
        upper_x = x[upper_valid]
        upper_y = upper_percentile[upper_valid].values
        
        upper_fit = np.polyfit(upper_x, upper_y, 1)
        upper_r_squared = self._calculate_r_squared(upper_x, upper_y, upper_fit)
        
        # 下边界线性拟合
        lower_valid = ~lower_percentile.isna()
        if lower_valid.sum() < 2:
            return None
        
        lower_x = x[lower_valid]
        lower_y = lower_percentile[lower_valid].values
        
        lower_fit = np.polyfit(lower_x, lower_y, 1)
        lower_r_squared = self._calculate_r_squared(lower_x, lower_y, lower_fit)
        
        # 创建趋势线对象
        upper_line = TrendLine(
            start_time=timestamps.iloc[0],
            end_time=timestamps.iloc[-1],
            start_price=upper_fit[0] * 0 + upper_fit[1],
            end_price=upper_fit[0] * (len(flag_data) - 1) + upper_fit[1],
            slope=upper_fit[0],
            r_squared=upper_r_squared
        )
        
        lower_line = TrendLine(
            start_time=timestamps.iloc[0],
            end_time=timestamps.iloc[-1],
            start_price=lower_fit[0] * 0 + lower_fit[1],
            end_price=lower_fit[0] * (len(flag_data) - 1) + lower_fit[1],
            slope=lower_fit[0],
            r_squared=lower_r_squared
        )
        
        return upper_line, lower_line
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, fit: np.ndarray) -> float:
        """计算R平方值"""
        y_pred = fit[0] * x + fit[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return max(0.0, 1 - (ss_res / ss_tot))
    
    def _validate_channel_width(self, upper_line: TrendLine, 
                               lower_line: TrendLine,
                               flagpole: Flagpole,
                               regime: MarketRegime) -> bool:
        """
        动态宽度检查
        通道归一化宽度应该在历史分布的合理范围内
        """
        # 计算通道宽度
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        avg_width = (start_width + end_width) / 2
        
        # 归一化宽度
        normalized_width = avg_width / flagpole.height if flagpole.height > 0 else 0
        
        # 获取历史宽度分布阈值
        width_p20 = self.baseline_manager.get_threshold(
            regime, IndicatorType.CHANNEL_WIDTH, 20, 0.2
        )
        width_p80 = self.baseline_manager.get_threshold(
            regime, IndicatorType.CHANNEL_WIDTH, 80, 0.8
        )
        
        # 检查是否在合理范围内
        if not (width_p20 <= normalized_width <= width_p80):
            logger.debug(f"Channel width {normalized_width:.3f} outside range "
                        f"[{width_p20:.3f}, {width_p80:.3f}]")
            return False
        
        return True
    
    def _classify_pattern_type(self, upper_line: TrendLine, 
                              lower_line: TrendLine,
                              flag_data: pd.DataFrame) -> str:
        """
        形态分类：矩形旗或三角旗
        """
        # 计算平行度和收敛度
        slope_diff = abs(upper_line.slope - lower_line.slope)
        avg_price = flag_data['close'].mean()
        
        # 归一化斜率差异
        normalized_slope_diff = slope_diff / avg_price if avg_price > 0 else float('inf')
        
        # 计算收敛程度
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width > 0:
            convergence_ratio = 1 - (end_width / start_width)
        else:
            convergence_ratio = 0
        
        # 分类逻辑
        if normalized_slope_diff < 0.001 and abs(convergence_ratio) < 0.3:
            return 'flag'  # 矩形旗（平行且不收敛）
        elif convergence_ratio > 0.4:
            return 'pennant'  # 三角旗（显著收敛）
        else:
            return 'invalid'  # 无效形态
    
    def _validate_pattern_stability(self, flag_data: pd.DataFrame,
                                   upper_line: TrendLine,
                                   lower_line: TrendLine) -> float:
        """
        形态稳固性校验
        返回稳固性评分（0-1）
        """
        scores = []
        
        # 1. 高驻留率（价格位于通道内的比例）
        containment_score = self._calculate_containment_score(flag_data, upper_line, lower_line)
        scores.append(containment_score * 0.4)
        
        # 2. 均衡触碰（上下轨触碰情况）
        touch_score = self._calculate_balanced_touch_score(flag_data, upper_line, lower_line)
        scores.append(touch_score * 0.3)
        
        # 3. 边界线拟合质量
        fit_quality = (upper_line.r_squared + lower_line.r_squared) / 2
        scores.append(fit_quality * 0.3)
        
        return sum(scores)
    
    def _calculate_containment_score(self, flag_data: pd.DataFrame,
                                   upper_line: TrendLine,
                                   lower_line: TrendLine) -> float:
        """计算通道包含性评分"""
        contained_count = 0
        
        for i, row in flag_data.iterrows():
            # 计算该点的理论边界
            time_ratio = i / (len(flag_data) - 1) if len(flag_data) > 1 else 0
            upper_boundary = upper_line.start_price + upper_line.slope * time_ratio
            lower_boundary = lower_line.start_price + lower_line.slope * time_ratio
            
            # 检查价格是否在边界内
            if lower_boundary <= row['close'] <= upper_boundary:
                contained_count += 1
        
        return contained_count / len(flag_data) if len(flag_data) > 0 else 0
    
    def _calculate_balanced_touch_score(self, flag_data: pd.DataFrame,
                                      upper_line: TrendLine,
                                      lower_line: TrendLine) -> float:
        """计算均衡触碰评分"""
        tolerance = flag_data['close'].std() * 0.5
        
        upper_touches = 0
        lower_touches = 0
        
        for i, row in flag_data.iterrows():
            time_ratio = i / (len(flag_data) - 1) if len(flag_data) > 1 else 0
            upper_boundary = upper_line.start_price + upper_line.slope * time_ratio
            lower_boundary = lower_line.start_price + lower_line.slope * time_ratio
            
            # 检查上轨触碰
            if abs(row['high'] - upper_boundary) <= tolerance:
                upper_touches += 1
            
            # 检查下轨触碰
            if abs(row['low'] - lower_boundary) <= tolerance:
                lower_touches += 1
        
        # 评分：要求上下轨都至少触碰2次
        min_touches = min(upper_touches, lower_touches)
        max_touches = max(upper_touches, lower_touches)
        
        if min_touches >= 2 and max_touches >= 2:
            # 均衡性：触碰次数越均衡越好
            balance_score = min_touches / max_touches if max_touches > 0 else 0
            return balance_score
        else:
            return 0.0
    
    def _detect_invalidation_signals(self, flag_data: pd.DataFrame,
                                   upper_line: TrendLine,
                                   lower_line: TrendLine,
                                   full_df: pd.DataFrame) -> List[InvalidationSignal]:
        """
        失效前置判别
        检测可能导致形态失效的信号
        """
        signals = []
        
        # 1. 假突破失效规则检测
        fake_breakout_signals = self._detect_fake_breakout_signals(
            flag_data, upper_line, lower_line
        )
        signals.extend(fake_breakout_signals)
        
        # 2. 波动衰减速率监控
        volatility_signals = self._detect_volatility_decay_signals(flag_data)
        signals.extend(volatility_signals)
        
        # 3. 成交量背离检测
        if 'volume' in flag_data.columns:
            volume_signals = self._detect_volume_divergence_signals(flag_data)
            signals.extend(volume_signals)
        
        return signals
    
    def _detect_fake_breakout_signals(self, flag_data: pd.DataFrame,
                                     upper_line: TrendLine,
                                     lower_line: TrendLine) -> List[InvalidationSignal]:
        """检测假突破信号"""
        signals = []
        
        for i in range(1, len(flag_data)):
            row = flag_data.iloc[i]
            time_ratio = i / (len(flag_data) - 1)
            
            upper_boundary = upper_line.start_price + upper_line.slope * time_ratio
            lower_boundary = lower_line.start_price + lower_line.slope * time_ratio
            
            # 检查是否有短暂突破
            if row['close'] > upper_boundary or row['close'] < lower_boundary:
                # 检查后续是否回到通道内
                if i < len(flag_data) - 1:
                    next_row = flag_data.iloc[i + 1]
                    next_time_ratio = (i + 1) / (len(flag_data) - 1)
                    next_upper = upper_line.start_price + upper_line.slope * next_time_ratio
                    next_lower = lower_line.start_price + lower_line.slope * next_time_ratio
                    
                    if next_lower <= next_row['close'] <= next_upper:
                        # 检查成交量是否背离
                        avg_volume = flag_data['volume'].mean() if 'volume' in flag_data.columns else 1
                        breakout_volume = row['volume'] if 'volume' in row else 1
                        
                        if breakout_volume < avg_volume * 1.2:  # 成交量不足
                            signal = InvalidationSignal(
                                signal_type='fake_breakout',
                                trigger_time=row['timestamp'],
                                trigger_price=row['close'],
                                severity=0.8,
                                description="Fake breakout with insufficient volume",
                                is_critical=True
                            )
                            signals.append(signal)
        
        return signals
    
    def _detect_volatility_decay_signals(self, flag_data: pd.DataFrame) -> List[InvalidationSignal]:
        """检测波动衰减异常信号"""
        signals = []
        
        if len(flag_data) >= 10:
            # 计算ATR变化趋势
            atr_window = max(3, len(flag_data) // 3)
            flag_data_copy = flag_data.copy()
            
            # 简单ATR计算
            flag_data_copy['tr'] = np.maximum(
                flag_data_copy['high'] - flag_data_copy['low'],
                np.maximum(
                    abs(flag_data_copy['high'] - flag_data_copy['close'].shift(1)),
                    abs(flag_data_copy['low'] - flag_data_copy['close'].shift(1))
                )
            )
            
            flag_data_copy['atr'] = flag_data_copy['tr'].rolling(window=atr_window).mean()
            
            # 检查ATR下降趋势
            atr_values = flag_data_copy['atr'].dropna()
            if len(atr_values) >= 3:
                # 计算线性回归斜率
                x = np.arange(len(atr_values))
                slope = np.polyfit(x, atr_values.values, 1)[0]
                
                # 波动衰减过慢可能表示趋势耗尽
                if slope >= 0:  # 波动不下降或上升
                    signal = InvalidationSignal(
                        signal_type='volatility_decay',
                        trigger_time=flag_data.iloc[-1]['timestamp'],
                        trigger_price=flag_data.iloc[-1]['close'],
                        severity=0.6,
                        description=f"Insufficient volatility decay (slope: {slope:.6f})",
                        is_critical=False
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_volume_divergence_signals(self, flag_data: pd.DataFrame) -> List[InvalidationSignal]:
        """检测成交量背离信号"""
        signals = []
        
        if 'volume' in flag_data.columns and len(flag_data) >= 5:
            # 计算成交量趋势
            volumes = flag_data['volume'].values
            x = np.arange(len(volumes))
            
            try:
                slope = np.polyfit(x, volumes, 1)[0]
                
                # 成交量不应该持续增长（应该衰减）
                if slope > 0:
                    avg_volume = np.mean(volumes)
                    relative_slope = slope / avg_volume if avg_volume > 0 else 0
                    
                    if relative_slope > 0.1:  # 成交量增长超过10%
                        signal = InvalidationSignal(
                            signal_type='volume_divergence',
                            trigger_time=flag_data.iloc[-1]['timestamp'],
                            trigger_price=flag_data.iloc[-1]['close'],
                            severity=0.5,
                            description="Volume increasing instead of contracting",
                            is_critical=False
                        )
                        signals.append(signal)
            except:
                pass
        
        return signals
    
    def _calculate_confidence_score(self, flag_data: pd.DataFrame,
                                  flagpole: Flagpole,
                                  upper_line: TrendLine,
                                  lower_line: TrendLine,
                                  pattern_type: str,
                                  stability_score: float) -> float:
        """计算综合置信度"""
        scores = []
        
        # 1. 几何质量（30%）
        geometric_score = (upper_line.r_squared + lower_line.r_squared) / 2
        scores.append(geometric_score * 0.3)
        
        # 2. 稳固性评分（25%）
        scores.append(stability_score * 0.25)
        
        # 3. 时间比例合理性（20%）
        time_proportion = len(flag_data) / flagpole.bars_count if flagpole.bars_count > 0 else 1
        time_score = min(1.0, 1.0 / (1.0 + abs(time_proportion - 2.0)))  # 理想比例约2:1
        scores.append(time_score * 0.2)
        
        # 4. 成交量模式（15%）
        volume_score = self._calculate_volume_pattern_score(flag_data)
        scores.append(volume_score * 0.15)
        
        # 5. 形态类型奖励（10%）
        type_bonus = 1.0 if pattern_type in ['flag', 'pennant'] else 0.5
        scores.append(type_bonus * 0.1)
        
        total_score = sum(scores)
        return min(1.0, max(0.0, total_score))
    
    def _calculate_volume_pattern_score(self, flag_data: pd.DataFrame) -> float:
        """计算成交量模式评分"""
        if 'volume' not in flag_data.columns or len(flag_data) < 3:
            return 0.5
        
        # 成交量应该逐渐衰减
        volumes = flag_data['volume'].values
        x = np.arange(len(volumes))
        
        try:
            slope = np.polyfit(x, volumes, 1)[0]
            avg_volume = np.mean(volumes)
            
            if avg_volume > 0:
                normalized_slope = slope / avg_volume
                
                # 轻微下降是理想的
                if -0.1 <= normalized_slope <= 0.05:
                    return 1.0
                elif normalized_slope < -0.1:
                    return max(0.3, 1.0 + normalized_slope * 5)
                else:
                    return max(0.2, 1.0 - normalized_slope * 10)
        except:
            pass
        
        return 0.5
    
    def _calculate_pole_volatility(self, flagpole: Flagpole, full_df: pd.DataFrame) -> float:
        """计算旗杆期间的波动率"""
        pole_data = full_df[
            (full_df['timestamp'] >= flagpole.start_time) &
            (full_df['timestamp'] <= flagpole.end_time)
        ]
        
        if len(pole_data) < 2:
            return 0.0
        
        return pole_data['close'].std()
    
    def _calculate_volume_trend(self, flag_data: pd.DataFrame) -> float:
        """计算成交量趋势"""
        if 'volume' not in flag_data.columns or len(flag_data) < 3:
            return 0.0
        
        volumes = flag_data['volume'].values
        x = np.arange(len(volumes))
        
        try:
            slope = np.polyfit(x, volumes, 1)[0]
            avg_volume = np.mean(volumes)
            return slope / avg_volume if avg_volume > 0 else 0
        except:
            return 0.0
    
    def _create_pattern_record(self, symbol: str,
                              flagpole: Flagpole,
                              boundaries: List[TrendLine],
                              duration: int,
                              confidence_score: float,
                              sub_type: FlagSubType,
                              regime: MarketRegime,
                              invalidation_signals: List[InvalidationSignal],
                              timeframe: str,
                              additional_metrics: Dict[str, Any]) -> PatternRecord:
        """创建形态记录"""
        import uuid
        
        # 获取使用的基线阈值
        used_baselines = {
            'retrace_depth_p75': self.baseline_manager.get_threshold(
                regime, IndicatorType.RETRACE_DEPTH, 75, np.nan
            ),
            'volume_contraction_p60': self.baseline_manager.get_threshold(
                regime, IndicatorType.VOLUME_CONTRACTION, 60, np.nan
            ),
            'volatility_drop_p75': self.baseline_manager.get_threshold(
                regime, IndicatorType.VOLATILITY_DROP, 75, np.nan
            )
        }
        
        # 计算突破准备度
        breakout_readiness = self._calculate_breakout_readiness(boundaries)
        
        pattern_record = PatternRecord(
            id=str(uuid.uuid4()),
            symbol=symbol,
            pattern_type=PatternType.FLAG_PATTERN.value,
            sub_type=sub_type.value,
            detection_date=datetime.now(),
            flagpole=flagpole,
            pattern_boundaries=boundaries,
            pattern_duration=duration,
            confidence_score=confidence_score,
            pattern_quality=self._classify_quality(confidence_score),
            market_regime=regime,
            used_baselines=used_baselines,
            invalidation_signals=invalidation_signals,
            breakout_readiness=breakout_readiness,
            timeframe=timeframe,
            additional_metrics=additional_metrics
        )
        
        return pattern_record
    
    def _classify_quality(self, confidence_score: float) -> str:
        """质量分类"""
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_breakout_readiness(self, boundaries: List[TrendLine]) -> float:
        """计算突破准备度"""
        if len(boundaries) < 2:
            return 0.5
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 基于通道收敛程度计算准备度
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width > 0:
            convergence_ratio = (start_width - end_width) / start_width
            # 收敛程度越高，突破准备度越高
            return min(1.0, max(0.0, convergence_ratio * 2))
        
        return 0.5