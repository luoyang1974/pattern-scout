"""
旗杆检测器（阶段1）
基于基线的"短暂而暴力"旗杆识别算法
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from loguru import logger

from src.data.models.base_models import (
    Flagpole, MarketRegime, IndicatorType
)
from src.patterns.base.market_regime_detector import BaselineManager
from src.patterns.indicators.technical_indicators import TechnicalIndicators


class FlagpoleDetector:
    """
    旗杆检测器
    实现基于基线的旗杆识别算法
    """
    
    def __init__(self, baseline_manager: BaselineManager):
        """
        初始化旗杆检测器
        
        Args:
            baseline_manager: 基线管理器
        """
        self.baseline_manager = baseline_manager
        self.tech_indicators = TechnicalIndicators()
        
        # 缺口处理配置
        self.gap_detection_config = {
            'min_gap_percent': 0.02,  # 最小缺口百分比2%
            'max_gap_percent': 0.1,   # 最大缺口百分比10%
            'confirmation_bars': 2,   # 缺口后确认K线数
        }
        
    def detect_flagpoles(self, df: pd.DataFrame, 
                        current_regime: MarketRegime,
                        timeframe: str = "15m") -> List[Flagpole]:
        """
        检测旗杆形态
        
        Args:
            df: OHLCV数据
            current_regime: 当前市场状态
            timeframe: 时间周期
            
        Returns:
            检测到的旗杆列表
        """
        if len(df) < 100:  # 需要足够的历史数据
            logger.warning(f"Insufficient data for flagpole detection: {len(df)} < 100")
            return []
        
        logger.info(f"Starting dynamic flagpole detection in {current_regime.value} regime")
        
        # 获取动态阈值
        dynamic_thresholds = self._get_dynamic_thresholds(current_regime)
        if not dynamic_thresholds:
            logger.warning("No dynamic thresholds available, using fallback values")
            dynamic_thresholds = self._get_fallback_thresholds()
        
        # 计算基础技术指标
        df_with_indicators = self._calculate_base_indicators(df)
        
        # 动态K线数量范围
        bars_range = self._get_dynamic_bars_range(timeframe, current_regime)
        min_bars, max_bars = bars_range['min'], bars_range['max']
        
        detected_flagpoles = []
        
        # 滑动窗口检测旗杆
        for bar_count in range(min_bars, max_bars + 1):
            for start_idx in range(len(df_with_indicators) - bar_count):
                end_idx = start_idx + bar_count
                
                # 提取候选区间
                candidate_data = df_with_indicators.iloc[start_idx:end_idx + 1]
                
                # 检测旗杆
                flagpole = self._analyze_flagpole_candidate(
                    candidate_data, dynamic_thresholds, current_regime, df_with_indicators
                )
                
                if flagpole:
                    # 检查与已检测旗杆的重叠
                    if not self._overlaps_with_existing(flagpole, detected_flagpoles):
                        detected_flagpoles.append(flagpole)
                        logger.info(f"Detected {flagpole.direction} flagpole: "
                                  f"height={flagpole.height_percent:.2%}, "
                                  f"slope_score={flagpole.slope_score:.2f}")
        
        # 按置信度排序并去重
        detected_flagpoles = self._remove_overlapping_flagpoles(detected_flagpoles)
        
        logger.info(f"Total flagpoles detected: {len(detected_flagpoles)}")
        return detected_flagpoles
    
    def _get_dynamic_thresholds(self, regime: MarketRegime) -> Dict[str, float]:
        """获取动态阈值"""
        thresholds = {}
        
        # 尝试获取各指标的动态阈值
        indicators_config = [
            ('slope_score_p90', IndicatorType.SLOPE_SCORE, 90),
            ('volume_burst_p85', IndicatorType.VOLUME_BURST, 85),
            ('retrace_depth_p75', IndicatorType.RETRACE_DEPTH, 75),
        ]
        
        for key, indicator_type, percentile in indicators_config:
            threshold = self.baseline_manager.get_threshold(
                regime, indicator_type, percentile, None
            )
            if not np.isnan(threshold):
                thresholds[key] = threshold
        
        return thresholds
    
    def _get_fallback_thresholds(self) -> Dict[str, float]:
        """获取备用阈值（调整为更适合实际数据的值）"""
        return {
            'slope_score_p90': 0.5,  # 大幅降低斜率阈值
            'volume_burst_p85': 1.5,  # 适当降低量能阈值  
            'retrace_depth_p75': 0.3,
        }
    
    def _get_dynamic_bars_range(self, timeframe: str, regime: MarketRegime) -> Dict[str, int]:
        """
        根据时间周期和市场状态获取动态K线数量范围
        
        Args:
            timeframe: 时间周期
            regime: 市场状态
            
        Returns:
            K线数量范围字典
        """
        base_ranges = {
            '1m': {'min': 8, 'max': 25},
            '5m': {'min': 6, 'max': 20},
            '15m': {'min': 4, 'max': 15},
            '1h': {'min': 3, 'max': 12},
            '4h': {'min': 2, 'max': 8},
            '1d': {'min': 2, 'max': 6},
        }
        
        # 根据市场状态调整
        base_range = base_ranges.get(timeframe, {'min': 4, 'max': 15})
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            # 高波动时缩短旗杆长度
            return {
                'min': max(2, base_range['min'] - 1),
                'max': max(base_range['min'] + 2, int(base_range['max'] * 0.8))
            }
        elif regime == MarketRegime.LOW_VOLATILITY:
            # 低波动时延长旗杆长度
            return {
                'min': base_range['min'],
                'max': min(int(base_range['max'] * 1.2), 25)
            }
        else:
            return base_range
    
    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        df_enhanced = df.copy()
        
        # 计算ATR
        df_enhanced['atr_14'] = self.tech_indicators.calculate_atr(df_enhanced, period=14)
        
        # 计算成交量移动平均
        if 'volume' in df_enhanced.columns:
            df_enhanced['volume_ma20'] = df_enhanced['volume'].rolling(window=20).mean()
        else:
            df_enhanced['volume_ma20'] = 1.0  # 默认值
        
        # 计算价格变化百分比
        df_enhanced['price_change_pct'] = df_enhanced['close'].pct_change()
        
        return df_enhanced
    
    def _analyze_flagpole_candidate(self, 
                                  candidate_data: pd.DataFrame,
                                  thresholds: Dict[str, float],
                                  regime: MarketRegime,
                                  full_df: pd.DataFrame) -> Optional[Flagpole]:
        """
        分析候选旗杆区间
        
        Args:
            candidate_data: 候选区间数据
            thresholds: 动态阈值
            regime: 市场状态
            full_df: 完整数据框
            
        Returns:
            旗杆对象或None
        """
        if len(candidate_data) < 2:
            return None
        
        # 基本信息
        start_time = candidate_data.iloc[0]['timestamp']
        end_time = candidate_data.iloc[-1]['timestamp']
        start_price = candidate_data.iloc[0]['close']
        end_price = candidate_data.iloc[-1]['close']
        
        # 计算方向和高度
        height = abs(end_price - start_price)
        height_percent = height / start_price if start_price > 0 else 0
        direction = 'up' if end_price > start_price else 'down'
        
        # 1. 动态斜率分验证
        slope_score = self._calculate_slope_score(candidate_data)
        slope_threshold = thresholds.get('slope_score_p90', 2.0)
        
        if slope_score < slope_threshold:
            logger.debug(f"Slope score {slope_score:.2f} below threshold {slope_threshold:.2f}")
            return None
        
        # 2. 动态量能爆发验证
        volume_burst = self._calculate_volume_burst(candidate_data)
        volume_threshold = thresholds.get('volume_burst_p85', 2.5)
        
        if volume_burst < volume_threshold:
            logger.debug(f"Volume burst {volume_burst:.2f} below threshold {volume_threshold:.2f}")
            return None
        
        # 3. 高动量K线占比验证（降低阈值到30%）
        impulse_bar_ratio = self._calculate_impulse_bar_ratio(candidate_data)
        
        if impulse_bar_ratio < 0.3:
            logger.debug(f"Impulse bar ratio {impulse_bar_ratio:.2%} below 30%")
            return None
        
        # 4. 低内部回撤验证（放宽到30%阈值）
        retracement_ratio = self._calculate_retracement_ratio(candidate_data)
        
        if retracement_ratio > 0.3:
            logger.debug(f"Retracement ratio {retracement_ratio:.2%} above 30%")
            return None
        
        # 5. 缺口处理
        has_gap, gap_info = self._detect_and_validate_gap(candidate_data, full_df)
        if has_gap and not self._validate_gap_continuation(gap_info, full_df):
            logger.debug("Gap continuation validation failed")
            return None
        
        # 6. 趋势强度验证
        trend_strength = self._calculate_trend_strength(candidate_data)
        
        # 创建旗杆对象
        flagpole = Flagpole(
            start_time=start_time,
            end_time=end_time,
            start_price=start_price,
            end_price=end_price,
            height=height,
            height_percent=height_percent,
            direction=direction,
            volume_ratio=volume_burst,
            bars_count=len(candidate_data),
            slope_score=slope_score,
            volume_burst=volume_burst,
            impulse_bar_ratio=impulse_bar_ratio,
            retracement_ratio=retracement_ratio,
            trend_strength=trend_strength
        )
        
        # 更新基线数据
        self._update_baseline_data(regime, slope_score, volume_burst, retracement_ratio)
        
        return flagpole
    
    def _calculate_slope_score(self, data: pd.DataFrame) -> float:
        """
        计算动态斜率分
        公式：((收盘价 - 开盘价) / K线数) / ATR14
        """
        if len(data) < 2:
            return 0.0
        
        start_price = data.iloc[0]['open']
        end_price = data.iloc[-1]['close']
        bars_count = len(data)
        avg_atr = data['atr_14'].mean()
        
        if avg_atr <= 0:
            return 0.0
        
        slope_score = abs((end_price - start_price) / bars_count) / avg_atr
        return slope_score
    
    def _calculate_volume_burst(self, data: pd.DataFrame) -> float:
        """
        计算动态量能爆发比
        公式：区间均量 / VOL20
        """
        if 'volume' not in data.columns or 'volume_ma20' not in data.columns:
            return 1.0
        
        interval_avg_volume = data['volume'].mean()
        vol20_avg = data['volume_ma20'].mean()
        
        if vol20_avg <= 0:
            return 1.0
        
        return interval_avg_volume / vol20_avg
    
    def _calculate_impulse_bar_ratio(self, data: pd.DataFrame) -> float:
        """
        计算高动量K线占比
        长实体定义：|收-开| ≥ 0.8*ATR14
        """
        if len(data) == 0:
            return 0.0
        
        impulse_count = 0
        
        for _, row in data.iterrows():
            body_size = abs(row['close'] - row['open'])
            atr_threshold = 0.8 * row['atr_14']
            
            if body_size >= atr_threshold:
                impulse_count += 1
        
        return impulse_count / len(data)
    
    def _calculate_retracement_ratio(self, data: pd.DataFrame) -> float:
        """
        计算低内部回撤比例
        公式：区间内最大回撤 / 总涨跌幅
        """
        if len(data) < 2:
            return 0.0
        
        start_price = data.iloc[0]['close']
        end_price = data.iloc[-1]['close']
        total_move = abs(end_price - start_price)
        
        if total_move <= 0:
            return 1.0  # 无变化时认为回撤比例为100%
        
        # 计算累积收益序列
        prices = data['close']
        cumulative_returns = (prices / start_price - 1)
        
        # 根据方向计算最大回撤
        if end_price > start_price:  # 上涨趋势
            running_max = cumulative_returns.expanding().max()
            drawdown = running_max - cumulative_returns
            max_drawdown = drawdown.max()
        else:  # 下跌趋势
            running_min = cumulative_returns.expanding().min()
            drawdown = cumulative_returns - running_min
            max_drawdown = drawdown.max()
        
        total_return = abs(cumulative_returns.iloc[-1])
        
        if total_return <= 0:
            return 1.0
        
        return max_drawdown / total_return
    
    def _detect_and_validate_gap(self, data: pd.DataFrame, 
                                full_df: pd.DataFrame) -> Tuple[bool, Optional[Dict]]:
        """
        检测和验证缺口
        
        Returns:
            (是否有缺口, 缺口信息)
        """
        if len(data) < 2:
            return False, None
        
        # 检查每个K线是否有显著缺口
        for i in range(1, len(data)):
            current_open = data.iloc[i]['open']
            prev_close = data.iloc[i-1]['close']
            
            gap_percent = abs(current_open - prev_close) / prev_close
            
            if (self.gap_detection_config['min_gap_percent'] <= 
                gap_percent <= 
                self.gap_detection_config['max_gap_percent']):
                
                gap_info = {
                    'gap_index': i,
                    'gap_percent': gap_percent,
                    'gap_direction': 'up' if current_open > prev_close else 'down',
                    'prev_close': prev_close,
                    'current_open': current_open
                }
                
                return True, gap_info
        
        return False, None
    
    def _validate_gap_continuation(self, gap_info: Dict, full_df: pd.DataFrame) -> bool:
        """
        验证缺口后的延续性
        要求缺口后的1-2根K线必须延续该方向
        """
        # 这里需要更复杂的逻辑来检查缺口后的确认
        # 简化实现：总是返回True
        return True
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        计算趋势强度（R²）
        使用线性回归拟合价格趋势
        """
        if len(data) < 3:
            return 0.0
        
        from scipy.stats import linregress
        
        x = np.arange(len(data))
        y = data['close'].values
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return r_value ** 2
        except:
            return 0.0
    
    def _update_baseline_data(self, regime: MarketRegime, 
                            slope_score: float, 
                            volume_burst: float,
                            retracement_ratio: float):
        """更新基线数据"""
        timestamp = datetime.now()
        
        # 更新各项指标的基线
        self.baseline_manager.update_baseline(
            regime, IndicatorType.SLOPE_SCORE, slope_score, timestamp
        )
        self.baseline_manager.update_baseline(
            regime, IndicatorType.VOLUME_BURST, volume_burst, timestamp
        )
        self.baseline_manager.update_baseline(
            regime, IndicatorType.RETRACE_DEPTH, retracement_ratio, timestamp
        )
    
    def _overlaps_with_existing(self, candidate: Flagpole, 
                               existing: List[Flagpole]) -> bool:
        """
        检查候选旗杆是否与已存在的旗杆重叠
        
        Args:
            candidate: 候选旗杆
            existing: 已存在的旗杆列表
            
        Returns:
            是否重叠
        """
        for existing_pole in existing:
            # 时间重叠检查
            if not (candidate.end_time < existing_pole.start_time or 
                   candidate.start_time > existing_pole.end_time):
                return True
        
        return False
    
    def _remove_overlapping_flagpoles(self, flagpoles: List[Flagpole]) -> List[Flagpole]:
        """
        去除重叠的旗杆，保留质量更高的
        
        Args:
            flagpoles: 旗杆列表
            
        Returns:
            去重后的旗杆列表
        """
        if len(flagpoles) <= 1:
            return flagpoles
        
        # 按斜率分排序（质量指标）
        sorted_poles = sorted(flagpoles, key=lambda x: x.slope_score, reverse=True)
        
        final_poles = []
        
        for candidate in sorted_poles:
            # 检查是否与已选择的旗杆重叠
            overlaps = False
            for selected in final_poles:
                if not (candidate.end_time < selected.start_time or 
                       candidate.start_time > selected.end_time):
                    overlaps = True
                    break
            
            if not overlaps:
                final_poles.append(candidate)
        
        return final_poles
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            'baseline_manager_active': self.baseline_manager.active_regime.value,
            'baseline_summary': self.baseline_manager.get_baseline_summary(),
            'gap_config': self.gap_detection_config
        }