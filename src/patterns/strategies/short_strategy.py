"""
短周期策略（15分钟-1小时）
标准的形态检测策略
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy import stats
from loguru import logger

from src.patterns.strategies.base_strategy import BaseStrategy
from src.data.models.base_models import Flagpole, TrendLine
from src.patterns.base.pattern_components import PatternComponents
from src.patterns.base.quality_scorer import QualityScorer


class ShortStrategy(BaseStrategy):
    """短周期策略实现"""
    
    def __init__(self):
        """初始化短周期策略"""
        self.pattern_components = PatternComponents()
        self.quality_scorer = QualityScorer()
    
    def get_category_name(self) -> str:
        """获取策略类别名称"""
        return 'short'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        短周期数据预处理
        - 标准技术指标
        - 适度噪音过滤
        - 保留原始特征
        """
        df = df.copy()
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        # 1. 轻度平滑（可选）
        if params.get('apply_smoothing', False):
            df['close_smooth'] = self.smooth_price_data(df['close'], window=5)
        else:
            df['close_smooth'] = df['close']
        
        # 2. 趋势强度指标
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # 3. 价格动量
        df['momentum'] = df['close'].pct_change(periods=5)
        
        logger.debug(f"Preprocessed short-term data: {len(df)} bars")
        
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """
        寻找平行边界（旗形）
        短周期：标准的平行线检测，平衡精度和灵活性
        """
        if len(df) < params.get('min_bars', 8):
            return None
        
        # 寻找支撑和阻力点
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 12),
            category='short'
        )
        
        if len(support_idx) < 3 or len(resistance_idx) < 3:
            return None
        
        # 拟合边界线
        upper_line = self.pattern_components.fit_trend_line(
            df, resistance_idx, 'high'
        )
        lower_line = self.pattern_components.fit_trend_line(
            df, support_idx, 'low'
        )
        
        if not upper_line or not lower_line:
            return None
        
        # 验证边界质量
        if not self.validate_boundaries(upper_line, lower_line, min_r_squared=0.4):
            return None
        
        # 验证平行度
        if not self._verify_parallel_standard(upper_line, lower_line, df):
            return None
        
        # 验证反向倾斜
        if not self._verify_opposite_slope_standard(flagpole, upper_line, lower_line, params):
            return None
        
        return [upper_line, lower_line]
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（三角旗形）
        短周期：标准的峰谷识别
        """
        # 使用标准参数
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 12),
            category='short'
        )
        
        # 确保有足够的点
        min_points = 3
        if len(support_idx) < min_points or len(resistance_idx) < min_points:
            # 尝试降低要求
            support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
                df,
                window_size=max(2, len(df) // 15),
                min_prominence=df['close'].std() * 0.001,
                category='short'
            )
        
        return support_idx, resistance_idx
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """
        评分旗形质量
        短周期：标准评分，各项指标均衡
        """
        # 基础评分
        base_score = self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, 'short'
        )
        
        # 短周期特定调整
        trend_consistency_bonus = self._calculate_trend_consistency_bonus(df, flagpole)
        breakout_readiness = self._assess_breakout_readiness(df, boundaries, flagpole)
        
        # 综合评分
        final_score = base_score * (1 + trend_consistency_bonus * 0.1) * breakout_readiness
        
        return min(1.0, max(0.0, final_score))
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """
        评分三角旗形质量
        短周期：标准评分，注重形态完整性
        """
        # 基础评分
        base_score = self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, 'short'
        )
        
        # 短周期特定调整
        convergence_quality = self._assess_convergence_quality(upper_line, lower_line, apex, df)
        volume_confirmation = self._check_volume_confirmation(df)
        
        # 综合评分
        final_score = base_score * convergence_quality * (1 + volume_confirmation * 0.1)
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算趋势强度"""
        window = 20
        trend_strengths = []
        
        for i in range(len(df)):
            if i < window - 1:
                trend_strengths.append(0)
            else:
                # 使用线性回归的R²值作为趋势强度
                y = df['close'].iloc[i-window+1:i+1].values
                x = np.arange(window)
                
                if len(y) == window:
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    trend_strengths.append(r_value ** 2)
                else:
                    trend_strengths.append(0)
        
        return pd.Series(trend_strengths, index=df.index)
    
    def _verify_parallel_standard(self, upper_line: TrendLine, 
                                lower_line: TrendLine,
                                df: pd.DataFrame) -> bool:
        """标准平行度验证"""
        # 计算标准化的斜率差异
        avg_price = df['close'].mean()
        if avg_price <= 0:
            return False
        
        slope_diff = abs(upper_line.slope - lower_line.slope)
        normalized_diff = slope_diff / avg_price
        
        # 短周期的标准容差
        tolerance = 0.15
        
        # 额外检查：通道宽度的一致性
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width > 0:
            width_ratio = min(start_width, end_width) / max(start_width, end_width)
            
            # 宽度变化不能太大
            if width_ratio < 0.7:
                return False
        
        return normalized_diff <= tolerance
    
    def _verify_opposite_slope_standard(self, flagpole: Flagpole,
                                      upper_line: TrendLine,
                                      lower_line: TrendLine,
                                      params: dict) -> bool:
        """标准反向倾斜验证"""
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price <= 0:
            return False
        
        # 计算倾斜角度
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price))
        
        # 从参数获取角度范围
        min_angle = params.get('pattern', {}).get('min_slope_angle', 0.5)
        max_angle = params.get('pattern', {}).get('max_slope_angle', 10)
        
        if flagpole.direction == 'up':
            # 上升旗形：旗面应该向下倾斜
            return -max_angle <= slope_angle <= -min_angle
        else:
            # 下降旗形：旗面应该向上倾斜
            return min_angle <= slope_angle <= max_angle
    
    def _calculate_trend_consistency_bonus(self, df: pd.DataFrame, 
                                         flagpole: Flagpole) -> float:
        """计算趋势一致性奖励"""
        if 'trend_strength' not in df.columns:
            return 0.0
        
        avg_trend_strength = df['trend_strength'].mean()
        
        # 高趋势强度给予奖励
        if avg_trend_strength > 0.7:
            return 0.2
        elif avg_trend_strength > 0.5:
            return 0.1
        else:
            return 0.0
    
    def _assess_breakout_readiness(self, df: pd.DataFrame,
                                 boundaries: List[TrendLine],
                                 flagpole: Flagpole) -> float:
        """评估突破准备度"""
        if len(boundaries) < 2:
            return 1.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 检查最近的价格位置
        recent_prices = df['close'].tail(3)
        channel_center = (upper_line.end_price + lower_line.end_price) / 2
        
        # 价格接近边界更容易突破
        if flagpole.direction == 'up':
            # 上升旗形，价格接近上边界更好
            distance_ratio = (recent_prices.mean() - channel_center) / (upper_line.end_price - channel_center)
        else:
            # 下降旗形，价格接近下边界更好
            distance_ratio = (channel_center - recent_prices.mean()) / (channel_center - lower_line.end_price)
        
        # 将距离比例转换为准备度分数
        if distance_ratio > 0.7:
            return 1.1  # 奖励
        elif distance_ratio > 0.3:
            return 1.0  # 标准
        else:
            return 0.9  # 轻微惩罚
    
    def _assess_convergence_quality(self, upper_line: TrendLine,
                                  lower_line: TrendLine,
                                  apex: Tuple[float, float],
                                  df: pd.DataFrame) -> float:
        """评估收敛质量"""
        if not apex:
            return 0.8
        
        apex_time, apex_price = apex
        pattern_length = len(df)
        
        # 检查收敛速度
        start_range = abs(upper_line.start_price - lower_line.start_price)
        end_range = abs(upper_line.end_price - lower_line.end_price)
        
        if start_range > 0:
            convergence_rate = (start_range - end_range) / start_range / pattern_length
        else:
            convergence_rate = 0
        
        # 理想的收敛速度
        ideal_rate = 0.05  # 每根K线收敛5%
        
        if convergence_rate > 0:
            quality = min(1.0, convergence_rate / ideal_rate)
        else:
            quality = 0.5
        
        # 检查apex位置的合理性
        if 0.5 <= apex_time / pattern_length <= 2.0:
            quality *= 1.0
        else:
            quality *= 0.8
        
        return quality
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> float:
        """检查成交量确认"""
        if 'volume_ratio' not in df.columns:
            return 0.0
        
        # 检查成交量是否递减
        first_half_volume = df['volume_ratio'].iloc[:len(df)//2].mean()
        second_half_volume = df['volume_ratio'].iloc[len(df)//2:].mean()
        
        if first_half_volume > 0:
            volume_decay = (first_half_volume - second_half_volume) / first_half_volume
            
            if volume_decay > 0.3:  # 30%以上的衰减
                return 0.2
            elif volume_decay > 0.1:
                return 0.1
            else:
                return 0.0
        
        return 0.0