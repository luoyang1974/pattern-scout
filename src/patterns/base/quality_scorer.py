"""
统一的质量评分系统
为不同形态和周期提供灵活的评分机制
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Tuple
from loguru import logger

from src.data.models.base_models import Flagpole, TrendLine


class QualityScorer:
    """统一的质量评分系统"""
    
    # 默认权重配置
    DEFAULT_WEIGHTS = {
        'flag': {
            'ultra_short': {
                'slope_direction': 0.20,
                'parallel_quality': 0.20,
                'volume_pattern': 0.30,
                'channel_containment': 0.20,
                'time_proportion': 0.10
            },
            'short': {
                'slope_direction': 0.25,
                'parallel_quality': 0.25,
                'volume_pattern': 0.25,
                'channel_containment': 0.15,
                'time_proportion': 0.10
            },
            'medium_long': {
                'slope_direction': 0.30,
                'parallel_quality': 0.30,
                'volume_pattern': 0.20,
                'channel_containment': 0.10,
                'time_proportion': 0.10
            }
        },
        'pennant': {
            'ultra_short': {
                'convergence_quality': 0.25,
                'symmetry': 0.20,
                'volume_pattern': 0.30,
                'size_proportion': 0.15,
                'apex_validity': 0.10
            },
            'short': {
                'convergence_quality': 0.30,
                'symmetry': 0.25,
                'volume_pattern': 0.25,
                'size_proportion': 0.10,
                'apex_validity': 0.10
            },
            'medium_long': {
                'convergence_quality': 0.35,
                'symmetry': 0.30,
                'volume_pattern': 0.20,
                'size_proportion': 0.05,
                'apex_validity': 0.10
            }
        }
    }
    
    def __init__(self, weights_config: dict = None):
        """
        初始化评分器
        
        Args:
            weights_config: 自定义权重配置
        """
        self.weights = weights_config or self.DEFAULT_WEIGHTS
    
    def score_flag_pattern(self, 
                         flag_data: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine],
                         category: str,
                         additional_metrics: dict = None) -> float:
        """
        旗形形态评分
        
        Args:
            flag_data: 旗面数据
            flagpole: 旗杆对象
            boundaries: [上边界, 下边界]
            category: 周期类别
            additional_metrics: 额外的评分指标
            
        Returns:
            综合得分（0-1）
        """
        weights = self.weights['flag'].get(category, self.weights['flag']['short'])
        scores = {}
        
        # 1. 反向倾斜方向评分
        scores['slope_direction'] = self._score_flag_slope_direction(
            flagpole, boundaries, category
        )
        
        # 2. 平行线质量评分
        scores['parallel_quality'] = self._score_parallel_quality(
            boundaries, flag_data, category
        )
        
        # 3. 成交量模式评分
        scores['volume_pattern'] = self._score_volume_pattern(
            flag_data, flagpole, 'flag', category
        )
        
        # 4. 通道包含度评分
        scores['channel_containment'] = self._score_channel_containment(
            flag_data, boundaries, category
        )
        
        # 5. 时间比例评分
        scores['time_proportion'] = self._score_time_proportion(
            flagpole, flag_data, 'flag', category
        )
        
        # 添加额外指标
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key in weights:
                    scores[key] = value
        
        # 计算加权总分
        total_score = 0
        total_weight = 0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0)
            total_score += score * weight
            total_weight += weight
        
        # 归一化
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        # 记录详细得分
        logger.debug(f"Flag pattern scores ({category}): {scores}")
        logger.debug(f"Final score: {final_score:.3f}")
        
        return min(1.0, max(0.0, final_score))
    
    def score_pennant_pattern(self,
                            pennant_data: pd.DataFrame,
                            flagpole: Flagpole,
                            boundaries: List[TrendLine],
                            apex: Tuple[float, float],
                            category: str,
                            additional_metrics: dict = None) -> float:
        """
        三角旗形（Pennant）形态评分
        
        Args:
            pennant_data: 三角旗面数据
            flagpole: 旗杆对象
            boundaries: [上边界, 下边界]
            apex: 收敛点 (时间偏移, 价格)
            category: 周期类别
            additional_metrics: 额外的评分指标
            
        Returns:
            综合得分（0-1）
        """
        weights = self.weights['pennant'].get(category, self.weights['pennant']['short'])
        scores = {}
        
        # 1. 收敛质量评分
        scores['convergence_quality'] = self._score_convergence_quality(
            boundaries, apex, pennant_data, category
        )
        
        # 2. 对称性评分
        scores['symmetry'] = self._score_pennant_symmetry(
            pennant_data, boundaries, category
        )
        
        # 3. 成交量模式评分
        scores['volume_pattern'] = self._score_volume_pattern(
            pennant_data, flagpole, 'pennant', category
        )
        
        # 4. 大小比例评分
        scores['size_proportion'] = self._score_size_proportion(
            flagpole, pennant_data, 'pennant', category
        )
        
        # 5. Apex有效性评分
        scores['apex_validity'] = self._score_apex_validity(
            apex, pennant_data, category
        )
        
        # 添加额外指标
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key in weights:
                    scores[key] = value
        
        # 计算加权总分
        total_score = 0
        total_weight = 0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0)
            total_score += score * weight
            total_weight += weight
        
        # 归一化
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        # 记录详细得分
        logger.debug(f"Pennant pattern scores ({category}): {scores}")
        logger.debug(f"Final score: {final_score:.3f}")
        
        return min(1.0, max(0.0, final_score))
    
    def _score_flag_slope_direction(self, flagpole: Flagpole, 
                                  boundaries: List[TrendLine],
                                  category: str) -> float:
        """评分旗形的反向倾斜"""
        if len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        
        # 计算平均价格用于归一化
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price <= 0:
            return 0.0
        
        # 计算倾斜角度
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price))
        
        # 根据周期调整理想角度范围
        if category == 'ultra_short':
            ideal_angle_range = (0.3, 8)  # 更宽松
        elif category == 'medium_long':
            ideal_angle_range = (1, 15)   # 更严格
        else:
            ideal_angle_range = (0.5, 10)
        
        # 评分逻辑
        if flagpole.direction == 'up':
            # 上升旗形应该向下倾斜
            if -ideal_angle_range[1] <= slope_angle <= -ideal_angle_range[0]:
                # 在理想范围内
                ideal_angle = -(ideal_angle_range[0] + ideal_angle_range[1]) / 2
                deviation = abs(slope_angle - ideal_angle)
                max_deviation = (ideal_angle_range[1] - ideal_angle_range[0]) / 2
                return max(0.0, 1.0 - deviation / max_deviation)
            elif slope_angle > 0:
                return 0.0  # 方向错误
            else:
                return 0.3  # 角度过大
        else:
            # 下降旗形应该向上倾斜
            if ideal_angle_range[0] <= slope_angle <= ideal_angle_range[1]:
                ideal_angle = (ideal_angle_range[0] + ideal_angle_range[1]) / 2
                deviation = abs(slope_angle - ideal_angle)
                max_deviation = (ideal_angle_range[1] - ideal_angle_range[0]) / 2
                return max(0.0, 1.0 - deviation / max_deviation)
            elif slope_angle < 0:
                return 0.0  # 方向错误
            else:
                return 0.3  # 角度过大
    
    def _score_parallel_quality(self, boundaries: List[TrendLine],
                              flag_data: pd.DataFrame,
                              category: str) -> float:
        """评分平行线质量"""
        if len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 1. 斜率相似度
        slope_diff = abs(upper_line.slope - lower_line.slope)
        avg_price = flag_data['close'].mean()
        normalized_slope_diff = slope_diff / avg_price if avg_price > 0 else float('inf')
        
        # 根据周期调整容忍度
        if category == 'ultra_short':
            tolerance = 0.2
        elif category == 'medium_long':
            tolerance = 0.1
        else:
            tolerance = 0.15
        
        slope_similarity = max(0, 1 - normalized_slope_diff / tolerance)
        
        # 2. 拟合质量
        fit_quality = (upper_line.r_squared + lower_line.r_squared) / 2
        
        # 3. 通道宽度一致性
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width > 0:
            width_ratio = min(start_width, end_width) / max(start_width, end_width)
        else:
            width_ratio = 0
        
        # 综合评分
        return (slope_similarity * 0.4 + fit_quality * 0.4 + width_ratio * 0.2)
    
    def _score_convergence_quality(self, boundaries: List[TrendLine],
                                 apex: Tuple[float, float],
                                 pennant_data: pd.DataFrame,
                                 category: str) -> float:
        """评分收敛质量"""
        if not apex or len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 1. 收敛程度
        start_range = abs(upper_line.start_price - lower_line.start_price)
        end_range = abs(upper_line.end_price - lower_line.end_price)
        
        if start_range > 0:
            convergence_ratio = 1 - (end_range / start_range)
            convergence_score = min(1.0, max(0.0, convergence_ratio))
        else:
            convergence_score = 0.0
        
        # 2. Apex位置合理性
        apex_time, apex_price = apex
        pattern_length = len(pennant_data)
        
        # Apex应该在形态结束后的合理范围内
        if category == 'ultra_short':
            ideal_apex_range = (0.5, 3.0)  # 相对于形态长度
        elif category == 'medium_long':
            ideal_apex_range = (0.3, 1.5)
        else:
            ideal_apex_range = (0.4, 2.0)
        
        apex_position_ratio = apex_time / pattern_length if pattern_length > 0 else float('inf')
        
        if ideal_apex_range[0] <= apex_position_ratio <= ideal_apex_range[1]:
            apex_score = 1.0
        elif apex_position_ratio < ideal_apex_range[0]:
            apex_score = apex_position_ratio / ideal_apex_range[0]
        else:
            apex_score = max(0.0, 1.0 - (apex_position_ratio - ideal_apex_range[1]) / ideal_apex_range[1])
        
        # 3. 边界线拟合质量
        fit_quality = (upper_line.r_squared + lower_line.r_squared) / 2
        
        # 综合评分
        return convergence_score * 0.5 + apex_score * 0.3 + fit_quality * 0.2
    
    def _score_pennant_symmetry(self, pennant_data: pd.DataFrame,
                              boundaries: List[TrendLine],
                              category: str) -> float:
        """评分三角旗形的对称性"""
        if len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 计算每个时间点的理论边界
        theoretical_prices = []
        
        for i in range(len(pennant_data)):
            progress = i / (len(pennant_data) - 1) if len(pennant_data) > 1 else 0
            
            upper_price = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            lower_price = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            theoretical_prices.append((upper_price, lower_price))
        
        # 计算对称性
        symmetry_scores = []
        
        for i, (upper, lower) in enumerate(theoretical_prices):
            actual_high = pennant_data.iloc[i]['high']
            actual_low = pennant_data.iloc[i]['low']
            
            mid_theoretical = (upper + lower) / 2
            mid_actual = (actual_high + actual_low) / 2
            
            # 上下偏离的对称性
            upper_deviation = upper - mid_theoretical
            lower_deviation = mid_theoretical - lower
            
            if upper_deviation > 0 and lower_deviation > 0:
                symmetry_ratio = min(upper_deviation, lower_deviation) / max(upper_deviation, lower_deviation)
                symmetry_scores.append(symmetry_ratio)
        
        base_symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.0
        
        # 根据周期调整
        if category == 'ultra_short':
            # 超短周期对对称性要求较低
            return min(1.0, base_symmetry * 1.2)
        elif category == 'medium_long':
            # 中长周期对对称性要求较高
            return base_symmetry * 0.9
        else:
            return base_symmetry
    
    def _score_volume_pattern(self, pattern_data: pd.DataFrame,
                            flagpole: Flagpole,
                            pattern_type: str,
                            category: str) -> float:
        """评分成交量模式"""
        # 获取旗杆期间的平均成交量（需要从完整数据中获取）
        flagpole_avg_volume = flagpole.volume_ratio  # 已经是比率
        
        # 形态期间的成交量
        pattern_volumes = pattern_data['volume']
        pattern_avg_volume = pattern_volumes.mean()
        
        # 计算成交量比率
        if 'volume_sma20' in pattern_data.columns:
            baseline_volume = pattern_data['volume_sma20'].mean()
        else:
            baseline_volume = pattern_volumes.rolling(20, min_periods=1).mean().mean()
        
        if baseline_volume > 0:
            pattern_volume_ratio = pattern_avg_volume / baseline_volume
        else:
            pattern_volume_ratio = 1.0
        
        # 理想的成交量关系
        if pattern_type == 'flag' or pattern_type == 'pennant':
            # 形态期间成交量应该显著低于旗杆
            ideal_ratio = 0.5  # 形态成交量是旗杆的50%
            
            if flagpole_avg_volume > 0:
                actual_ratio = pattern_volume_ratio / flagpole_avg_volume
            else:
                actual_ratio = pattern_volume_ratio
            
            # 计算得分
            if actual_ratio <= ideal_ratio:
                volume_contrast_score = 1.0
            elif actual_ratio <= ideal_ratio * 2:
                volume_contrast_score = 1.0 - (actual_ratio - ideal_ratio) / ideal_ratio
            else:
                volume_contrast_score = 0.3
        else:
            volume_contrast_score = 0.5
        
        # 成交量递减趋势
        if len(pattern_volumes) >= 3:
            x = np.arange(len(pattern_volumes))
            slope, _, r_value, _, _ = stats.linregress(x, pattern_volumes)
            
            if slope < 0:  # 递减
                trend_score = abs(r_value)  # 一致性越高越好
            else:
                trend_score = 0.3
        else:
            trend_score = 0.5
        
        # 根据周期调整权重
        if category == 'ultra_short':
            # 超短周期成交量可能噪音较多
            return volume_contrast_score * 0.6 + trend_score * 0.4
        elif category == 'medium_long':
            # 中长周期成交量模式更重要
            return volume_contrast_score * 0.7 + trend_score * 0.3
        else:
            return volume_contrast_score * 0.65 + trend_score * 0.35
    
    def _score_channel_containment(self, flag_data: pd.DataFrame,
                                 boundaries: List[TrendLine],
                                 category: str) -> float:
        """评分价格在通道内的包含度"""
        if len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        containment_count = 0
        total_count = len(flag_data)
        
        for i in range(len(flag_data)):
            progress = i / (len(flag_data) - 1) if len(flag_data) > 1 else 0
            
            # 计算理论边界
            theoretical_upper = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            theoretical_lower = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            row = flag_data.iloc[i]
            
            # 根据周期调整容忍度
            if category == 'ultra_short':
                tolerance = 0.03  # 3%
            elif category == 'medium_long':
                tolerance = 0.01  # 1%
            else:
                tolerance = 0.02  # 2%
            
            # 检查价格是否在通道内
            high_ok = row['high'] <= theoretical_upper * (1 + tolerance)
            low_ok = row['low'] >= theoretical_lower * (1 - tolerance)
            
            if high_ok and low_ok:
                containment_count += 1
        
        return containment_count / total_count if total_count > 0 else 0
    
    def _score_time_proportion(self, flagpole: Flagpole,
                             pattern_data: pd.DataFrame,
                             pattern_type: str,
                             category: str) -> float:
        """评分时间比例关系"""
        flagpole_duration = (flagpole.end_time - flagpole.start_time).total_seconds()
        pattern_duration = len(pattern_data)  # 以K线数量表示
        
        # 需要知道时间周期来转换
        # 这里简化处理，假设已经是相同单位
        
        if flagpole_duration <= 0:
            return 0.5
        
        # 理想的时间比例
        if pattern_type == 'flag':
            if category == 'ultra_short':
                ideal_ratio_range = (1.0, 4.0)
            elif category == 'medium_long':
                ideal_ratio_range = (0.8, 2.5)
            else:
                ideal_ratio_range = (1.0, 3.0)
        else:  # pennant
            if category == 'ultra_short':
                ideal_ratio_range = (0.8, 3.0)
            elif category == 'medium_long':
                ideal_ratio_range = (0.7, 2.0)
            else:
                ideal_ratio_range = (0.8, 2.5)
        
        # 简化的比例计算
        time_ratio = pattern_duration / 10  # 假设旗杆平均10根K线
        
        if ideal_ratio_range[0] <= time_ratio <= ideal_ratio_range[1]:
            return 1.0
        elif time_ratio < ideal_ratio_range[0]:
            return time_ratio / ideal_ratio_range[0]
        else:
            return max(0.3, 1.0 - (time_ratio - ideal_ratio_range[1]) / ideal_ratio_range[1])
    
    def _score_size_proportion(self, flagpole: Flagpole,
                             pattern_data: pd.DataFrame,
                             pattern_type: str,
                             category: str) -> float:
        """评分形态大小比例"""
        # 旗杆的价格变动
        flagpole_height = abs(flagpole.height_percent) / 100
        
        # 形态的价格范围
        pattern_high = pattern_data['high'].max()
        pattern_low = pattern_data['low'].min()
        pattern_range = (pattern_high - pattern_low) / pattern_data['close'].mean()
        
        # 理想的大小比例
        if pattern_type == 'pennant':
            # Pennant应该比较小
            if category == 'ultra_short':
                ideal_ratio_range = (0.1, 0.4)
            elif category == 'medium_long':
                ideal_ratio_range = (0.15, 0.35)
            else:
                ideal_ratio_range = (0.1, 0.35)
        else:
            ideal_ratio_range = (0.2, 0.5)
        
        if flagpole_height > 0:
            size_ratio = pattern_range / flagpole_height
        else:
            return 0.5
        
        if ideal_ratio_range[0] <= size_ratio <= ideal_ratio_range[1]:
            return 1.0
        elif size_ratio < ideal_ratio_range[0]:
            return size_ratio / ideal_ratio_range[0]
        else:
            return max(0.3, 1.0 - (size_ratio - ideal_ratio_range[1]) / ideal_ratio_range[1])
    
    def _score_apex_validity(self, apex: Tuple[float, float],
                           pennant_data: pd.DataFrame,
                           category: str) -> float:
        """评分Apex（收敛点）的有效性"""
        if not apex:
            return 0.0
        
        apex_time, apex_price = apex
        pattern_length = len(pennant_data)
        
        # Apex时间位置
        if pattern_length > 0:
            apex_position_ratio = apex_time / pattern_length
        else:
            return 0.0
        
        # 理想的Apex位置
        if category == 'ultra_short':
            ideal_range = (0.5, 3.0)
        elif category == 'medium_long':
            ideal_range = (0.3, 1.5)
        else:
            ideal_range = (0.4, 2.0)
        
        if ideal_range[0] <= apex_position_ratio <= ideal_range[1]:
            position_score = 1.0
        elif apex_position_ratio < 0:
            position_score = 0.0  # Apex在过去
        elif apex_position_ratio < ideal_range[0]:
            position_score = apex_position_ratio / ideal_range[0]
        else:
            position_score = max(0.0, 1.0 - (apex_position_ratio - ideal_range[1]) / ideal_range[1])
        
        # Apex价格合理性
        pattern_mean = pennant_data['close'].mean()
        price_deviation = abs(apex_price - pattern_mean) / pattern_mean
        
        if price_deviation < 0.1:  # 10%以内
            price_score = 1.0
        elif price_deviation < 0.2:
            price_score = 0.7
        else:
            price_score = 0.4
        
        return position_score * 0.7 + price_score * 0.3