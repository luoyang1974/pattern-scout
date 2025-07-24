"""
中长周期策略（4小时-日线）
严格的形态检测，高质量要求
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


class MediumLongStrategy(BaseStrategy):
    """中长周期策略实现"""
    
    def __init__(self):
        """初始化中长周期策略"""
        self.pattern_components = PatternComponents()
        self.quality_scorer = QualityScorer()
    
    def get_category_name(self) -> str:
        """获取策略类别名称"""
        return 'medium_long'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        中长周期数据预处理
        - 最小噪音处理
        - 保留原始特征
        - 增强趋势分析
        """
        df = df.copy()
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        # 1. 不进行平滑，保留原始数据
        df['close_smooth'] = df['close']
        
        # 2. 长期趋势指标
        df['long_trend'] = self._calculate_long_trend(df)
        
        # 3. 支撑阻力强度
        df['sr_strength'] = self._calculate_support_resistance_strength(df)
        
        # 4. 形态清晰度指标
        df['pattern_clarity'] = self._calculate_pattern_clarity(df)
        
        logger.debug(f"Preprocessed medium-long term data: {len(df)} bars")
        
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """
        寻找平行边界（旗形）
        中长周期：严格的平行线要求，清晰的边界
        """
        if len(df) < params.get('min_bars', 5):
            return None
        
        # 寻找清晰的支撑和阻力点
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 15),
            min_prominence=df['close'].std() * 0.003,  # 更高的突出度要求
            category='medium_long'
        )
        
        # 要求更多的触及点
        min_touches = params.get('pattern', {}).get('min_touches', 4)
        if len(support_idx) < min_touches or len(resistance_idx) < min_touches:
            return None
        
        # 拟合高质量边界线
        upper_line = self._fit_high_quality_boundary(df, resistance_idx, 'high')
        lower_line = self._fit_high_quality_boundary(df, support_idx, 'low')
        
        if not upper_line or not lower_line:
            return None
        
        # 严格的边界验证
        if not self.validate_boundaries(upper_line, lower_line, min_r_squared=0.6):
            return None
        
        # 严格的平行度验证
        if not self._verify_parallel_strict(upper_line, lower_line, df):
            return None
        
        # 清晰的反向倾斜
        if not self._verify_opposite_slope_strict(flagpole, upper_line, lower_line, params):
            return None
        
        # 额外验证：边界的有效性
        if not self._verify_boundary_effectiveness(df, upper_line, lower_line):
            return None
        
        return [upper_line, lower_line]
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（三角旗形）
        中长周期：只选择最显著的峰谷
        """
        # 使用更严格的参数
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 15),
            min_prominence=df['close'].std() * 0.005,  # 高突出度要求
            category='medium_long'
        )
        
        # 过滤掉不够显著的点
        support_idx = self._filter_significant_points(df, support_idx, 'support')
        resistance_idx = self._filter_significant_points(df, resistance_idx, 'resistance')
        
        return support_idx, resistance_idx
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """
        评分旗形质量
        中长周期：严格评分，强调形态完整性
        """
        # 基础评分
        base_score = self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, 'medium_long'
        )
        
        # 中长周期特定评估
        pattern_clarity = self._assess_pattern_clarity(df, boundaries)
        institutional_behavior = self._check_institutional_behavior(df, flagpole)
        trend_context = self._evaluate_trend_context(df, flagpole)
        
        # 严格的质量要求
        if pattern_clarity < 0.7:
            base_score *= 0.8
        
        # 综合评分
        final_score = base_score * pattern_clarity * institutional_behavior * trend_context
        
        return min(1.0, max(0.0, final_score))
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """
        评分三角旗形质量
        中长周期：最严格的评分标准
        """
        # 基础评分
        base_score = self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, 'medium_long'
        )
        
        # 中长周期特定评估
        convergence_precision = self._assess_convergence_precision(upper_line, lower_line, apex)
        symmetry_quality = self._evaluate_symmetry_quality(df, upper_line, lower_line)
        volume_profile = self._analyze_volume_profile(df)
        
        # 严格要求
        if convergence_precision < 0.8 or symmetry_quality < 0.7:
            base_score *= 0.85
        
        # 综合评分
        final_score = base_score * convergence_precision * symmetry_quality * (1 + volume_profile * 0.1)
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_long_trend(self, df: pd.DataFrame) -> pd.Series:
        """计算长期趋势"""
        # 使用50周期移动平均的斜率
        ma50 = df['close'].rolling(window=50, min_periods=20).mean()
        
        trend_values = []
        window = 10
        
        for i in range(len(df)):
            if i < window - 1 or pd.isna(ma50.iloc[i]):
                trend_values.append(0)
            else:
                # 计算MA的斜率
                y = ma50.iloc[i-window+1:i+1].values
                x = np.arange(window)
                
                if len(y) == window and not np.any(np.isnan(y)):
                    slope, _, _, _, _ = stats.linregress(x, y)
                    # 标准化斜率
                    normalized_slope = slope / ma50.iloc[i] * 100 if ma50.iloc[i] > 0 else 0
                    trend_values.append(normalized_slope)
                else:
                    trend_values.append(0)
        
        return pd.Series(trend_values, index=df.index)
    
    def _calculate_support_resistance_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算支撑阻力强度"""
        # 使用价格在某个水平附近的聚集程度
        window = 20
        strengths = []
        
        for i in range(len(df)):
            if i < window:
                strengths.append(0)
            else:
                # 计算价格在当前水平附近的频率
                current_price = df['close'].iloc[i]
                price_range = df['close'].iloc[i-window:i]
                
                # 计算在±0.5%范围内的价格数量
                tolerance = current_price * 0.005
                nearby_count = ((price_range >= current_price - tolerance) & 
                              (price_range <= current_price + tolerance)).sum()
                
                strength = nearby_count / window
                strengths.append(strength)
        
        return pd.Series(strengths, index=df.index)
    
    def _calculate_pattern_clarity(self, df: pd.DataFrame) -> pd.Series:
        """计算形态清晰度"""
        # 基于价格运动的规律性
        window = 10
        clarities = []
        
        for i in range(len(df)):
            if i < window:
                clarities.append(0.5)
            else:
                # 计算价格变化的一致性
                changes = df['close'].iloc[i-window:i].pct_change().dropna()
                
                if len(changes) > 0:
                    # 变异系数的倒数作为清晰度
                    cv = changes.std() / abs(changes.mean()) if changes.mean() != 0 else float('inf')
                    clarity = 1 / (1 + cv)
                    clarities.append(clarity)
                else:
                    clarities.append(0.5)
        
        return pd.Series(clarities, index=df.index)
    
    def _fit_high_quality_boundary(self, df: pd.DataFrame,
                                 point_indices: List[int],
                                 price_type: str) -> Optional[TrendLine]:
        """拟合高质量边界线"""
        if len(point_indices) < 4:  # 要求更多点
            return None
        
        # 提取价格
        prices = df.iloc[point_indices][price_type].values
        timestamps = df.iloc[point_indices]['timestamp'].values
        
        # 使用稳健回归方法
        x = np.arange(len(prices))
        
        # 尝试RANSAC风格的拟合
        best_line = None
        best_score = 0
        
        # 多次尝试，选择最佳拟合
        for _ in range(5):
            # 随机选择子集
            if len(point_indices) > 4:
                sample_size = max(4, int(len(point_indices) * 0.8))
                sample_indices = np.random.choice(len(prices), sample_size, replace=False)
                sample_x = x[sample_indices]
                sample_y = prices[sample_indices]
            else:
                sample_x = x
                sample_y = prices
            
            # 线性回归
            slope, intercept, r_value, _, _ = stats.linregress(sample_x, sample_y)
            
            # 计算所有点的拟合误差
            predicted = slope * x + intercept
            errors = np.abs(prices - predicted)
            inliers = errors < np.percentile(errors, 75)
            
            # 评分基于内点数量和拟合质量
            score = inliers.sum() * abs(r_value)
            
            if score > best_score:
                best_score = score
                best_line = (slope, intercept, r_value)
        
        if best_line is None:
            return None
        
        slope, intercept, r_value = best_line
        
        # 重新计算起止价格
        start_price = intercept
        end_price = slope * (len(prices) - 1) + intercept
        
        return TrendLine(
            start_time=timestamps[0],
            end_time=timestamps[-1],
            start_price=start_price,
            end_price=end_price,
            slope=slope,
            r_squared=r_value ** 2
        )
    
    def _verify_parallel_strict(self, upper_line: TrendLine,
                              lower_line: TrendLine,
                              df: pd.DataFrame) -> bool:
        """严格的平行度验证"""
        avg_price = df['close'].mean()
        if avg_price <= 0:
            return False
        
        # 斜率差异
        slope_diff = abs(upper_line.slope - lower_line.slope)
        normalized_diff = slope_diff / avg_price
        
        # 中长周期要求更严格的平行度
        if normalized_diff > 0.1:
            return False
        
        # 通道宽度一致性
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width > 0:
            width_ratio = min(start_width, end_width) / max(start_width, end_width)
            
            # 要求更高的一致性
            if width_ratio < 0.85:
                return False
        
        # 检查拟合质量
        min_r_squared = 0.7
        if upper_line.r_squared < min_r_squared or lower_line.r_squared < min_r_squared:
            return False
        
        return True
    
    def _verify_opposite_slope_strict(self, flagpole: Flagpole,
                                    upper_line: TrendLine,
                                    lower_line: TrendLine,
                                    params: dict) -> bool:
        """严格的反向倾斜验证"""
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price <= 0:
            return False
        
        # 计算倾斜角度
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price))
        
        # 中长周期的角度要求
        min_angle = params.get('pattern', {}).get('min_slope_angle', 1.0)
        max_angle = params.get('pattern', {}).get('max_slope_angle', 15.0)
        
        # 严格检查方向
        if flagpole.direction == 'up':
            # 必须明确向下倾斜
            if not (-max_angle <= slope_angle <= -min_angle):
                return False
        else:
            # 必须明确向上倾斜
            if not (min_angle <= slope_angle <= max_angle):
                return False
        
        return True
    
    def _verify_boundary_effectiveness(self, df: pd.DataFrame,
                                     upper_line: TrendLine,
                                     lower_line: TrendLine) -> bool:
        """验证边界的有效性"""
        touches_upper = 0
        touches_lower = 0
        
        for i in range(len(df)):
            progress = i / (len(df) - 1) if len(df) > 1 else 0
            
            # 计算理论边界
            theoretical_upper = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            theoretical_lower = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            # 检查触及（1%容差）
            if abs(df.iloc[i]['high'] - theoretical_upper) / theoretical_upper < 0.01:
                touches_upper += 1
            
            if abs(df.iloc[i]['low'] - theoretical_lower) / theoretical_lower < 0.01:
                touches_lower += 1
        
        # 每条边界至少3次有效触及
        return touches_upper >= 3 and touches_lower >= 3
    
    def _filter_significant_points(self, df: pd.DataFrame,
                                 point_indices: List[int],
                                 point_type: str) -> List[int]:
        """过滤出显著的支撑/阻力点"""
        if len(point_indices) <= 4:
            return point_indices
        
        # 计算每个点的显著性
        significances = []
        
        for idx in point_indices:
            if point_type == 'support':
                price = df.iloc[idx]['low']
                # 检查前后的价格
                window = 5
                start = max(0, idx - window)
                end = min(len(df), idx + window + 1)
                
                nearby_lows = df.iloc[start:end]['low']
                significance = (nearby_lows >= price).sum() / len(nearby_lows)
            else:  # resistance
                price = df.iloc[idx]['high']
                window = 5
                start = max(0, idx - window)
                end = min(len(df), idx + window + 1)
                
                nearby_highs = df.iloc[start:end]['high']
                significance = (nearby_highs <= price).sum() / len(nearby_highs)
            
            significances.append(significance)
        
        # 选择最显著的点
        threshold = 0.7
        significant_points = [point_indices[i] for i, sig in enumerate(significances) if sig >= threshold]
        
        # 如果过滤后太少，返回原始点
        if len(significant_points) < 4:
            return point_indices
        
        return significant_points
    
    def _assess_pattern_clarity(self, df: pd.DataFrame,
                              boundaries: List[TrendLine]) -> float:
        """评估形态清晰度"""
        if 'pattern_clarity' in df.columns:
            avg_clarity = df['pattern_clarity'].mean()
        else:
            avg_clarity = 0.7
        
        # 检查边界的拟合质量
        if len(boundaries) >= 2:
            boundary_quality = (boundaries[0].r_squared + boundaries[1].r_squared) / 2
        else:
            boundary_quality = 0.5
        
        return avg_clarity * 0.5 + boundary_quality * 0.5
    
    def _check_institutional_behavior(self, df: pd.DataFrame,
                                    flagpole: Flagpole) -> float:
        """检查机构行为特征"""
        # 检查成交量模式
        if 'volume' in df.columns:
            # 大成交量往往表示机构参与
            volume_percentile = df['volume'].quantile(0.9)
            large_volume_bars = (df['volume'] > volume_percentile).sum()
            
            # 机构交易特征：大成交量集中在特定时段
            if large_volume_bars > 0 and large_volume_bars < len(df) * 0.2:
                return 1.1  # 奖励
            else:
                return 1.0
        
        return 1.0
    
    def _evaluate_trend_context(self, df: pd.DataFrame,
                              flagpole: Flagpole) -> float:
        """评估趋势背景"""
        if 'long_trend' in df.columns:
            avg_trend = df['long_trend'].mean()
            
            # 检查是否与主趋势一致
            if flagpole.direction == 'up' and avg_trend > 0:
                return 1.05
            elif flagpole.direction == 'down' and avg_trend < 0:
                return 1.05
            else:
                return 0.95
        
        return 1.0
    
    def _assess_convergence_precision(self, upper_line: TrendLine,
                                    lower_line: TrendLine,
                                    apex: Tuple[float, float]) -> float:
        """评估收敛精度"""
        if not apex:
            return 0.7
        
        # 检查收敛的线性程度
        # 理想情况下，两条线应该稳定地收敛
        start_gap = abs(upper_line.start_price - lower_line.start_price)
        end_gap = abs(upper_line.end_price - lower_line.end_price)
        
        if start_gap > 0:
            convergence_ratio = 1 - (end_gap / start_gap)
            
            # 中长周期要求明确的收敛
            if convergence_ratio > 0.5:
                return min(1.0, convergence_ratio * 1.2)
            else:
                return convergence_ratio
        
        return 0.5
    
    def _evaluate_symmetry_quality(self, df: pd.DataFrame,
                                 upper_line: TrendLine,
                                 lower_line: TrendLine) -> float:
        """评估对称性质量"""
        # 计算上下边界的斜率对称性
        upper_angle = np.degrees(np.arctan(upper_line.slope))
        lower_angle = np.degrees(np.arctan(lower_line.slope))
        
        # 理想情况：斜率大小相等，方向相反
        ideal_symmetry = abs(upper_angle + lower_angle) / max(abs(upper_angle), abs(lower_angle))
        
        # 对称性越好，得分越高
        if ideal_symmetry < 0.2:  # 非常对称
            return 1.0
        elif ideal_symmetry < 0.4:
            return 0.9
        elif ideal_symmetry < 0.6:
            return 0.8
        else:
            return 0.7
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> float:
        """分析成交量轮廓"""
        if 'volume' not in df.columns:
            return 0.0
        
        # 分析成交量分布
        volumes = df['volume']
        
        # 检查递减趋势
        first_third = volumes.iloc[:len(volumes)//3].mean()
        middle_third = volumes.iloc[len(volumes)//3:2*len(volumes)//3].mean()
        last_third = volumes.iloc[2*len(volumes)//3:].mean()
        
        # 理想：逐步递减
        if first_third > middle_third > last_third:
            decay_score = 0.3
        elif first_third > last_third:
            decay_score = 0.2
        else:
            decay_score = 0.0
        
        # 检查成交量的稳定性（不要太多spike）
        volume_cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else float('inf')
        
        if volume_cv < 0.5:  # 稳定
            stability_score = 0.2
        elif volume_cv < 1.0:
            stability_score = 0.1
        else:
            stability_score = 0.0
        
        return decay_score + stability_score