"""
5分钟周期策略（高频交易）
针对5分钟数据的优化处理
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


class MinuteFiveStrategy(BaseStrategy):
    """5分钟周期策略实现"""
    
    def __init__(self):
        """初始化5分钟策略"""
        self.pattern_components = PatternComponents()
        self.quality_scorer = QualityScorer()
        self.noise_threshold = 0.0015  # 0.15%的噪音阈值
    
    def get_category_name(self) -> str:
        """获取策略类别名称"""
        return '5m'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        5分钟数据预处理
        - 强噪音过滤
        - 适度平滑处理
        - 高频特征提取
        """
        df = df.copy()
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        # 1. 异常值检测和处理（使用价格平滑）
        if len(df) > 10:
            # 对价格数据进行适度平滑，去除异常值影响
            df['close'] = self.smooth_price_data(df['close'], window=2)
            df['high'] = self.smooth_price_data(df['high'], window=2)
            df['low'] = self.smooth_price_data(df['low'], window=2)
        
        # 2. 价格平滑（保留更多原始特征）
        df['close_smooth'] = self.smooth_price_data(df['close'], window=3)
        df['high_smooth'] = self.smooth_price_data(df['high'], window=2)
        df['low_smooth'] = self.smooth_price_data(df['low'], window=2)
        
        # 3. 噪音指标
        df['noise_level'] = self._calculate_noise_level(df)
        
        # 4. 短期动量指标
        df['short_momentum'] = self._calculate_short_momentum(df)
        
        # 5. 波动性指标
        df['volatility'] = self._calculate_volatility(df)
        
        logger.debug(f"Preprocessed 5-minute data: {len(df)} bars")
        
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """
        寻找平行边界（旗形）
        5分钟：平衡精度和容忍度
        """
        if len(df) < params.get('min_bars', 8):
            return None
        
        # 寻找支撑和阻力点
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df, 
            window_size=max(2, len(df) // 10),
            min_prominence=df['close'].std() * 0.001,
            category='5m'
        )
        
        # 确保有足够的点
        if len(support_idx) < 2 or len(resistance_idx) < 2:
            return None
        
        # 拟合边界线
        upper_line = self._fit_boundary_with_robustness(
            df, resistance_idx, 'high_smooth', 'resistance'
        )
        lower_line = self._fit_boundary_with_robustness(
            df, support_idx, 'low_smooth', 'support'
        )
        
        if not upper_line or not lower_line:
            return None
        
        # 验证平行度
        if not self._verify_parallel_moderate(upper_line, lower_line, df):
            return None
        
        return [upper_line, lower_line]
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（三角旗形）
        5分钟：密集采样，较低突出度要求
        """
        # 使用适中的参数
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 12),
            min_prominence=df['close'].std() * 0.0008,
            category='5m'
        )
        
        # 确保有足够的点
        if len(support_idx) < 2 or len(resistance_idx) < 2:
            # 使用更宽松的要求
            support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
                df,
                window_size=max(1, len(df) // 15),
                min_prominence=df['close'].std() * 0.0003,
                category='5m'
            )
        
        return support_idx, resistance_idx
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """
        评分旗形质量
        5分钟：平衡各项指标权重
        """
        base_score = self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '5m'
        )
        
        # 5分钟特殊调整
        momentum_consistency = self._check_momentum_consistency(df, flagpole)
        volatility_penalty = self._calculate_volatility_penalty(df)
        pattern_integrity = self._assess_pattern_integrity(df, boundaries)
        
        adjusted_score = base_score * momentum_consistency * (1 - volatility_penalty) * pattern_integrity
        
        return min(1.0, max(0.0, adjusted_score))
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """
        评分三角旗形质量
        5分钟：注重收敛质量和对称性
        """
        base_score = self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '5m'
        )
        
        # 5分钟特殊调整
        convergence_stability = self._assess_convergence_stability(upper_line, lower_line, df)
        volume_pattern_score = self._evaluate_volume_pattern(df)
        
        adjusted_score = base_score * convergence_stability * (1 + volume_pattern_score)
        
        return min(1.0, max(0.0, adjusted_score))
    
    def _calculate_noise_level(self, df: pd.DataFrame) -> pd.Series:
        """计算噪音水平（5分钟优化）"""
        # 使用高低价波动率和价格跳跃的组合
        hl_range = (df['high'] - df['low']) / df['close']
        price_changes = df['close'].pct_change().abs()
        
        # 综合噪音指标
        noise = (hl_range + price_changes * 2).rolling(window=4).std()
        
        return noise.fillna(noise.mean())
    
    def _calculate_short_momentum(self, df: pd.DataFrame) -> pd.Series:
        """计算短期动量"""
        # 使用3期ROC
        roc = df['close'].pct_change(periods=3)
        
        # 动量平滑
        momentum = roc.rolling(window=3).mean()
        
        return momentum.fillna(0)
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """计算波动性"""
        # 使用真实波动幅度
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        volatility = true_range.rolling(window=5).mean() / df['close']
        
        return volatility.fillna(volatility.mean())
    
    def _fit_boundary_with_robustness(self, df: pd.DataFrame,
                                    point_indices: List[int],
                                    price_col: str,
                                    boundary_type: str) -> Optional[TrendLine]:
        """拟合边界线（增强鲁棒性）"""
        if len(point_indices) < 2:
            return None
        
        prices = df.iloc[point_indices][price_col].values
        timestamps = df.iloc[point_indices]['timestamp'].values
        
        # 使用加权回归，给中间的点更高权重
        x = np.arange(len(prices))
        weights = np.ones(len(prices))
        
        # 中间部分权重更高
        if len(prices) > 4:
            mid_start = len(prices) // 4
            mid_end = 3 * len(prices) // 4
            weights[mid_start:mid_end] = 1.5
        
        # 加权线性回归
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.utils import check_array
            
            X = x.reshape(-1, 1)
            y = prices
            
            model = LinearRegression()
            model.fit(X, y, sample_weight=weights)
            
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # 计算R²
            y_pred = model.predict(X)
            r_value = np.corrcoef(y, y_pred)[0, 1]
            
        except ImportError:
            # 回退到普通线性回归
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        
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
    
    def _verify_parallel_moderate(self, upper_line: TrendLine,
                                lower_line: TrendLine,
                                df: pd.DataFrame) -> bool:
        """验证平行度（适中要求）"""
        avg_price = df['close'].mean()
        if avg_price <= 0:
            return False
        
        slope_diff = abs(upper_line.slope - lower_line.slope)
        normalized_diff = slope_diff / avg_price
        
        # 5分钟适中的容忍度
        tolerance = 0.28
        
        return normalized_diff <= tolerance
    
    def _check_momentum_consistency(self, df: pd.DataFrame, flagpole: Flagpole) -> float:
        """检查动量一致性"""
        if 'short_momentum' not in df.columns:
            return 1.0
        
        momentum_values = df['short_momentum'].dropna()
        if len(momentum_values) == 0:
            return 1.0
        
        # 检查动量方向的一致性
        if flagpole.direction == 'up':
            # 上升旗形，期望轻微负动量或平稳
            positive_momentum_ratio = (momentum_values > 0.001).sum() / len(momentum_values)
            if positive_momentum_ratio < 0.3:  # 大部分时间动量不强
                return 1.1
            else:
                return 0.95
        else:
            # 下降旗形，期望轻微正动量或平稳
            negative_momentum_ratio = (momentum_values < -0.001).sum() / len(momentum_values)
            if negative_momentum_ratio < 0.3:
                return 1.1
            else:
                return 0.95
    
    def _calculate_volatility_penalty(self, df: pd.DataFrame) -> float:
        """计算波动性惩罚"""
        if 'volatility' not in df.columns:
            return 0.0
        
        avg_volatility = df['volatility'].mean()
        
        # 5分钟数据的波动性阈值
        if avg_volatility > 0.008:  # 0.8%以上
            return min(0.25, (avg_volatility - 0.008) * 20)
        else:
            return 0.0
    
    def _assess_pattern_integrity(self, df: pd.DataFrame, boundaries: List[TrendLine]) -> float:
        """评估形态完整性"""
        if len(boundaries) < 2:
            return 0.8
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 检查价格在通道内的比例
        within_channel_count = 0
        total_count = len(df)
        
        for i in range(len(df)):
            progress = i / (len(df) - 1) if len(df) > 1 else 0
            
            upper_price = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            lower_price = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # 检查是否在通道内（允许5%溢出）
            tolerance = (upper_price - lower_price) * 0.05
            
            if (current_low >= lower_price - tolerance and 
                current_high <= upper_price + tolerance):
                within_channel_count += 1
        
        integrity_ratio = within_channel_count / total_count if total_count > 0 else 0
        
        # 转换为评分
        if integrity_ratio > 0.8:
            return 1.0
        elif integrity_ratio > 0.6:
            return 0.95
        else:
            return 0.9
    
    def _assess_convergence_stability(self, upper_line: TrendLine,
                                    lower_line: TrendLine,
                                    df: pd.DataFrame) -> float:
        """评估收敛稳定性"""
        # 检查收敛是否稳定进行
        pattern_length = len(df)
        
        # 计算理论通道宽度变化
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width <= 0:
            return 0.8
        
        # 理论收敛比例
        theoretical_convergence = (start_width - end_width) / start_width
        
        # 检查实际价格的收敛情况
        first_half_width = []
        second_half_width = []
        
        mid_point = pattern_length // 2
        
        for i in range(pattern_length):
            progress = i / (pattern_length - 1) if pattern_length > 1 else 0
            
            upper_theoretical = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            lower_theoretical = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            actual_width = df.iloc[i]['high'] - df.iloc[i]['low']
            
            if i < mid_point:
                first_half_width.append(actual_width)
            else:
                second_half_width.append(actual_width)
        
        if first_half_width and second_half_width:
            avg_first_width = np.mean(first_half_width)
            avg_second_width = np.mean(second_half_width)
            
            if avg_first_width > 0:
                actual_convergence = (avg_first_width - avg_second_width) / avg_first_width
                
                # 比较理论和实际收敛
                convergence_match = 1 - abs(theoretical_convergence - actual_convergence)
                
                return max(0.7, min(1.0, convergence_match))
        
        return 0.8
    
    def _evaluate_volume_pattern(self, df: pd.DataFrame) -> float:
        """评估成交量模式"""
        if 'volume' not in df.columns:
            return 0.0
        
        volumes = df['volume']
        
        # 检查成交量递减模式
        if len(volumes) >= 6:
            first_third = volumes.iloc[:len(volumes)//3].mean()
            second_third = volumes.iloc[len(volumes)//3:2*len(volumes)//3].mean()
            last_third = volumes.iloc[2*len(volumes)//3:].mean()
            
            # 理想模式：逐步递减
            if first_third > second_third > last_third:
                return 0.15
            elif first_third > last_third:
                return 0.1
            else:
                return 0.05
        
        return 0.0