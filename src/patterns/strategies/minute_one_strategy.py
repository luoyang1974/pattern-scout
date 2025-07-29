"""
1分钟周期策略（超高频交易）
针对1分钟数据的特殊处理
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


class MinuteOneStrategy(BaseStrategy):
    """1分钟周期策略实现"""
    
    def __init__(self):
        """初始化1分钟策略"""
        self.pattern_components = PatternComponents()
        self.quality_scorer = QualityScorer()
        self.noise_threshold = 0.001  # 0.1%的噪音阈值
    
    def get_category_name(self) -> str:
        """获取策略类别名称"""
        return '1m'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        1分钟数据预处理
        - 超强噪音过滤
        - 异常值处理
        - 多层平滑处理
        """
        df = df.copy()
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        # 1. 异常值检测和处理
        df = self._handle_outliers(df)
        
        # 2. 多层价格平滑
        df['close_smooth_1'] = self.smooth_price_data(df['close'], window=2)
        df['close_smooth_2'] = self.smooth_price_data(df['close_smooth_1'], window=3)
        df['close_smooth'] = df['close_smooth_2']
        
        # 对高低价也进行平滑
        df['high_smooth'] = self.smooth_price_data(df['high'], window=2)
        df['low_smooth'] = self.smooth_price_data(df['low'], window=2)
        
        # 3. 超短期噪音指标
        df['noise_level'] = self._calculate_ultra_short_noise(df)
        
        # 4. 微观结构指标
        df['micro_trend'] = self._calculate_micro_trend(df, window=3)
        
        # 5. 高频特征
        df['price_velocity'] = self._calculate_price_velocity(df)
        
        logger.debug(f"Preprocessed 1-minute data: {len(df)} bars")
        
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """
        寻找平行边界（旗形）
        1分钟：使用区域概念，允许最大偏差
        """
        if len(df) < params.get('min_bars', 8):
            return None
        
        # 使用平滑后的数据寻找关键点
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df, 
            window_size=max(1, len(df) // 10),
            min_prominence=df['close'].std() * 0.0005,  # 极低的突出度
            category='1m'
        )
        
        # 确保有足够的点
        if len(support_idx) < 2 or len(resistance_idx) < 2:
            return None
        
        # 拟合边界线（使用平滑数据）
        upper_line = self._fit_boundary_ultra_tolerant(
            df, resistance_idx, 'high_smooth', 'resistance'
        )
        lower_line = self._fit_boundary_ultra_tolerant(
            df, support_idx, 'low_smooth', 'support'
        )
        
        if not upper_line or not lower_line:
            return None
        
        # 验证平行度（最宽松要求）
        if not self._verify_parallel_ultra_tolerant(upper_line, lower_line, df):
            return None
        
        return [upper_line, lower_line]
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值（1分钟特化）"""
        # 使用更严格的Z-score方法
        for col in ['high', 'low', 'close']:
            z_scores = np.abs(stats.zscore(df[col].ffill()))
            
            # 标记异常值（Z-score > 2.5，更严格）
            outliers = z_scores > 2.5
            
            if outliers.any():
                # 使用滚动中位数替换异常值
                df.loc[outliers, col] = df[col].rolling(window=5, center=True).median()
        
        return df
    
    def _calculate_ultra_short_noise(self, df: pd.DataFrame) -> pd.Series:
        """计算超短期噪音水平"""
        # 使用高低价范围和成交量的组合
        hl_range = (df['high'] - df['low']) / df['close']
        
        # 计算价格跳跃
        price_jumps = df['close'].diff().abs() / df['close'].shift(1)
        
        # 综合噪音指标
        noise = (hl_range + price_jumps).rolling(window=3).std()
        
        return noise.fillna(noise.mean())
    
    def _calculate_price_velocity(self, df: pd.DataFrame) -> pd.Series:
        """计算价格速度"""
        # 价格变化的加速度
        returns = df['close'].pct_change()
        velocity = returns.rolling(window=2).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0)
        
        return velocity.fillna(0)
    
    def _fit_boundary_ultra_tolerant(self, df: pd.DataFrame,
                                   point_indices: List[int],
                                   price_col: str,
                                   boundary_type: str) -> Optional[TrendLine]:
        """拟合边界线（超高容忍度）"""
        if len(point_indices) < 2:
            return None
        
        prices = df.iloc[point_indices][price_col].values
        timestamps = df.iloc[point_indices]['timestamp'].values
        
        # 简单的首尾连线法（对1分钟数据最稳定）
        if len(prices) >= 2:
            slope = (prices[-1] - prices[0]) / (len(prices) - 1) if len(prices) > 1 else 0
            intercept = prices[0]
            
            # 计算拟合质量（使用所有点）
            x = np.arange(len(prices))
            predicted = slope * x + intercept
            r_value = np.corrcoef(prices, predicted)[0, 1] if len(prices) > 2 else 0.5
            
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
        
        return None
    
    def _verify_parallel_ultra_tolerant(self, upper_line: TrendLine,
                                      lower_line: TrendLine,
                                      df: pd.DataFrame) -> bool:
        """验证平行度（超高容忍度）"""
        # 1分钟数据允许最大的偏差
        avg_price = df['close'].mean()
        if avg_price <= 0:
            return False
        
        slope_diff = abs(upper_line.slope - lower_line.slope)
        normalized_diff = slope_diff / avg_price
        
        # 超高容忍度
        return normalized_diff <= 0.4
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """
        评分旗形质量
        1分钟：降低几何要求，提高速度和成交量权重
        """
        base_score = self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '1m'
        )
        
        # 1分钟特殊调整
        noise_penalty = self._calculate_noise_penalty(df) * 0.5  # 减少噪音惩罚
        speed_bonus = self._calculate_formation_speed_bonus(df) * 1.5  # 增加速度奖励
        volume_consistency = self._check_volume_consistency(df)
        
        adjusted_score = base_score * (1 - noise_penalty) * (1 + speed_bonus) * volume_consistency
        
        return min(1.0, max(0.0, adjusted_score))
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """
        评分三角旗形质量
        1分钟：注重快速形成，降低精确度要求
        """
        base_score = self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '1m'
        )
        
        # 1分钟特殊调整
        formation_speed_bonus = self._calculate_formation_speed_bonus(df) * 2.0
        micro_structure_score = self._evaluate_micro_structure(df)
        
        adjusted_score = base_score * (1 + formation_speed_bonus) * micro_structure_score
        
        return min(1.0, max(0.0, adjusted_score))
    
    def _calculate_formation_speed_bonus(self, df: pd.DataFrame) -> float:
        """计算形成速度奖励（1分钟优化）"""
        formation_bars = len(df)
        
        if formation_bars <= 10:  # 极快
            return 0.3
        elif formation_bars <= 15:  # 很快
            return 0.2
        elif formation_bars <= 20:  # 快
            return 0.1
        else:
            return 0.0
    
    def _check_volume_consistency(self, df: pd.DataFrame) -> float:
        """检查成交量一致性"""
        if 'volume' not in df.columns:
            return 1.0
        
        # 检查成交量的稳定性（避免异常spike）
        volume_cv = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else float('inf')
        
        if volume_cv < 0.8:  # 相对稳定
            return 1.1
        elif volume_cv < 1.5:
            return 1.0
        else:
            return 0.9
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（三角旗形）
        1分钟：最密集的采样，最低的突出度要求
        """
        # 使用更小的窗口和突出度
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(1, len(df) // 12),
            min_prominence=df['close'].std() * 0.0002,  # 极极低的突出度
            category='1m'
        )
        
        # 如果点太少，降低要求再试一次
        if len(support_idx) < 2 or len(resistance_idx) < 2:
            # 使用简单的局部极值
            support_idx = self._find_local_minima(df['low_smooth'] if 'low_smooth' in df else df['low'], window=2)
            resistance_idx = self._find_local_maxima(df['high_smooth'] if 'high_smooth' in df else df['high'], window=2)
        
        return support_idx, resistance_idx
    
    def _find_local_minima(self, prices: pd.Series, window: int = 2) -> List[int]:
        """寻找局部最小值"""
        minima = []
        
        for i in range(window, len(prices) - window):
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                minima.append(i)
        
        # 添加首尾
        minima = [0] + minima + [len(prices) - 1]
        
        return sorted(list(set(minima)))
    
    def _find_local_maxima(self, prices: pd.Series, window: int = 2) -> List[int]:
        """寻找局部最大值"""
        maxima = []
        
        for i in range(window, len(prices) - window):
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                maxima.append(i)
        
        # 添加首尾
        maxima = [0] + maxima + [len(prices) - 1]
        
        return sorted(list(set(maxima)))