"""
15分钟周期策略（短线交易）
专门为15分钟数据优化的策略
"""
import pandas as pd
from typing import List, Tuple, Optional
from src.patterns.strategies.base_strategy import BaseStrategy
from src.data.models.base_models import Flagpole, TrendLine


class MinuteFifteenStrategy(BaseStrategy):
    """
    15分钟周期策略实现
    专门为15分钟数据优化，适合短线交易
    """
    
    def get_category_name(self) -> str:
        return '15m'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        15分钟数据预处理
        适度平滑，保持敏感性
        """
        df = df.copy()
        df = self.add_technical_indicators(df)
        
        # 适度平滑，去除高频噪音
        if len(df) > 10:
            # 对关键价格序列进行平滑
            df['close'] = self.smooth_price_data(df['close'], window=3)
            df['high'] = self.smooth_price_data(df['high'], window=3)
            df['low'] = self.smooth_price_data(df['low'], window=3)
        
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """寻找平行边界（旗形）"""
        return self.pattern_components.find_parallel_channel(
            df, flagpole, params, category='15m'
        )
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（三角旗形）
        15分钟：标准的峰谷识别
        """
        # 使用标准参数
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 12),
            category='15m'
        )
        
        # 确保有足够的点
        min_points = 3
        if len(support_idx) < min_points or len(resistance_idx) < min_points:
            # 尝试降低要求
            support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
                df,
                window_size=max(2, len(df) // 15),
                min_prominence=df['close'].std() * 0.001,
                category='15m'
            )
        
        return support_idx, resistance_idx
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """评分旗形质量"""
        return self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '15m'
        )
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """评分三角旗形质量"""
        return self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '15m'
        )