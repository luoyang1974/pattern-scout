"""
其他时间周期策略（1小时、4小时、日线、周线、月线）
提供基础实现，继承自BaseStrategy的默认行为
"""
import pandas as pd
from typing import List, Tuple, Optional
from src.patterns.strategies.base_strategy import BaseStrategy
from src.data.models.base_models import Flagpole, TrendLine


class HourOneStrategy(BaseStrategy):
    """1小时周期策略实现"""
    
    def get_category_name(self) -> str:
        return '1h'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """1小时数据预处理"""
        df = df.copy()
        df = self.add_technical_indicators(df)
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """寻找平行边界（旗形）"""
        return self.pattern_components.find_parallel_channel(
            df, flagpole, params, category='1h'
        )
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """寻找关键支撑/阻力点"""
        return self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 12),
            category='1h'
        )
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """评分旗形质量"""
        return self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '1h'
        )
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """评分三角旗形质量"""
        return self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '1h'
        )


class HourFourStrategy(BaseStrategy):
    """4小时周期策略实现"""
    
    def get_category_name(self) -> str:
        return '4h'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """4小时数据预处理"""
        df = df.copy()
        df = self.add_technical_indicators(df)
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """寻找平行边界（旗形）"""
        return self.pattern_components.find_parallel_channel(
            df, flagpole, params, category='4h'
        )
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """寻找关键支撑/阻力点"""
        return self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 15),
            min_prominence=df['close'].std() * 0.005,
            category='4h'
        )
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """评分旗形质量"""
        return self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '4h'
        )
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """评分三角旗形质量"""
        return self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '4h'
        )


class DayOneStrategy(BaseStrategy):
    """日线周期策略实现"""
    
    def get_category_name(self) -> str:
        return '1d'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """日线数据预处理"""
        df = df.copy()
        df = self.add_technical_indicators(df)
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """寻找平行边界（旗形）"""
        return self.pattern_components.find_parallel_channel(
            df, flagpole, params, category='1d'
        )
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """寻找关键支撑/阻力点"""
        return self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 15),
            min_prominence=df['close'].std() * 0.008,
            category='1d'
        )
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """评分旗形质量"""
        return self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '1d'
        )
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """评分三角旗形质量"""
        return self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '1d'
        )


class WeekOneStrategy(BaseStrategy):
    """周线周期策略实现"""
    
    def get_category_name(self) -> str:
        return '1w'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """周线数据预处理"""
        df = df.copy()
        df = self.add_technical_indicators(df)
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """寻找平行边界（旗形）"""
        return self.pattern_components.find_parallel_channel(
            df, flagpole, params, category='1w'
        )
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """寻找关键支撑/阻力点"""
        return self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 20),
            min_prominence=df['close'].std() * 0.01,
            category='1w'
        )
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """评分旗形质量"""
        return self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '1w'
        )
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """评分三角旗形质量"""
        return self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '1w'
        )


class MonthOneStrategy(BaseStrategy):
    """月线周期策略实现"""
    
    def get_category_name(self) -> str:
        return '1M'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """月线数据预处理"""
        df = df.copy()
        df = self.add_technical_indicators(df)
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """寻找平行边界（旗形）"""
        return self.pattern_components.find_parallel_channel(
            df, flagpole, params, category='1M'
        )
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """寻找关键支撑/阻力点"""
        return self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 25),
            min_prominence=df['close'].std() * 0.015,
            category='1M'
        )
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """评分旗形质量"""
        return self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, '1M'
        )
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """评分三角旗形质量"""
        return self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, '1M'
        )