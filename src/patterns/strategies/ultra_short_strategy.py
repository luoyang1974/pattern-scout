"""
超短周期策略（1-5分钟）
针对高频数据的特殊处理
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


class UltraShortStrategy(BaseStrategy):
    """超短周期策略实现"""
    
    def __init__(self):
        """初始化超短周期策略"""
        self.pattern_components = PatternComponents()
        self.quality_scorer = QualityScorer()
        self.noise_threshold = 0.002  # 0.2%的噪音阈值
    
    def get_category_name(self) -> str:
        """获取策略类别名称"""
        return 'ultra_short'
    
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        超短周期数据预处理
        - 强噪音过滤
        - 异常值处理
        - 平滑处理
        """
        df = df.copy()
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        # 1. 异常值检测和处理
        df = self._handle_outliers(df)
        
        # 2. 价格平滑（保留原始数据）
        df['close_smooth'] = self.smooth_price_data(df['close'], window=3)
        df['high_smooth'] = self.smooth_price_data(df['high'], window=3)
        df['low_smooth'] = self.smooth_price_data(df['low'], window=3)
        
        # 3. 噪音指标
        df['noise_level'] = self._calculate_noise_level(df)
        
        # 4. 微观结构指标
        df['micro_trend'] = self._calculate_micro_trend(df)
        
        logger.debug(f"Preprocessed ultra-short data: {len(df)} bars")
        
        return df
    
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """
        寻找平行边界（旗形）
        超短周期特点：使用区域而非精确线，允许更大的偏差
        """
        if len(df) < params.get('min_bars', 10):
            return None
        
        # 使用平滑后的数据寻找关键点
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df, 
            window_size=max(2, len(df) // 8),
            min_prominence=df['close'].std() * 0.001,  # 很小的突出度
            category='ultra_short'
        )
        
        # 确保有足够的点
        if len(support_idx) < 2 or len(resistance_idx) < 2:
            return None
        
        # 拟合边界线（使用平滑数据）
        upper_line = self._fit_boundary_with_tolerance(
            df, resistance_idx, 'high_smooth', 'resistance'
        )
        lower_line = self._fit_boundary_with_tolerance(
            df, support_idx, 'low_smooth', 'support'
        )
        
        if not upper_line or not lower_line:
            return None
        
        # 验证平行度（放宽要求）
        if not self._verify_parallel_with_tolerance(upper_line, lower_line, df, tolerance=0.25):
            return None
        
        # 验证反向倾斜
        if not self._verify_opposite_slope_ultra_short(flagpole, upper_line, lower_line):
            return None
        
        return [upper_line, lower_line]
    
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（三角旗形）
        超短周期：更密集的采样，更低的突出度要求
        """
        # 使用更小的窗口和突出度
        support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
            df,
            window_size=max(2, len(df) // 10),
            min_prominence=df['close'].std() * 0.0005,  # 极低的突出度
            category='ultra_short'
        )
        
        # 如果点太少，降低要求再试一次
        if len(support_idx) < 3 or len(resistance_idx) < 3:
            # 使用简单的局部极值
            support_idx = self._find_local_minima(df['low_smooth'] if 'low_smooth' in df else df['low'])
            resistance_idx = self._find_local_maxima(df['high_smooth'] if 'high_smooth' in df else df['high'])
        
        return support_idx, resistance_idx
    
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """
        评分旗形质量
        超短周期：降低形态完整性要求，增加成交量和突破确认权重
        """
        # 使用专门的超短周期评分
        base_score = self.quality_scorer.score_flag_pattern(
            df, flagpole, boundaries, 'ultra_short'
        )
        
        # 额外的超短周期调整
        noise_penalty = self._calculate_noise_penalty(df)
        momentum_bonus = self._calculate_momentum_bonus(df, flagpole)
        
        # 调整最终得分
        adjusted_score = base_score * (1 - noise_penalty) * (1 + momentum_bonus)
        
        return min(1.0, max(0.0, adjusted_score))
    
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """
        评分三角旗形质量
        超短周期：更宽松的收敛要求，重视快速形成
        """
        # 使用专门的超短周期评分
        base_score = self.quality_scorer.score_pennant_pattern(
            df, flagpole, [upper_line, lower_line], apex, 'ultra_short'
        )
        
        # 超短周期特殊调整
        formation_speed_bonus = self._calculate_formation_speed_bonus(df, flagpole)
        volatility_penalty = self._calculate_volatility_penalty(df)
        
        # 调整最终得分
        adjusted_score = base_score * (1 + formation_speed_bonus) * (1 - volatility_penalty)
        
        return min(1.0, max(0.0, adjusted_score))
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        # 使用Z-score方法检测异常值
        for col in ['high', 'low', 'close']:
            z_scores = np.abs(stats.zscore(df[col].ffill()))
            
            # 标记异常值（Z-score > 3）
            outliers = z_scores > 3
            
            if outliers.any():
                # 使用前后值的平均替换异常值
                df.loc[outliers, col] = df[col].rolling(window=3, center=True).mean()
        
        return df
    
    def _calculate_noise_level(self, df: pd.DataFrame) -> pd.Series:
        """计算噪音水平"""
        # 使用高低价波动率作为噪音指标
        hl_range = (df['high'] - df['low']) / df['close']
        noise = hl_range.rolling(window=5).std()
        
        return noise.fillna(noise.mean())
    
    def _calculate_micro_trend(self, df: pd.DataFrame) -> pd.Series:
        """计算微观趋势"""
        # 使用短期线性回归斜率
        window = 5
        micro_trends = []
        
        for i in range(len(df)):
            if i < window - 1:
                micro_trends.append(0)
            else:
                y = df['close'].iloc[i-window+1:i+1].values
                x = np.arange(window)
                
                if len(y) == window:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    micro_trends.append(slope)
                else:
                    micro_trends.append(0)
        
        return pd.Series(micro_trends, index=df.index)
    
    def _fit_boundary_with_tolerance(self, df: pd.DataFrame, 
                                   point_indices: List[int],
                                   price_col: str,
                                   boundary_type: str) -> Optional[TrendLine]:
        """拟合边界线（带容差）"""
        if len(point_indices) < 2:
            return None
        
        # 提取价格
        prices = df.iloc[point_indices][price_col].values
        timestamps = df.iloc[point_indices]['timestamp'].values
        
        # 尝试多种拟合方法
        # 1. 标准线性回归
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        
        # 2. 如果拟合质量太差，使用稳健回归
        if abs(r_value) < 0.3:
            # 使用RANSAC或其他稳健方法
            # 这里简化使用首尾连线
            slope = (prices[-1] - prices[0]) / (len(prices) - 1) if len(prices) > 1 else 0
            intercept = prices[0]
            r_value = 0.5  # 给一个中等的拟合度
        
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
    
    def _verify_parallel_with_tolerance(self, upper_line: TrendLine, 
                                      lower_line: TrendLine,
                                      df: pd.DataFrame,
                                      tolerance: float = 0.25) -> bool:
        """验证平行度（带容差）"""
        # 计算标准化的斜率差异
        avg_price = df['close'].mean()
        if avg_price <= 0:
            return False
        
        slope_diff = abs(upper_line.slope - lower_line.slope)
        normalized_diff = slope_diff / avg_price
        
        # 超短周期允许更大的偏差
        return normalized_diff <= tolerance
    
    def _verify_opposite_slope_ultra_short(self, flagpole: Flagpole,
                                         upper_line: TrendLine,
                                         lower_line: TrendLine) -> bool:
        """验证反向倾斜（超短周期版本）"""
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        
        # 超短周期的角度要求更宽松
        if flagpole.direction == 'up':
            # 上升旗形，旗面应该向下倾斜，但允许轻微向上
            return avg_slope < 0.001  # 允许轻微正斜率
        else:
            # 下降旗形，旗面应该向上倾斜，但允许轻微向下
            return avg_slope > -0.001  # 允许轻微负斜率
    
    def _find_local_minima(self, prices: pd.Series, window: int = 3) -> List[int]:
        """寻找局部最小值"""
        minima = []
        
        for i in range(window, len(prices) - window):
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                minima.append(i)
        
        # 添加首尾
        minima = [0] + minima + [len(prices) - 1]
        
        return sorted(list(set(minima)))
    
    def _find_local_maxima(self, prices: pd.Series, window: int = 3) -> List[int]:
        """寻找局部最大值"""
        maxima = []
        
        for i in range(window, len(prices) - window):
            if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                maxima.append(i)
        
        # 添加首尾
        maxima = [0] + maxima + [len(prices) - 1]
        
        return sorted(list(set(maxima)))
    
    def _calculate_noise_penalty(self, df: pd.DataFrame) -> float:
        """计算噪音惩罚"""
        if 'noise_level' not in df.columns:
            return 0.0
        
        avg_noise = df['noise_level'].mean()
        
        # 根据噪音水平计算惩罚
        if avg_noise < 0.002:  # 低噪音
            return 0.0
        elif avg_noise < 0.005:  # 中等噪音
            return 0.1
        else:  # 高噪音
            return 0.2
    
    def _calculate_momentum_bonus(self, df: pd.DataFrame, flagpole: Flagpole) -> float:
        """计算动量奖励"""
        if 'micro_trend' not in df.columns:
            return 0.0
        
        # 检查微观趋势是否与旗杆方向一致
        avg_micro_trend = df['micro_trend'].mean()
        
        if flagpole.direction == 'up' and avg_micro_trend > 0:
            return min(0.2, avg_micro_trend * 10)
        elif flagpole.direction == 'down' and avg_micro_trend < 0:
            return min(0.2, abs(avg_micro_trend) * 10)
        else:
            return 0.0
    
    def _calculate_formation_speed_bonus(self, df: pd.DataFrame, 
                                       flagpole: Flagpole) -> float:
        """计算形成速度奖励"""
        # 形态形成越快越好（对于超短周期）
        formation_bars = len(df)
        
        if formation_bars <= 15:  # 非常快
            return 0.2
        elif formation_bars <= 25:  # 快
            return 0.1
        else:
            return 0.0
    
    def _calculate_volatility_penalty(self, df: pd.DataFrame) -> float:
        """计算波动率惩罚"""
        # 计算价格波动率
        returns = df['close'].pct_change().dropna()
        
        if len(returns) > 0:
            volatility = returns.std()
            
            # 过高的波动率要惩罚
            if volatility > 0.01:  # 1%以上
                return min(0.3, (volatility - 0.01) * 10)
            else:
                return 0.0
        
        return 0.0