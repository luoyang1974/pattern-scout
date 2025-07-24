"""
周期策略基类
定义不同周期策略的通用接口
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple, Optional
from scipy.signal import savgol_filter

from src.data.models.base_models import Flagpole, TrendLine


class BaseStrategy(ABC):
    """周期策略基类"""
    
    @abstractmethod
    def get_category_name(self) -> str:
        """获取策略类别名称"""
        pass
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            df: 原始OHLCV数据
            params: 参数配置
            
        Returns:
            预处理后的数据
        """
        pass
    
    @abstractmethod
    def find_parallel_boundaries(self, df: pd.DataFrame, 
                               flagpole: Flagpole, 
                               params: dict) -> Optional[List[TrendLine]]:
        """
        寻找平行边界（用于旗形）
        
        Args:
            df: 形态数据
            flagpole: 旗杆对象
            params: 参数配置
            
        Returns:
            [上边界, 下边界] 或 None
        """
        pass
    
    @abstractmethod
    def find_key_points(self, df: pd.DataFrame, 
                       flagpole: Flagpole) -> Tuple[List[int], List[int]]:
        """
        寻找关键支撑/阻力点（用于三角旗形）
        
        Args:
            df: 形态数据
            flagpole: 旗杆对象
            
        Returns:
            (支撑点索引, 阻力点索引)
        """
        pass
    
    @abstractmethod
    def score_flag_quality(self, df: pd.DataFrame,
                         flagpole: Flagpole,
                         boundaries: List[TrendLine]) -> float:
        """
        评分旗形质量
        
        Args:
            df: 形态数据
            flagpole: 旗杆对象
            boundaries: 边界线
            
        Returns:
            质量得分（0-1）
        """
        pass
    
    @abstractmethod
    def score_pennant_quality(self, df: pd.DataFrame,
                            flagpole: Flagpole,
                            upper_line: TrendLine,
                            lower_line: TrendLine,
                            apex: Tuple[float, float]) -> float:
        """
        评分三角旗形质量
        
        Args:
            df: 形态数据
            flagpole: 旗杆对象
            upper_line: 上边界线
            lower_line: 下边界线
            apex: 收敛点
            
        Returns:
            质量得分（0-1）
        """
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加通用技术指标"""
        df = df.copy()
        
        # 成交量指标
        if 'volume' in df.columns:
            df['volume_sma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20'].fillna(1)
        
        # 价格变化
        df['price_change'] = df['close'].pct_change()
        
        # ATR（平均真实范围）
        df['atr'] = self.calculate_atr(df)
        
        # 价格范围
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def smooth_price_data(self, prices: pd.Series, window: int = None) -> pd.Series:
        """平滑价格数据（用于降噪）"""
        if window is None:
            window = max(3, len(prices) // 20)
        
        if len(prices) < window:
            return prices
        
        # 使用Savitzky-Golay滤波器进行平滑
        try:
            window_length = window if window % 2 == 1 else window + 1
            polyorder = min(3, window_length - 1)
            smoothed = savgol_filter(prices, window_length, polyorder)
            return pd.Series(smoothed, index=prices.index)
        except:
            # 如果失败，使用简单移动平均
            return prices.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    def validate_boundaries(self, upper_line: TrendLine, 
                          lower_line: TrendLine,
                          min_r_squared: float = 0.3) -> bool:
        """验证边界线质量"""
        # 检查拟合质量
        if upper_line.r_squared < min_r_squared or lower_line.r_squared < min_r_squared:
            return False
        
        # 检查边界不交叉
        if upper_line.start_price <= lower_line.start_price:
            return False
        
        if upper_line.end_price <= lower_line.end_price:
            return False
        
        return True