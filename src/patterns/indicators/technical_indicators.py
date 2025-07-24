import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy import stats
import talib


class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def moving_average(prices: pd.Series, window: int) -> pd.Series:
        """计算移动平均线"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        try:
            return pd.Series(talib.RSI(prices.values, timeperiod=window), index=prices.index)
        except:
            # 备用计算方法
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        try:
            macd_line, macd_signal, macd_histogram = talib.MACD(
                prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return (
                pd.Series(macd_line, index=prices.index),
                pd.Series(macd_signal, index=prices.index),
                pd.Series(macd_histogram, index=prices.index)
            )
        except:
            # 备用计算方法
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line, macd_signal, macd_histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        try:
            upper, middle, lower = talib.BBANDS(
                prices.values, timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev
            )
            return (
                pd.Series(upper, index=prices.index),
                pd.Series(middle, index=prices.index),
                pd.Series(lower, index=prices.index)
            )
        except:
            # 备用计算方法
            middle = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """计算随机指标"""
        try:
            slowk, slowd = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=k_window, slowk_period=3, slowd_period=d_window
            )
            return (
                pd.Series(slowk, index=close.index),
                pd.Series(slowd, index=close.index)
            )
        except:
            # 备用计算方法
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_window).mean()
            return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        try:
            atr_values = talib.ATR(high.values, low.values, close.values, timeperiod=window)
            return pd.Series(atr_values, index=close.index)
        except:
            # 备用计算方法
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=window).mean()
    
    @staticmethod
    def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
        """计算成交量移动平均"""
        return volume.rolling(window=window).mean()
    
    @staticmethod
    def price_momentum(prices: pd.Series, window: int = 10) -> pd.Series:
        """计算价格动量"""
        return prices / prices.shift(window) - 1
    
    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        """计算成交量比率"""
        volume_avg = volume.rolling(window=window).mean()
        return volume / volume_avg


class TrendAnalyzer:
    """趋势分析器"""
    
    @staticmethod
    def detect_trend_direction(prices: pd.Series, window: int = 20) -> str:
        """检测趋势方向"""
        if len(prices) < window:
            return "unknown"
        
        # 使用线性回归检测趋势
        recent_prices = prices.tail(window)
        x = np.arange(len(recent_prices))
        slope, _, r_value, _, _ = stats.linregress(x, recent_prices)
        
        # 根据斜率和R²值判断趋势
        if abs(r_value) > 0.7:  # 相关性足够强
            if slope > 0:
                return "uptrend"
            else:
                return "downtrend"
        else:
            return "sideways"
    
    @staticmethod
    def calculate_trend_strength(prices: pd.Series, window: int = 20) -> float:
        """计算趋势强度"""
        if len(prices) < window:
            return 0.0
        
        recent_prices = prices.tail(window)
        x = np.arange(len(recent_prices))
        _, _, r_value, _, _ = stats.linregress(x, recent_prices)
        
        return abs(r_value)
    
    @staticmethod
    def find_support_resistance_levels(prices: pd.Series, window: int = 20, 
                                     min_touches: int = 2) -> Tuple[List[float], List[float]]:
        """寻找支撑阻力位"""
        if len(prices) < window * 2:
            return [], []
        
        # 找到局部极值点
        highs = []
        lows = []
        
        for i in range(window, len(prices) - window):
            price_window = prices.iloc[i-window:i+window+1]
            current_price = prices.iloc[i]
            
            # 检查是否为局部最高点
            if current_price == price_window.max():
                highs.append(current_price)
            
            # 检查是否为局部最低点
            if current_price == price_window.min():
                lows.append(current_price)
        
        # 聚类相近的价格水平
        def cluster_levels(levels: List[float], tolerance: float = 0.01) -> List[float]:
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= min_touches:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))
            
            return clusters
        
        resistance_levels = cluster_levels(highs)
        support_levels = cluster_levels(lows)
        
        return support_levels, resistance_levels


class PatternIndicators:
    """形态相关指标"""
    
    @staticmethod
    def calculate_price_range(high: pd.Series, low: pd.Series) -> pd.Series:
        """计算价格区间"""
        return (high - low) / low * 100
    
    @staticmethod
    def calculate_body_size(open_price: pd.Series, close: pd.Series) -> pd.Series:
        """计算实体大小"""
        return abs(close - open_price) / open_price * 100
    
    @staticmethod
    def calculate_upper_shadow(high: pd.Series, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """计算上影线"""
        body_high = pd.concat([open_price, close], axis=1).max(axis=1)
        return (high - body_high) / body_high * 100
    
    @staticmethod
    def calculate_lower_shadow(low: pd.Series, open_price: pd.Series, close: pd.Series) -> pd.Series:
        """计算下影线"""
        body_low = pd.concat([open_price, close], axis=1).min(axis=1)
        return (body_low - low) / body_low * 100
    
    @staticmethod
    def detect_high_volume_bars(volume: pd.Series, window: int = 20, threshold: float = 1.5) -> pd.Series:
        """检测高成交量K线"""
        volume_avg = volume.rolling(window=window).mean()
        return volume > (volume_avg * threshold)
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """计算波动率"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # 年化波动率