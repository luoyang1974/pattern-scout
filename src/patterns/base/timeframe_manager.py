"""
时间周期管理器
负责周期检测、分类和数据重采样
"""
import pandas as pd
from typing import Optional, Dict, Tuple
from loguru import logger


class TimeframeManager:
    """智能时间周期管理"""
    
    # 周期分类定义
    CATEGORIES = {
        'ultra_short': {
            'timeframes': ['1m', '3m', '5m'],
            'bars_per_hour': [60, 20, 12],
            'description': '超短周期（1-5分钟）'
        },
        'short': {
            'timeframes': ['15m', '30m', '1h'],
            'bars_per_hour': [4, 2, 1],
            'description': '短周期（15分钟-1小时）'
        },
        'medium_long': {
            'timeframes': ['4h', '1d', '1w'],
            'bars_per_hour': [0.25, 0.042, 0.006],
            'description': '中长周期（4小时-1周）'
        }
    }
    
    # 时间周期映射（秒数）
    TIMEFRAME_SECONDS = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800
    }
    
    # 重采样规则
    RESAMPLE_RULES = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    def detect_timeframe(self, df: pd.DataFrame) -> str:
        """
        基于数据间隔智能检测时间周期
        
        Args:
            df: OHLCV数据
            
        Returns:
            检测到的时间周期字符串（如 '15m'）
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            logger.warning("Cannot detect timeframe: insufficient data")
            return '15m'  # 默认返回15分钟
        
        # 计算时间间隔
        timestamps = pd.to_datetime(df['timestamp'])
        intervals = timestamps.diff().dropna()
        
        # 过滤异常值（周末、节假日等导致的大间隔）
        intervals_seconds = intervals.dt.total_seconds()
        
        # 使用中位数更稳定
        median_interval = intervals_seconds.median()
        
        # 找到最接近的标准周期
        closest_timeframe = '15m'
        min_diff = float('inf')
        
        for tf, seconds in self.TIMEFRAME_SECONDS.items():
            diff = abs(median_interval - seconds)
            if diff < min_diff:
                min_diff = diff
                closest_timeframe = tf
        
        # 容错：如果偏差太大，使用更智能的方法
        if min_diff > median_interval * 0.2:  # 20%容错
            closest_timeframe = self._detect_timeframe_fuzzy(intervals_seconds)
        
        logger.debug(f"Detected timeframe: {closest_timeframe} "
                    f"(median interval: {median_interval:.0f}s)")
        
        return closest_timeframe
    
    def _detect_timeframe_fuzzy(self, intervals: pd.Series) -> str:
        """模糊时间周期检测（处理不规则数据）"""
        # 使用聚类方法找到主要的时间间隔
        # 这里简化处理，实际可以使用更复杂的聚类算法
        
        # 计算常见间隔的出现频率
        interval_counts = {}
        for tf, seconds in self.TIMEFRAME_SECONDS.items():
            # 计算在该周期±10%范围内的间隔数量
            tolerance = seconds * 0.1
            count = ((intervals >= seconds - tolerance) & 
                    (intervals <= seconds + tolerance)).sum()
            interval_counts[tf] = count
        
        # 返回出现最多的周期
        return max(interval_counts, key=interval_counts.get)
    
    def get_category(self, timeframe: str) -> str:
        """
        获取时间周期所属类别
        
        Args:
            timeframe: 时间周期字符串
            
        Returns:
            类别名称（'ultra_short', 'short', 'medium_long'）
        """
        for category, info in self.CATEGORIES.items():
            if timeframe in info['timeframes']:
                return category
        
        # 默认返回短周期
        logger.warning(f"Unknown timeframe {timeframe}, defaulting to 'short' category")
        return 'short'
    
    def normalize_params(self, params: dict, timeframe: str) -> dict:
        """
        将基于时间的参数转换为K线根数
        
        Args:
            params: 原始参数字典
            timeframe: 时间周期
            
        Returns:
            转换后的参数字典
        """
        normalized = params.copy()
        
        # 获取每小时的K线数
        category = self.get_category(timeframe)
        bars_per_hour = None
        
        for cat, info in self.CATEGORIES.items():
            if cat == category:
                tf_index = info['timeframes'].index(timeframe) if timeframe in info['timeframes'] else 0
                bars_per_hour = info['bars_per_hour'][tf_index]
                break
        
        if bars_per_hour is None:
            bars_per_hour = 4  # 默认15分钟
        
        # 转换时间参数
        time_params = ['min_duration_hours', 'max_duration_hours', 
                      'min_formation_hours', 'max_formation_hours']
        
        for param in time_params:
            if param in normalized:
                # 转换小时数为K线根数
                bar_param = param.replace('_hours', '_bars')
                normalized[bar_param] = int(normalized[param] * bars_per_hour)
        
        return normalized
    
    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> Optional[pd.DataFrame]:
        """
        将数据重采样到目标时间周期
        
        Args:
            df: 原始OHLCV数据
            target_timeframe: 目标时间周期
            
        Returns:
            重采样后的数据，如果失败返回None
        """
        try:
            # 确保时间戳列是datetime类型
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 检测当前周期
            current_timeframe = self.detect_timeframe(df)
            current_seconds = self.TIMEFRAME_SECONDS.get(current_timeframe, 900)
            target_seconds = self.TIMEFRAME_SECONDS.get(target_timeframe, 900)
            
            # 如果目标周期小于当前周期，无法重采样
            if target_seconds < current_seconds:
                logger.error(f"Cannot resample from {current_timeframe} to {target_timeframe} "
                           "(target timeframe is smaller)")
                return None
            
            # 如果周期相同，直接返回
            if target_seconds == current_seconds:
                return df
            
            # 设置时间戳为索引
            df.set_index('timestamp', inplace=True)
            
            # 转换周期字符串为pandas格式
            resample_freq = self._get_pandas_freq(target_timeframe)
            
            # 执行重采样
            resampled = df.resample(resample_freq).agg(self.RESAMPLE_RULES)
            
            # 过滤空行
            resampled = resampled.dropna()
            
            # 重置索引
            resampled.reset_index(inplace=True)
            
            # 保留symbol列
            if 'symbol' in df.columns:
                resampled['symbol'] = df['symbol'].iloc[0]
            
            logger.info(f"Resampled data from {current_timeframe} to {target_timeframe}: "
                       f"{len(df)} -> {len(resampled)} bars")
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return None
    
    def _get_pandas_freq(self, timeframe: str) -> str:
        """将时间周期转换为pandas频率字符串"""
        freq_map = {
            '1m': '1min',
            '3m': '3min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D',
            '1w': '1W'
        }
        return freq_map.get(timeframe, '15min')
    
    def get_lookback_periods(self, timeframe: str, pattern_type: str) -> Dict[str, int]:
        """
        获取不同形态在特定周期下的回看周期数
        
        Args:
            timeframe: 时间周期
            pattern_type: 形态类型（'flag' 或 'pennant'）
            
        Returns:
            包含各种回看周期的字典
        """
        category = self.get_category(timeframe)
        
        # 基础回看周期配置
        lookback_configs = {
            'ultra_short': {
                'flag': {'min_total_bars': 20, 'max_total_bars': 80, 'pre_pattern_bars': 20},
                'pennant': {'min_total_bars': 20, 'max_total_bars': 80, 'pre_pattern_bars': 20}
            },
            'short': {
                'flag': {'min_total_bars': 15, 'max_total_bars': 60, 'pre_pattern_bars': 15},
                'pennant': {'min_total_bars': 15, 'max_total_bars': 60, 'pre_pattern_bars': 15}
            },
            'medium_long': {
                'flag': {'min_total_bars': 10, 'max_total_bars': 40, 'pre_pattern_bars': 10},
                'pennant': {'min_total_bars': 10, 'max_total_bars': 40, 'pre_pattern_bars': 10}
            }
        }
        
        return lookback_configs.get(category, {}).get(pattern_type, {})
    
    def validate_data_sufficiency(self, df: pd.DataFrame, timeframe: str, 
                                pattern_type: str) -> Tuple[bool, str]:
        """
        验证数据是否足够进行形态检测
        
        Args:
            df: OHLCV数据
            timeframe: 时间周期
            pattern_type: 形态类型
            
        Returns:
            (是否足够, 错误信息)
        """
        lookback = self.get_lookback_periods(timeframe, pattern_type)
        min_bars = lookback.get('min_total_bars', 20)
        
        if len(df) < min_bars:
            return False, f"Insufficient data: {len(df)} bars < {min_bars} required"
        
        # 检查时间连续性
        timestamps = pd.to_datetime(df['timestamp'])
        intervals = timestamps.diff().dropna()
        
        # 检查是否有大的时间间隔（可能的数据缺失）
        expected_interval = self.TIMEFRAME_SECONDS.get(timeframe, 900)
        max_gap = expected_interval * 5  # 允许5倍的间隔（考虑周末等）
        
        large_gaps = (intervals.dt.total_seconds() > max_gap).sum()
        if large_gaps > len(df) * 0.1:  # 超过10%的大间隔
            return False, f"Too many time gaps in data: {large_gaps} large gaps found"
        
        return True, "Data is sufficient"