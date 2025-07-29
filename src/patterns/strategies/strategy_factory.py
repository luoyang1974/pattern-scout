"""
策略工厂
根据时间周期自动选择合适的策略
"""
from typing import Optional
from loguru import logger

from src.patterns.strategies.base_strategy import BaseStrategy
from src.patterns.strategies.minute_one_strategy import MinuteOneStrategy
from src.patterns.strategies.minute_five_strategy import MinuteFiveStrategy
from src.patterns.strategies.minute_fifteen_strategy import MinuteFifteenStrategy
from src.patterns.strategies.timeframe_strategies import (
    HourOneStrategy,
    HourFourStrategy,
    DayOneStrategy,
    WeekOneStrategy,
    MonthOneStrategy
)


class StrategyFactory:
    """策略工厂类"""
    
    # 8个具体周期到策略的映射
    STRATEGY_MAP = {
        '1m': MinuteOneStrategy,
        '5m': MinuteFiveStrategy,
        '15m': MinuteFifteenStrategy,
        '1h': HourOneStrategy,
        '4h': HourFourStrategy,
        '1d': DayOneStrategy,
        '1w': WeekOneStrategy,
        '1M': MonthOneStrategy,
    }
    
    @classmethod
    def get_strategy(cls, timeframe: str) -> Optional[BaseStrategy]:
        """
        根据时间周期获取策略
        
        Args:
            timeframe: 时间周期字符串，如 '1m', '15m', '1h' 等
            
        Returns:
            对应的策略实例
        """
        if timeframe in cls.STRATEGY_MAP:
            strategy_class = cls.STRATEGY_MAP[timeframe]
            logger.debug(f"Selected strategy {strategy_class.__name__} for timeframe {timeframe}")
            return strategy_class()
        
        # 如果没有找到，提供默认策略
        logger.warning(f"No specific strategy found for timeframe {timeframe}, using 15m strategy as default")
        return MinuteFifteenStrategy()
    
    @classmethod
    def list_supported_timeframes(cls) -> list:
        """获取支持的时间周期列表"""
        return list(cls.STRATEGY_MAP.keys())