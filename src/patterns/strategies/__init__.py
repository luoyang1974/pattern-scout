"""
周期策略包
提供8个具体时间周期的形态检测策略
"""
from .base_strategy import BaseStrategy
from .minute_one_strategy import MinuteOneStrategy
from .minute_five_strategy import MinuteFiveStrategy
from .minute_fifteen_strategy import MinuteFifteenStrategy
from .timeframe_strategies import (
    HourOneStrategy,
    HourFourStrategy, 
    DayOneStrategy,
    WeekOneStrategy,
    MonthOneStrategy
)

# 策略工厂
from .strategy_factory import StrategyFactory

__all__ = [
    'BaseStrategy',
    # 8个具体周期策略
    'MinuteOneStrategy',
    'MinuteFiveStrategy',
    'MinuteFifteenStrategy',
    'HourOneStrategy',
    'HourFourStrategy',
    'DayOneStrategy',
    'WeekOneStrategy',
    'MonthOneStrategy',
    # 策略工厂
    'StrategyFactory'
]