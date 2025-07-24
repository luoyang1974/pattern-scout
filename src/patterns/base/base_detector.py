"""
基础形态检测器抽象类
为所有形态检测器提供统一接口和共享功能
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
from datetime import datetime
import uuid
from loguru import logger

from src.data.models.base_models import PatternRecord, Flagpole, TrendLine, PatternType
from src.patterns.base.timeframe_manager import TimeframeManager
from src.patterns.base.pattern_components import PatternComponents


class BasePatternDetector(ABC):
    """所有形态检测器的基类"""
    
    def __init__(self, config: dict = None):
        """
        初始化基础检测器
        
        Args:
            config: 配置字典
        """
        self.config = config or self.get_default_config()
        self.timeframe_manager = TimeframeManager()
        self.pattern_components = PatternComponents()
        self._strategy_cache = {}
        
    @abstractmethod
    def get_default_config(self) -> dict:
        """获取默认配置（子类实现）"""
        pass
    
    @abstractmethod
    def get_pattern_type(self) -> PatternType:
        """获取形态类型（子类实现）"""
        pass
    
    def detect(self, df: pd.DataFrame, timeframe: str = None) -> List[PatternRecord]:
        """
        统一的形态检测接口
        
        Args:
            df: OHLCV数据
            timeframe: 时间周期（如 '15m', '1h'）。如果为 None，自动检测
            
        Returns:
            检测到的形态列表
        """
        # 数据验证
        if len(df) < self._get_min_data_points():
            logger.warning(f"Insufficient data for {self.get_pattern_type()} detection")
            return []
        
        # 1. 时间周期处理
        timeframe = self._resolve_timeframe(df, timeframe)
        category = self.timeframe_manager.get_category(timeframe)
        
        logger.info(f"Detecting {self.get_pattern_type()} patterns - "
                   f"Timeframe: {timeframe}, Category: {category}")
        
        # 2. 获取策略和参数
        strategy = self._get_strategy(category)
        params = self._get_params(category, timeframe)
        
        # 3. 数据预处理
        df_processed = strategy.preprocess(df, params)
        
        # 4. 检测旗杆（共享逻辑）
        flagpoles = self.pattern_components.detect_flagpoles_adaptive(
            df_processed, params['flagpole'], category
        )
        
        logger.info(f"Detected {len(flagpoles)} potential flagpoles")
        
        # 5. 检测具体形态（子类实现）
        patterns = self._detect_pattern_formation(
            df_processed, flagpoles, params, strategy
        )
        
        logger.info(f"Total {self.get_pattern_type()} patterns detected: {len(patterns)}")
        
        return patterns
    
    def detect_multi_timeframe(self, df: pd.DataFrame, 
                             timeframes: List[str]) -> Dict[str, List[PatternRecord]]:
        """
        多周期检测
        
        Args:
            df: 原始 OHLCV 数据（通常是最小周期）
            timeframes: 要检测的周期列表，如 ['15m', '1h', '4h']
            
        Returns:
            每个周期的检测结果
        """
        results = {}
        
        for tf in timeframes:
            logger.info(f"Processing timeframe: {tf}")
            
            # 将数据重采样到目标周期
            df_resampled = self.timeframe_manager.resample_data(df, tf)
            
            if df_resampled is not None and len(df_resampled) >= self._get_min_data_points():
                # 在该周期上检测
                patterns = self.detect(df_resampled, tf)
                results[tf] = patterns
            else:
                logger.warning(f"Insufficient data for timeframe {tf}")
                results[tf] = []
        
        return results
    
    @abstractmethod
    def _detect_pattern_formation(self, df: pd.DataFrame, 
                                flagpoles: List[Flagpole],
                                params: dict,
                                strategy) -> List[PatternRecord]:
        """
        检测具体的形态（子类实现）
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            strategy: 周期策略对象
            
        Returns:
            检测到的形态列表
        """
        pass
    
    def _resolve_timeframe(self, df: pd.DataFrame, timeframe: str = None) -> str:
        """解析时间周期"""
        if timeframe is None:
            timeframe = self.timeframe_manager.detect_timeframe(df)
            logger.info(f"Auto-detected timeframe: {timeframe}")
        return timeframe
    
    def _get_strategy(self, category: str):
        """获取周期策略对象（使用缓存）"""
        if category not in self._strategy_cache:
            if category == 'ultra_short':
                from src.patterns.strategies.ultra_short_strategy import UltraShortStrategy
                self._strategy_cache[category] = UltraShortStrategy()
            elif category == 'short':
                from src.patterns.strategies.short_strategy import ShortStrategy
                self._strategy_cache[category] = ShortStrategy()
            else:  # medium_long
                from src.patterns.strategies.medium_long_strategy import MediumLongStrategy
                self._strategy_cache[category] = MediumLongStrategy()
        
        return self._strategy_cache[category]
    
    def _get_params(self, category: str, timeframe: str) -> dict:
        """获取周期相关参数"""
        # 从配置中获取该类别的参数
        category_config = self.config.get('pattern_detection', {}).get(category, {})
        
        # 合并默认参数
        params = {
            'timeframe': timeframe,
            'category': category,
            'flagpole': category_config.get('flagpole', {}),
            'pattern': category_config.get(self.get_pattern_type().lower(), {}),
            'scoring': category_config.get('scoring', {}),
            'min_confidence': self.config.get('scoring', {}).get('min_confidence_score', 0.6)
        }
        
        return params
    
    def _get_min_data_points(self) -> int:
        """获取最小数据点数量"""
        return self.config.get('global', {}).get('min_data_points', 60)
    
    def _create_pattern_record(self, symbol: str, flagpole: Flagpole,
                             boundaries: List[TrendLine], duration: int,
                             confidence_score: float,
                             additional_info: dict = None) -> PatternRecord:
        """创建形态记录"""
        pattern_record = PatternRecord(
            id=str(uuid.uuid4()),
            symbol=symbol,
            pattern_type=self.get_pattern_type(),
            detection_date=datetime.now(),
            flagpole=flagpole,
            pattern_boundaries=boundaries,
            pattern_duration=duration,
            confidence_score=confidence_score,
            pattern_quality=self._classify_pattern_quality(confidence_score)
        )
        
        # 添加额外信息
        if additional_info:
            for key, value in additional_info.items():
                setattr(pattern_record, key, value)
        
        return pattern_record
    
    def _classify_pattern_quality(self, confidence_score: float) -> str:
        """根据置信度分类形态质量"""
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.65:
            return 'medium'
        else:
            return 'low'