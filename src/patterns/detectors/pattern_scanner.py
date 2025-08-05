"""
统一的形态扫描器
提供便捷的接口来检测多种形态
"""
import pandas as pd
from typing import List, Dict, Optional
from loguru import logger

from src.data.models.base_models import PatternRecord, PatternType, FlagSubType
from src.patterns.detectors import FlagDetector


class PatternScanner:
    """
    统一的形态扫描器
    支持单一或多种形态的检测
    """
    
    def __init__(self, config: dict = None):
        """
        初始化扫描器
        
        Args:
            config: 配置字典（会传递给各个检测器）
        """
        self.config = config or {}
        
        # 初始化统一的旗形检测器
        self.flag_detector = FlagDetector(config)
        
        # 保持向后兼容的检测器映射
        self.detectors = {
            PatternType.FLAG_PATTERN: self.flag_detector
        }
    
    def scan(self, 
             df: pd.DataFrame,
             pattern_types: Optional[List[str]] = None,
             timeframe: Optional[str] = None) -> Dict[str, List[PatternRecord]]:
        """
        扫描指定的形态
        
        Args:
            df: OHLCV数据
            pattern_types: 要检测的形态类型列表，None表示检测所有
            timeframe: 时间周期，None表示自动检测
            
        Returns:
            按形态类型分组的检测结果
        """
        if pattern_types is None:
            # 默认检测旗形形态（包含矩形旗和三角旗）
            pattern_types = [PatternType.FLAG_PATTERN]
        
        results = {}
        
        for pattern_type in pattern_types:
            if pattern_type in self.detectors:
                logger.info(f"Scanning for {pattern_type} patterns...")
                
                detector = self.detectors[pattern_type]
                patterns = detector.detect(df, timeframe)
                
                results[pattern_type] = patterns
                
                logger.info(f"Found {len(patterns)} {pattern_type} patterns")
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}")
        
        return results
    
    def scan_multi_timeframe(self,
                           df: pd.DataFrame,
                           timeframes: List[str],
                           pattern_types: Optional[List[str]] = None) -> Dict[str, Dict[str, List[PatternRecord]]]:
        """
        多周期扫描
        
        Args:
            df: 原始OHLCV数据（最小周期）
            timeframes: 要扫描的时间周期列表
            pattern_types: 要检测的形态类型列表
            
        Returns:
            嵌套字典：{pattern_type: {timeframe: patterns}}
        """
        if pattern_types is None:
            pattern_types = [PatternType.FLAG_PATTERN]
        
        results = {}
        
        for pattern_type in pattern_types:
            if pattern_type in self.detectors:
                logger.info(f"Multi-timeframe scanning for {pattern_type} patterns...")
                
                detector = self.detectors[pattern_type]
                timeframe_results = detector.detect_multi_timeframe(df, timeframes)
                
                results[pattern_type] = timeframe_results
                
                # 统计结果
                total_patterns = sum(len(patterns) for patterns in timeframe_results.values())
                logger.info(f"Found {total_patterns} {pattern_type} patterns across {len(timeframes)} timeframes")
        
        return results
    
    def scan_single_pattern(self,
                          df: pd.DataFrame,
                          pattern_type: str,
                          timeframe: Optional[str] = None) -> List[PatternRecord]:
        """
        扫描单一形态类型（便捷方法）
        
        Args:
            df: OHLCV数据
            pattern_type: 形态类型
            timeframe: 时间周期
            
        Returns:
            检测到的形态列表
        """
        if pattern_type not in self.detectors:
            logger.error(f"Unknown pattern type: {pattern_type}")
            return []
        
        detector = self.detectors[pattern_type]
        return detector.detect(df, timeframe)
    
    def get_detector(self, pattern_type: str):
        """
        获取特定的检测器实例
        
        Args:
            pattern_type: 形态类型
            
        Returns:
            检测器实例或None
        """
        return self.detectors.get(pattern_type)
    
    def update_config(self, config: dict, pattern_type: Optional[str] = None):
        """
        更新配置
        
        Args:
            config: 新的配置字典
            pattern_type: 如果指定，只更新该类型的检测器配置
        """
        if pattern_type:
            if pattern_type in self.detectors:
                self.detectors[pattern_type].config.update(config)
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}")
        else:
            # 更新所有检测器的配置
            self.config.update(config)
            self.flag_detector.config.update(config)
    
    # 向后兼容性方法 - 支持旧的分离式API
    def scan_flags(self, df: pd.DataFrame, timeframe: Optional[str] = None) -> List[PatternRecord]:
        """
        扫描矩形旗形态（向后兼容）
        
        Args:
            df: OHLCV数据
            timeframe: 时间周期
            
        Returns:
            检测到的矩形旗形态列表
        """
        logger.info("Using legacy flag detection API - consider using scan() with pattern_types=[PatternType.FLAG_PATTERN]")
        all_patterns = self.flag_detector.detect(df, timeframe)
        return [p for p in all_patterns if p.sub_type == FlagSubType.FLAG]
    
    def scan_pennants(self, df: pd.DataFrame, timeframe: Optional[str] = None) -> List[PatternRecord]:
        """
        扫描三角旗形态（向后兼容）
        
        Args:
            df: OHLCV数据
            timeframe: 时间周期
            
        Returns:
            检测到的三角旗形态列表
        """
        logger.info("Using legacy pennant detection API - consider using scan() with pattern_types=[PatternType.FLAG_PATTERN]")
        all_patterns = self.flag_detector.detect(df, timeframe)
        return [p for p in all_patterns if p.sub_type == FlagSubType.PENNANT]