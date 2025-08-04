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
from src.patterns.base.atr_adaptive_manager import ATRAdaptiveManager
from src.patterns.base.parameter_adapter import ParameterAdapter


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
        self.atr_adaptive_manager = ATRAdaptiveManager()
        self.parameter_adapter = ParameterAdapter(self.config)
        
        # ATR自适应开关（可以在配置中控制）
        self.enable_atr_adaptation = self.config.get('global', {}).get('enable_atr_adaptation', True)
        
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
        
        # 2. 获取参数
        pattern_type_name = self.get_pattern_type().lower()
        params = self.parameter_adapter.get_timeframe_params(timeframe, pattern_type_name)
        
        # 2.5 ATR自适应参数调整
        if self.enable_atr_adaptation:
            params = self._apply_atr_adaptation(df, params, timeframe)
        
        # 3. 数据预处理（简单预处理，移除strategy依赖）
        df_processed = self._preprocess_data(df, params)
        
        # 4. 检测旗杆（共享逻辑）
        flagpoles = self.pattern_components.detect_flagpoles_adaptive(
            df_processed, params['flagpole'], timeframe
        )
        
        logger.info(f"Detected {len(flagpoles)} potential flagpoles")
        
        # 5. 检测具体形态（子类实现）
        patterns = self._detect_pattern_formation(
            df_processed, flagpoles, params
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
                                params: dict) -> List[PatternRecord]:
        """
        检测具体的形态（子类实现）
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            
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
    
    def _preprocess_data(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        数据预处理（替代strategy.preprocess）
        
        Args:
            df: 原始OHLCV数据
            params: 参数配置
            
        Returns:
            预处理后的数据
        """
        df = df.copy()
        
        # 添加技术指标
        df = self._add_technical_indicators(df)
        
        # 根据时间周期进行不同程度的平滑处理
        timeframe = params.get('timeframe', '15m')
        if timeframe in ['1m', '5m']:  # 短周期需要更多平滑
            df = self._smooth_price_data(df, window=5)
        elif timeframe in ['15m', '1h']:  # 中等周期适度平滑
            df = self._smooth_price_data(df, window=3)
        # 长周期数据本身比较平滑，不需要额外处理
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加通用技术指标"""
        df = df.copy()
        
        # 成交量指标
        if 'volume' in df.columns:
            df['volume_sma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20'].fillna(1)
        
        # 价格变化
        df['price_change'] = df['close'].pct_change()
        
        # ATR（平均真实范围）
        df['atr'] = self._calculate_atr(df)
        
        # 价格范围
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    
    def _smooth_price_data(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """平滑价格数据（用于降噪）"""
        df = df.copy()
        
        if len(df) > window:
            # 对关键价格序列进行平滑
            df['close'] = df['close'].rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            df['high'] = df['high'].rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            df['low'] = df['low'].rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _apply_atr_adaptation(self, df: pd.DataFrame, params: dict, timeframe: str) -> dict:
        """
        应用ATR自适应参数调整
        
        Args:
            df: OHLCV数据
            params: 原始参数
            timeframe: 时间周期（如'15m'）
            
        Returns:
            调整后的参数
        """
        try:
            # 分析市场波动率
            volatility_analysis = self.atr_adaptive_manager.analyze_market_volatility(df)
            
            # 应用自适应调整
            adapted_params = self.atr_adaptive_manager.adapt_parameters(
                params, volatility_analysis, timeframe
            )
            
            # 记录调整信息
            if volatility_analysis.get('volatility_level'):
                logger.info(f"ATR adaptation applied - Volatility: {volatility_analysis['volatility_level']}, "
                          f"Normalized ATR: {volatility_analysis['atr_normalized']:.3%}")
            
            return adapted_params
            
        except Exception as e:
            logger.error(f"ATR adaptation failed: {e}")
            # 返回原始参数
            return params
    
    def get_volatility_report(self, df: pd.DataFrame) -> str:
        """
        获取波动率分析报告
        
        Args:
            df: OHLCV数据
            
        Returns:
            波动率报告字符串
        """
        return self.atr_adaptive_manager.get_volatility_report(df)
    
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