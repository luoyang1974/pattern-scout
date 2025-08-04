"""
简单参数适配器
替代strategies目录，直接从配置文件读取时间周期相关参数
"""
from typing import Dict, Any, Optional
from loguru import logger


class ParameterAdapter:
    """
    时间周期参数适配器
    简单直接地从配置文件读取参数，无需复杂的策略类
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化参数适配器
        
        Args:
            config: 完整配置字典
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """验证配置文件结构"""
        if 'pattern_detection' not in self.config:
            raise ValueError("Configuration missing 'pattern_detection' section")
        
        # 检查支持的时间周期
        supported_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
        pattern_detection = self.config['pattern_detection']
        
        for tf in supported_timeframes:
            if tf not in pattern_detection:
                logger.warning(f"Missing configuration for timeframe {tf}")
    
    def get_timeframe_params(self, timeframe: str, pattern_type: str) -> Dict[str, Any]:
        """
        获取指定时间周期和形态类型的参数
        
        Args:
            timeframe: 时间周期 ('1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M')
            pattern_type: 形态类型 ('flag', 'pennant')
            
        Returns:
            包含所有必要参数的字典
        """
        # 获取该时间周期的配置
        timeframe_config = self.config.get('pattern_detection', {}).get(timeframe, {})
        
        if not timeframe_config:
            logger.warning(f"No configuration found for timeframe {timeframe}, using 15m as default")
            timeframe_config = self.config.get('pattern_detection', {}).get('15m', {})
        
        # 构建完整参数字典
        params = {
            'timeframe': timeframe,
            'flagpole': timeframe_config.get('flagpole', {}),
            'pattern': timeframe_config.get(pattern_type, {}),
            'scoring': timeframe_config.get('scoring', {}),
            'min_confidence': self.config.get('scoring', {}).get('min_confidence_score', 0.6)
        }
        
        return params
    
    def get_flagpole_params(self, timeframe: str) -> Dict[str, Any]:
        """获取旗杆检测参数"""
        timeframe_config = self.config.get('pattern_detection', {}).get(timeframe, {})
        return timeframe_config.get('flagpole', {})
    
    def get_flag_params(self, timeframe: str) -> Dict[str, Any]:
        """获取旗形检测参数"""
        timeframe_config = self.config.get('pattern_detection', {}).get(timeframe, {})
        return timeframe_config.get('flag', {})
    
    def get_pennant_params(self, timeframe: str) -> Dict[str, Any]:
        """获取三角旗形检测参数"""
        timeframe_config = self.config.get('pattern_detection', {}).get(timeframe, {})
        return timeframe_config.get('pennant', {})
    
    def get_scoring_params(self, timeframe: str) -> Dict[str, Any]:
        """获取质量评分参数"""
        timeframe_config = self.config.get('pattern_detection', {}).get(timeframe, {})
        return timeframe_config.get('scoring', {})
    
    def get_supported_timeframes(self) -> list:
        """获取支持的时间周期列表"""
        return list(self.config.get('pattern_detection', {}).keys())
    
    def has_timeframe_config(self, timeframe: str) -> bool:
        """检查是否存在指定时间周期的配置"""
        return timeframe in self.config.get('pattern_detection', {})