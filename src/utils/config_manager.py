"""
增强的配置管理器
支持多周期配置和动态参数选择
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class ConfigManager:
    """增强的配置管理器"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = Path(config_file)
        self.config = {}
        self._load_config()
        self._load_env_variables()
        
        # 检查是否使用多周期配置
        self.is_multi_timeframe = 'timeframe_categories' in self.config
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.warning(f"Configuration file not found: {self.config_file}")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = self._get_default_config()
    
    def _load_env_variables(self):
        """加载环境变量"""
        # MongoDB配置
        if 'MONGODB_USERNAME' in os.environ:
            self.config.setdefault('data_sources', {}).setdefault('mongodb', {})['username'] = os.environ['MONGODB_USERNAME']
        if 'MONGODB_PASSWORD' in os.environ:
            self.config.setdefault('data_sources', {}).setdefault('mongodb', {})['password'] = os.environ['MONGODB_PASSWORD']
        if 'API_KEY' in os.environ:
            self.config.setdefault('api', {})['key'] = os.environ['API_KEY']
    
    def get_timeframe_category(self, timeframe: str) -> str:
        """
        获取时间周期对应的类别
        
        Args:
            timeframe: 时间周期（如 '15m'）
            
        Returns:
            类别名称（如 'short'）
        """
        if not self.is_multi_timeframe:
            return 'default'
        
        categories = self.config.get('timeframe_categories', {})
        
        for category, info in categories.items():
            if timeframe in info.get('timeframes', []):
                return category
        
        # 默认返回 short
        logger.warning(f"Unknown timeframe {timeframe}, using 'short' category")
        return 'short'
    
    def get_pattern_config(self, pattern_type: str, timeframe: str = None, category: str = None) -> Dict[str, Any]:
        """
        获取特定形态的配置
        
        Args:
            pattern_type: 形态类型（'flag', 'pennant'等）
            timeframe: 时间周期
            category: 周期类别（如果不提供，根据timeframe自动判断）
            
        Returns:
            形态配置字典
        """
        if not self.is_multi_timeframe:
            # 使用旧配置格式
            return self.config.get('pattern_detection', {}).get(pattern_type, {})
        
        # 确定类别
        if category is None and timeframe is not None:
            category = self.get_timeframe_category(timeframe)
        elif category is None:
            category = 'short'  # 默认
        
        # 获取该类别的配置
        category_config = self.config.get('pattern_detection', {}).get(category, {})
        return category_config.get(pattern_type, {})
    
    def get_scoring_weights(self, pattern_type: str, category: str) -> Dict[str, float]:
        """
        获取评分权重
        
        Args:
            pattern_type: 形态类型
            category: 周期类别
            
        Returns:
            权重字典
        """
        if not self.is_multi_timeframe:
            # 使用默认权重
            return self._get_default_weights(pattern_type)
        
        weights = self.config.get('scoring_weights', {}).get(pattern_type, {}).get(category, {})
        
        if not weights:
            logger.warning(f"No weights found for {pattern_type}/{category}, using defaults")
            return self._get_default_weights(pattern_type)
        
        return weights
    
    def get_all_timeframe_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有周期类别的配置
        
        Returns:
            {category: config} 字典
        """
        if not self.is_multi_timeframe:
            return {'default': self.config.get('pattern_detection', {})}
        
        result = {}
        pattern_detection = self.config.get('pattern_detection', {})
        
        for category in self.config.get('timeframe_categories', {}).keys():
            if category in pattern_detection:
                result[category] = pattern_detection[category]
        
        return result
    
    def get_supported_timeframes(self) -> List[str]:
        """
        获取所有支持的时间周期
        
        Returns:
            时间周期列表
        """
        if not self.is_multi_timeframe:
            return ['15m']  # 默认
        
        timeframes = []
        for category_info in self.config.get('timeframe_categories', {}).values():
            timeframes.extend(category_info.get('timeframes', []))
        
        return timeframes
    
    def _get_default_weights(self, pattern_type: str) -> Dict[str, float]:
        """获取默认权重"""
        if pattern_type == 'flag':
            return {
                'slope_direction': 0.25,
                'parallel_quality': 0.25,
                'volume_pattern': 0.25,
                'channel_containment': 0.15,
                'time_proportion': 0.10
            }
        elif pattern_type == 'pennant':
            return {
                'convergence_quality': 0.30,
                'symmetry': 0.25,
                'volume_pattern': 0.25,
                'size_proportion': 0.10,
                'apex_validity': 0.10
            }
        else:
            return {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'global': {
                'min_data_points': 60,
                'enable_multi_timeframe': False
            },
            'data_sources': {
                'csv': {
                    'enabled': True,
                    'directory': 'data/csv/'
                },
                'mongodb': {
                    'enabled': False,
                    'host': 'localhost',
                    'port': 27017,
                    'database': 'market_data',
                    'collection': 'ohlcv'
                }
            },
            'pattern_detection': {
                'flagpole': {
                    'min_bars': 4,
                    'max_bars': 10,
                    'min_height_percent': 1.5,
                    'max_height_percent': 10.0,
                    'volume_surge_ratio': 2.0,
                    'max_retracement': 0.3,
                    'min_trend_strength': 0.7
                },
                'flag': {
                    'min_bars': 8,
                    'max_bars': 30,
                    'min_slope_angle': 0.5,
                    'max_slope_angle': 10,
                    'retracement_range': [0.2, 0.6],
                    'volume_decay_threshold': 0.7,
                    'parallel_tolerance': 0.15,
                    'min_touches': 3
                },
                'pennant': {
                    'min_bars': 8,
                    'max_bars': 30,
                    'min_touches': 3,
                    'convergence_ratio': 0.6,
                    'apex_distance_range': [0.4, 2.0],
                    'symmetry_tolerance': 0.3,
                    'volume_decay_threshold': 0.7
                }
            },
            'indicators': {
                'moving_averages': [5, 10, 20, 50],
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'bollinger_std': 2
            },
            'scoring': {
                'min_confidence_score': 0.6
            },
            'output': {
                'base_path': 'output',
                'data_path': 'output/data',
                'charts_path': 'output/charts',
                'reports_path': 'output/reports',
                'charts': {
                    'format': 'png',
                    'width': 1200,
                    'height': 800,
                    'dpi': 300
                },
                'dataset': {
                    'format': 'json',
                    'backup_enabled': True
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/pattern_scout.log',
                'max_size': '10MB',
                'backup_count': 5
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键（支持点号分隔的嵌套键）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config_dict = self.config
        
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        config_dict[keys[-1]] = value
    
    def save_config(self, file_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            file_path: 保存路径，默认为原配置文件
        """
        save_path = Path(file_path) if file_path else self.config_file
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必需的配置项
            if self.is_multi_timeframe:
                required_keys = [
                    'global',
                    'data_sources',
                    'timeframe_categories',
                    'pattern_detection',
                    'output'
                ]
            else:
                required_keys = [
                    'data_sources',
                    'pattern_detection',
                    'output'
                ]
            
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"Missing required configuration section: {key}")
                    return False
            
            # 验证数据源配置
            data_sources = self.config['data_sources']
            if not any(data_sources.get(source, {}).get('enabled', False) 
                      for source in ['csv', 'mongodb']):
                logger.error("No data source is enabled")
                return False
            
            # 验证输出目录
            output_config = self.config.get('output', {})
            base_path = output_config.get('base_path', 'output')
            
            for subdir in ['data', 'charts', 'reports']:
                dir_path = Path(base_path) / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 创建日志目录
            log_file = self.config.get('logging', {}).get('file', 'logs/pattern_scout.log')
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def print_config_summary(self):
        """打印配置摘要"""
        logger.info("=== Configuration Summary ===")
        
        if self.is_multi_timeframe:
            logger.info("Multi-timeframe mode: ENABLED")
            categories = self.config.get('timeframe_categories', {})
            for cat, info in categories.items():
                logger.info(f"  {cat}: {info.get('timeframes', [])}")
        else:
            logger.info("Multi-timeframe mode: DISABLED")
        
        # 数据源
        data_sources = self.config.get('data_sources', {})
        enabled_sources = [src for src, cfg in data_sources.items() 
                          if cfg.get('enabled', False)]
        logger.info(f"Enabled data sources: {enabled_sources}")
        
        # 输出设置
        output = self.config.get('output', {})
        logger.info(f"Output base path: {output.get('base_path', 'output')}")
        logger.info(f"Chart format: {output.get('charts', {}).get('format', 'png')}")
        
        logger.info("=============================")