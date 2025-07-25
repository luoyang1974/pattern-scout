"""
ATR自适应参数管理器
基于市场波动率动态调整检测参数
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger

from src.patterns.indicators.technical_indicators import TechnicalIndicators


class ATRAdaptiveManager:
    """
    ATR自适应参数管理器
    根据市场波动率动态调整形态检测参数
    """
    
    def __init__(self, atr_period: int = 14):
        """
        初始化ATR自适应管理器
        
        Args:
            atr_period: ATR计算周期
        """
        self.atr_period = atr_period
        
        # ATR分类阈值（标准化后的ATR值）
        self.volatility_thresholds = {
            'very_low': 0.005,    # 0.5%
            'low': 0.01,          # 1%
            'medium': 0.02,       # 2%
            'high': 0.035,        # 3.5%
            'very_high': 0.05     # 5%
        }
        
        # 基础参数配置（用于动态调整）
        self.base_params = {
            'swing_detection': {
                'very_low': {'window_multiplier': 1.5, 'prominence_multiplier': 0.8},
                'low': {'window_multiplier': 1.2, 'prominence_multiplier': 0.6},
                'medium': {'window_multiplier': 1.0, 'prominence_multiplier': 0.5},
                'high': {'window_multiplier': 0.8, 'prominence_multiplier': 0.4},
                'very_high': {'window_multiplier': 0.6, 'prominence_multiplier': 0.3}
            },
            'flagpole_detection': {
                'very_low': {'height_multiplier': 0.8, 'strength_multiplier': 1.2},
                'low': {'height_multiplier': 0.9, 'strength_multiplier': 1.1},
                'medium': {'height_multiplier': 1.0, 'strength_multiplier': 1.0},
                'high': {'height_multiplier': 1.1, 'strength_multiplier': 0.9},
                'very_high': {'height_multiplier': 1.2, 'strength_multiplier': 0.8}
            },
            'pattern_validation': {
                'very_low': {'tolerance_multiplier': 1.3, 'confidence_multiplier': 0.9},
                'low': {'tolerance_multiplier': 1.2, 'confidence_multiplier': 0.95},
                'medium': {'tolerance_multiplier': 1.0, 'confidence_multiplier': 1.0},
                'high': {'tolerance_multiplier': 0.9, 'confidence_multiplier': 1.05},
                'very_high': {'tolerance_multiplier': 0.8, 'confidence_multiplier': 1.1}
            }
        }
    
    def analyze_market_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析市场波动率
        
        Args:
            df: OHLCV数据
            
        Returns:
            波动率分析结果
        """
        try:
            # 计算ATR
            atr = TechnicalIndicators.calculate_atr(df, self.atr_period)
            
            if len(atr) < self.atr_period:
                logger.warning(f"Insufficient data for ATR calculation: {len(atr)} < {self.atr_period}")
                return self._get_fallback_volatility_analysis(df)
            
            # 获取最近的ATR值
            recent_atr = atr.iloc[-1]
            recent_price = df['close'].iloc[-1]
            
            # 标准化ATR（相对于价格）
            normalized_atr = recent_atr / recent_price if recent_price > 0 else 0
            
            # 计算ATR统计信息
            atr_mean = atr.mean()
            atr_std = atr.std()
            atr_percentile = (atr.iloc[-20:] < recent_atr).sum() / min(20, len(atr)) * 100
            
            # 分类波动率级别
            volatility_level = self._classify_volatility(normalized_atr)
            
            # 计算趋势波动率（价格变化的标准差）
            price_changes = df['close'].pct_change().dropna()
            price_volatility = price_changes.std() if len(price_changes) > 0 else 0
            
            return {
                'atr_raw': recent_atr,
                'atr_normalized': normalized_atr,
                'atr_mean': atr_mean,
                'atr_std': atr_std,
                'atr_percentile': atr_percentile,
                'volatility_level': volatility_level,
                'price_volatility': price_volatility,
                'is_high_volatility': normalized_atr > self.volatility_thresholds['high'],
                'volatility_trend': self._calculate_volatility_trend(atr),
                'data_quality': {
                    'sufficient_data': len(atr) >= self.atr_period,
                    'atr_series_length': len(atr),
                    'recent_data_points': min(20, len(atr))
                }
            }
        
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return self._get_fallback_volatility_analysis(df)
    
    def _get_fallback_volatility_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取回退的波动率分析（当ATR计算失败时）"""
        try:
            # 使用价格范围作为波动率估计
            price_range = (df['high'] - df['low']) / df['close']
            avg_range = price_range.mean() if len(price_range) > 0 else 0.02
            
            volatility_level = self._classify_volatility(avg_range)
            
            return {
                'atr_raw': avg_range * df['close'].iloc[-1],
                'atr_normalized': avg_range,
                'atr_mean': avg_range,
                'atr_std': 0,
                'atr_percentile': 50,
                'volatility_level': volatility_level,
                'price_volatility': avg_range,
                'is_high_volatility': avg_range > self.volatility_thresholds['high'],
                'volatility_trend': 'stable',
                'data_quality': {
                    'sufficient_data': False,
                    'atr_series_length': 0,
                    'recent_data_points': len(df)
                }
            }
        except Exception as e:
            logger.error(f"Fallback volatility analysis failed: {e}")
            # 返回默认值
            return {
                'atr_raw': 1.0,
                'atr_normalized': 0.02,
                'atr_mean': 1.0,
                'atr_std': 0,
                'atr_percentile': 50,
                'volatility_level': 'medium',
                'price_volatility': 0.02,
                'is_high_volatility': False,
                'volatility_trend': 'stable',
                'data_quality': {
                    'sufficient_data': False,
                    'atr_series_length': 0,
                    'recent_data_points': len(df)
                }
            }
    
    def _classify_volatility(self, normalized_atr: float) -> str:
        """
        分类波动率级别
        
        Args:
            normalized_atr: 标准化的ATR值
            
        Returns:
            波动率级别
        """
        if normalized_atr <= self.volatility_thresholds['very_low']:
            return 'very_low'
        elif normalized_atr <= self.volatility_thresholds['low']:
            return 'low'
        elif normalized_atr <= self.volatility_thresholds['medium']:
            return 'medium'
        elif normalized_atr <= self.volatility_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_volatility_trend(self, atr_series: pd.Series) -> str:
        """
        计算波动率趋势
        
        Args:
            atr_series: ATR时间序列
            
        Returns:
            趋势方向: 'increasing', 'decreasing', 'stable'
        """
        if len(atr_series) < 5:
            return 'stable'
        
        # 使用最近5个值计算趋势
        recent_atr = atr_series.iloc[-5:]
        
        # 计算线性趋势
        x = np.arange(len(recent_atr))
        y = recent_atr.values
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            
            # 相对于平均值的斜率阈值
            avg_atr = recent_atr.mean()
            slope_threshold = avg_atr * 0.05  # 5%变化阈值
            
            if slope > slope_threshold:
                return 'increasing'
            elif slope < -slope_threshold:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'stable'
    
    def adapt_parameters(self, base_params: Dict[str, Any], 
                        volatility_analysis: Dict[str, Any],
                        timeframe_category: str = 'short') -> Dict[str, Any]:
        """
        根据波动率分析自适应调整参数
        
        Args:
            base_params: 基础参数配置
            volatility_analysis: 波动率分析结果
            timeframe_category: 时间周期类别
            
        Returns:
            调整后的参数
        """
        try:
            volatility_level = volatility_analysis['volatility_level']
            adapted_params = base_params.copy()
            
            logger.info(f"Adapting parameters for volatility level: {volatility_level}")
            
            # 调整摆动点检测参数
            if 'pattern' in adapted_params:
                self._adapt_swing_detection_params(
                    adapted_params['pattern'], volatility_level, volatility_analysis
                )
            
            # 调整旗杆检测参数
            if 'flagpole' in adapted_params:
                self._adapt_flagpole_detection_params(
                    adapted_params['flagpole'], volatility_level, volatility_analysis
                )
            
            # 调整形态验证参数
            self._adapt_pattern_validation_params(
                adapted_params, volatility_level, volatility_analysis, timeframe_category
            )
            
            # 添加ATR相关的元数据
            adapted_params['atr_adaptive'] = {
                'volatility_level': volatility_level,
                'normalized_atr': volatility_analysis['atr_normalized'],
                'volatility_trend': volatility_analysis['volatility_trend'],
                'adaptation_applied': True
            }
            
            return adapted_params
            
        except Exception as e:
            logger.error(f"Error in parameter adaptation: {e}")
            # 返回原始参数，但添加错误信息
            base_params['atr_adaptive'] = {
                'volatility_level': 'medium',
                'normalized_atr': 0.02,
                'volatility_trend': 'stable',
                'adaptation_applied': False,
                'error': str(e)
            }
            return base_params
    
    def _adapt_swing_detection_params(self, pattern_params: Dict[str, Any], 
                                    volatility_level: str, 
                                    volatility_analysis: Dict[str, Any]) -> None:
        """调整摆动点检测参数"""
        swing_params = self.base_params['swing_detection'][volatility_level]
        
        # 调整ATR突出度参数
        if 'min_prominence_atr' in pattern_params:
            base_prominence = pattern_params['min_prominence_atr']
            adapted_prominence = base_prominence * swing_params['prominence_multiplier']
            pattern_params['min_prominence_atr'] = adapted_prominence
            
            logger.debug(f"Adapted prominence ATR: {base_prominence:.3f} -> {adapted_prominence:.3f}")
        
        # 调整窗口大小相关参数（间接影响）
        window_multiplier = swing_params['window_multiplier']
        
        # 记录自适应信息
        pattern_params['swing_adaptation'] = {
            'window_multiplier': window_multiplier,
            'prominence_multiplier': swing_params['prominence_multiplier']
        }
    
    def _adapt_flagpole_detection_params(self, flagpole_params: Dict[str, Any], 
                                       volatility_level: str, 
                                       volatility_analysis: Dict[str, Any]) -> None:
        """调整旗杆检测参数"""
        flagpole_adapt = self.base_params['flagpole_detection'][volatility_level]
        
        # 调整高度阈值
        if 'min_height_percent' in flagpole_params:
            base_height = flagpole_params['min_height_percent']
            adapted_height = base_height * flagpole_adapt['height_multiplier']
            flagpole_params['min_height_percent'] = adapted_height
            
            logger.debug(f"Adapted min height: {base_height:.2f}% -> {adapted_height:.2f}%")
        
        if 'max_height_percent' in flagpole_params:
            base_max_height = flagpole_params['max_height_percent']
            adapted_max_height = base_max_height * flagpole_adapt['height_multiplier']
            flagpole_params['max_height_percent'] = adapted_max_height
        
        # 调整趋势强度要求
        if 'min_trend_strength' in flagpole_params:
            base_strength = flagpole_params['min_trend_strength']
            adapted_strength = base_strength * flagpole_adapt['strength_multiplier']
            # 确保在合理范围内
            adapted_strength = max(0.3, min(0.95, adapted_strength))
            flagpole_params['min_trend_strength'] = adapted_strength
            
            logger.debug(f"Adapted min trend strength: {base_strength:.2f} -> {adapted_strength:.2f}")
    
    def _adapt_pattern_validation_params(self, all_params: Dict[str, Any], 
                                       volatility_level: str, 
                                       volatility_analysis: Dict[str, Any],
                                       timeframe_category: str) -> None:
        """调整形态验证参数"""
        validation_adapt = self.base_params['pattern_validation'][volatility_level]
        
        # 调整平行度容忍度（旗形）
        if 'pattern' in all_params and 'parallel_tolerance' in all_params['pattern']:
            base_tolerance = all_params['pattern']['parallel_tolerance']
            adapted_tolerance = base_tolerance * validation_adapt['tolerance_multiplier']
            all_params['pattern']['parallel_tolerance'] = adapted_tolerance
            
            logger.debug(f"Adapted parallel tolerance: {base_tolerance:.3f} -> {adapted_tolerance:.3f}")
        
        # 调整收敛比例（三角旗形）
        if 'pattern' in all_params and 'convergence_ratio' in all_params['pattern']:
            base_convergence = all_params['pattern']['convergence_ratio']
            adapted_convergence = base_convergence * validation_adapt['tolerance_multiplier']
            # 确保在合理范围内
            adapted_convergence = max(0.3, min(0.8, adapted_convergence))
            all_params['pattern']['convergence_ratio'] = adapted_convergence
        
        # 调整最小置信度
        if 'min_confidence' in all_params:
            base_confidence = all_params['min_confidence']
            adapted_confidence = base_confidence * validation_adapt['confidence_multiplier']
            # 确保在合理范围内
            adapted_confidence = max(0.3, min(0.9, adapted_confidence))
            all_params['min_confidence'] = adapted_confidence
            
            logger.debug(f"Adapted min confidence: {base_confidence:.2f} -> {adapted_confidence:.2f}")
    
    def get_atr_based_thresholds(self, df: pd.DataFrame, 
                               base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """
        基于ATR计算动态阈值
        
        Args:
            df: OHLCV数据
            base_thresholds: 基础阈值字典
            
        Returns:
            调整后的阈值字典
        """
        try:
            volatility_analysis = self.analyze_market_volatility(df)
            normalized_atr = volatility_analysis['atr_normalized']
            
            # ATR倍数映射
            atr_multipliers = {
                'very_low': 0.5,    # 低波动时使用更严格的阈值
                'low': 0.75,
                'medium': 1.0,
                'high': 1.3,        # 高波动时放宽阈值
                'very_high': 1.6
            }
            
            volatility_level = volatility_analysis['volatility_level']
            multiplier = atr_multipliers[volatility_level]
            
            adapted_thresholds = {}
            for key, base_value in base_thresholds.items():
                # 对于百分比类型的阈值，使用ATR调整
                if 'percent' in key.lower() or 'ratio' in key.lower():
                    adapted_thresholds[key] = base_value * multiplier
                else:
                    # 对于绝对值类型的阈值，直接基于ATR
                    avg_price = df['close'].mean()
                    adapted_thresholds[key] = normalized_atr * avg_price * base_value
            
            logger.info(f"ATR-based thresholds adapted for {volatility_level} volatility (multiplier: {multiplier:.2f})")
            
            return adapted_thresholds
            
        except Exception as e:
            logger.error(f"Error calculating ATR-based thresholds: {e}")
            return base_thresholds
    
    def get_volatility_report(self, df: pd.DataFrame) -> str:
        """
        生成波动率分析报告
        
        Args:
            df: OHLCV数据
            
        Returns:
            格式化的报告字符串
        """
        try:
            analysis = self.analyze_market_volatility(df)
            
            report = f"""
ATR波动率分析报告
==================
时间范围: {df['timestamp'].iloc[0]} 至 {df['timestamp'].iloc[-1]}
数据点数: {len(df)}

波动率指标:
- ATR值: {analysis['atr_raw']:.4f}
- 标准化ATR: {analysis['atr_normalized']:.3%}
- ATR百分位: {analysis['atr_percentile']:.1f}%
- 波动率级别: {analysis['volatility_level'].upper()}
- 波动率趋势: {analysis['volatility_trend']}
- 高波动标记: {'是' if analysis['is_high_volatility'] else '否'}

数据质量:
- 数据充足性: {'是' if analysis['data_quality']['sufficient_data'] else '否'}
- ATR序列长度: {analysis['data_quality']['atr_series_length']}
- 分析数据点: {analysis['data_quality']['recent_data_points']}

建议:
- 适合检测: {'高频形态' if analysis['volatility_level'] in ['high', 'very_high'] else '标准形态'}
- 参数调整: {'放宽阈值' if analysis['is_high_volatility'] else '标准阈值'}
"""
            return report
            
        except Exception as e:
            return f"波动率分析报告生成失败: {e}"