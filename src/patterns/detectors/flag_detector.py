"""
改进版旗形形态检测器
支持多时间周期自适应检测
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger

from src.data.models.base_models import PatternRecord, Flagpole, TrendLine, PatternType
from src.patterns.base import BasePatternDetector


class FlagDetector(BasePatternDetector):
    """
    旗形形态检测器
    继承自基类，实现多周期自适应检测
    """
    
    def get_pattern_type(self) -> PatternType:
        """获取形态类型"""
        return PatternType.FLAG
    
    def get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'global': {
                'min_data_points': 60,
                'enable_multi_timeframe': False,
                'enable_atr_adaptation': True,  # 启用ATR自适应
                'enable_ransac_fitting': True   # 启用RANSAC拟合
            },
            'timeframe_configs': {
                'ultra_short': {
                    'flagpole': {
                        'min_bars': 5,
                        'max_bars': 15,
                        'min_height_percent': 0.5,
                        'max_height_percent': 3.0,
                        'volume_surge_ratio': 1.5,
                        'max_retracement': 0.4,
                        'min_trend_strength': 0.6
                    },
                    'flag': {
                        'min_bars': 10,
                        'max_bars': 30,
                        'min_slope_angle': 0.3,
                        'max_slope_angle': 8,
                        'retracement_range': (0.15, 0.6),
                        'volume_decay_threshold': 0.8,
                        'parallel_tolerance': 0.25,
                        'min_touches': 2
                    }
                },
                'short': {
                    'flagpole': {
                        'min_bars': 5,
                        'max_bars': 12,
                        'min_height_percent': 1.5,
                        'max_height_percent': 10.0,
                        'volume_surge_ratio': 2.0,
                        'max_retracement': 0.3,
                        'min_trend_strength': 0.7
                    },
                    'flag': {
                        'min_bars': 10,
                        'max_bars': 25,
                        'min_slope_angle': 0.5,
                        'max_slope_angle': 10,
                        'retracement_range': (0.2, 0.6),
                        'volume_decay_threshold': 0.7,
                        'parallel_tolerance': 0.15,
                        'min_touches': 3
                    }
                },
                'medium_long': {
                    'flagpole': {
                        'min_bars': 3,
                        'max_bars': 8,
                        'min_height_percent': 2.0,
                        'max_height_percent': 15.0,
                        'volume_surge_ratio': 2.5,
                        'max_retracement': 0.2,
                        'min_trend_strength': 0.8
                    },
                    'flag': {
                        'min_bars': 5,
                        'max_bars': 20,
                        'min_slope_angle': 1,
                        'max_slope_angle': 15,
                        'retracement_range': (0.2, 0.5),
                        'volume_decay_threshold': 0.6,
                        'parallel_tolerance': 0.1,
                        'min_touches': 4
                    }
                }
            },
            'scoring': {
                'min_confidence_score': 0.6
            }
        }
    
    def _detect_pattern_formation(self, df: pd.DataFrame, 
                                flagpoles: List[Flagpole],
                                params: dict,
                                strategy) -> List[PatternRecord]:
        """
        检测旗形形态
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            strategy: 周期策略对象
            
        Returns:
            检测到的旗形形态列表
        """
        patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for flagpole in flagpoles:
            # 在旗杆后寻找旗面
            flag_result = self._detect_flag_after_pole(df, flagpole, params, strategy)
            
            if flag_result:
                # 创建形态记录
                pattern_record = self._create_pattern_record(
                    symbol=symbol,
                    flagpole=flagpole,
                    boundaries=flag_result['boundaries'],
                    duration=flag_result['duration'],
                    confidence_score=flag_result['confidence_score'],
                    additional_info={
                        'volume_pattern': flag_result.get('volume_pattern', {}),
                        'slope_direction_valid': flag_result.get('slope_direction_valid', True),
                        'parallel_quality': flag_result.get('parallel_quality', 0),
                        'category': params['category'],
                        'timeframe': params['timeframe']
                    }
                )
                
                patterns.append(pattern_record)
                
                logger.info(f"Detected flag pattern: {pattern_record.id} "
                          f"with confidence {pattern_record.confidence_score:.3f}")
        
        return patterns
    
    def _detect_flag_after_pole(self, df: pd.DataFrame, 
                              flagpole: Flagpole,
                              params: dict,
                              strategy) -> Optional[dict]:
        """检测旗杆后的旗面形态"""
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        if flagpole_end_idx >= len(df) - params['pattern']['min_bars']:
            return None
        
        min_flag_bars = params['pattern']['min_bars']
        max_flag_bars = min(
            params['pattern']['max_bars'],
            len(df) - flagpole_end_idx - 1
        )
        
        best_flag = None
        best_score = 0
        
        # 尝试不同长度的旗面
        for flag_duration in range(min_flag_bars, max_flag_bars + 1):
            flag_start_idx = flagpole_end_idx + 1
            flag_end_idx = flag_start_idx + flag_duration
            
            if flag_end_idx >= len(df):
                continue
            
            flag_data = df.iloc[flag_start_idx:flag_end_idx + 1].copy()
            
            # 分析旗面形态
            flag_result = self._analyze_flag_pattern(
                flag_data, flagpole, params, strategy, df
            )
            
            if flag_result and flag_result['confidence_score'] > best_score:
                best_score = flag_result['confidence_score']
                best_flag = flag_result
                best_flag['duration'] = flag_duration
        
        if best_flag and best_score >= params['min_confidence']:
            return best_flag
        
        return None
    
    def _analyze_flag_pattern(self, flag_data: pd.DataFrame,
                            flagpole: Flagpole,
                            params: dict,
                            strategy,
                            full_df: pd.DataFrame) -> Optional[dict]:
        """分析旗面形态"""
        if len(flag_data) < params['pattern']['min_bars']:
            return None
        
        # 1. 使用智能摆动点检测代替简单的边界寻找
        window_size = max(2, len(flag_data) // 5)
        swing_highs, swing_lows = self.pattern_components.find_swing_points(
            flag_data, 
            window=window_size,
            min_prominence_atr_multiple=params['pattern'].get('min_prominence_atr', 0.1)
        )
        
        # 如果摆动点太少，使用原始策略
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            boundaries = strategy.find_parallel_boundaries(flag_data, flagpole, params)
        else:
            # 检查是否启用RANSAC拟合
            use_ransac = params.get('use_ransac_fitting', True)
            
            # 使用摆动点拟合边界（支持RANSAC）
            upper_line = self.pattern_components.fit_trend_line_ransac(
                flag_data, swing_highs, 'high', use_ransac=use_ransac
            )
            lower_line = self.pattern_components.fit_trend_line_ransac(
                flag_data, swing_lows, 'low', use_ransac=use_ransac
            )
            boundaries = [upper_line, lower_line] if upper_line and lower_line else None
        
        if not boundaries or len(boundaries) < 2:
            return None
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        
        # 2. 验证核心特征
        # 2.1 验证反向倾斜（旗形的核心特征）
        if not self._verify_opposite_slope(flagpole, upper_line, lower_line, params):
            logger.debug("Failed opposite slope verification")
            return None
        
        # 2.2 验证平行度
        parallel_quality = self._calculate_parallel_quality(
            upper_line, lower_line, flag_data, params
        )
        
        if parallel_quality < 0.5:  # 最低要求
            logger.debug(f"Poor parallel quality: {parallel_quality:.3f}")
            return None
        
        # 2.3 新增：验证通道不发散
        if not self._verify_no_divergence(upper_line, lower_line, flag_data):
            logger.debug("Channel is diverging")
            return None
        
        # 2.4 使用增强的成交量分析
        flag_start_idx = full_df[full_df['timestamp'] >= flag_data.iloc[0]['timestamp']].index[0]
        flag_end_idx = full_df[full_df['timestamp'] <= flag_data.iloc[-1]['timestamp']].index[-1]
        flagpole_start_idx = full_df[full_df['timestamp'] >= flagpole.start_time].index[0]
        flagpole_end_idx = full_df[full_df['timestamp'] <= flagpole.end_time].index[-1]
        
        volume_analysis = self.pattern_components.analyze_volume_pattern_enhanced(
            full_df,
            flag_start_idx,
            flag_end_idx,
            flagpole_start_idx,
            flagpole_end_idx
        )
        
        if volume_analysis['health_score'] < 0.5:
            logger.debug(f"Poor volume health score: {volume_analysis['health_score']:.3f}")
            return None
        
        # 2.5 新增：验证突破准备度
        breakout_readiness = self._verify_breakout_preparation(
            flag_data, upper_line, lower_line, params
        )
        
        # 3. 计算综合置信度
        base_confidence = strategy.score_flag_quality(
            flag_data, flagpole, boundaries
        )
        
        # 根据新增验证调整置信度
        adjusted_confidence = base_confidence * 0.7 + \
                            volume_analysis['health_score'] * 0.2 + \
                            breakout_readiness * 0.1
        
        return {
            'boundaries': boundaries,
            'confidence_score': adjusted_confidence,
            'volume_pattern': volume_analysis,
            'slope_direction_valid': True,
            'parallel_quality': parallel_quality,
            'breakout_readiness': breakout_readiness,
            'swing_points': {
                'highs': swing_highs,
                'lows': swing_lows
            }
        }
    
    def _verify_opposite_slope(self, flagpole: Flagpole,
                             upper_line: TrendLine,
                             lower_line: TrendLine,
                             params: dict) -> bool:
        """验证旗面是否向主趋势相反方向倾斜"""
        # 计算平均斜率
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        
        # 计算平均价格用于归一化
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price <= 0:
            return False
        
        # 计算倾斜角度
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price))
        
        min_angle = params['pattern']['min_slope_angle']
        max_angle = params['pattern']['max_slope_angle']
        
        # 验证方向和角度
        if flagpole.direction == 'up':
            # 上升旗形：旗面必须向下倾斜
            valid = -max_angle <= slope_angle <= -min_angle
            logger.debug(f"Up flag slope angle: {slope_angle:.1f}° "
                        f"(expected: -{max_angle}° to -{min_angle}°)")
        else:
            # 下降旗形：旗面必须向上倾斜
            valid = min_angle <= slope_angle <= max_angle
            logger.debug(f"Down flag slope angle: {slope_angle:.1f}° "
                        f"(expected: {min_angle}° to {max_angle}°)")
        
        return valid
    
    def _calculate_parallel_quality(self, upper_line: TrendLine,
                                  lower_line: TrendLine,
                                  flag_data: pd.DataFrame,
                                  params: dict) -> float:
        """计算平行线质量"""
        # 1. 斜率相似度
        avg_price = flag_data['close'].mean()
        if avg_price <= 0:
            return 0.0
        
        slope_diff = abs(upper_line.slope - lower_line.slope)
        normalized_slope_diff = slope_diff / avg_price
        
        tolerance = params['pattern']['parallel_tolerance']
        slope_similarity = max(0, 1 - normalized_slope_diff / tolerance)
        
        # 2. 拟合质量
        fit_quality = (upper_line.r_squared + lower_line.r_squared) / 2
        
        # 3. 通道宽度一致性
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width > 0:
            width_ratio = min(start_width, end_width) / max(start_width, end_width)
        else:
            width_ratio = 0
        
        # 综合评分
        return slope_similarity * 0.4 + fit_quality * 0.4 + width_ratio * 0.2
    
    def _verify_volume_pattern(self, flag_data: pd.DataFrame,
                             flagpole: Flagpole,
                             full_df: pd.DataFrame,
                             params: dict) -> bool:
        """验证成交量模式"""
        # 获取旗杆期间的平均成交量
        flagpole_start_idx = full_df[full_df['timestamp'] >= flagpole.start_time].index[0]
        flagpole_end_idx = full_df[full_df['timestamp'] <= flagpole.end_time].index[-1]
        
        if flagpole_start_idx < len(full_df) and flagpole_end_idx < len(full_df):
            flagpole_volume = full_df.iloc[flagpole_start_idx:flagpole_end_idx + 1]['volume'].mean()
        else:
            flagpole_volume = full_df['volume'].mean() * 2  # 估计值
        
        # 旗面平均成交量
        flag_volume = flag_data['volume'].mean()
        
        # 旗面成交量应该显著小于旗杆成交量
        if flagpole_volume > 0:
            volume_decay_ratio = flag_volume / flagpole_volume
            threshold = params['pattern']['volume_decay_threshold']
            
            is_valid = volume_decay_ratio <= threshold
            
            logger.debug(f"Volume decay ratio: {volume_decay_ratio:.2f} "
                        f"(threshold: {threshold})")
            
            return is_valid
        
        return True  # 如果无法计算，默认通过
    
    def _verify_no_divergence(self, upper_line: TrendLine,
                             lower_line: TrendLine,
                             flag_data: pd.DataFrame) -> bool:
        """验证通道不发散"""
        # 计算通道开始和结束的宽度
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        # 允许轻微的宽度增加（最多20%）
        max_expansion_ratio = 1.2
        
        if start_width > 0:
            expansion_ratio = end_width / start_width
            is_valid = expansion_ratio <= max_expansion_ratio
            
            logger.debug(f"Channel expansion ratio: {expansion_ratio:.2f} "
                        f"(max allowed: {max_expansion_ratio})")
            
            return is_valid
        
        return True
    
    def _verify_breakout_preparation(self, flag_data: pd.DataFrame,
                                   upper_line: TrendLine,
                                   lower_line: TrendLine,
                                   params: dict) -> float:
        """
        验证是否有突破准备迹象
        
        Returns:
            突破准备度评分（0-1）
        """
        # 1. 价格接近边界的程度
        last_prices = flag_data['close'].iloc[-5:]  # 最后5根K线
        channel_width = abs(upper_line.end_price - lower_line.end_price)
        
        # 计算价格到上下边界的距离
        avg_price = last_prices.mean()
        distance_to_upper = abs(avg_price - upper_line.end_price)
        distance_to_lower = abs(avg_price - lower_line.end_price)
        min_distance = min(distance_to_upper, distance_to_lower)
        
        # 距离越近得分越高
        proximity_score = 1.0 - (min_distance / channel_width) if channel_width > 0 else 0.5
        
        # 2. 成交量是否开始放大
        if len(flag_data) >= 8:
            recent_volume = flag_data['volume'].iloc[-3:].mean()
            earlier_volume = flag_data['volume'].iloc[:-3].mean()
            volume_expansion_ratio = recent_volume / earlier_volume if earlier_volume > 0 else 1.0
            
            # 成交量放大是积极信号
            volume_score = min(1.0, (volume_expansion_ratio - 0.8) / 0.4)  # 0.8-1.2映射到0-1
        else:
            volume_score = 0.5
        
        # 3. 价格波动收窄（即将选择方向）
        recent_volatility = flag_data['close'].iloc[-5:].std()
        overall_volatility = flag_data['close'].std()
        
        if overall_volatility > 0:
            volatility_contraction = 1.0 - (recent_volatility / overall_volatility)
            volatility_score = max(0, min(1, volatility_contraction * 2))  # 放大效果
        else:
            volatility_score = 0.5
        
        # 综合评分
        breakout_readiness = (proximity_score * 0.5 + 
                            volume_score * 0.3 + 
                            volatility_score * 0.2)
        
        logger.debug(f"Breakout readiness - proximity: {proximity_score:.2f}, "
                    f"volume: {volume_score:.2f}, volatility: {volatility_score:.2f}, "
                    f"total: {breakout_readiness:.2f}")
        
        return breakout_readiness