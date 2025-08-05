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
                                params: dict) -> List[PatternRecord]:
        """
        检测旗形形态
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            
        Returns:
            检测到的旗形形态列表
        """
        patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for flagpole in flagpoles:
            # 在旗杆后寻找旗面
            flag_result = self._detect_flag_after_pole(df, flagpole, params)
            
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
                        'category': params['timeframe'],
                        'timeframe': params['timeframe']
                    }
                )
                
                patterns.append(pattern_record)
                
                logger.info(f"Detected flag pattern: {pattern_record.id} "
                          f"with confidence {pattern_record.confidence_score:.3f}")
        
        return patterns
    
    def _detect_flag_after_pole(self, df: pd.DataFrame, 
                              flagpole: Flagpole,
                              params: dict) -> Optional[dict]:
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
                flag_data, flagpole, params, df
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
        
        # 如果摆动点太少，使用基础方法寻找边界
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            boundaries = self._find_simple_boundaries(flag_data, flagpole, params)
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
        base_confidence = self._calculate_flag_quality_score(
            flag_data, flagpole, boundaries, params
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
    
    def _calculate_flag_quality_score(self, flag_data: pd.DataFrame,
                                    flagpole: Flagpole,
                                    boundaries: List[TrendLine],
                                    params: dict) -> float:
        """
        计算旗形质量评分（替代strategy.score_flag_quality）
        
        Args:
            flag_data: 旗面数据
            flagpole: 旗杆对象
            boundaries: 边界线列表 [上边界, 下边界]
            params: 参数配置
            
        Returns:
            质量评分（0-1）
        """
        if not boundaries or len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        scores = []
        
        # 1. 几何特征评分（40%权重）
        # 1.1 R²拟合质量
        r_squared_score = (upper_line.r_squared + lower_line.r_squared) / 2
        scores.append(('r_squared', r_squared_score, 0.15))
        
        # 1.2 平行度质量
        parallel_score = self._calculate_parallel_quality(upper_line, lower_line, flag_data, params)
        scores.append(('parallel', parallel_score, 0.15))
        
        # 1.3 回撤比例合理性
        retracement_score = self._calculate_retracement_score(flag_data, flagpole, params)
        scores.append(('retracement', retracement_score, 0.10))
        
        # 2. 技术特征评分（35%权重）
        # 2.1 成交量模式
        volume_score = self._calculate_volume_score(flag_data, params)
        scores.append(('volume', volume_score, 0.20))
        
        # 2.2 倾斜角度合理性
        slope_score = self._calculate_slope_score(flagpole, upper_line, lower_line, params)
        scores.append(('slope', slope_score, 0.15))
        
        # 3. 形态完整性评分（25%权重）
        # 3.1 持续时间合理性
        duration_score = self._calculate_duration_score(flag_data, params)
        scores.append(('duration', duration_score, 0.10))
        
        # 3.2 边界触及次数
        touch_score = self._calculate_boundary_touch_score(flag_data, boundaries, params)
        scores.append(('touch', touch_score, 0.15))
        
        # 计算加权总分
        total_score = sum(score * weight for _, score, weight in scores)
        
        # 记录详细评分
        score_details = {name: f"{score:.3f}" for name, score, _ in scores}
        logger.debug(f"Flag quality scores: {score_details}, total: {total_score:.3f}")
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_retracement_score(self, flag_data: pd.DataFrame, 
                                   flagpole: Flagpole, params: dict) -> float:
        """计算回撤比例评分"""
        retracement_range = params['pattern'].get('retracement_range', [0.2, 0.6])
        min_retracement, max_retracement = retracement_range
        
        # 计算实际回撤比例
        flag_height = abs(flag_data['high'].max() - flag_data['low'].min())
        flagpole_height = abs(flagpole.end_price - flagpole.start_price)
        
        if flagpole_height == 0:
            return 0.5
        
        actual_retracement = flag_height / flagpole_height
        
        # 评分：在理想范围内得分最高
        if min_retracement <= actual_retracement <= max_retracement:
            return 1.0
        elif actual_retracement < min_retracement:
            return max(0.0, actual_retracement / min_retracement)
        else:
            return max(0.0, 1.0 - (actual_retracement - max_retracement) / max_retracement)
    
    def _calculate_volume_score(self, flag_data: pd.DataFrame, params: dict) -> float:
        """计算成交量评分"""
        if 'volume' not in flag_data.columns:
            return 0.5
        
        # 成交量应该相对平稳或递减
        volume_trend = np.polyfit(range(len(flag_data)), flag_data['volume'], 1)[0]
        volume_mean = flag_data['volume'].mean()
        
        if volume_mean > 0:
            normalized_trend = volume_trend / volume_mean
            # 轻微下降是理想的
            if -0.1 <= normalized_trend <= 0.05:
                return 1.0
            elif normalized_trend < -0.1:
                return max(0.3, 1.0 + normalized_trend * 5)  # 下降过快扣分
            else:
                return max(0.2, 1.0 - normalized_trend * 10)  # 上升扣分
        
        return 0.5
    
    def _calculate_slope_score(self, flagpole: Flagpole, upper_line: TrendLine,
                             lower_line: TrendLine, params: dict) -> float:
        """计算倾斜角度评分"""
        # 计算平均斜率
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price <= 0:
            return 0.5
        
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price))
        min_angle = params['pattern']['min_slope_angle']
        max_angle = params['pattern']['max_slope_angle']
        
        # 检查方向和角度
        if flagpole.direction == 'up':
            target_range = [-max_angle, -min_angle]
            actual_angle = slope_angle
        else:
            target_range = [min_angle, max_angle]
            actual_angle = slope_angle
        
        # 在目标范围内得满分
        if target_range[0] <= actual_angle <= target_range[1]:
            return 1.0
            
        # 计算偏离程度
        if actual_angle < target_range[0]:
            deviation = abs(actual_angle - target_range[0]) / abs(target_range[0])
        else:
            deviation = abs(actual_angle - target_range[1]) / abs(target_range[1])
        
        return max(0.0, 1.0 - deviation * 2)
    
    def _calculate_duration_score(self, flag_data: pd.DataFrame, params: dict) -> float:
        """计算持续时间评分"""
        actual_duration = len(flag_data)
        min_bars = params['pattern']['min_bars']
        max_bars = params['pattern']['max_bars']
        optimal_duration = (min_bars + max_bars) / 2
        
        if min_bars <= actual_duration <= max_bars:
            # 在有效范围内，接近最优值得分最高
            deviation = abs(actual_duration - optimal_duration) / (max_bars - min_bars)
            return 1.0 - deviation * 0.3  # 最多扣30%
        else:
            return 0.0
    
    def _calculate_boundary_touch_score(self, flag_data: pd.DataFrame,
                                      boundaries: List[TrendLine], params: dict) -> float:
        """计算边界触及次数评分"""
        if not boundaries or len(boundaries) < 2:
            return 0.0
        
        upper_line, lower_line = boundaries[0], boundaries[1]
        min_touches = params['pattern'].get('min_touches', 2)
        
        # 计算价格触及边界的次数
        tolerance = flag_data['close'].std() * 0.5  # 容忍度
        
        upper_touches = 0
        lower_touches = 0
        
        for i, row in flag_data.iterrows():
            # 计算该时间点的边界价格
            time_ratio = i / (len(flag_data) - 1) if len(flag_data) > 1 else 0
            upper_price = upper_line.start_price + upper_line.slope * time_ratio
            lower_price = lower_line.start_price + lower_line.slope * time_ratio
            
            # 检查是否触及边界
            if abs(row['high'] - upper_price) <= tolerance:
                upper_touches += 1
            if abs(row['low'] - lower_price) <= tolerance:
                lower_touches += 1
        
        total_touches = upper_touches + lower_touches
        
        if total_touches >= min_touches * 2:
            return 1.0
        elif total_touches >= min_touches:
            return 0.7
        else:
            return max(0.0, total_touches / min_touches * 0.5)
    
    def _find_simple_boundaries(self, flag_data: pd.DataFrame, 
                              flagpole: Flagpole, params: dict) -> Optional[List[TrendLine]]:
        """
        简单的边界线寻找方法（替代strategy.find_parallel_boundaries）
        """
        if len(flag_data) < 4:
            return None
        
        from scipy import stats
        
        # 使用简单的线性回归拟合上下边界
        x = np.arange(len(flag_data))
        
        # 拟合上边界（使用high价格）
        upper_prices = flag_data['high'].values
        upper_slope, upper_intercept, upper_r, _, _ = stats.linregress(x, upper_prices)
        
        # 拟合下边界（使用low价格）  
        lower_prices = flag_data['low'].values
        lower_slope, lower_intercept, lower_r, _, _ = stats.linregress(x, lower_prices)
        
        # 创建趋势线对象
        upper_line = TrendLine(
            start_time=flag_data.iloc[0]['timestamp'],
            end_time=flag_data.iloc[-1]['timestamp'],
            start_price=upper_intercept,
            end_price=upper_intercept + upper_slope * (len(flag_data) - 1),
            slope=upper_slope,
            r_squared=upper_r ** 2
        )
        
        lower_line = TrendLine(
            start_time=flag_data.iloc[0]['timestamp'],
            end_time=flag_data.iloc[-1]['timestamp'],
            start_price=lower_intercept,
            end_price=lower_intercept + lower_slope * (len(flag_data) - 1),
            slope=lower_slope,
            r_squared=lower_r ** 2
        )
        
        # 验证基本的平行性
        if self._validate_basic_parallelism(upper_line, lower_line, params):
            return [upper_line, lower_line]
        else:
            return None
    
    def _validate_basic_parallelism(self, upper_line: TrendLine, 
                                  lower_line: TrendLine, params: dict) -> bool:
        """验证基本的平行性"""
        # 检查斜率是否相似
        slope_diff = abs(upper_line.slope - lower_line.slope)
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price > 0:
            slope_diff_pct = slope_diff / avg_price
            # 斜率差异不超过0.1%
            if slope_diff_pct > 0.001:
                return False
        
        # 检查R²值
        min_r_squared = 0.3
        if upper_line.r_squared < min_r_squared or lower_line.r_squared < min_r_squared:
            return False
        
        # 检查边界不交叉
        if upper_line.start_price <= lower_line.start_price:
            return False
        if upper_line.end_price <= lower_line.end_price:
            return False
            
        return True