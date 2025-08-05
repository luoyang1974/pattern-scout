"""
统一的旗形形态检测器
支持矩形旗（Flag）和三角旗（Pennant）的统一检测
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from scipy import stats
from loguru import logger

from src.data.models.base_models import PatternRecord, Flagpole, TrendLine, PatternType, FlagSubType
from src.patterns.base import BasePatternDetector


class FlagDetector(BasePatternDetector):
    """
    统一的旗形形态检测器
    支持同时检测矩形旗（Flag）和三角旗（Pennant）
    """
    
    def get_pattern_type(self) -> str:
        """获取形态类型"""
        return PatternType.FLAG_PATTERN
    
    def get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'global': {
                'min_data_points': 60,
                'enable_multi_timeframe': False,
                'enable_atr_adaptation': True,
                'enable_ransac_fitting': True
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
                        'retracement_range': [0.15, 0.6],
                        'volume_decay_threshold': 0.8,
                        'parallel_tolerance': 0.25,
                        'min_touches': 2
                    },
                    'pennant': {
                        'min_bars': 10,
                        'max_bars': 40,
                        'min_touches': 2,
                        'convergence_ratio': 0.5,
                        'apex_distance_range': [0.5, 3.0],
                        'symmetry_tolerance': 0.4,
                        'volume_decay_threshold': 0.8
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
                        'retracement_range': [0.2, 0.5],
                        'volume_decay_threshold': 0.6,
                        'parallel_tolerance': 0.1,
                        'min_touches': 4
                    },
                    'pennant': {
                        'min_bars': 5,
                        'max_bars': 25,
                        'min_touches': 4,
                        'convergence_ratio': 0.7,
                        'apex_distance_range': [0.3, 1.5],
                        'symmetry_tolerance': 0.2,
                        'volume_decay_threshold': 0.6
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
        检测旗形形态（包含Flag和Pennant）
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            
        Returns:
            检测到的旗形形态列表
        """
        all_patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for flagpole in flagpoles:
            # 检测矩形旗（Flag）
            flag_patterns = self._detect_flag_formation(df, flagpole, params)
            for pattern in flag_patterns:
                all_patterns.append(pattern)
            
            # 检测三角旗（Pennant）
            pennant_patterns = self._detect_pennant_formation(df, flagpole, params)
            for pattern in pennant_patterns:
                all_patterns.append(pattern)
        
        # 按置信度排序，如果同一位置有多个形态，选择置信度最高的
        all_patterns.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # 去重：如果检测位置重叠，保留置信度更高的形态
        final_patterns = self._remove_overlapping_patterns(all_patterns)
        
        logger.info(f"Total flag patterns detected: {len(final_patterns)}")
        return final_patterns
    
    def _detect_flag_formation(self, df: pd.DataFrame, 
                             flagpole: Flagpole,
                             params: dict) -> List[PatternRecord]:
        """检测矩形旗形态"""
        patterns = []
        
        # 在旗杆后寻找旗面
        flag_result = self._detect_flag_after_pole(df, flagpole, params)
        
        if flag_result:
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
            
            # 创建形态记录
            pattern_record = self._create_pattern_record(
                symbol=symbol,
                flagpole=flagpole,
                boundaries=flag_result['boundaries'],
                duration=flag_result['duration'],
                confidence_score=flag_result['confidence_score'],
                sub_type=FlagSubType.FLAG,
                additional_info={
                    'volume_pattern': flag_result.get('volume_pattern', {}),
                    'slope_direction_valid': flag_result.get('slope_direction_valid', True),
                    'parallel_quality': flag_result.get('parallel_quality', 0),
                    'timeframe': params.get('timeframe', 'unknown')
                }
            )
            
            patterns.append(pattern_record)
            
            logger.info(f"Detected flag pattern: {pattern_record.id} "
                      f"with confidence {pattern_record.confidence_score:.3f}")
        
        return patterns
    
    def _detect_pennant_formation(self, df: pd.DataFrame,
                                flagpole: Flagpole,
                                params: dict) -> List[PatternRecord]:
        """检测三角旗形态"""
        patterns = []
        
        # 在旗杆后寻找收敛的三角形
        pennant_result = self._detect_pennant_after_pole(df, flagpole, params)
        
        if pennant_result:
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
            
            # 创建形态记录
            pattern_record = self._create_pattern_record(
                symbol=symbol,
                flagpole=flagpole,
                boundaries=pennant_result['boundaries'],
                duration=pennant_result['duration'],
                confidence_score=pennant_result['confidence_score'],
                sub_type=FlagSubType.PENNANT,
                additional_info={
                    'apex': pennant_result.get('apex'),
                    'convergence_quality': pennant_result.get('convergence_quality', 0),
                    'symmetry_score': pennant_result.get('symmetry_score', 0),
                    'volume_pattern': pennant_result.get('volume_pattern', {}),
                    'triangle_type': pennant_result.get('triangle_type', 'symmetric'),
                    'timeframe': params.get('timeframe', 'unknown')
                }
            )
            
            patterns.append(pattern_record)
            
            logger.info(f"Detected pennant pattern: {pattern_record.id} "
                      f"with confidence {pattern_record.confidence_score:.3f}")
        
        return patterns
    
    def _detect_flag_after_pole(self, df: pd.DataFrame, 
                              flagpole: Flagpole,
                              params: dict) -> Optional[dict]:
        """检测旗杆后的矩形旗面形态"""
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        flag_params = params.get('flag', params.get('pattern', {}))
        if flagpole_end_idx >= len(df) - flag_params['min_bars']:
            return None
        
        min_flag_bars = flag_params['min_bars']
        max_flag_bars = min(
            flag_params['max_bars'],
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
        
        min_confidence = params.get('scoring', {}).get('min_confidence_score', 0.6)
        if best_flag and best_score >= min_confidence:
            return best_flag
        
        return None
    
    def _detect_pennant_after_pole(self, df: pd.DataFrame,
                                 flagpole: Flagpole,
                                 params: dict) -> Optional[dict]:
        """检测旗杆后的三角旗形"""
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        pennant_params = params.get('pennant', params.get('pattern', {}))
        if flagpole_end_idx >= len(df) - pennant_params['min_bars']:
            return None
        
        min_pennant_bars = pennant_params['min_bars']
        max_pennant_bars = min(
            pennant_params['max_bars'],
            len(df) - flagpole_end_idx - 1
        )
        
        best_pennant = None
        best_score = 0
        
        # 尝试不同长度的三角旗面
        for pennant_duration in range(min_pennant_bars, max_pennant_bars + 1):
            pennant_start_idx = flagpole_end_idx + 1
            pennant_end_idx = pennant_start_idx + pennant_duration
            
            if pennant_end_idx >= len(df):
                continue
            
            pennant_data = df.iloc[pennant_start_idx:pennant_end_idx + 1].copy()
            
            # 分析三角旗形
            pennant_result = self._analyze_pennant_pattern(
                pennant_data, flagpole, params, df
            )
            
            if pennant_result and pennant_result['confidence_score'] > best_score:
                best_score = pennant_result['confidence_score']
                best_pennant = pennant_result
                best_pennant['duration'] = pennant_duration
        
        min_confidence = params.get('scoring', {}).get('min_confidence_score', 0.6)
        if best_pennant and best_score >= min_confidence:
            return best_pennant
        
        return None
    
    def _analyze_flag_pattern(self, flag_data: pd.DataFrame,
                            flagpole: Flagpole,
                            params: dict,
                            full_df: pd.DataFrame) -> Optional[dict]:
        """分析矩形旗面形态"""
        flag_params = params.get('flag', params.get('pattern', {}))
        if len(flag_data) < flag_params['min_bars']:
            return None
        
        # 1. 使用智能摆动点检测代替简单的边界寻找
        window_size = max(2, len(flag_data) // 5)
        swing_highs, swing_lows = self.pattern_components.find_swing_points(
            flag_data, 
            window=window_size,
            min_prominence_atr_multiple=flag_params.get('min_prominence_atr', 0.1)
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
        if not self._verify_opposite_slope(flagpole, upper_line, lower_line, flag_params):
            logger.debug("Failed opposite slope verification")
            return None
        
        # 2.2 验证平行度
        parallel_quality = self._calculate_parallel_quality(
            upper_line, lower_line, flag_data, flag_params
        )
        
        if parallel_quality < 0.5:  # 最低要求
            logger.debug(f"Poor parallel quality: {parallel_quality:.3f}")
            return None
        
        # 2.3 验证通道不发散
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
        
        # 2.5 验证突破准备度
        breakout_readiness = self._verify_breakout_preparation(
            flag_data, upper_line, lower_line, flag_params
        )
        
        # 3. 计算综合置信度
        base_confidence = self._calculate_flag_quality_score(
            flag_data, flagpole, boundaries, flag_params
        )
        
        # 根据验证调整置信度
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
    
    def _analyze_pennant_pattern(self, pennant_data: pd.DataFrame,
                               flagpole: Flagpole,
                               params: dict,
                               full_df: pd.DataFrame) -> Optional[dict]:
        """分析三角旗形形态"""
        pennant_params = params.get('pennant', params.get('pattern', {}))
        if len(pennant_data) < pennant_params['min_bars']:
            return None
        
        # 1. 使用智能摆动点检测
        window_size = max(2, len(pennant_data) // 5)
        swing_highs, swing_lows = self.pattern_components.find_swing_points(
            pennant_data, 
            window=window_size,
            min_prominence_atr_multiple=pennant_params.get('min_prominence_atr', 0.1)
        )
        
        # 如果摆动点太少，使用原始策略
        if len(swing_highs) < pennant_params['min_touches'] or \
           len(swing_lows) < pennant_params['min_touches']:
            support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
                pennant_data,
                window_size=max(2, len(pennant_data) // 12),
                category=params.get('timeframe', 'medium')
            )
        else:
            support_idx, resistance_idx = swing_lows, swing_highs
        
        if len(support_idx) < pennant_params['min_touches'] or \
           len(resistance_idx) < pennant_params['min_touches']:
            logger.debug("Insufficient support/resistance points")
            return None
        
        # 2. 拟合趋势线
        upper_line = self._fit_convergent_line(pennant_data, resistance_idx, 'high')
        lower_line = self._fit_convergent_line(pennant_data, support_idx, 'low')
        
        if not upper_line or not lower_line:
            logger.debug("Failed to fit trend lines")
            return None
        
        # 3. 分类三角形类型
        triangle_type = self._classify_triangle(upper_line, lower_line, pennant_data)
        
        if triangle_type == 'invalid':
            logger.debug("Invalid triangle type")
            return None
        
        # 4. 计算收敛点（仅对收敛形态）
        if triangle_type in ['symmetric', 'ascending', 'descending']:
            apex = self.pattern_components.calculate_convergence_point(upper_line, lower_line)
        else:
            apex = None
        
        # 5. 根据类型进行验证
        if triangle_type == 'symmetric':
            if not self._verify_symmetric_triangle(upper_line, lower_line, apex, pennant_data, pennant_params):
                return None
        elif triangle_type == 'ascending':
            if not self._verify_ascending_triangle(upper_line, lower_line, pennant_data, pennant_params):
                return None
        elif triangle_type == 'descending':
            if not self._verify_descending_triangle(upper_line, lower_line, pennant_data, pennant_params):
                return None
        
        # 6. 使用增强的成交量分析
        pennant_start_idx = full_df[full_df['timestamp'] >= pennant_data.iloc[0]['timestamp']].index[0]
        pennant_end_idx = full_df[full_df['timestamp'] <= pennant_data.iloc[-1]['timestamp']].index[-1]
        flagpole_start_idx = full_df[full_df['timestamp'] >= flagpole.start_time].index[0]
        flagpole_end_idx = full_df[full_df['timestamp'] <= flagpole.end_time].index[-1]
        
        volume_analysis = self.pattern_components.analyze_volume_pattern_enhanced(
            full_df,
            pennant_start_idx,
            pennant_end_idx,
            flagpole_start_idx,
            flagpole_end_idx
        )
        
        if volume_analysis['health_score'] < 0.5:
            logger.debug(f"Poor volume health score: {volume_analysis['health_score']:.3f}")
            return None
        
        # 7. 计算各项质量指标
        convergence_quality = self._calculate_convergence_quality_enhanced(
            upper_line, lower_line, apex, pennant_data, pennant_params, triangle_type
        )
        
        symmetry_score = self._calculate_symmetry_enhanced(
            pennant_data, upper_line, lower_line, triangle_type
        )
        
        # 8. 计算支撑/阻力质量
        sr_quality = self._assess_support_resistance_quality(
            pennant_data, support_idx, resistance_idx
        )
        
        # 9. 计算总体置信度（根据形态类型调整权重）
        if triangle_type == 'symmetric':
            base_confidence = self._calculate_pennant_quality_score(
                pennant_data, flagpole, upper_line, lower_line, apex, pennant_params
            )
            confidence_score = (base_confidence * 0.5 + 
                              convergence_quality * 0.2 + 
                              symmetry_score * 0.15 +
                              volume_analysis['health_score'] * 0.1 +
                              sr_quality * 0.05)
        else:
            # 非对称三角形降低对称性权重，增加支撑阻力权重
            base_confidence = self._calculate_pennant_quality_score(
                pennant_data, flagpole, upper_line, lower_line, apex, pennant_params
            )
            confidence_score = (base_confidence * 0.5 + 
                              convergence_quality * 0.25 + 
                              symmetry_score * 0.05 +
                              volume_analysis['health_score'] * 0.1 +
                              sr_quality * 0.1)
        
        return {
            'boundaries': [upper_line, lower_line],
            'apex': apex,
            'confidence_score': confidence_score,
            'convergence_quality': convergence_quality,
            'symmetry_score': symmetry_score,
            'volume_pattern': volume_analysis,
            'triangle_type': triangle_type,
            'support_resistance_quality': sr_quality,
            'swing_points': {
                'highs': swing_highs,
                'lows': swing_lows
            }
        }
    
    def _remove_overlapping_patterns(self, patterns: List[PatternRecord]) -> List[PatternRecord]:
        """去除重叠的形态，保留置信度更高的"""
        if len(patterns) <= 1:
            return patterns
        
        final_patterns = []
        
        for i, pattern in enumerate(patterns):
            is_overlapping = False
            
            for j, other_pattern in enumerate(patterns):
                if i == j:
                    continue
                
                # 检查是否重叠（基于旗杆和形态时间范围）
                if self._patterns_overlap(pattern, other_pattern):
                    # 如果当前形态置信度更低，跳过
                    if pattern.confidence_score <= other_pattern.confidence_score:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                final_patterns.append(pattern)
        
        return final_patterns
    
    def _patterns_overlap(self, pattern1: PatternRecord, pattern2: PatternRecord) -> bool:
        """检查两个形态是否重叠"""
        # 简单检查：如果旗杆时间重叠，认为是重叠形态
        pole1_start = pattern1.flagpole.start_time
        pole1_end = pattern1.flagpole.end_time
        pole2_start = pattern2.flagpole.start_time
        pole2_end = pattern2.flagpole.end_time
        
        # 检查时间段是否重叠
        return not (pole1_end < pole2_start or pole2_end < pole1_start)
    
    def _create_pattern_record(self, symbol: str, flagpole: Flagpole,
                             boundaries: List[TrendLine], duration: int,
                             confidence_score: float, sub_type: str,
                             additional_info: dict = None) -> PatternRecord:
        """创建形态记录"""
        import uuid
        from datetime import datetime
        
        pattern_record = PatternRecord(
            id=str(uuid.uuid4()),
            symbol=symbol,
            pattern_type=self.get_pattern_type(),
            sub_type=sub_type,
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
    
    # 以下方法沿用原有的Flag和Pennant检测逻辑
    # 这里只列出方法签名，实际实现会从原文件中复制
    
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
        
        min_angle = params['min_slope_angle']
        max_angle = params['max_slope_angle']
        
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
        
        tolerance = params['parallel_tolerance']
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
    
    # Pennant相关方法
    def _fit_convergent_line(self, pennant_data: pd.DataFrame,
                            point_indices: List[int],
                            price_type: str) -> Optional[TrendLine]:
        """拟合收敛趋势线（支持RANSAC）"""
        if len(point_indices) < 2:
            return None
        
        # 检查是否启用RANSAC拟合
        use_ransac = getattr(self, 'enable_ransac_fitting', True)
        
        if use_ransac and len(point_indices) >= 3:
            # 使用RANSAC拟合（对噪音更鲁棒）
            trend_line = self.pattern_components.fit_trend_line_ransac(
                pennant_data, point_indices, price_type, use_ransac=True
            )
            
            if trend_line is not None:
                return trend_line
        
        # 回退到传统方法
        # 提取价格和时间
        prices = pennant_data.iloc[point_indices][price_type].values
        timestamps = pennant_data.iloc[point_indices]['timestamp'].values
        
        # 使用线性回归拟合
        x = np.arange(len(prices))
        
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        except Exception:
            return None
        
        # 对于Pennant，我们接受较低的拟合度（因为是收敛形态）
        if abs(r_value) < 0.3:
            # 尝试使用首尾连线
            if len(prices) >= 2:
                slope = (prices[-1] - prices[0]) / (len(prices) - 1)
                intercept = prices[0]
                r_value = 0.5  # 给一个中等拟合度
            else:
                return None
        
        start_price = intercept
        end_price = slope * (len(prices) - 1) + intercept
        
        return TrendLine(
            start_time=timestamps[0],
            end_time=timestamps[-1],
            start_price=start_price,
            end_price=end_price,
            slope=slope,
            r_squared=r_value ** 2
        )
    
    def _classify_triangle(self, upper_line: TrendLine, 
                          lower_line: TrendLine,
                          pennant_data: pd.DataFrame) -> str:
        """
        智能三角形分类
        
        Returns:
            'symmetric': 对称三角形
            'ascending': 上升三角形
            'descending': 下降三角形
            'invalid': 无效形态
        """
        # 计算标准化斜率（相对于平均价格）
        avg_price = pennant_data['close'].mean()
        if avg_price <= 0:
            return 'invalid'
        
        upper_slope_normalized = upper_line.slope / avg_price
        lower_slope_normalized = lower_line.slope / avg_price
        
        # 水平判定阈值
        horizontal_threshold = 0.0005  # 0.05%每单位时间
        
        # 检查是否发散
        if upper_slope_normalized > 0 and lower_slope_normalized < 0:
            # 上边界上升，下边界下降 - 发散
            return 'invalid'
        
        # 分类逻辑
        upper_horizontal = abs(upper_slope_normalized) < horizontal_threshold
        lower_horizontal = abs(lower_slope_normalized) < horizontal_threshold
        
        if upper_horizontal and not lower_horizontal and lower_slope_normalized > 0:
            # 上边界水平，下边界上升
            return 'ascending'
        elif lower_horizontal and not upper_horizontal and upper_slope_normalized < 0:
            # 下边界水平，上边界下降
            return 'descending'
        elif upper_slope_normalized < -horizontal_threshold and lower_slope_normalized > horizontal_threshold:
            # 上边界下降，下边界上升 - 收敛
            # 检查是否对称
            slope_sum = abs(upper_slope_normalized) + abs(lower_slope_normalized)
            if slope_sum > 0:
                symmetry_ratio = min(abs(upper_slope_normalized), abs(lower_slope_normalized)) / \
                               max(abs(upper_slope_normalized), abs(lower_slope_normalized))
                if symmetry_ratio > 0.5:  # 斜率相似度超过50%
                    return 'symmetric'
            return 'symmetric'  # 默认为对称
        else:
            return 'invalid'
    
    def _verify_symmetric_triangle(self, upper_line: TrendLine,
                                 lower_line: TrendLine,
                                 apex: Optional[Tuple[float, float]],
                                 pennant_data: pd.DataFrame,
                                 params: dict) -> bool:
        """验证对称三角形"""
        # 使用收敛性验证
        return self._verify_convergence(upper_line, lower_line, apex, pennant_data, params)
        
    def _verify_convergence(self, upper_line: TrendLine,
                          lower_line: TrendLine,
                          apex: Optional[Tuple[float, float]],
                          pennant_data: pd.DataFrame,
                          params: dict) -> bool:
        """验证收敛性"""
        # 1. 检查是否有交点（apex）
        if not apex:
            # 如果平行，则不是收敛形态
            logger.debug("No convergence point (parallel lines)")
            return False
        
        apex_time, apex_price = apex
        
        # 2. 检查apex位置的合理性
        pattern_length = len(pennant_data)
        apex_distance_ratio = apex_time / pattern_length if pattern_length > 0 else float('inf')
        
        min_ratio, max_ratio = params.get('apex_distance_range', [0.5, 3.0])
        
        if not (min_ratio <= apex_distance_ratio <= max_ratio):
            logger.debug(f"Apex distance ratio {apex_distance_ratio:.2f} "
                        f"outside range [{min_ratio}, {max_ratio}]")
            return False
        
        # 3. 检查收敛程度
        start_range = abs(upper_line.start_price - lower_line.start_price)
        end_range = abs(upper_line.end_price - lower_line.end_price)
        
        if start_range <= 0:
            return False
        
        convergence_ratio = 1 - (end_range / start_range)
        min_convergence = params.get('convergence_ratio', 0.5)
        
        if convergence_ratio < min_convergence:
            logger.debug(f"Convergence ratio {convergence_ratio:.2f} "
                        f"below minimum {min_convergence}")
            return False
        
        # 4. 检查价格范围合理性
        avg_price = pennant_data['close'].mean()
        if abs(apex_price - avg_price) / avg_price > 0.5:  # Apex价格偏离过大
            logger.debug("Apex price too far from average")
            return False
        
        return True
    
    def _verify_ascending_triangle(self, upper_line: TrendLine,
                                  lower_line: TrendLine,
                                  pennant_data: pd.DataFrame,
                                  params: dict) -> bool:
        """验证上升三角形"""
        # 1. 验证上边界是否足够水平
        avg_price = pennant_data['close'].mean()
        upper_slope_pct = abs(upper_line.slope / avg_price) * 100
        
        if upper_slope_pct > 0.1:  # 斜率超过0.1%
            logger.debug(f"Upper boundary not horizontal enough: {upper_slope_pct:.3f}%")
            return False
        
        # 2. 验证下边界是否上升
        if lower_line.slope <= 0:
            logger.debug("Lower boundary not ascending")
            return False
        
        # 3. 验证价格多次触及水平阻力
        resistance_price = upper_line.start_price  # 使用起始价格作为阻力位
        high_prices = pennant_data['high'].values
        touches = sum(1 for high in high_prices if abs(high - resistance_price) / resistance_price < 0.005)
        
        if touches < params.get('min_touches', 2):
            logger.debug(f"Insufficient resistance touches: {touches}")
            return False
        
        return True
    
    def _verify_descending_triangle(self, upper_line: TrendLine,
                                   lower_line: TrendLine,
                                   pennant_data: pd.DataFrame,
                                   params: dict) -> bool:
        """验证下降三角形"""
        # 1. 验证下边界是否足够水平
        avg_price = pennant_data['close'].mean()
        lower_slope_pct = abs(lower_line.slope / avg_price) * 100
        
        if lower_slope_pct > 0.1:  # 斜率超过0.1%
            logger.debug(f"Lower boundary not horizontal enough: {lower_slope_pct:.3f}%")
            return False
        
        # 2. 验证上边界是否下降
        if upper_line.slope >= 0:
            logger.debug("Upper boundary not descending")
            return False
        
        # 3. 验证价格多次触及水平支撑
        support_price = lower_line.start_price  # 使用起始价格作为支撑位
        low_prices = pennant_data['low'].values
        touches = sum(1 for low in low_prices if abs(low - support_price) / support_price < 0.005)
        
        if touches < params.get('min_touches', 2):
            logger.debug(f"Insufficient support touches: {touches}")
            return False
        
        return True
    
    def _calculate_convergence_quality_enhanced(self, upper_line: TrendLine,
                                              lower_line: TrendLine,
                                              apex: Optional[Tuple[float, float]],
                                              pennant_data: pd.DataFrame,
                                              params: dict,
                                              triangle_type: str) -> float:
        """增强的收敛质量计算"""
        if triangle_type in ['ascending', 'descending']:
            # 非对称三角形的质量主要看水平线质量和触及次数
            return self._calculate_asymmetric_quality(
                upper_line, lower_line, pennant_data, triangle_type
            )
        else:
            # 对称三角形使用收敛质量计算
            return self._calculate_convergence_quality(
                upper_line, lower_line, apex, pennant_data, params
            )
    
    def _calculate_asymmetric_quality(self, upper_line: TrendLine,
                                    lower_line: TrendLine,
                                    pennant_data: pd.DataFrame,
                                    triangle_type: str) -> float:
        """计算非对称三角形质量"""
        avg_price = pennant_data['close'].mean()
        
        if triangle_type == 'ascending':
            # 检查上边界水平度
            horizontal_line = upper_line
            trending_line = lower_line
            reference_price = upper_line.start_price
            price_data = pennant_data['high']
        else:  # descending
            horizontal_line = lower_line
            trending_line = upper_line
            reference_price = lower_line.start_price
            price_data = pennant_data['low']
        
        # 1. 水平线质量
        horizontal_quality = 1.0 - min(1.0, abs(horizontal_line.slope / avg_price) * 100)
        
        # 2. 趋势线质量（R²）
        trend_quality = trending_line.r_squared
        
        # 3. 触及质量
        touches = sum(1 for price in price_data if abs(price - reference_price) / reference_price < 0.005)
        touch_quality = min(1.0, touches / 5)  # 5次触及得满分
        
        # 4. 形态紧凑度
        pattern_range = pennant_data['high'].max() - pennant_data['low'].min()
        compactness = 1.0 - min(1.0, pattern_range / avg_price / 0.1)  # 10%范围内得满分
        
        return (horizontal_quality * 0.4 + 
                trend_quality * 0.3 + 
                touch_quality * 0.2 + 
                compactness * 0.1)
                
    def _calculate_convergence_quality(self, upper_line: TrendLine,
                                     lower_line: TrendLine,
                                     apex: Optional[Tuple[float, float]],
                                     pennant_data: pd.DataFrame,
                                     params: dict) -> float:
        """计算收敛质量"""
        if not apex:
            return 0.0
        
        # 1. 收敛程度
        start_range = abs(upper_line.start_price - lower_line.start_price)
        end_range = abs(upper_line.end_price - lower_line.end_price)
        
        if start_range > 0:
            convergence_ratio = 1 - (end_range / start_range)
            convergence_score = min(1.0, convergence_ratio / params.get('convergence_ratio', 0.5))
        else:
            convergence_score = 0.0
        
        # 2. 收敛的线性程度（两条线是否稳定收敛）
        # 检查中间点
        n_checks = min(5, len(pennant_data) - 1)
        linear_scores = []
        
        for i in range(1, n_checks):
            progress = i / n_checks
            expected_range = start_range * (1 - progress * convergence_ratio)
            
            # 计算实际范围
            idx = int(progress * (len(pennant_data) - 1))
            upper_price = upper_line.start_price + upper_line.slope * idx
            lower_price = lower_line.start_price + lower_line.slope * idx
            actual_range = abs(upper_price - lower_price)
            
            # 计算偏差
            if expected_range > 0:
                deviation = abs(actual_range - expected_range) / expected_range
                linear_scores.append(max(0, 1 - deviation))
        
        linearity_score = np.mean(linear_scores) if linear_scores else 0.5
        
        # 3. Apex位置质量
        apex_time, _ = apex
        pattern_length = len(pennant_data)
        apex_ratio = apex_time / pattern_length if pattern_length > 0 else float('inf')
        
        min_ratio, max_ratio = params.get('apex_distance_range', [0.5, 3.0])
        ideal_ratio = (min_ratio + max_ratio) / 2
        
        if min_ratio <= apex_ratio <= max_ratio:
            # 越接近理想位置越好
            apex_score = 1 - abs(apex_ratio - ideal_ratio) / (max_ratio - min_ratio)
        else:
            apex_score = 0.0
        
        # 综合评分
        return convergence_score * 0.5 + linearity_score * 0.3 + apex_score * 0.2
    
    def _calculate_symmetry_enhanced(self, pennant_data: pd.DataFrame,
                                   upper_line: TrendLine,
                                   lower_line: TrendLine,
                                   triangle_type: str) -> float:
        """增强的对称性计算"""
        if triangle_type in ['ascending', 'descending']:
            # 非对称三角形不需要对称性，返回中等分数
            return 0.5
        else:
            # 对称三角形使用对称性计算
            return self._calculate_symmetry(pennant_data, upper_line, lower_line)
            
    def _calculate_symmetry(self, pennant_data: pd.DataFrame,
                          upper_line: TrendLine,
                          lower_line: TrendLine) -> float:
        """计算对称性"""
        # 1. 斜率对称性
        upper_slope_angle = np.degrees(np.arctan(upper_line.slope))
        lower_slope_angle = np.degrees(np.arctan(lower_line.slope))
        
        # 理想情况：斜率大小相近，方向相反
        if upper_slope_angle * lower_slope_angle < 0:  # 方向相反
            slope_symmetry = 1 - abs(abs(upper_slope_angle) - abs(lower_slope_angle)) / \
                             max(abs(upper_slope_angle), abs(lower_slope_angle), 1)
        else:
            # 方向相同也可以，但对称性较差
            slope_symmetry = 0.5
        
        # 2. 触及点分布对称性
        high_touches = []
        low_touches = []
        
        for i in range(len(pennant_data)):
            # 理论边界
            theoretical_upper = upper_line.start_price + upper_line.slope * i
            theoretical_lower = lower_line.start_price + lower_line.slope * i
            
            # 检查触及（2%容差）
            if abs(pennant_data.iloc[i]['high'] - theoretical_upper) / theoretical_upper < 0.02:
                high_touches.append(i)
            
            if abs(pennant_data.iloc[i]['low'] - theoretical_lower) / theoretical_lower < 0.02:
                low_touches.append(i)
        
        # 触及点数量的对称性
        if len(high_touches) > 0 and len(low_touches) > 0:
            touch_ratio = min(len(high_touches), len(low_touches)) / \
                         max(len(high_touches), len(low_touches))
        else:
            touch_ratio = 0.0
        
        # 3. 价格分布对称性
        midline = (upper_line.start_price + lower_line.start_price) / 2
        price_deviations = pennant_data['close'] - midline
        
        # 检查正负偏离的平衡
        positive_dev = (price_deviations > 0).sum()
        negative_dev = (price_deviations < 0).sum()
        
        if positive_dev + negative_dev > 0:
            balance_score = 1 - abs(positive_dev - negative_dev) / (positive_dev + negative_dev)
        else:
            balance_score = 0.5
        
        # 综合对称性得分
        return slope_symmetry * 0.4 + touch_ratio * 0.3 + balance_score * 0.3
    
    def _assess_support_resistance_quality(self, pennant_data: pd.DataFrame,
                                         support_idx: List[int],
                                         resistance_idx: List[int]) -> float:
        """评估支撑/阻力点质量"""
        if not support_idx or not resistance_idx:
            return 0.0
        
        # 1. 点的数量
        support_count = len(support_idx)
        resistance_count = len(resistance_idx)
        count_score = min(1.0, (support_count + resistance_count) / 10)  # 10个点得满分
        
        # 2. 分布均匀性
        pattern_length = len(pennant_data)
        
        # 支撑点分布
        support_distribution = [idx / pattern_length for idx in support_idx]
        support_gaps = [support_distribution[i+1] - support_distribution[i] 
                       for i in range(len(support_distribution)-1)]
        support_uniformity = 1.0 - (np.std(support_gaps) if support_gaps else 0.5)
        
        # 阻力点分布
        resistance_distribution = [idx / pattern_length for idx in resistance_idx]
        resistance_gaps = [resistance_distribution[i+1] - resistance_distribution[i] 
                          for i in range(len(resistance_distribution)-1)]
        resistance_uniformity = 1.0 - (np.std(resistance_gaps) if resistance_gaps else 0.5)
        
        uniformity_score = (support_uniformity + resistance_uniformity) / 2
        
        # 3. 边界清晰度（价格是否明确触及边界）
        high_prices = pennant_data['high'].values
        low_prices = pennant_data['low'].values
        
        # 检查阻力点的清晰度
        resistance_clarity_scores = []
        for idx in resistance_idx[:5]:  # 最多检查5个点
            if idx < len(high_prices):
                # 检查是否是局部最高点
                window = 2
                start = max(0, idx - window)
                end = min(len(high_prices), idx + window + 1)
                is_peak = high_prices[idx] == max(high_prices[start:end])
                resistance_clarity_scores.append(1.0 if is_peak else 0.5)
        
        # 检查支撑点的清晰度
        support_clarity_scores = []
        for idx in support_idx[:5]:  # 最多检查5个点
            if idx < len(low_prices):
                # 检查是否是局部最低点
                window = 2
                start = max(0, idx - window)
                end = min(len(low_prices), idx + window + 1)
                is_trough = low_prices[idx] == min(low_prices[start:end])
                support_clarity_scores.append(1.0 if is_trough else 0.5)
        
        clarity_score = np.mean(resistance_clarity_scores + support_clarity_scores) if \
                       (resistance_clarity_scores or support_clarity_scores) else 0.5
        
        # 综合评分
        return count_score * 0.3 + uniformity_score * 0.3 + clarity_score * 0.4
    
    def _calculate_pennant_quality_score(self, pennant_data: pd.DataFrame,
                                       flagpole: Flagpole,
                                       upper_line: TrendLine,
                                       lower_line: TrendLine,
                                       apex: Tuple[float, float],
                                       params: dict) -> float:
        """
        计算三角旗形质量评分（替代strategy.score_pennant_quality）
        
        Args:
            pennant_data: 三角旗形数据
            flagpole: 旗杆对象
            upper_line: 上边界线
            lower_line: 下边界线
            apex: 收敛点
            params: 参数配置
            
        Returns:
            质量评分（0-1）
        """
        scores = []
        
        # 1. 几何特征评分（45%权重）
        # 1.1 R²拟合质量
        r_squared_score = (upper_line.r_squared + lower_line.r_squared) / 2
        scores.append(('r_squared', r_squared_score, 0.15))
        
        # 1.2 收敛质量
        convergence_score = self._calculate_convergence_score(upper_line, lower_line, apex, pennant_data, params)
        scores.append(('convergence', convergence_score, 0.20))
        
        # 1.3 形态完整性
        completeness_score = self._calculate_completeness_score(pennant_data, params)
        scores.append(('completeness', completeness_score, 0.10))
        
        # 2. 技术特征评分（35%权重）
        # 2.1 成交量模式
        volume_score = self._calculate_pennant_volume_score(pennant_data, params)
        scores.append(('volume', volume_score, 0.20))
        
        # 2.2 价格波动收敛
        volatility_score = self._calculate_volatility_convergence_score(pennant_data)
        scores.append(('volatility', volatility_score, 0.15))
        
        # 3. 支撑阻力质量评分（20%权重）
        # 3.1 边界触及次数和质量
        boundary_score = self._calculate_boundary_quality_score(pennant_data, upper_line, lower_line, params)
        scores.append(('boundary', boundary_score, 0.20))
        
        # 计算加权总分
        total_score = sum(score * weight for _, score, weight in scores)
        
        # 记录详细评分
        score_details = {name: f"{score:.3f}" for name, score, _ in scores}
        logger.debug(f"Pennant quality scores: {score_details}, total: {total_score:.3f}")
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_convergence_score(self, upper_line: TrendLine, lower_line: TrendLine,
                                   apex: Tuple[float, float], pennant_data: pd.DataFrame,
                                   params: dict) -> float:
        """计算收敛质量评分"""
        # 1. 收敛比例
        start_width = abs(upper_line.start_price - lower_line.start_price)
        end_width = abs(upper_line.end_price - lower_line.end_price)
        
        if start_width == 0:
            return 0.0
        
        convergence_ratio = end_width / start_width
        min_convergence = params.get('convergence_ratio', 0.5)
        
        # 收敛比例评分
        if convergence_ratio <= min_convergence:
            convergence_score = 1.0
        else:
            convergence_score = max(0.0, (1.0 - convergence_ratio) / (1.0 - min_convergence))
        
        # 2. 顶点距离合理性
        apex_distance_range = params.get('apex_distance_range', [0.5, 3.0])
        min_distance, max_distance = apex_distance_range
        
        pattern_length = len(pennant_data)
        if pattern_length > 0:
            apex_distance = apex[0] / pattern_length  # 归一化距离
            
            if min_distance <= apex_distance <= max_distance:
                distance_score = 1.0
            elif apex_distance < min_distance:
                distance_score = max(0.0, apex_distance / min_distance)
            else:
                distance_score = max(0.0, 1.0 - (apex_distance - max_distance) / max_distance)
        else:
            distance_score = 0.5
        
        return (convergence_score * 0.7 + distance_score * 0.3)
    
    def _calculate_completeness_score(self, pennant_data: pd.DataFrame, params: dict) -> float:
        """计算形态完整性评分"""
        actual_duration = len(pennant_data)
        min_bars = params['min_bars']
        max_bars = params['max_bars']
        
        if min_bars <= actual_duration <= max_bars:
            # 在有效范围内，偏向较短的形态（更典型）
            optimal_ratio = 0.6  # 偏向范围的60%位置
            optimal_duration = min_bars + (max_bars - min_bars) * optimal_ratio
            deviation = abs(actual_duration - optimal_duration) / (max_bars - min_bars)
            return max(0.5, 1.0 - deviation)
        else:
            return 0.0
    
    def _calculate_pennant_volume_score(self, pennant_data: pd.DataFrame, params: dict) -> float:
        """计算三角旗形成交量评分"""
        if 'volume' not in pennant_data.columns:
            return 0.5
        
        # 成交量应该逐渐萖缩
        volumes = pennant_data['volume'].values
        
        # 计算成交量趋势
        x = np.arange(len(volumes))
        slope, _, r_value, _, _ = stats.linregress(x, volumes)
        
        # 成交量下降是好的信号
        volume_mean = volumes.mean()
        if volume_mean > 0:
            normalized_slope = slope / volume_mean
            
            if normalized_slope < -0.05:  # 明显下降
                trend_score = 1.0
            elif -0.05 <= normalized_slope <= 0.02:  # 轻微下降或持平
                trend_score = 0.8
            else:  # 上升
                trend_score = max(0.2, 0.8 - normalized_slope * 10)
            
            # R²评分：趋势越明显越好
            r_squared_score = r_value ** 2
            
            return trend_score * 0.7 + r_squared_score * 0.3
        
        return 0.5
    
    def _calculate_volatility_convergence_score(self, pennant_data: pd.DataFrame) -> float:
        """计算价格波动收敛评分"""
        if len(pennant_data) < 6:
            return 0.5
        
        # 计算前半部分和后半部分的波动率
        mid_point = len(pennant_data) // 2
        early_volatility = pennant_data['close'].iloc[:mid_point].std()
        late_volatility = pennant_data['close'].iloc[mid_point:].std()
        
        if early_volatility == 0:
            return 0.5
        
        # 波动率收敛比例
        convergence_ratio = late_volatility / early_volatility
        
        # 理想情况：后期波动率是前期的30%-70%
        if 0.3 <= convergence_ratio <= 0.7:
            return 1.0
        elif convergence_ratio < 0.3:
            return max(0.3, convergence_ratio / 0.3)
        else:
            return max(0.2, 1.0 - (convergence_ratio - 0.7) / 0.3)
    
    def _calculate_boundary_quality_score(self, pennant_data: pd.DataFrame,
                                        upper_line: TrendLine, lower_line: TrendLine,
                                        params: dict) -> float:
        """计算边界质量评分"""
        min_touches = params.get('min_touches', 2)
        
        # 计算价格与边界的接触质量
        tolerance = pennant_data['close'].std() * 0.3
        
        upper_touches = 0
        lower_touches = 0
        total_deviation = 0
        
        for i, row in pennant_data.iterrows():
            time_ratio = i / (len(pennant_data) - 1) if len(pennant_data) > 1 else 0
            
            # 计算理论边界价格
            upper_price = upper_line.start_price + upper_line.slope * time_ratio
            lower_price = lower_line.start_price + lower_line.slope * time_ratio
            
            # 检查触及情况
            upper_distance = abs(row['high'] - upper_price)
            lower_distance = abs(row['low'] - lower_price)
            
            if upper_distance <= tolerance:
                upper_touches += 1
            if lower_distance <= tolerance:
                lower_touches += 1
            
            # 累计偏离度
            min_distance = min(
                abs(row['close'] - upper_price),
                abs(row['close'] - lower_price)
            )
            total_deviation += min_distance
        
        # 触及次数评分
        total_touches = upper_touches + lower_touches
        touch_score = min(1.0, total_touches / (min_touches * 2))
        
        # 偏离度评分（越小越好）
        avg_deviation = total_deviation / len(pennant_data)
        channel_width = abs(upper_line.start_price - lower_line.start_price)
        
        if channel_width > 0:
            deviation_ratio = avg_deviation / channel_width
            deviation_score = max(0.0, 1.0 - deviation_ratio * 2)
        else:
            deviation_score = 0.5
        
        return touch_score * 0.6 + deviation_score * 0.4
    
    # Flag相关的辅助方法
    def _calculate_retracement_score(self, flag_data: pd.DataFrame, 
                                   flagpole: Flagpole, params: dict) -> float:
        """计算回撤比例评分"""
        retracement_range = params.get('retracement_range', [0.2, 0.6])
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
        min_angle = params['min_slope_angle']
        max_angle = params['max_slope_angle']
        
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
        min_bars = params['min_bars']
        max_bars = params['max_bars']
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
        min_touches = params.get('min_touches', 2)
        
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