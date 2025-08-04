"""
三角旗形（Pennant）形态检测器
正确实现收敛三角形的检测
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from scipy import stats
from loguru import logger

from src.data.models.base_models import PatternRecord, Flagpole, TrendLine, PatternType
from src.patterns.base import BasePatternDetector


class PennantDetector(BasePatternDetector):
    """
    三角旗形（Pennant）形态检测器
    检测收敛的三角形形态
    """
    
    def get_pattern_type(self) -> PatternType:
        """获取形态类型"""
        return PatternType.PENNANT
    
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
                    'pennant': {
                        'min_bars': 10,
                        'max_bars': 40,
                        'min_touches': 2,
                        'convergence_ratio': 0.5,  # 最小收敛比例
                        'apex_distance_range': (0.5, 3.0),  # Apex距离范围（相对于形态长度）
                        'symmetry_tolerance': 0.4,
                        'volume_decay_threshold': 0.8
                    }
                },
                'short': {
                    'flagpole': {
                        'min_bars': 4,
                        'max_bars': 10,
                        'min_height_percent': 1.0,
                        'max_height_percent': 10.0,
                        'volume_surge_ratio': 2.0,
                        'max_retracement': 0.3,
                        'min_trend_strength': 0.7
                    },
                    'pennant': {
                        'min_bars': 8,
                        'max_bars': 30,
                        'min_touches': 3,
                        'convergence_ratio': 0.6,
                        'apex_distance_range': (0.4, 2.0),
                        'symmetry_tolerance': 0.3,
                        'volume_decay_threshold': 0.7
                    }
                },
                'medium_long': {
                    'flagpole': {
                        'min_bars': 3,
                        'max_bars': 6,
                        'min_height_percent': 2.0,
                        'max_height_percent': 15.0,
                        'volume_surge_ratio': 2.5,
                        'max_retracement': 0.2,
                        'min_trend_strength': 0.8
                    },
                    'pennant': {
                        'min_bars': 5,
                        'max_bars': 25,
                        'min_touches': 4,
                        'convergence_ratio': 0.7,
                        'apex_distance_range': (0.3, 1.5),
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
        检测三角旗形（Pennant）形态
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            
        Returns:
            检测到的Pennant形态列表
        """
        patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for flagpole in flagpoles:
            # 在旗杆后寻找收敛的三角形
            pennant_result = self._detect_pennant_after_pole(
                df, flagpole, params
            )
            
            if pennant_result:
                # 创建形态记录
                pattern_record = self._create_pattern_record(
                    symbol=symbol,
                    flagpole=flagpole,
                    boundaries=pennant_result['boundaries'],
                    duration=pennant_result['duration'],
                    confidence_score=pennant_result['confidence_score'],
                    additional_info={
                        'apex': pennant_result.get('apex'),
                        'convergence_quality': pennant_result.get('convergence_quality', 0),
                        'symmetry_score': pennant_result.get('symmetry_score', 0),
                        'volume_pattern': pennant_result.get('volume_pattern', {}),
                        'category': params['timeframe'],
                        'timeframe': params['timeframe']
                    }
                )
                
                patterns.append(pattern_record)
                
                logger.info(f"Detected pennant pattern: {pattern_record.id} "
                          f"with confidence {pattern_record.confidence_score:.3f}")
        
        return patterns
    
    def _detect_pennant_after_pole(self, df: pd.DataFrame,
                                 flagpole: Flagpole,
                                 params: dict) -> Optional[dict]:
        """检测旗杆后的三角旗形"""
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        if flagpole_end_idx >= len(df) - params['pattern']['min_bars']:
            return None
        
        min_pennant_bars = params['pattern']['min_bars']
        max_pennant_bars = min(
            params['pattern']['max_bars'],
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
        
        if best_pennant and best_score >= params['min_confidence']:
            return best_pennant
        
        return None
    
    def _analyze_pennant_pattern(self, pennant_data: pd.DataFrame,
                               flagpole: Flagpole,
                               params: dict,
                               full_df: pd.DataFrame) -> Optional[dict]:
        """分析三角旗形形态"""
        if len(pennant_data) < params['pattern']['min_bars']:
            return None
        
        # 1. 使用智能摆动点检测
        window_size = max(2, len(pennant_data) // 5)
        swing_highs, swing_lows = self.pattern_components.find_swing_points(
            pennant_data, 
            window=window_size,
            min_prominence_atr_multiple=params['pattern'].get('min_prominence_atr', 0.1)
        )
        
        # 如果摆动点太少，使用原始策略
        if len(swing_highs) < params['pattern']['min_touches'] or \
           len(swing_lows) < params['pattern']['min_touches']:
            support_idx, resistance_idx = self.pattern_components.find_support_resistance_points(
                pennant_data,
                window_size=max(2, len(pennant_data) // 12),
                category=params['timeframe']
            )
        else:
            support_idx, resistance_idx = swing_lows, swing_highs
        
        if len(support_idx) < params['pattern']['min_touches'] or \
           len(resistance_idx) < params['pattern']['min_touches']:
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
            if not self._verify_symmetric_triangle(upper_line, lower_line, apex, pennant_data, params):
                return None
        elif triangle_type == 'ascending':
            if not self._verify_ascending_triangle(upper_line, lower_line, pennant_data, params):
                return None
        elif triangle_type == 'descending':
            if not self._verify_descending_triangle(upper_line, lower_line, pennant_data, params):
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
            upper_line, lower_line, apex, pennant_data, params, triangle_type
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
                pennant_data, flagpole, upper_line, lower_line, apex, params
            )
            confidence_score = (base_confidence * 0.5 + 
                              convergence_quality * 0.2 + 
                              symmetry_score * 0.15 +
                              volume_analysis['health_score'] * 0.1 +
                              sr_quality * 0.05)
        else:
            # 非对称三角形降低对称性权重，增加支撑阻力权重
            base_confidence = self._calculate_pennant_quality_score(
                pennant_data, flagpole, upper_line, lower_line, apex, params
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
    
    def _fit_convergent_line(self, pennant_data: pd.DataFrame,
                            point_indices: List[int],
                            price_type: str) -> Optional[TrendLine]:
        """拟合收敛趋势线（支持RANSAC）"""
        if len(point_indices) < 2:
            return None
        
        # 检查是否启用RANSAC拟合
        use_ransac = getattr(self, '_use_ransac_fitting', True)
        
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
        
        min_ratio, max_ratio = params['pattern']['apex_distance_range']
        
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
        min_convergence = params['pattern']['convergence_ratio']
        
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
    
    def _verify_volume_pattern(self, pennant_data: pd.DataFrame,
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
            flagpole_volume = full_df['volume'].mean() * 2
        
        # Pennant期间的平均成交量
        pennant_volume = pennant_data['volume'].mean()
        
        # Pennant成交量应该显著小于旗杆
        if flagpole_volume > 0:
            volume_decay_ratio = pennant_volume / flagpole_volume
            threshold = params['pattern']['volume_decay_threshold']
            
            is_valid = volume_decay_ratio <= threshold
            
            logger.debug(f"Volume decay ratio: {volume_decay_ratio:.2f} "
                        f"(threshold: {threshold})")
            
            return is_valid
        
        return True
    
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
            convergence_score = min(1.0, convergence_ratio / params['pattern']['convergence_ratio'])
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
        
        min_ratio, max_ratio = params['pattern']['apex_distance_range']
        ideal_ratio = (min_ratio + max_ratio) / 2
        
        if min_ratio <= apex_ratio <= max_ratio:
            # 越接近理想位置越好
            apex_score = 1 - abs(apex_ratio - ideal_ratio) / (max_ratio - min_ratio)
        else:
            apex_score = 0.0
        
        # 综合评分
        return convergence_score * 0.5 + linearity_score * 0.3 + apex_score * 0.2
    
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
        # 使用原有的收敛性验证
        return self._verify_convergence(upper_line, lower_line, apex, pennant_data, params)
    
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
        
        if touches < params['pattern']['min_touches']:
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
        
        if touches < params['pattern']['min_touches']:
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
        if triangle_type == 'ascending' or triangle_type == 'descending':
            # 非对称三角形的质量主要看水平线质量和触及次数
            return self._calculate_asymmetric_quality(
                upper_line, lower_line, pennant_data, triangle_type
            )
        else:
            # 对称三角形使用原有逻辑
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
    
    def _calculate_symmetry_enhanced(self, pennant_data: pd.DataFrame,
                                   upper_line: TrendLine,
                                   lower_line: TrendLine,
                                   triangle_type: str) -> float:
        """增强的对称性计算"""
        if triangle_type in ['ascending', 'descending']:
            # 非对称三角形不需要对称性，返回中等分数
            return 0.5
        else:
            # 对称三角形使用原有逻辑
            return self._calculate_symmetry(pennant_data, upper_line, lower_line)
    
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
        min_convergence = params['pattern'].get('convergence_ratio', 0.5)
        
        # 收敛比例评分
        if convergence_ratio <= min_convergence:
            convergence_score = 1.0
        else:
            convergence_score = max(0.0, (1.0 - convergence_ratio) / (1.0 - min_convergence))
        
        # 2. 顶点距离合理性
        apex_distance_range = params['pattern'].get('apex_distance_range', [0.5, 3.0])
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
        min_bars = params['pattern']['min_bars']
        max_bars = params['pattern']['max_bars']
        
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
        
        # 成交量应该逐渐萎缩
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
        min_touches = params['pattern'].get('min_touches', 2)
        
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