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
                'enable_multi_timeframe': False
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
                                params: dict,
                                strategy) -> List[PatternRecord]:
        """
        检测三角旗形（Pennant）形态
        
        Args:
            df: 预处理后的数据
            flagpoles: 检测到的旗杆
            params: 周期相关参数
            strategy: 周期策略对象
            
        Returns:
            检测到的Pennant形态列表
        """
        patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for flagpole in flagpoles:
            # 在旗杆后寻找收敛的三角形
            pennant_result = self._detect_pennant_after_pole(
                df, flagpole, params, strategy
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
                        'category': params['category'],
                        'timeframe': params['timeframe']
                    }
                )
                
                patterns.append(pattern_record)
                
                logger.info(f"Detected pennant pattern: {pattern_record.id} "
                          f"with confidence {pattern_record.confidence_score:.3f}")
        
        return patterns
    
    def _detect_pennant_after_pole(self, df: pd.DataFrame,
                                 flagpole: Flagpole,
                                 params: dict,
                                 strategy) -> Optional[dict]:
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
                pennant_data, flagpole, params, strategy, df
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
                               strategy,
                               full_df: pd.DataFrame) -> Optional[dict]:
        """分析三角旗形形态"""
        if len(pennant_data) < params['pattern']['min_bars']:
            return None
        
        # 1. 使用策略寻找关键支撑/阻力点
        support_idx, resistance_idx = strategy.find_key_points(pennant_data, flagpole)
        
        if len(support_idx) < params['pattern']['min_touches'] or \
           len(resistance_idx) < params['pattern']['min_touches']:
            logger.debug("Insufficient support/resistance points")
            return None
        
        # 2. 拟合收敛的趋势线（不限制方向）
        upper_line = self._fit_convergent_line(pennant_data, resistance_idx, 'high')
        lower_line = self._fit_convergent_line(pennant_data, support_idx, 'low')
        
        if not upper_line or not lower_line:
            logger.debug("Failed to fit trend lines")
            return None
        
        # 3. 验证收敛性
        apex = self.pattern_components.calculate_convergence_point(upper_line, lower_line)
        
        if not self._verify_convergence(upper_line, lower_line, apex, pennant_data, params):
            logger.debug("Failed convergence verification")
            return None
        
        # 4. 验证成交量模式
        if not self._verify_volume_pattern(pennant_data, flagpole, full_df, params):
            logger.debug("Invalid volume pattern")
            return None
        
        # 5. 计算各项质量指标
        convergence_quality = self._calculate_convergence_quality(
            upper_line, lower_line, apex, pennant_data, params
        )
        
        symmetry_score = self._calculate_symmetry(
            pennant_data, upper_line, lower_line
        )
        
        # 6. 计算总体置信度
        confidence_score = strategy.score_pennant_quality(
            pennant_data, flagpole, upper_line, lower_line, apex
        )
        
        # 7. 分析成交量详情
        volume_pattern = self.pattern_components.analyze_volume_pattern(
            pennant_data['volume'], 'decreasing'
        )
        
        return {
            'boundaries': [upper_line, lower_line],
            'apex': apex,
            'confidence_score': confidence_score,
            'convergence_quality': convergence_quality,
            'symmetry_score': symmetry_score,
            'volume_pattern': volume_pattern
        }
    
    def _fit_convergent_line(self, pennant_data: pd.DataFrame,
                            point_indices: List[int],
                            price_type: str) -> Optional[TrendLine]:
        """拟合收敛趋势线（不限制方向）"""
        if len(point_indices) < 2:
            return None
        
        # 提取价格和时间
        prices = pennant_data.iloc[point_indices][price_type].values
        timestamps = pennant_data.iloc[point_indices]['timestamp'].values
        
        # 使用线性回归拟合
        x = np.arange(len(prices))
        
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        except:
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
            progress = i / (len(pennant_data) - 1) if len(pennant_data) > 1 else 0
            
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