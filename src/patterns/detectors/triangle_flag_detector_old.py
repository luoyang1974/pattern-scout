import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from scipy import stats
import uuid

from src.data.models.base_models import PatternRecord, Flagpole, TrendLine, PatternType
from src.patterns.indicators.technical_indicators import TechnicalIndicators, TrendAnalyzer
from loguru import logger


class TriangleFlagDetector:
    """三角旗形形态检测器"""
    
    def __init__(self, config: dict = None):
        """
        初始化三角旗形检测器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'flagpole': {
                'min_height_percent': 3.0,
                'max_height_percent': 50.0,
                'min_duration_days': 1,
                'max_duration_days': 5
            },
            'pennant': {
                'min_duration_days': 2,
                'max_duration_days': 10,
                'convergence_ratio': 0.7,  # 收敛比例要求
                'min_convergence_angle': 5,  # 最小收敛角度（度）
                'max_convergence_angle': 45  # 最大收敛角度（度）
            },
            'volume': {
                'flagpole_volume_threshold': 1.2,
                'pennant_volume_decrease': 0.8
            },
            'scoring': {
                'min_confidence_score': 0.6
            }
        }
    
    def detect_triangle_flags(self, df: pd.DataFrame) -> List[PatternRecord]:
        """
        检测三角旗形形态
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            检测到的三角旗形形态列表
        """
        if len(df) < 25:  # 至少需要25个数据点
            logger.warning("Insufficient data for triangle flag detection")
            return []
        
        patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        # 检测所有可能的旗杆
        flagpoles = self._detect_flagpoles(df)
        logger.info(f"Detected {len(flagpoles)} potential flagpoles for triangle flags")
        
        # 为每个旗杆寻找对应的三角旗面
        for flagpole in flagpoles:
            triangle_pattern = self._detect_triangle_after_pole(df, flagpole)
            if triangle_pattern:
                # 创建形态记录
                pattern_record = PatternRecord(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    pattern_type=PatternType.TRIANGLE_FLAG,
                    detection_date=datetime.now(),
                    flagpole=flagpole,
                    pattern_boundaries=triangle_pattern['boundaries'],
                    pattern_duration=triangle_pattern['duration'],
                    confidence_score=triangle_pattern['confidence_score'],
                    pattern_quality=self._classify_pattern_quality(triangle_pattern['confidence_score'])
                )
                patterns.append(pattern_record)
                logger.info(f"Detected triangle flag pattern: {pattern_record.id} with confidence {pattern_record.confidence_score:.2f}")
        
        logger.info(f"Total triangle flag patterns detected: {len(patterns)}")
        return patterns
    
    def _detect_flagpoles(self, df: pd.DataFrame) -> List[Flagpole]:
        """检测旗杆（与旗形检测器相同的逻辑）"""
        flagpoles = []
        
        # 计算价格变化和成交量
        df = df.copy()
        df['price_change'] = df['close'].pct_change()
        df['volume_sma'] = TechnicalIndicators.volume_sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma'].fillna(1)
        
        min_duration = self.config['flagpole']['min_duration_days']
        max_duration = self.config['flagpole']['max_duration_days']
        min_height = self.config['flagpole']['min_height_percent'] / 100
        max_height = self.config['flagpole']['max_height_percent'] / 100
        
        # 滑动窗口检测旗杆
        for i in range(len(df) - max_duration - 10):
            for duration in range(min_duration, min(max_duration + 1, len(df) - i - 10)):
                start_idx = i
                end_idx = i + duration
                
                if end_idx >= len(df):
                    continue
                
                # 计算价格变化幅度
                start_price = df.iloc[start_idx]['close']
                end_price = df.iloc[end_idx]['close']
                height_percent = abs(end_price - start_price) / start_price
                
                # 检查价格变化是否符合要求
                if min_height <= height_percent <= max_height:
                    # 检查成交量确认
                    period_volume_ratio = df.iloc[start_idx:end_idx+1]['volume_ratio'].mean()
                    
                    if period_volume_ratio >= self.config['volume']['flagpole_volume_threshold']:
                        # 检查价格运动的一致性
                        if self._is_consistent_price_movement(df.iloc[start_idx:end_idx+1]):
                            direction = 'up' if end_price > start_price else 'down'
                            
                            flagpole = Flagpole(
                                start_time=df.iloc[start_idx]['timestamp'],
                                end_time=df.iloc[end_idx]['timestamp'],
                                start_price=start_price,
                                end_price=end_price,
                                height_percent=height_percent * 100,
                                direction=direction,
                                volume_ratio=period_volume_ratio
                            )
                            flagpoles.append(flagpole)
        
        # 去除重叠的旗杆，保留最佳的
        return self._filter_overlapping_flagpoles(flagpoles)
    
    def _is_consistent_price_movement(self, df_segment: pd.DataFrame) -> bool:
        """检查价格运动的一致性"""
        prices = df_segment['close']
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        return abs(r_value) > 0.8
    
    def _filter_overlapping_flagpoles(self, flagpoles: List[Flagpole]) -> List[Flagpole]:
        """过滤重叠的旗杆，保留最佳的"""
        if not flagpoles:
            return []
        
        flagpoles.sort(key=lambda x: x.start_time)
        filtered = []
        
        for flagpole in flagpoles:
            overlaps = False
            for selected in filtered:
                if (flagpole.start_time <= selected.end_time and 
                    flagpole.end_time >= selected.start_time):
                    if flagpole.height_percent * flagpole.volume_ratio > \
                       selected.height_percent * selected.volume_ratio:
                        filtered.remove(selected)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                filtered.append(flagpole)
        
        return filtered
    
    def _detect_triangle_after_pole(self, df: pd.DataFrame, flagpole: Flagpole) -> Optional[dict]:
        """检测旗杆后的三角旗面形态"""
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        if flagpole_end_idx >= len(df) - 8:  # 没有足够数据
            return None
        
        min_triangle_duration = self.config['pennant']['min_duration_days']
        max_triangle_duration = min(
            self.config['pennant']['max_duration_days'],
            len(df) - flagpole_end_idx - 1
        )
        
        # 尝试不同的三角旗面长度
        best_triangle = None
        best_score = 0
        
        for triangle_duration in range(min_triangle_duration, max_triangle_duration + 1):
            triangle_start_idx = flagpole_end_idx + 1
            triangle_end_idx = triangle_start_idx + triangle_duration
            
            if triangle_end_idx >= len(df):
                continue
            
            triangle_data = df.iloc[triangle_start_idx:triangle_end_idx + 1]
            
            # 检测三角旗面形态
            triangle_result = self._analyze_triangle_pattern(triangle_data, flagpole)
            
            if triangle_result and triangle_result['confidence_score'] > best_score:
                best_score = triangle_result['confidence_score']
                best_triangle = triangle_result
                best_triangle['duration'] = triangle_duration
        
        # 检查最佳三角旗面是否满足最低要求
        if best_triangle and best_score >= self.config['scoring']['min_confidence_score']:
            return best_triangle
        
        return None
    
    def _analyze_triangle_pattern(self, triangle_data: pd.DataFrame, flagpole: Flagpole) -> Optional[dict]:
        """分析三角旗面形态"""
        if len(triangle_data) < 4:  # 至少需要4个数据点
            return None
        
        # 拟合上下边界线
        upper_line = self._fit_triangle_trend_line(triangle_data, 'high', 'descending')
        lower_line = self._fit_triangle_trend_line(triangle_data, 'low', 'ascending')
        
        if not upper_line or not lower_line:
            return None
        
        # 检查收敛性
        convergence_score = self._check_convergence(upper_line, lower_line, triangle_data)
        
        if convergence_score < 0.5:  # 收敛性不足
            return None
        
        # 计算置信度评分
        confidence_score = self._calculate_triangle_confidence(
            triangle_data, flagpole, upper_line, lower_line, convergence_score
        )
        
        if confidence_score < self.config['scoring']['min_confidence_score']:
            return None
        
        return {
            'boundaries': [upper_line, lower_line],
            'confidence_score': confidence_score,
            'convergence_score': convergence_score,
            'volume_pattern': self._analyze_volume_pattern(triangle_data['volume'].values)
        }
    
    def _fit_triangle_trend_line(self, triangle_data: pd.DataFrame, price_type: str, 
                                expected_direction: str) -> Optional[TrendLine]:
        """拟合三角形趋势线"""
        prices = triangle_data[price_type].values
        timestamps = triangle_data['timestamp'].values
        
        if len(prices) < 3:
            return None
        
        # 使用线性回归拟合趋势线
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        
        # 检查拟合度
        if abs(r_value) < 0.4:  # 相对宽松的拟合度要求
            return None
        
        # 检查趋势方向是否符合预期
        if expected_direction == 'descending' and slope >= 0:
            return None
        elif expected_direction == 'ascending' and slope <= 0:
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
    
    def _check_convergence(self, upper_line: TrendLine, lower_line: TrendLine, 
                          triangle_data: pd.DataFrame) -> float:
        """检查收敛性"""
        # 计算起始和结束时的价格区间
        start_range = upper_line.start_price - lower_line.start_price
        end_range = upper_line.end_price - lower_line.end_price
        
        if start_range <= 0 or end_range <= 0:
            return 0.0
        
        # 计算收敛比例
        convergence_ratio = 1 - (end_range / start_range)
        
        # 检查是否满足最小收敛比例要求
        min_convergence = self.config['pennant']['convergence_ratio']
        
        if convergence_ratio < min_convergence:
            return 0.0
        
        # 计算收敛角度
        triangle_length = len(triangle_data)
        avg_price = triangle_data['close'].mean()
        price_convergence = start_range - end_range
        
        # 避免除零错误
        if triangle_length <= 1:
            return 0.0
        
        convergence_angle = np.degrees(np.arctan(price_convergence / avg_price / triangle_length))
        
        min_angle = self.config['pennant']['min_convergence_angle']
        max_angle = self.config['pennant']['max_convergence_angle']
        
        if not (min_angle <= convergence_angle <= max_angle):
            return convergence_ratio * 0.5  # 角度不合适，降低评分
        
        return min(1.0, convergence_ratio)
    
    def _calculate_triangle_confidence(self, triangle_data: pd.DataFrame, flagpole: Flagpole,
                                     upper_line: TrendLine, lower_line: TrendLine,
                                     convergence_score: float) -> float:
        """计算三角旗形形态置信度"""
        scores = []
        
        # 1. 收敛性评分 (35%)
        scores.append(('convergence', convergence_score, 0.35))
        
        # 2. 边界线拟合度 (25%)
        fit_score = (upper_line.r_squared + lower_line.r_squared) / 2
        scores.append(('fit_quality', fit_score, 0.25))
        
        # 3. 价格在三角形内的比例 (20%)
        containment_score = self._calculate_triangle_containment(triangle_data, upper_line, lower_line)
        scores.append(('triangle_containment', containment_score, 0.20))
        
        # 4. 成交量模式 (10%)
        volume_score = self._score_volume_pattern(triangle_data['volume'])
        scores.append(('volume_pattern', volume_score, 0.10))
        
        # 5. 与旗杆的关系 (10%)
        relationship_score = self._score_pole_triangle_relationship(flagpole, triangle_data)
        scores.append(('pole_relationship', relationship_score, 0.10))
        
        # 计算加权总分
        total_score = sum(score * weight for _, score, weight in scores)
        
        logger.debug(f"Triangle flag confidence scores: {[(name, f'{score:.2f}') for name, score, _ in scores]}")
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_triangle_containment(self, triangle_data: pd.DataFrame, 
                                      upper_line: TrendLine, lower_line: TrendLine) -> float:
        """计算价格在三角形内的比例"""
        containment_count = 0
        total_count = len(triangle_data)
        
        for i, row in triangle_data.iterrows():
            # 计算该时点的理论上下边界
            progress = i / (len(triangle_data) - 1) if len(triangle_data) > 1 else 0
            theoretical_upper = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            theoretical_lower = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            # 检查高低价是否在边界内
            high_in_triangle = row['high'] <= theoretical_upper * 1.01  # 允许1%的突破
            low_in_triangle = row['low'] >= theoretical_lower * 0.99
            
            if high_in_triangle and low_in_triangle:
                containment_count += 1
        
        return containment_count / total_count if total_count > 0 else 0
    
    def _score_volume_pattern(self, volumes: pd.Series) -> float:
        """评分成交量模式（三角旗面期间成交量应该递减）"""
        if len(volumes) < 3:
            return 0.5
        
        # 理想的三角旗面成交量应该逐渐递减
        trend_direction = TrendAnalyzer.detect_trend_direction(volumes)
        
        if trend_direction == "downtrend":
            return 0.9
        elif trend_direction == "sideways":
            return 0.6
        else:  # uptrend
            return 0.3
    
    def _score_pole_triangle_relationship(self, flagpole: Flagpole, triangle_data: pd.DataFrame) -> float:
        """评分旗杆与三角旗面的关系"""
        triangle_price_range = triangle_data['high'].max() - triangle_data['low'].min()
        triangle_avg_price = triangle_data['close'].mean()
        
        # 三角旗面的价格波动应该相对旗杆较小，且随时间收敛
        triangle_volatility = triangle_price_range / triangle_avg_price * 100
        pole_height = flagpole.height_percent
        
        # 理想情况下，三角旗面初始波动应该是旗杆高度的15%-25%
        ideal_ratio = 0.2
        actual_ratio = triangle_volatility / pole_height if pole_height > 0 else 1
        
        # 计算偏离度
        deviation = abs(actual_ratio - ideal_ratio) / ideal_ratio
        score = max(0, 1 - deviation)
        
        return score
    
    def _analyze_volume_pattern(self, volumes: np.ndarray) -> dict:
        """分析成交量模式"""
        if len(volumes) < 3:
            return {'trend': 'unknown', 'consistency': 0}
        
        # 计算成交量趋势
        x = np.arange(len(volumes))
        slope, _, r_value, _, _ = stats.linregress(x, volumes)
        
        trend = 'decreasing' if slope < 0 else 'increasing' if slope > 0 else 'flat'
        consistency = abs(r_value)
        
        return {
            'trend': trend,
            'consistency': consistency,
            'slope': slope
        }
    
    def _classify_pattern_quality(self, confidence_score: float) -> str:
        """根据置信度分类形态质量"""
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.65:
            return 'medium'
        else:
            return 'low'