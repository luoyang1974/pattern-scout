"""
改进版旗形形态检测器
基于技术分析理论正确实现旗形检测算法
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
import uuid

from src.data.models.base_models import PatternRecord, Flagpole, TrendLine, PatternType
from loguru import logger


class FlagDetector:
    """改进版旗形形态检测器"""
    
    def __init__(self, config: dict = None):
        """
        初始化改进版旗形检测器
        
        Args:
            config: 配置参数字典
        """
        if config and 'pattern_detection' in config:
            # 使用传入的配置文件
            pattern_config = config['pattern_detection']
            self.config = {
                'flagpole': pattern_config.get('flagpole', {}),
                'flag': pattern_config.get('flag', {}),
                'scoring': pattern_config.get('scoring', {})
            }
        else:
            # 使用默认配置
            self.config = self._get_improved_config()
        
    def _get_improved_config(self) -> dict:
        """获取改进的配置参数（针对15分钟数据优化）"""
        return {
            'flagpole': {
                'min_bars': 4,              # 最少4个15分钟K线（1小时）
                'max_bars': 20,             # 最多20个15分钟K线（5小时）
                'min_height_percent': 1.5,  # 最小涨跌幅1.5%
                'max_height_percent': 10.0, # 最大涨跌幅10%
                'volume_surge_ratio': 2.0,  # 成交量激增倍数
                'max_retracement': 0.3,     # 最大回撤30%
                'min_trend_strength': 0.7   # 最小趋势强度（R²）
            },
            'flag': {
                'min_bars': 8,              # 最少8个15分钟K线（2小时）
                'max_bars': 48,             # 最多48个15分钟K线（12小时）
                'min_slope_angle': 0.5,     # 最小倾斜角度0.5度（适应15分钟数据）
                'max_slope_angle': 10,      # 最大倾斜角度10度（适应15分钟数据）
                'retracement_range': (0.2, 0.6),  # 回撤范围20%-60%
                'volume_decay_threshold': 0.7,     # 成交量衰减阈值
                'parallel_tolerance': 0.15,       # 平行度容忍度
                'min_touches': 2                  # 最少触及边界次数
            },
            'scoring': {
                'min_confidence_score': 0.5
            }
        }
    
    def detect_flags(self, df: pd.DataFrame) -> List[PatternRecord]:
        """
        检测旗形形态
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            检测到的旗形形态列表
        """
        if len(df) < 60:  # 至少需要60个数据点（15小时数据）
            logger.warning("Insufficient data for improved flag detection")
            return []
        
        patterns = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        # 预处理数据
        df_processed = self._preprocess_data(df)
        
        # 改进的旗杆检测
        flagpoles = self._detect_flagpoles_improved(df_processed)
        logger.info(f"Detected {len(flagpoles)} potential flagpoles with improved algorithm")
        
        # 为每个旗杆寻找对应的旗面
        for flagpole in flagpoles:
            flag_result = self._detect_flag_after_pole_improved(df_processed, flagpole)
            if flag_result:
                # 创建形态记录
                pattern_record = PatternRecord(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    pattern_type=PatternType.FLAG,
                    detection_date=datetime.now(),
                    flagpole=flagpole,
                    pattern_boundaries=flag_result['boundaries'],
                    pattern_duration=flag_result['duration'],
                    confidence_score=flag_result['confidence_score'],
                    pattern_quality=self._classify_pattern_quality(flag_result['confidence_score'])
                )
                patterns.append(pattern_record)
                logger.info(f"Detected improved flag pattern: {pattern_record.id} with confidence {pattern_record.confidence_score:.3f}")
        
        logger.info(f"Total improved flag patterns detected: {len(patterns)}")
        return patterns
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据，添加必要的技术指标"""
        df = df.copy()
        
        # 计算技术指标
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma20'].fillna(1)
        df['price_change'] = df['close'].pct_change()
        df['atr'] = self._calculate_atr(df, period=14)
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算平均真实范围（ATR）"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _detect_flagpoles_improved(self, df: pd.DataFrame) -> List[Flagpole]:
        """改进的旗杆检测算法"""
        flagpoles = []
        
        min_bars = self.config['flagpole']['min_bars']
        max_bars = self.config['flagpole']['max_bars']
        min_height = self.config['flagpole']['min_height_percent'] / 100
        max_height = self.config['flagpole']['max_height_percent'] / 100
        
        # 滑动窗口检测潜在旗杆
        for i in range(len(df) - max_bars - 20):  # 留足空间检测旗面
            for duration in range(min_bars, min(max_bars + 1, len(df) - i - 20)):
                start_idx = i
                end_idx = i + duration
                
                # 提取旗杆数据
                pole_data = df.iloc[start_idx:end_idx + 1].copy()
                
                if len(pole_data) < min_bars:
                    continue
                
                # 1. 基本价格变化检查
                start_price = pole_data.iloc[0]['close']
                end_price = pole_data.iloc[-1]['close']
                height_percent = abs(end_price - start_price) / start_price
                
                if not (min_height <= height_percent <= max_height):
                    continue
                
                # 2. 检查价格运动的急促性和一致性
                if not self._verify_rapid_consistent_movement(pole_data):
                    continue
                
                # 3. 检查成交量激增
                if not self._verify_volume_surge(pole_data):
                    continue
                
                # 4. 检查突破性（突破前期重要价位）
                if not self._verify_breakout_nature(df, start_idx, end_idx):
                    continue
                
                # 5. 验证单向性（限制回撤）
                if not self._verify_unidirectional_movement(pole_data):
                    continue
                
                # 创建旗杆对象
                direction = 'up' if end_price > start_price else 'down'
                volume_ratio = pole_data['volume_ratio'].mean()
                
                flagpole = Flagpole(
                    start_time=pole_data.iloc[0]['timestamp'],
                    end_time=pole_data.iloc[-1]['timestamp'],
                    start_price=start_price,
                    end_price=end_price,
                    height_percent=height_percent * 100,
                    direction=direction,
                    volume_ratio=volume_ratio
                )
                flagpoles.append(flagpole)
        
        # 过滤重叠的旗杆
        return self._filter_overlapping_flagpoles_improved(flagpoles)
    
    def _verify_rapid_consistent_movement(self, pole_data: pd.DataFrame) -> bool:
        """验证价格运动的急促性和一致性"""
        prices = pole_data['close']
        
        # 使用线性回归检查趋势一致性
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # 趋势强度必须足够（R²值）
        min_strength = self.config['flagpole']['min_trend_strength']
        if abs(r_value) < min_strength:
            return False
        
        # 检查价格运动的连续性（不能有大的反向跳空）
        price_changes = prices.pct_change().dropna()
        if len(price_changes) == 0:
            return False
            
        # 计算异常价格变化的数量
        std_change = price_changes.std()
        mean_change = price_changes.mean()
        
        # 异常变化定义：偏离均值超过2个标准差
        anomalies = abs(price_changes - mean_change) > 2 * std_change
        anomaly_ratio = anomalies.sum() / len(price_changes)
        
        # 异常变化不能超过20%
        return anomaly_ratio <= 0.2
    
    def _verify_volume_surge(self, pole_data: pd.DataFrame) -> bool:
        """验证成交量激增"""
        volume_ratio = pole_data['volume_ratio'].mean()
        surge_threshold = self.config['flagpole']['volume_surge_ratio']
        
        return volume_ratio >= surge_threshold
    
    def _verify_breakout_nature(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """验证突破性质"""
        # 获取旗杆前的数据（用于判断突破）
        lookback_period = 20
        pre_start = max(0, start_idx - lookback_period)
        
        if pre_start >= start_idx:
            return True  # 没有足够的历史数据，暂时通过
        
        pre_data = df.iloc[pre_start:start_idx]
        pole_data = df.iloc[start_idx:end_idx + 1]
        
        if len(pre_data) < 5 or len(pole_data) < 2:
            return True  # 数据不足，暂时通过
        
        pole_direction = 'up' if pole_data.iloc[-1]['close'] > pole_data.iloc[0]['close'] else 'down'
        
        # 检查是否突破了前期的重要价位
        if pole_direction == 'up':
            pre_high = pre_data['high'].max()
            pole_high = pole_data['high'].max()
            return pole_high > pre_high * 1.001  # 至少突破0.1%
        else:
            pre_low = pre_data['low'].min()
            pole_low = pole_data['low'].min()
            return pole_low < pre_low * 0.999  # 至少跌破0.1%
    
    def _verify_unidirectional_movement(self, pole_data: pd.DataFrame) -> bool:
        """验证单向性运动（限制最大回撤）"""
        prices = pole_data['close']
        direction = 'up' if prices.iloc[-1] > prices.iloc[0] else 'down'
        
        if direction == 'up':
            # 上升旗杆：计算最大回撤
            peak = prices.expanding().max()
            drawdown = (peak - prices) / peak
            max_drawdown = drawdown.max()
        else:
            # 下降旗杆：计算最大反弹
            trough = prices.expanding().min()
            bounce = (prices - trough) / trough
            max_drawdown = bounce.max()
        
        max_allowed = self.config['flagpole']['max_retracement']
        return max_drawdown <= max_allowed
    
    def _filter_overlapping_flagpoles_improved(self, flagpoles: List[Flagpole]) -> List[Flagpole]:
        """改进的重叠旗杆过滤"""
        if not flagpoles:
            return []
        
        # 按质量评分排序（高度×成交量×时间效率）
        flagpoles.sort(key=lambda x: x.height_percent * x.volume_ratio / 
                      ((x.end_time - x.start_time).total_seconds() / 3600), reverse=True)
        
        filtered = []
        for flagpole in flagpoles:
            # 检查时间重叠
            overlaps = any(
                (flagpole.start_time <= selected.end_time and 
                 flagpole.end_time >= selected.start_time)
                for selected in filtered
            )
            
            if not overlaps:
                filtered.append(flagpole)
        
        return filtered
    
    def _detect_flag_after_pole_improved(self, df: pd.DataFrame, flagpole: Flagpole) -> Optional[dict]:
        """改进的旗面检测"""
        # 找到旗杆结束位置
        flagpole_end_idx = df[df['timestamp'] <= flagpole.end_time].index[-1]
        
        if flagpole_end_idx >= len(df) - 20:  # 需要足够的数据
            return None
        
        min_flag_bars = self.config['flag']['min_bars']
        max_flag_bars = min(
            self.config['flag']['max_bars'],
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
            flag_result = self._analyze_flag_pattern_improved(flag_data, flagpole, df)
            
            if flag_result and flag_result['confidence_score'] > best_score:
                best_score = flag_result['confidence_score']
                best_flag = flag_result
                best_flag['duration'] = flag_duration
        
        if best_flag and best_score >= self.config['scoring']['min_confidence_score']:
            return best_flag
        
        return None
    
    def _analyze_flag_pattern_improved(self, flag_data: pd.DataFrame, flagpole: Flagpole, 
                                     full_df: pd.DataFrame) -> Optional[dict]:
        """改进的旗面形态分析"""
        if len(flag_data) < self.config['flag']['min_bars']:
            return None
        
        # 1. 识别关键的高低点（不是所有点）
        key_highs, key_lows = self._find_key_extremes(flag_data)
        
        if len(key_highs) < 2 or len(key_lows) < 2:
            return None
        
        # 2. 构建上下边界线
        upper_line = self._fit_boundary_line(flag_data, key_highs, 'resistance')
        lower_line = self._fit_boundary_line(flag_data, key_lows, 'support')
        
        if not upper_line or not lower_line:
            return None
        
        # 3. 核心验证：检查旗面是否向主趋势相反方向倾斜
        if not self._verify_flag_slope_direction(flagpole, upper_line, lower_line):
            return None
        
        # 4. 验证成交量模式（旗面应该缩量）
        if not self._verify_flag_volume_pattern(flag_data, flagpole, full_df):
            return None
        
        # 5. 计算改进的置信度
        confidence_score = self._calculate_improved_confidence(
            flag_data, flagpole, upper_line, lower_line, full_df
        )
        
        if confidence_score < self.config['scoring']['min_confidence_score']:
            return None
        
        return {
            'boundaries': [upper_line, lower_line],
            'confidence_score': confidence_score,
            'volume_pattern': self._analyze_volume_pattern_detailed(flag_data),
            'slope_direction_valid': True
        }
    
    def _find_key_extremes(self, flag_data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """寻找关键的高低点"""
        highs = flag_data['high'].values
        lows = flag_data['low'].values
        
        # 使用scipy.signal.find_peaks寻找峰值和谷值
        # 设置最小间距，避免噪音
        min_distance = max(2, len(flag_data) // 8)
        
        # 寻找高点
        high_peaks, _ = find_peaks(highs, distance=min_distance, prominence=np.std(highs) * 0.5)
        
        # 寻找低点（对负值寻找峰值）
        low_peaks, _ = find_peaks(-lows, distance=min_distance, prominence=np.std(lows) * 0.5)
        
        # 确保至少有首末点
        high_indices = sorted(list(set([0] + list(high_peaks) + [len(flag_data) - 1])))
        low_indices = sorted(list(set([0] + list(low_peaks) + [len(flag_data) - 1])))
        
        return high_indices, low_indices
    
    def _fit_boundary_line(self, flag_data: pd.DataFrame, key_points: List[int], 
                          line_type: str) -> Optional[TrendLine]:
        """拟合边界线（支撑或阻力）"""
        if len(key_points) < 2:
            return None
        
        # 提取关键点的价格和时间
        if line_type == 'resistance':
            prices = [flag_data.iloc[i]['high'] for i in key_points]
        else:
            prices = [flag_data.iloc[i]['low'] for i in key_points]
        
        timestamps = [flag_data.iloc[i]['timestamp'] for i in key_points]
        
        # 使用线性回归拟合
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        
        # 检查拟合质量
        if abs(r_value) < 0.4:  # 降低要求，因为旗面可能不是完美线性
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
    
    def _verify_flag_slope_direction(self, flagpole: Flagpole, upper_line: TrendLine, 
                                   lower_line: TrendLine) -> bool:
        """验证旗面倾斜方向（核心功能）"""
        # 计算旗面的平均斜率
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        
        # 计算斜率对应的角度
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price)) if avg_price > 0 else 0
        
        min_angle = self.config['flag']['min_slope_angle']
        max_angle = self.config['flag']['max_slope_angle']
        
        # 验证倾斜方向和角度
        if flagpole.direction == 'up':
            # 上升旗形：旗面必须向下倾斜
            if slope_angle >= -max_angle and slope_angle <= -min_angle:
                logger.debug(f"Valid upward flag: slope angle {slope_angle:.1f}°")
                return True
            else:
                logger.debug(f"Invalid upward flag: slope angle {slope_angle:.1f}° (should be -{max_angle}° to -{min_angle}°)")
                return False
        else:
            # 下降旗形：旗面必须向上倾斜
            if slope_angle >= min_angle and slope_angle <= max_angle:
                logger.debug(f"Valid downward flag: slope angle {slope_angle:.1f}°")
                return True
            else:
                logger.debug(f"Invalid downward flag: slope angle {slope_angle:.1f}° (should be {min_angle}° to {max_angle}°)")
                return False
    
    def _verify_flag_volume_pattern(self, flag_data: pd.DataFrame, flagpole: Flagpole, 
                                  full_df: pd.DataFrame) -> bool:
        """验证旗面成交量模式"""
        # 获取旗杆期间的平均成交量
        flagpole_start_idx = full_df[full_df['timestamp'] <= flagpole.start_time].index[-1]
        flagpole_end_idx = full_df[full_df['timestamp'] <= flagpole.end_time].index[-1]
        flagpole_volume = full_df.iloc[flagpole_start_idx:flagpole_end_idx + 1]['volume'].mean()
        
        # 旗面平均成交量
        flag_volume = flag_data['volume'].mean()
        
        # 旗面成交量应该显著小于旗杆成交量
        volume_decay_ratio = flag_volume / flagpole_volume if flagpole_volume > 0 else 1
        threshold = self.config['flag']['volume_decay_threshold']
        
        is_valid = volume_decay_ratio <= threshold
        logger.debug(f"Volume decay ratio: {volume_decay_ratio:.2f} (threshold: {threshold})")
        
        return is_valid
    
    def _calculate_improved_confidence(self, flag_data: pd.DataFrame, flagpole: Flagpole,
                                     upper_line: TrendLine, lower_line: TrendLine,
                                     full_df: pd.DataFrame) -> float:
        """计算改进的置信度评分"""
        scores = []
        
        # 1. 反向倾斜验证 (25%) - 最重要的特征
        slope_score = self._score_slope_direction(flagpole, upper_line, lower_line)
        scores.append(('slope_direction', slope_score, 0.25))
        
        # 2. 成交量对比 (20%) - 旗杆放量vs旗面缩量
        volume_score = self._score_volume_contrast(flag_data, flagpole, full_df)
        scores.append(('volume_contrast', volume_score, 0.20))
        
        # 3. 形态完整性 (20%) - 价格在通道内的表现
        channel_score = self._score_channel_integrity(flag_data, upper_line, lower_line)
        scores.append(('channel_integrity', channel_score, 0.20))
        
        # 4. 边界线质量 (15%) - 平行度和触及次数
        boundary_score = self._score_boundary_quality(upper_line, lower_line, flag_data)
        scores.append(('boundary_quality', boundary_score, 0.15))
        
        # 5. 时间比例 (10%) - 旗面与旗杆的时间关系
        time_ratio_score = self._score_time_relationship(flagpole, flag_data)
        scores.append(('time_ratio', time_ratio_score, 0.10))
        
        # 6. 价格回撤 (10%) - 旗面回撤幅度
        retracement_score = self._score_retracement(flagpole, flag_data)
        scores.append(('retracement', retracement_score, 0.10))
        
        # 计算加权总分
        total_score = sum(score * weight for _, score, weight in scores)
        
        logger.debug(f"Improved flag confidence scores: {[(name, f'{score:.3f}') for name, score, _ in scores]}")
        logger.debug(f"Total confidence: {total_score:.3f}")
        
        return min(1.0, max(0.0, total_score))
    
    def _score_slope_direction(self, flagpole: Flagpole, upper_line: TrendLine, lower_line: TrendLine) -> float:
        """评分斜率方向正确性"""
        avg_slope = (upper_line.slope + lower_line.slope) / 2
        avg_price = (upper_line.start_price + lower_line.start_price) / 2
        
        if avg_price <= 0:
            return 0.0
        
        slope_angle = np.degrees(np.arctan(avg_slope / avg_price))
        
        min_angle = self.config['flag']['min_slope_angle']
        max_angle = self.config['flag']['max_slope_angle']
        
        if flagpole.direction == 'up':
            # 上升旗形：期望负斜率
            if slope_angle <= -max_angle:
                return 0.5  # 太陡
            elif slope_angle >= 0:
                return 0.0  # 方向错误
            else:
                # 在合理范围内，角度越接近理想值得分越高
                ideal_angle = -(min_angle + max_angle) / 2
                deviation = abs(slope_angle - ideal_angle)
                max_deviation = max_angle - min_angle
                return max(0.0, 1.0 - deviation / max_deviation)
        else:
            # 下降旗形：期望正斜率
            if slope_angle >= max_angle:
                return 0.5  # 太陡
            elif slope_angle <= 0:
                return 0.0  # 方向错误
            else:
                # 在合理范围内
                ideal_angle = (min_angle + max_angle) / 2
                deviation = abs(slope_angle - ideal_angle)
                max_deviation = max_angle - min_angle
                return max(0.0, 1.0 - deviation / max_deviation)
    
    def _score_volume_contrast(self, flag_data: pd.DataFrame, flagpole: Flagpole, full_df: pd.DataFrame) -> float:
        """评分成交量对比"""
        try:
            # 获取旗杆成交量
            flagpole_start_idx = full_df[full_df['timestamp'] <= flagpole.start_time].index[-1]
            flagpole_end_idx = full_df[full_df['timestamp'] <= flagpole.end_time].index[-1]
            flagpole_volume = full_df.iloc[flagpole_start_idx:flagpole_end_idx + 1]['volume'].mean()
            
            # 旗面成交量
            flag_volume = flag_data['volume'].mean()
            
            if flagpole_volume <= 0:
                return 0.5
            
            volume_ratio = flag_volume / flagpole_volume
            ideal_ratio = 0.5  # 理想情况下旗面成交量是旗杆的50%
            
            # 计算得分
            if volume_ratio <= ideal_ratio:
                # 成交量递减，好现象
                return 1.0 - (ideal_ratio - volume_ratio) / ideal_ratio * 0.3
            else:
                # 成交量没有递减，扣分
                excess = volume_ratio - ideal_ratio
                penalty = min(excess / ideal_ratio, 1.0)
                return max(0.0, 1.0 - penalty)
        
        except Exception as e:
            logger.debug(f"Volume contrast scoring error: {e}")
            return 0.5
    
    def _score_channel_integrity(self, flag_data: pd.DataFrame, upper_line: TrendLine, lower_line: TrendLine) -> float:
        """评分通道完整性"""
        containment_count = 0
        total_count = len(flag_data)
        
        for i in range(len(flag_data)):
            progress = i / (len(flag_data) - 1) if len(flag_data) > 1 else 0
            
            # 计算理论边界位置
            theoretical_upper = upper_line.start_price + (upper_line.end_price - upper_line.start_price) * progress
            theoretical_lower = lower_line.start_price + (lower_line.end_price - lower_line.start_price) * progress
            
            row = flag_data.iloc[i]
            
            # 检查价格是否在通道内（允许小幅突破）
            tolerance = 0.02  # 2%突破容忍度
            high_ok = row['high'] <= theoretical_upper * (1 + tolerance)
            low_ok = row['low'] >= theoretical_lower * (1 - tolerance)
            
            if high_ok and low_ok:
                containment_count += 1
        
        return containment_count / total_count if total_count > 0 else 0
    
    def _score_boundary_quality(self, upper_line: TrendLine, lower_line: TrendLine, flag_data: pd.DataFrame) -> float:
        """评分边界线质量"""
        # 1. 平行度评分
        slope_diff = abs(upper_line.slope - lower_line.slope)
        avg_price = flag_data['close'].mean()
        normalized_slope_diff = slope_diff / avg_price * 100 if avg_price > 0 else float('inf')
        
        parallel_tolerance = self.config['flag']['parallel_tolerance']
        parallel_score = max(0, 1 - normalized_slope_diff / parallel_tolerance)
        
        # 2. 拟合质量评分
        fit_quality = (upper_line.r_squared + lower_line.r_squared) / 2
        
        # 综合评分
        return (parallel_score + fit_quality) / 2
    
    def _score_time_relationship(self, flagpole: Flagpole, flag_data: pd.DataFrame) -> float:
        """评分时间关系"""
        flagpole_duration = (flagpole.end_time - flagpole.start_time).total_seconds() / 3600  # 小时
        flag_duration = len(flag_data) * 0.25  # 15分钟数据转换为小时
        
        if flagpole_duration <= 0:
            return 0
        
        time_ratio = flag_duration / flagpole_duration
        
        # 理想的时间比例是1:1到3:1
        if 1.0 <= time_ratio <= 3.0:
            return 1.0
        elif 0.5 <= time_ratio < 1.0:
            return 0.8
        elif 3.0 < time_ratio <= 5.0:
            return 0.6
        else:
            return 0.3
    
    def _score_retracement(self, flagpole: Flagpole, flag_data: pd.DataFrame) -> float:
        """评分价格回撤"""
        flagpole_height = flagpole.height_percent / 100
        
        if flagpole.direction == 'up':
            # 上升旗形的回撤
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()
            retracement = (flag_high - flag_low) / flagpole.end_price
        else:
            # 下降旗形的反弹
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()
            retracement = (flag_high - flag_low) / flagpole.start_price
        
        min_ret, max_ret = self.config['flag']['retracement_range']
        
        if min_ret <= retracement <= max_ret:
            return 1.0
        elif retracement < min_ret:
            return 0.7  # 回撤太小
        else:
            return max(0.0, 1.0 - (retracement - max_ret) / max_ret)
    
    def _analyze_volume_pattern_detailed(self, flag_data: pd.DataFrame) -> dict:
        """详细分析成交量模式"""
        volumes = flag_data['volume']
        
        if len(volumes) < 3:
            return {'trend': 'unknown', 'consistency': 0, 'decay_rate': 0}
        
        # 计算成交量趋势
        x = np.arange(len(volumes))
        slope, _, r_value, _, _ = stats.linregress(x, volumes)
        
        # 计算衰减率
        first_half = volumes.iloc[:len(volumes)//2].mean()
        second_half = volumes.iloc[len(volumes)//2:].mean()
        decay_rate = (first_half - second_half) / first_half if first_half > 0 else 0
        
        return {
            'trend': 'decreasing' if slope < 0 else 'increasing' if slope > 0 else 'flat',
            'consistency': abs(r_value),
            'slope': slope,
            'decay_rate': decay_rate
        }
    
    def _classify_pattern_quality(self, confidence_score: float) -> str:
        """根据置信度分类形态质量"""
        if confidence_score >= 0.75:
            return 'high'
        elif confidence_score >= 0.60:
            return 'medium'
        else:
            return 'low'