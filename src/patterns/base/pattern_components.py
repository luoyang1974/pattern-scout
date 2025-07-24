"""
共享的形态检测组件
包含旗杆检测、支撑阻力识别等通用功能
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks

from src.data.models.base_models import Flagpole, TrendLine


class PatternComponents:
    """共享的形态组件"""
    
    @staticmethod
    def detect_flagpoles_adaptive(df: pd.DataFrame, params: dict, 
                                category: str) -> List[Flagpole]:
        """
        自适应的旗杆检测
        根据周期类别调整检测逻辑
        
        Args:
            df: 预处理后的OHLCV数据
            params: 旗杆参数
            category: 周期类别
            
        Returns:
            检测到的旗杆列表
        """
        flagpoles = []
        
        min_bars = params.get('min_bars', 4)
        max_bars = params.get('max_bars', 10)
        min_height = params.get('min_height_percent', 1.5) / 100
        max_height = params.get('max_height_percent', 10.0) / 100
        
        # 根据周期调整参数
        if category == 'ultra_short':
            # 超短周期：更宽松的要求
            volume_surge_ratio = params.get('volume_surge_ratio', 1.5)
            min_trend_strength = params.get('min_trend_strength', 0.6)
            max_retracement = params.get('max_retracement', 0.4)
        elif category == 'medium_long':
            # 中长周期：更严格的要求
            volume_surge_ratio = params.get('volume_surge_ratio', 2.5)
            min_trend_strength = params.get('min_trend_strength', 0.8)
            max_retracement = params.get('max_retracement', 0.2)
        else:
            # 短周期：标准要求
            volume_surge_ratio = params.get('volume_surge_ratio', 2.0)
            min_trend_strength = params.get('min_trend_strength', 0.7)
            max_retracement = params.get('max_retracement', 0.3)
        
        # 添加必要的技术指标
        if 'volume_ratio' not in df.columns:
            df['volume_sma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20'].fillna(1)
        
        # 滑动窗口检测
        for i in range(len(df) - max_bars - 10):  # 留空间检测形态
            for duration in range(min_bars, min(max_bars + 1, len(df) - i - 10)):
                start_idx = i
                end_idx = i + duration
                
                pole_data = df.iloc[start_idx:end_idx + 1].copy()
                
                if len(pole_data) < min_bars:
                    continue
                
                # 1. 基本价格变化检查
                start_price = pole_data.iloc[0]['close']
                end_price = pole_data.iloc[-1]['close']
                height_percent = abs(end_price - start_price) / start_price
                
                if not (min_height <= height_percent <= max_height):
                    continue
                
                # 2. 趋势强度检查
                if not PatternComponents._verify_trend_strength(
                    pole_data, min_trend_strength, category):
                    continue
                
                # 3. 成交量检查
                if not PatternComponents._verify_volume_surge(
                    pole_data, volume_surge_ratio, category):
                    continue
                
                # 4. 单向性检查
                if not PatternComponents._verify_unidirectional(
                    pole_data, max_retracement):
                    continue
                
                # 5. 突破性检查（可选）
                if category != 'ultra_short':  # 超短周期跳过突破检查
                    if not PatternComponents._verify_breakout(
                        df, start_idx, end_idx):
                        continue
                
                # 创建旗杆
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
        return PatternComponents._filter_overlapping_flagpoles(flagpoles)
    
    @staticmethod
    def _verify_trend_strength(pole_data: pd.DataFrame, 
                              min_strength: float, 
                              category: str) -> bool:
        """验证趋势强度"""
        prices = pole_data['close'].values
        
        if category == 'ultra_short':
            # 超短周期使用移动平均平滑
            if len(prices) >= 3:
                prices = pd.Series(prices).rolling(3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # 线性回归
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        return abs(r_value) >= min_strength
    
    @staticmethod
    def _verify_volume_surge(pole_data: pd.DataFrame, 
                           surge_ratio: float,
                           category: str) -> bool:
        """验证成交量激增"""
        volume_ratio = pole_data['volume_ratio'].mean()
        
        if category == 'ultra_short':
            # 超短周期可能有更多噪音，使用中位数
            volume_ratio = pole_data['volume_ratio'].median()
        
        return volume_ratio >= surge_ratio
    
    @staticmethod
    def _verify_unidirectional(pole_data: pd.DataFrame, 
                             max_retracement: float) -> bool:
        """验证单向性运动"""
        prices = pole_data['close'].values
        direction = 'up' if prices[-1] > prices[0] else 'down'
        
        if direction == 'up':
            # 计算最大回撤
            peak = pd.Series(prices).expanding().max()
            drawdown = (peak - prices) / peak
            max_dd = drawdown.max()
        else:
            # 计算最大反弹
            trough = pd.Series(prices).expanding().min()
            bounce = (prices - trough) / trough
            max_dd = bounce.max()
        
        return max_dd <= max_retracement
    
    @staticmethod
    def _verify_breakout(df: pd.DataFrame, start_idx: int, 
                        end_idx: int) -> bool:
        """验证突破性质"""
        lookback = 20
        pre_start = max(0, start_idx - lookback)
        
        if pre_start >= start_idx:
            return True
        
        pre_data = df.iloc[pre_start:start_idx]
        pole_data = df.iloc[start_idx:end_idx + 1]
        
        if len(pre_data) < 5:
            return True
        
        direction = 'up' if pole_data.iloc[-1]['close'] > pole_data.iloc[0]['close'] else 'down'
        
        if direction == 'up':
            pre_high = pre_data['high'].max()
            pole_high = pole_data['high'].max()
            return pole_high > pre_high * 1.001
        else:
            pre_low = pre_data['low'].min()
            pole_low = pole_data['low'].min()
            return pole_low < pre_low * 0.999
    
    @staticmethod
    def _filter_overlapping_flagpoles(flagpoles: List[Flagpole]) -> List[Flagpole]:
        """过滤重叠的旗杆"""
        if not flagpoles:
            return []
        
        # 按质量评分排序
        flagpoles.sort(
            key=lambda x: x.height_percent * x.volume_ratio,
            reverse=True
        )
        
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
    
    @staticmethod
    def find_support_resistance_points(df: pd.DataFrame, 
                                     window_size: int = None,
                                     min_prominence: float = None,
                                     category: str = 'short') -> Tuple[List[int], List[int]]:
        """
        寻找支撑和阻力点
        
        Args:
            df: OHLCV数据
            window_size: 窗口大小（None则自动计算）
            min_prominence: 最小突出度（None则自动计算）
            category: 周期类别
            
        Returns:
            (支撑点索引列表, 阻力点索引列表)
        """
        highs = df['high'].values
        lows = df['low'].values
        
        # 根据周期调整参数
        if window_size is None:
            if category == 'ultra_short':
                window_size = max(3, len(df) // 10)
            elif category == 'medium_long':
                window_size = max(2, len(df) // 15)
            else:
                window_size = max(2, len(df) // 12)
        
        if min_prominence is None:
            price_range = highs.max() - lows.min()
            if category == 'ultra_short':
                min_prominence = price_range * 0.001  # 0.1%
            elif category == 'medium_long':
                min_prominence = price_range * 0.003  # 0.3%
            else:
                min_prominence = price_range * 0.002  # 0.2%
        
        # 寻找阻力点（高点）
        resistance_idx, properties = find_peaks(
            highs,
            distance=window_size,
            prominence=min_prominence
        )
        
        # 寻找支撑点（低点）
        support_idx, properties = find_peaks(
            -lows,
            distance=window_size,
            prominence=min_prominence
        )
        
        # 确保包含首尾点
        resistance_idx = sorted(list(set([0] + list(resistance_idx) + [len(df) - 1])))
        support_idx = sorted(list(set([0] + list(support_idx) + [len(df) - 1])))
        
        return support_idx, resistance_idx
    
    @staticmethod
    def fit_trend_line(df: pd.DataFrame, point_indices: List[int], 
                      price_type: str = 'close') -> Optional[TrendLine]:
        """
        拟合趋势线
        
        Args:
            df: OHLCV数据
            point_indices: 要拟合的点的索引
            price_type: 价格类型（'high', 'low', 'close'）
            
        Returns:
            趋势线对象，如果拟合失败返回None
        """
        if len(point_indices) < 2:
            return None
        
        # 提取价格和时间
        prices = df.iloc[point_indices][price_type].values
        timestamps = df.iloc[point_indices]['timestamp'].values
        
        # 线性回归
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        
        # 计算起止价格
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
    
    @staticmethod
    def calculate_convergence_point(upper_line: TrendLine, 
                                  lower_line: TrendLine) -> Optional[Tuple[float, float]]:
        """
        计算两条趋势线的收敛点（交点）
        
        Args:
            upper_line: 上边界线
            lower_line: 下边界线
            
        Returns:
            (时间偏移量, 价格) 或 None（如果平行）
        """
        # 如果斜率相同（平行线），返回None
        if abs(upper_line.slope - lower_line.slope) < 1e-10:
            return None
        
        # 计算交点
        # y1 = slope1 * x + intercept1
        # y2 = slope2 * x + intercept2
        # 交点: slope1 * x + intercept1 = slope2 * x + intercept2
        
        # 使用线性方程求解
        # 需要先将时间转换为数值
        if hasattr((upper_line.end_time - upper_line.start_time), 'total_seconds'):
            time_diff = (upper_line.end_time - upper_line.start_time).total_seconds()
        else:
            # 处理 numpy.timedelta64 类型
            time_diff = pd.Timedelta(upper_line.end_time - upper_line.start_time).total_seconds()
        
        if time_diff <= 0:
            return None
        
        # 归一化时间
        upper_intercept = upper_line.start_price
        lower_intercept = lower_line.start_price
        
        # 求解交点的x坐标（时间）
        x_intersect = (lower_intercept - upper_intercept) / (upper_line.slope - lower_line.slope)
        
        # 计算交点的y坐标（价格）
        y_intersect = upper_line.slope * x_intersect + upper_intercept
        
        # 检查交点是否在合理范围内
        if x_intersect < 0:  # 交点在过去
            return None
        
        # 返回相对于结束时间的偏移量和价格
        return (x_intersect, y_intersect)
    
    @staticmethod
    def calculate_pattern_symmetry(upper_points: List[float], 
                                 lower_points: List[float]) -> float:
        """
        计算形态的对称性
        
        Args:
            upper_points: 上边界点
            lower_points: 下边界点
            
        Returns:
            对称性得分（0-1）
        """
        if len(upper_points) != len(lower_points) or len(upper_points) < 2:
            return 0.0
        
        # 计算中线
        midline = [(upper + lower) / 2 for upper, lower in zip(upper_points, lower_points)]
        
        # 计算上下边界到中线的距离
        upper_distances = [upper - mid for upper, mid in zip(upper_points, midline)]
        lower_distances = [mid - lower for mid, lower in zip(midline, lower_points)]
        
        # 计算对称性（距离的相似度）
        symmetry_scores = []
        for ud, ld in zip(upper_distances, lower_distances):
            if ud > 0 and ld > 0:
                ratio = min(ud, ld) / max(ud, ld)
                symmetry_scores.append(ratio)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    @staticmethod
    def analyze_volume_pattern(volumes: pd.Series, 
                             pattern_type: str = 'decreasing') -> dict:
        """
        分析成交量模式
        
        Args:
            volumes: 成交量序列
            pattern_type: 期望的模式类型
            
        Returns:
            分析结果字典
        """
        if len(volumes) < 3:
            return {'trend': 'unknown', 'consistency': 0, 'score': 0}
        
        # 线性回归分析趋势
        x = np.arange(len(volumes))
        slope, _, r_value, _, _ = stats.linregress(x, volumes)
        
        # 判断趋势
        if slope < -volumes.mean() * 0.01:
            trend = 'decreasing'
        elif slope > volumes.mean() * 0.01:
            trend = 'increasing'
        else:
            trend = 'flat'
        
        # 计算一致性
        consistency = abs(r_value)
        
        # 计算得分
        if pattern_type == 'decreasing' and trend == 'decreasing':
            score = consistency
        elif pattern_type == 'increasing' and trend == 'increasing':
            score = consistency
        elif pattern_type == 'flat' and trend == 'flat':
            score = 1 - consistency  # 平坦模式希望波动小
        else:
            score = 0.3  # 不匹配期望模式
        
        return {
            'trend': trend,
            'consistency': consistency,
            'slope': slope,
            'score': score
        }