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
from src.patterns.indicators.technical_indicators import TechnicalIndicators
from src.patterns.base.ransac_trend_fitter import RANSACTrendLineFitter


class PatternComponents:
    """共享的形态组件"""
    
    def __init__(self):
        """初始化组件"""
        # 初始化RANSAC拟合器（使用更宽松的参数）
        self.ransac_fitter = RANSACTrendLineFitter(
            max_iterations=1000,
            min_inliers_ratio=0.4,  # 降低最小内点比例以适应小数据集
            confidence=0.99
        )
    
    @staticmethod
    def find_swing_points(df: pd.DataFrame, window: int, 
                         min_prominence_atr_multiple: float = None,
                         atr_period: int = 14) -> Tuple[List[int], List[int]]:
        """
        智能摆动点检测算法
        
        Args:
            df: OHLCV数据
            window: 回看窗口大小
            min_prominence_atr_multiple: 最小突出度（ATR的倍数），None则不过滤
            atr_period: ATR计算周期
            
        Returns:
            (swing_highs, swing_lows): 摆动高点和低点的索引列表
        """
        swing_highs = []
        swing_lows = []
        
        # 确保窗口大小合理
        window = max(1, min(window, len(df) // 4))
        
        # 步骤1: 基础摆动点识别
        for i in range(window, len(df) - window):
            # 获取窗口数据
            window_high = df['high'].iloc[i-window:i+window+1]
            window_low = df['low'].iloc[i-window:i+window+1]
            
            # 摆动高点：当前高点是窗口内最高点
            if df['high'].iloc[i] == window_high.max():
                # 确保是唯一最高点
                if list(window_high).count(df['high'].iloc[i]) == 1:
                    swing_highs.append(i)
            
            # 摆动低点：当前低点是窗口内最低点
            if df['low'].iloc[i] == window_low.min():
                # 确保是唯一最低点
                if list(window_low).count(df['low'].iloc[i]) == 1:
                    swing_lows.append(i)
        
        # 步骤2: 基于ATR的突出度过滤
        if min_prominence_atr_multiple is not None and min_prominence_atr_multiple > 0:
            # 计算ATR
            atr = TechnicalIndicators.calculate_atr(df, period=atr_period)
            if atr is not None and len(atr) > 0:
                # 过滤高点
                filtered_highs = []
                for idx in swing_highs:
                    if idx < len(atr) and not pd.isna(atr.iloc[idx]):
                        # 计算与邻近点的价格差异
                        prominence = PatternComponents._calculate_point_prominence(
                            df, idx, 'high', window
                        )
                        if prominence >= min_prominence_atr_multiple * atr.iloc[idx]:
                            filtered_highs.append(idx)
                swing_highs = filtered_highs
                
                # 过滤低点
                filtered_lows = []
                for idx in swing_lows:
                    if idx < len(atr) and not pd.isna(atr.iloc[idx]):
                        prominence = PatternComponents._calculate_point_prominence(
                            df, idx, 'low', window
                        )
                        if prominence >= min_prominence_atr_multiple * atr.iloc[idx]:
                            filtered_lows.append(idx)
                swing_lows = filtered_lows
        
        # 步骤3: 时间间隔过滤（避免过于密集的摆动点）
        min_distance = max(1, window // 2)
        swing_highs = PatternComponents._filter_by_time_distance(
            swing_highs, min_distance, df, 'high'
        )
        swing_lows = PatternComponents._filter_by_time_distance(
            swing_lows, min_distance, df, 'low'
        )
        
        return swing_highs, swing_lows
    
    @staticmethod
    def _calculate_point_prominence(df: pd.DataFrame, idx: int, 
                                  price_type: str, window: int) -> float:
        """计算点的突出度"""
        if price_type == 'high':
            # 对于高点，计算与左右低点的差异
            left_low = df['low'].iloc[max(0, idx-window):idx].min() if idx > 0 else df['low'].iloc[idx]
            right_low = df['low'].iloc[idx+1:min(len(df), idx+window+1)].min() if idx < len(df)-1 else df['low'].iloc[idx]
            prominence = df['high'].iloc[idx] - max(left_low, right_low)
        else:
            # 对于低点，计算与左右高点的差异
            left_high = df['high'].iloc[max(0, idx-window):idx].max() if idx > 0 else df['high'].iloc[idx]
            right_high = df['high'].iloc[idx+1:min(len(df), idx+window+1)].max() if idx < len(df)-1 else df['high'].iloc[idx]
            prominence = min(left_high, right_high) - df['low'].iloc[idx]
        
        return prominence
    
    @staticmethod
    def _filter_by_time_distance(indices: List[int], min_distance: int,
                               df: pd.DataFrame = None, price_type: str = None) -> List[int]:
        """按时间间隔过滤，保留最显著的点"""
        if not indices or min_distance <= 0:
            return indices
        
        if df is None or price_type is None:
            # 简单过滤
            filtered = [indices[0]]
            for idx in indices[1:]:
                if idx - filtered[-1] >= min_distance:
                    filtered.append(idx)
            return filtered
        
        # 基于价格显著性的过滤
        filtered = []
        i = 0
        while i < len(indices):
            # 查找当前点附近的所有点
            group = [indices[i]]
            j = i + 1
            while j < len(indices) and indices[j] - indices[i] < min_distance:
                group.append(indices[j])
                j += 1
            
            # 从组中选择最显著的点
            if len(group) == 1:
                filtered.append(group[0])
            else:
                # 选择价格最极端的点
                if price_type == 'high':
                    best_idx = max(group, key=lambda x: df['high'].iloc[x])
                else:
                    best_idx = min(group, key=lambda x: df['low'].iloc[x])
                filtered.append(best_idx)
            
            i = j
        
        return filtered
    
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
    
    def fit_trend_line_ransac(self, df: pd.DataFrame, point_indices: List[int], 
                             price_type: str = 'close',
                             use_ransac: bool = True) -> Optional[TrendLine]:
        """
        使用RANSAC算法拟合鲁棒趋势线
        
        Args:
            df: OHLCV数据
            point_indices: 要拟合的点的索引
            price_type: 价格类型（'high', 'low', 'close'）
            use_ransac: 是否使用RANSAC，False时使用传统OLS
            
        Returns:
            趋势线对象，如果拟合失败返回None
        """
        if not use_ransac:
            # 回退到传统方法
            return self.fit_trend_line(df, point_indices, price_type)
        
        return self.ransac_fitter.fit_trend_line(df, point_indices, price_type)
    
    def get_ransac_statistics(self) -> dict:
        """
        获取最后一次RANSAC拟合的统计信息
        
        Returns:
            统计信息字典
        """
        return self.ransac_fitter.get_fit_statistics()
    
    def compare_fitting_methods(self, df: pd.DataFrame, point_indices: List[int], 
                               price_type: str = 'close') -> dict:
        """
        比较传统OLS与RANSAC拟合方法的效果
        
        Args:
            df: OHLCV数据
            point_indices: 点索引
            price_type: 价格类型
            
        Returns:
            比较结果字典
        """
        return self.ransac_fitter.compare_with_ols(df, point_indices, price_type)
    
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
    
    @staticmethod
    def analyze_volume_pattern_enhanced(df: pd.DataFrame, 
                                      pattern_start: int, 
                                      pattern_end: int,
                                      flagpole_start: int = None,
                                      flagpole_end: int = None) -> dict:
        """
        增强的成交量模式分析
        
        Args:
            df: OHLCV数据
            pattern_start: 形态开始索引
            pattern_end: 形态结束索引
            flagpole_start: 旗杆开始索引（可选）
            flagpole_end: 旗杆结束索引（可选）
            
        Returns:
            详细的成交量分析结果
        """
        pattern_df = df.iloc[pattern_start:pattern_end + 1]
        pattern_volumes = pattern_df['volume']
        
        # 1. 基础趋势分析
        basic_analysis = PatternComponents.analyze_volume_pattern(pattern_volumes, 'decreasing')
        
        # 2. 相对强度分析
        if flagpole_start is not None and flagpole_end is not None:
            flagpole_avg_volume = df.iloc[flagpole_start:flagpole_end + 1]['volume'].mean()
            pattern_avg_volume = pattern_volumes.mean()
            volume_contraction_ratio = pattern_avg_volume / flagpole_avg_volume if flagpole_avg_volume > 0 else 1
        else:
            # 使用形态前的历史数据
            lookback = min(20, pattern_start)
            if lookback > 0:
                pre_pattern_avg = df.iloc[max(0, pattern_start - lookback):pattern_start]['volume'].mean()
                pattern_avg_volume = pattern_volumes.mean()
                volume_contraction_ratio = pattern_avg_volume / pre_pattern_avg if pre_pattern_avg > 0 else 1
            else:
                volume_contraction_ratio = 1.0
        
        # 3. 波动性分析
        volume_volatility = pattern_volumes.std() / pattern_volumes.mean() if pattern_volumes.mean() > 0 else 0
        
        # 4. 极值分析（检测异常成交量）
        volume_spikes = PatternComponents._detect_volume_spikes(pattern_df)
        
        # 5. 流动性检查
        min_liquidity_threshold = 0.3  # 最低成交量不应低于平均值的30%
        liquidity_healthy = pattern_volumes.min() >= pattern_volumes.mean() * min_liquidity_threshold
        
        # 6. 综合健康度评分
        health_score = PatternComponents._calculate_volume_health_score(
            basic_analysis['slope'],
            basic_analysis['consistency'],
            volume_contraction_ratio,
            volume_volatility,
            len(volume_spikes),
            liquidity_healthy
        )
        
        return {
            'trend': basic_analysis['trend'],
            'trend_slope': basic_analysis['slope'],
            'trend_r2': basic_analysis['consistency'] ** 2,
            'contraction_ratio': volume_contraction_ratio,
            'volatility': volume_volatility,
            'spike_count': len(volume_spikes),
            'spike_indices': volume_spikes,
            'liquidity_healthy': liquidity_healthy,
            'health_score': health_score,
            'basic_score': basic_analysis['score']
        }
    
    @staticmethod
    def _detect_volume_spikes(df: pd.DataFrame, threshold: float = 2.0) -> List[int]:
        """
        检测成交量异常峰值
        
        Args:
            df: 包含volume的DataFrame
            threshold: 异常阈值（标准差的倍数）
            
        Returns:
            异常索引列表
        """
        volumes = df['volume']
        mean_volume = volumes.mean()
        std_volume = volumes.std()
        
        spike_indices = []
        for i in range(len(volumes)):
            if volumes.iloc[i] > mean_volume + threshold * std_volume:
                spike_indices.append(i)
        
        return spike_indices
    
    @staticmethod
    def _calculate_volume_health_score(slope: float, consistency: float,
                                     contraction_ratio: float, volatility: float,
                                     spike_count: int, liquidity_healthy: bool) -> float:
        """
        计算成交量健康度综合评分
        
        Returns:
            0-1之间的健康度评分
        """
        scores = []
        weights = []
        
        # 1. 趋势得分（期望递减）
        if slope < 0:
            trend_score = min(1.0, abs(slope) * 100)  # 斜率越负越好
        else:
            trend_score = 0.3
        scores.append(trend_score)
        weights.append(0.3)
        
        # 2. 一致性得分
        scores.append(consistency)
        weights.append(0.2)
        
        # 3. 收缩比例得分（期望 0.3-0.7）
        if 0.3 <= contraction_ratio <= 0.7:
            contraction_score = 1.0
        elif contraction_ratio < 0.3:
            contraction_score = contraction_ratio / 0.3
        else:
            contraction_score = max(0, 1 - (contraction_ratio - 0.7) / 0.3)
        scores.append(contraction_score)
        weights.append(0.25)
        
        # 4. 波动性得分（越低越好）
        volatility_score = max(0, 1 - volatility)
        scores.append(volatility_score)
        weights.append(0.15)
        
        # 5. 流动性得分
        liquidity_score = 1.0 if liquidity_healthy else 0.5
        scores.append(liquidity_score)
        weights.append(0.1)
        
        # 计算加权平均
        total_score = sum(s * w for s, w in zip(scores, weights))
        
        # 根据异常峰值数量进行惩罚
        if spike_count > 2:
            total_score *= 0.8
        elif spike_count > 0:
            total_score *= 0.9
        
        return max(0, min(1, total_score))