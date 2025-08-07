"""
稳健化统计工具
提供缩尾处理、MAD去极值等稳健化统计方法
支持动态基线算法的三层防护机制
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from loguru import logger


class RobustStatistics:
    """稳健化统计处理器"""
    
    def __init__(self, 
                 winsorize_limits: Tuple[float, float] = (0.02, 0.98), 
                 mad_threshold: float = 2.5):
        """
        初始化稳健化统计处理器
        
        Args:
            winsorize_limits: 缩尾处理的百分位数范围（更保守的默认值）
            mad_threshold: MAD异常值检测阈值（更严格的默认值）
        """
        self.winsorize_limits = winsorize_limits
        self.mad_threshold = mad_threshold
        
    def mad_filter(self, data: pd.Series, threshold: Optional[float] = None) -> pd.Series:
        """
        使用中位数绝对偏差(MAD)方法过滤异常值
        
        Args:
            data: 输入数据序列
            threshold: 自定义阈值，如果为None则使用实例默认值
            
        Returns:
            过滤异常值后的数据序列
        """
        if len(data) == 0:
            return data
            
        threshold = threshold or self.mad_threshold
        
        # 计算中位数和MAD
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            # 如果MAD为0，使用标准差方法
            std = data.std()
            if std == 0:
                # 如果标准差也为0，数据没有变化，直接返回
                return data
            lower_bound = median - threshold * std
            upper_bound = median + threshold * std
        else:
            # 标准MAD阈值（使用修正因子1.4826来与正态分布一致）
            mad_adjusted = mad * 1.4826
            lower_bound = median - threshold * mad_adjusted
            upper_bound = median + threshold * mad_adjusted
        
        # 标记异常值
        mask = (data >= lower_bound) & (data <= upper_bound)
        filtered_data = data[mask]
        
        removed_count = len(data) - len(filtered_data)
        if removed_count > 0:
            logger.debug(f"MAD filtered data: median={median:.4f}, mad={mad:.4f}, "
                        f"bounds=[{lower_bound:.4f}, {upper_bound:.4f}], removed {removed_count} outliers")
        
        return filtered_data
        
    def winsorize(self, data: pd.Series, limits: Optional[Tuple[float, float]] = None) -> pd.Series:
        """
        对数据进行缩尾处理
        
        Args:
            data: 输入数据序列
            limits: 自定义缩尾限制，如果为None则使用实例默认值
            
        Returns:
            经过缩尾处理的数据序列
        """
        if len(data) == 0:
            return data
            
        limits = limits or self.winsorize_limits
        lower_percentile = limits[0] * 100
        upper_percentile = limits[1] * 100
        
        # 处理非数值
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return data
        
        lower_bound = np.percentile(clean_data, lower_percentile)
        upper_bound = np.percentile(clean_data, upper_percentile)
        
        # 将极值替换为边界值
        result = data.copy()
        result = result.clip(lower=lower_bound, upper=upper_bound)
        
        clipped_count = ((data < lower_bound) | (data > upper_bound)).sum()
        if clipped_count > 0:
            logger.debug(f"Winsorized data: bounds=[{lower_bound:.4f}, {upper_bound:.4f}], "
                        f"clipped {clipped_count} values")
        
        return result

    def robust_percentiles(self, data: pd.Series, 
                          percentiles: List[int] = [75, 85, 90, 95],
                          use_three_layer_protection: bool = True) -> Dict[int, float]:
        """
        计算稳健化百分位数
        
        Args:
            data: 输入数据序列
            percentiles: 要计算的百分位数列表
            use_three_layer_protection: 是否使用三层防护机制
            
        Returns:
            百分位数字典
        """
        if len(data) == 0:
            return {p: np.nan for p in percentiles}
        
        processed_data = data.copy()
        
        if use_three_layer_protection:
            # 三层防护机制
            logger.debug(f"Applying three-layer protection to {len(processed_data)} data points")
            
            # 第一层：MAD过滤
            original_size = len(processed_data)
            processed_data = self.mad_filter(processed_data)
            mad_removed = original_size - len(processed_data)
            
            # 第二层：缩尾处理
            processed_data = self.winsorize(processed_data)
            
            # 第三层：动态阈值调整（如果数据量过少，放宽限制）
            if len(processed_data) < 50:
                # 数据量不足时使用更宽松的处理
                processed_data = self.winsorize(data, limits=(0.05, 0.95))
                logger.warning(f"Insufficient data after filtering ({len(processed_data)} points), "
                             f"using relaxed winsorization")
            
            logger.debug(f"Three-layer protection: removed {mad_removed} outliers, "
                        f"final sample size: {len(processed_data)}")
        
        # 计算百分位数
        result = {}
        if len(processed_data) > 0:
            for p in percentiles:
                result[p] = np.percentile(processed_data, p)
        else:
            # 如果处理后没有数据，使用原始数据的百分位数
            logger.warning("No data remaining after robust processing, using raw percentiles")
            for p in percentiles:
                result[p] = np.percentile(data.dropna(), p) if len(data.dropna()) > 0 else np.nan
        
        return result

    def detect_regime_change(self, data: pd.Series, window: int = 50) -> bool:
        """
        检测数据分布的制度变化
        
        Args:
            data: 输入数据序列
            window: 检测窗口大小
            
        Returns:
            是否检测到制度变化
        """
        if len(data) < window * 2:
            return False
        
        # 分别计算前后两个窗口的统计特征
        recent_data = data.iloc[-window:]
        earlier_data = data.iloc[-2*window:-window]
        
        # 计算两个窗口的稳健统计量
        recent_median = recent_data.median()
        earlier_median = earlier_data.median()
        
        recent_mad = np.median(np.abs(recent_data - recent_median))
        earlier_mad = np.median(np.abs(earlier_data - earlier_median))
        
        # 检测均值和方差的显著变化
        median_change_ratio = abs(recent_median - earlier_median) / (earlier_median + 1e-8)
        mad_change_ratio = abs(recent_mad - earlier_mad) / (earlier_mad + 1e-8)
        
        # 阈值判断
        significant_median_change = median_change_ratio > 0.2  # 20%变化
        significant_mad_change = mad_change_ratio > 0.5       # 50%变化
        
        regime_change_detected = significant_median_change or significant_mad_change
        
        if regime_change_detected:
            logger.info(f"Regime change detected: median_change={median_change_ratio:.3f}, "
                       f"mad_change={mad_change_ratio:.3f}")
        
        return regime_change_detected

    def adaptive_threshold_adjustment(self, 
                                   baseline_percentiles: Dict[int, float],
                                   current_data: pd.Series,
                                   adjustment_factor: float = 0.1) -> Dict[int, float]:
        """
        自适应阈值调整
        
        Args:
            baseline_percentiles: 当前基线百分位数
            current_data: 最新数据
            adjustment_factor: 调整因子
            
        Returns:
            调整后的百分位数
        """
        if len(current_data) < 10:
            return baseline_percentiles
        
        # 计算当前数据的百分位数
        current_percentiles = self.robust_percentiles(current_data, 
                                                    list(baseline_percentiles.keys()), 
                                                    use_three_layer_protection=False)
        
        # 渐进式调整
        adjusted_percentiles = {}
        for p in baseline_percentiles.keys():
            baseline_value = baseline_percentiles[p]
            current_value = current_percentiles.get(p, baseline_value)
            
            if not np.isnan(current_value) and not np.isnan(baseline_value):
                # 使用指数平滑进行调整
                adjusted_value = baseline_value * (1 - adjustment_factor) + current_value * adjustment_factor
                adjusted_percentiles[p] = adjusted_value
                
                logger.debug(f"P{p} adjusted: {baseline_value:.4f} -> {adjusted_value:.4f} "
                           f"(current: {current_value:.4f})")
            else:
                adjusted_percentiles[p] = baseline_value
        
        return adjusted_percentiles

    def calculate_stability_score(self, 
                                data: pd.Series, 
                                window: int = 100,
                                lookback_periods: int = 5) -> float:
        """
        计算数据稳定性得分
        
        Args:
            data: 输入数据序列
            window: 分析窗口大小
            lookback_periods: 回看周期数
            
        Returns:
            稳定性得分 (0-1，1表示最稳定)
        """
        if len(data) < window * lookback_periods:
            return 0.5  # 数据不足，返回中等稳定性
        
        # 计算不同时期的统计特征
        medians = []
        mads = []
        
        for i in range(lookback_periods):
            start_idx = len(data) - (i + 1) * window
            end_idx = len(data) - i * window if i > 0 else len(data)
            
            period_data = data.iloc[start_idx:end_idx]
            medians.append(period_data.median())
            mads.append(np.median(np.abs(period_data - period_data.median())))
        
        # 计算稳定性指标
        median_stability = 1.0 - (np.std(medians) / (np.mean(medians) + 1e-8))
        mad_stability = 1.0 - (np.std(mads) / (np.mean(mads) + 1e-8))
        
        # 综合稳定性得分
        stability_score = (median_stability + mad_stability) / 2
        stability_score = max(0.0, min(1.0, stability_score))
        
        logger.debug(f"Stability score: {stability_score:.3f} "
                    f"(median_stability: {median_stability:.3f}, mad_stability: {mad_stability:.3f})")
        
        return stability_score

    def get_robust_summary_stats(self, data: pd.Series) -> Dict[str, float]:
        """
        获取稳健的汇总统计信息
        
        Args:
            data: 输入数据序列
            
        Returns:
            统计信息字典
        """
        if len(data) == 0:
            return {
                'median': np.nan,
                'mad': np.nan,
                'iqr': np.nan,
                'robust_mean': np.nan,
                'robust_std': np.nan,
                'outlier_ratio': np.nan
            }
        
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return {
                'median': np.nan,
                'mad': np.nan,
                'iqr': np.nan,
                'robust_mean': np.nan,
                'robust_std': np.nan,
                'outlier_ratio': np.nan
            }
        
        median = clean_data.median()
        mad = np.median(np.abs(clean_data - median))
        
        q25 = np.percentile(clean_data, 25)
        q75 = np.percentile(clean_data, 75)
        iqr = q75 - q25
        
        # 稳健均值（去除极值后的均值）
        filtered_data = self.mad_filter(clean_data)
        robust_mean = filtered_data.mean() if len(filtered_data) > 0 else median
        robust_std = filtered_data.std() if len(filtered_data) > 0 else mad * 1.4826
        
        # 异常值比例
        outlier_ratio = (len(clean_data) - len(filtered_data)) / len(clean_data)
        
        return {
            'median': median,
            'mad': mad,
            'iqr': iqr,
            'robust_mean': robust_mean,
            'robust_std': robust_std,
            'outlier_ratio': outlier_ratio
        }