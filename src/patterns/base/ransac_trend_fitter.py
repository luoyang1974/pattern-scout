"""
RANSAC趋势线拟合器
提供鲁棒的趋势线拟合功能，能够处理噪音和异常值
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from scipy import stats
from loguru import logger

from src.data.models.base_models import TrendLine


class RANSACTrendLineFitter:
    """
    RANSAC趋势线拟合器
    使用随机抽样一致性算法进行鲁棒的趋势线拟合
    """
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 distance_threshold: float = None,
                 min_samples: int = 2,
                 min_inliers_ratio: float = 0.6,
                 confidence: float = 0.99):
        """
        初始化RANSAC拟合器
        
        Args:
            max_iterations: 最大迭代次数
            distance_threshold: 距离阈值（自动计算时为None）
            min_samples: 最小样本数（线性拟合需要2个点）
            min_inliers_ratio: 最小内点比例
            confidence: 置信度
        """
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.min_samples = min_samples
        self.min_inliers_ratio = min_inliers_ratio
        self.confidence = confidence
        
        # RANSAC统计信息
        self.last_fit_stats = {}
    
    def fit_trend_line(self, df: pd.DataFrame, point_indices: List[int], 
                      price_type: str = 'close',
                      adaptive_threshold: bool = True) -> Optional[TrendLine]:
        """
        使用RANSAC算法拟合趋势线
        
        Args:
            df: OHLCV数据
            point_indices: 要拟合的点的索引
            price_type: 价格类型
            adaptive_threshold: 是否使用自适应阈值
            
        Returns:
            趋势线对象，如果拟合失败返回None
        """
        if len(point_indices) < self.min_samples:
            logger.debug(f"Insufficient points for RANSAC fitting: {len(point_indices)}")
            return None
        
        try:
            # 提取数据
            prices = df.iloc[point_indices][price_type].values
            timestamps = df.iloc[point_indices]['timestamp'].values
            x_coords = np.arange(len(prices))
            
            # 数据预处理：标准化
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            if price_std == 0:
                price_std = 1.0
            prices_normalized = (prices - price_mean) / price_std
            
            # 计算自适应阈值
            if adaptive_threshold or self.distance_threshold is None:
                threshold = self._calculate_adaptive_threshold(prices_normalized)
            else:
                threshold = self.distance_threshold / price_std
            
            # 执行RANSAC
            best_model, inliers = self._ransac_fit(x_coords, prices_normalized, threshold)
            
            if best_model is None:
                logger.debug("RANSAC failed to find a valid model")
                return None
            
            # 反标准化模型参数
            slope_normalized, intercept_normalized = best_model
            slope = slope_normalized * price_std
            intercept = intercept_normalized * price_std + price_mean
            
            # 使用内点重新拟合以获得更准确的参数和R²
            inlier_x = x_coords[inliers]
            inlier_prices = prices[inliers]
            
            if len(inlier_x) >= 2:
                slope_refined, intercept_refined, r_value, _, _ = stats.linregress(inlier_x, inlier_prices)
                r_squared = r_value ** 2
            else:
                slope_refined, intercept_refined = slope, intercept
                r_squared = 0.0
            
            # 计算起止价格
            start_price = intercept_refined
            end_price = slope_refined * (len(prices) - 1) + intercept_refined
            
            # 记录统计信息
            self.last_fit_stats = {
                'total_points': len(point_indices),
                'inliers_count': np.sum(inliers),
                'inliers_ratio': np.sum(inliers) / len(point_indices),
                'iterations_used': getattr(self, '_last_iterations', 0),
                'threshold_used': threshold * price_std,
                'r_squared': r_squared,
                'outliers_indices': [point_indices[i] for i in range(len(inliers)) if not inliers[i]]
            }
            
            logger.debug(f"RANSAC fit completed: {self.last_fit_stats['inliers_count']}/{self.last_fit_stats['total_points']} inliers, "
                        f"R²={r_squared:.3f}")
            
            return TrendLine(
                start_time=timestamps[0],
                end_time=timestamps[-1],
                start_price=start_price,
                end_price=end_price,
                slope=slope_refined,
                r_squared=r_squared
            )
            
        except Exception as e:
            logger.error(f"Error in RANSAC trend line fitting: {e}")
            return None
    
    def _calculate_adaptive_threshold(self, prices: np.ndarray) -> float:
        """
        计算自适应距离阈值
        
        Args:
            prices: 标准化后的价格数据
            
        Returns:
            适应性阈值
        """
        # 方法1: 基于数据的标准差
        std_threshold = np.std(prices) * 0.5
        
        # 方法2: 基于中位数绝对偏差 (MAD)
        median = np.median(prices)
        mad = np.median(np.abs(prices - median))
        mad_threshold = 1.4826 * mad * 0.8  # 1.4826是正态分布的MAD到标准差的转换因子
        
        # 方法3: 基于价格范围
        price_range = np.max(prices) - np.min(prices)
        range_threshold = price_range * 0.05  # 5%的价格范围
        
        # 选择中等的阈值
        thresholds = [std_threshold, mad_threshold, range_threshold]
        thresholds = [t for t in thresholds if t > 0]
        
        if not thresholds:
            return 0.1  # 默认阈值
        
        # 使用中位数作为最终阈值
        adaptive_threshold = np.median(thresholds)
        
        # 确保阈值在合理范围内
        min_threshold = 0.01
        max_threshold = 2.0
        adaptive_threshold = max(min_threshold, min(max_threshold, adaptive_threshold))
        
        return adaptive_threshold
    
    def _ransac_fit(self, x: np.ndarray, y: np.ndarray, 
                   threshold: float) -> Tuple[Optional[Tuple[float, float]], Optional[np.ndarray]]:
        """
        执行RANSAC算法
        
        Args:
            x: x坐标
            y: y坐标（标准化后）
            threshold: 距离阈值
            
        Returns:
            (最佳模型参数, 内点掩码)
        """
        n_points = len(x)
        min_inliers = max(2, int(n_points * self.min_inliers_ratio))
        
        best_model = None
        best_inliers = None
        best_score = 0
        
        # 计算理论上需要的迭代次数
        theoretical_iterations = self._calculate_iterations_needed(n_points, min_inliers)
        actual_iterations = min(self.max_iterations, theoretical_iterations)
        
        self._last_iterations = 0
        
        for iteration in range(actual_iterations):
            self._last_iterations = iteration + 1
            
            # 1. 随机选择最小样本集
            sample_indices = np.random.choice(n_points, self.min_samples, replace=False)
            sample_x = x[sample_indices]
            sample_y = y[sample_indices]
            
            # 2. 拟合模型
            try:
                if len(np.unique(sample_x)) < 2:  # 避免相同x坐标
                    continue
                    
                slope, intercept, _, _, _ = stats.linregress(sample_x, sample_y)
                
                # 检查模型有效性
                if np.isnan(slope) or np.isnan(intercept) or np.isinf(slope) or np.isinf(intercept):
                    continue
                
            except Exception:
                continue
            
            # 3. 计算所有点到直线的距离
            distances = self._point_to_line_distance(x, y, slope, intercept)
            
            # 4. 确定内点
            inliers = distances <= threshold
            n_inliers = np.sum(inliers)
            
            # 5. 评估模型
            if n_inliers >= min_inliers:
                # 计算内点的拟合质量
                if n_inliers >= 2:
                    inlier_x = x[inliers]
                    inlier_y = y[inliers]
                    _, _, r_value, _, _ = stats.linregress(inlier_x, inlier_y)
                    fit_quality = r_value ** 2 if not np.isnan(r_value) else 0
                else:
                    fit_quality = 0
                
                # 综合评分：内点数量 + 拟合质量
                score = n_inliers + fit_quality * 10
                
                if score > best_score:
                    best_score = score
                    best_model = (slope, intercept)
                    best_inliers = inliers.copy()
                    
                    # 早期停止条件：如果找到了非常好的模型
                    if n_inliers / n_points > 0.9 and fit_quality > 0.95:
                        logger.debug(f"Early stopping at iteration {iteration + 1} with excellent fit")
                        break
        
        return best_model, best_inliers
    
    def _point_to_line_distance(self, x: np.ndarray, y: np.ndarray, 
                               slope: float, intercept: float) -> np.ndarray:
        """
        计算点到直线的距离
        
        Args:
            x, y: 点坐标
            slope, intercept: 直线参数
            
        Returns:
            距离数组
        """
        # 直线方程: y = slope * x + intercept
        # 转换为一般式: slope * x - y + intercept = 0
        # 点到直线距离公式: |ax + by + c| / sqrt(a² + b²)
        
        a = slope
        b = -1
        c = intercept
        
        distances = np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
        return distances
    
    def _calculate_iterations_needed(self, n_points: int, min_inliers: int) -> int:
        """
        根据置信度计算理论上需要的迭代次数
        
        Args:
            n_points: 总点数
            min_inliers: 最小内点数
            
        Returns:
            理论迭代次数
        """
        if min_inliers <= 0 or n_points <= 0:
            return self.max_iterations
        
        # 估计内点比例
        inlier_ratio = min_inliers / n_points
        
        # 避免数值问题
        inlier_ratio = max(0.01, min(0.99, inlier_ratio))
        
        # 计算选择到全部内点的概率
        prob_all_inliers = inlier_ratio ** self.min_samples
        
        if prob_all_inliers <= 0:
            return self.max_iterations
        
        # 计算需要的迭代次数
        # P(至少一次成功) = 1 - (1 - p)^n = confidence
        # n = log(1 - confidence) / log(1 - p)
        try:
            iterations = int(np.log(1 - self.confidence) / np.log(1 - prob_all_inliers))
            return max(10, min(self.max_iterations, iterations))
        except:
            return self.max_iterations
    
    def get_fit_statistics(self) -> Dict[str, Any]:
        """
        获取最后一次拟合的统计信息
        
        Returns:
            统计信息字典
        """
        return self.last_fit_stats.copy()
    
    def fit_multiple_trend_lines(self, df: pd.DataFrame, 
                                point_sets: List[List[int]], 
                                price_type: str = 'close') -> List[Optional[TrendLine]]:
        """
        批量拟合多条趋势线
        
        Args:
            df: OHLCV数据
            point_sets: 多组点索引列表
            price_type: 价格类型
            
        Returns:
            趋势线列表
        """
        results = []
        total_stats = {
            'successful_fits': 0,
            'failed_fits': 0,
            'avg_inlier_ratio': 0,
            'avg_r_squared': 0
        }
        
        inlier_ratios = []
        r_squared_values = []
        
        for i, point_indices in enumerate(point_sets):
            trend_line = self.fit_trend_line(df, point_indices, price_type)
            results.append(trend_line)
            
            if trend_line is not None:
                total_stats['successful_fits'] += 1
                stats = self.get_fit_statistics()
                inlier_ratios.append(stats.get('inliers_ratio', 0))
                r_squared_values.append(stats.get('r_squared', 0))
            else:
                total_stats['failed_fits'] += 1
        
        # 计算平均统计
        if inlier_ratios:
            total_stats['avg_inlier_ratio'] = np.mean(inlier_ratios)
        if r_squared_values:
            total_stats['avg_r_squared'] = np.mean(r_squared_values)
        
        logger.info(f"Batch RANSAC fitting completed: {total_stats['successful_fits']}/{len(point_sets)} successful, "
                   f"avg inlier ratio: {total_stats['avg_inlier_ratio']:.2f}, "
                   f"avg R²: {total_stats['avg_r_squared']:.3f}")
        
        return results
    
    def compare_with_ols(self, df: pd.DataFrame, point_indices: List[int], 
                        price_type: str = 'close') -> Dict[str, Any]:
        """
        比较RANSAC与普通最小二乘法(OLS)的拟合效果
        
        Args:
            df: OHLCV数据
            point_indices: 点索引
            price_type: 价格类型
            
        Returns:
            比较结果
        """
        if len(point_indices) < 2:
            return {'error': 'Insufficient points for comparison'}
        
        # RANSAC拟合
        ransac_line = self.fit_trend_line(df, point_indices, price_type)
        ransac_stats = self.get_fit_statistics()
        
        # OLS拟合
        try:
            prices = df.iloc[point_indices][price_type].values
            timestamps = df.iloc[point_indices]['timestamp'].values
            x = np.arange(len(prices))
            
            slope_ols, intercept_ols, r_value_ols, _, _ = stats.linregress(x, prices)
            
            ols_line = TrendLine(
                start_time=timestamps[0],
                end_time=timestamps[-1],
                start_price=intercept_ols,
                end_price=slope_ols * (len(prices) - 1) + intercept_ols,
                slope=slope_ols,
                r_squared=r_value_ols ** 2
            )
            
        except Exception as e:
            return {'error': f'OLS fitting failed: {e}'}
        
        # 比较结果
        comparison = {
            'ransac': {
                'r_squared': ransac_line.r_squared if ransac_line else 0,
                'slope': ransac_line.slope if ransac_line else 0,
                'inliers_ratio': ransac_stats.get('inliers_ratio', 0),
                'outliers_count': len(ransac_stats.get('outliers_indices', []))
            },
            'ols': {
                'r_squared': ols_line.r_squared,
                'slope': ols_line.slope,
                'inliers_ratio': 1.0,  # OLS使用所有点
                'outliers_count': 0
            },
            'improvement': {
                'r_squared_diff': (ransac_line.r_squared if ransac_line else 0) - ols_line.r_squared,
                'outliers_detected': len(ransac_stats.get('outliers_indices', [])),
                'robustness_gain': ransac_stats.get('inliers_ratio', 0)
            }
        }
        
        return comparison