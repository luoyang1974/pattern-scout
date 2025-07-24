import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.data.models.base_models import PatternRecord, BreakthroughResult, AnalysisResult
from src.patterns.indicators.technical_indicators import TechnicalIndicators, TrendAnalyzer
from loguru import logger


class BreakthroughAnalyzer:
    """突破分析引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化突破分析引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'breakthrough': {
                'min_breakthrough_percent': 0.5,    # 最小突破幅度
                'max_breakthrough_days': 10,        # 突破确认最大天数
                'volume_confirmation_ratio': 1.2,   # 成交量确认比例
                'false_breakthrough_threshold': 0.3  # 假突破阈值
            },
            'follow_through': {
                'tracking_days': 20,                # 后续跟踪天数
                'continuation_threshold': 2.0,      # 持续突破阈值(%)
                'reversal_threshold': -1.5,         # 反转阈值(%)
                'sideways_range': 1.0               # 横盘区间(%)
            },
            'technical_indicators': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'volume_sma_period': 20
            },
            'scoring_weights': {
                'breakthrough_strength': 0.25,      # 突破强度
                'volume_confirmation': 0.20,        # 成交量确认
                'technical_indicators': 0.20,       # 技术指标
                'pattern_quality': 0.15,            # 形态质量
                'market_context': 0.10,             # 市场环境
                'follow_through': 0.10              # 后续表现
            }
        }
    
    def analyze_breakthrough(self, pattern: PatternRecord, price_data_after: pd.DataFrame) -> AnalysisResult:
        """
        分析形态突破
        
        Args:
            pattern: 形态记录
            price_data_after: 形态完成后的价格数据
            
        Returns:
            分析结果
        """
        logger.info(f"Analyzing breakthrough for pattern {pattern.id}")
        
        try:
            # 1. 检测突破点
            breakthrough = self._detect_breakthrough(pattern, price_data_after)
            
            if not breakthrough:
                logger.warning(f"No breakthrough detected for pattern {pattern.id}")
                return AnalysisResult(
                    pattern=pattern,
                    breakthrough=None,
                    scores={'breakthrough_detected': False},
                    metadata={'analysis_date': datetime.now()}
                )
            
            # 2. 多维度分析评分
            scores = self._comprehensive_analysis(pattern, price_data_after, breakthrough)
            
            # 3. 综合评估
            final_score = self._calculate_weighted_score(scores)
            result_type = self._classify_result(final_score, scores)
            
            # 4. 创建突破结果
            breakthrough_result = BreakthroughResult(
                pattern_id=pattern.id,
                breakthrough_date=breakthrough['date'],
                breakthrough_price=breakthrough['price'],
                result_type=result_type,
                confidence_score=final_score,
                analysis_details=scores
            )
            
            # 5. 创建完整分析结果
            analysis_result = AnalysisResult(
                pattern=pattern,
                breakthrough=breakthrough_result,
                scores=scores,
                metadata={
                    'analysis_date': datetime.now(),
                    'breakthrough_strength': breakthrough['strength'],
                    'tracking_days': len(price_data_after)
                }
            )
            
            logger.info(f"Breakthrough analysis completed for pattern {pattern.id}: {result_type} (confidence: {final_score:.2f})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing breakthrough for pattern {pattern.id}: {e}")
            return AnalysisResult(
                pattern=pattern,
                breakthrough=None,
                scores={'error': str(e)},
                metadata={'analysis_date': datetime.now(), 'error': True}
            )
    
    def _detect_breakthrough(self, pattern: PatternRecord, price_data_after: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """检测突破点"""
        if price_data_after.empty:
            return None
        
        # 确定突破方向和关键价位
        breakthrough_direction = self._determine_breakthrough_direction(pattern)
        key_levels = self._get_key_levels(pattern)
        
        min_breakthrough = self.config['breakthrough']['min_breakthrough_percent'] / 100
        max_days = min(self.config['breakthrough']['max_breakthrough_days'], len(price_data_after))
        
        # 在前几天内寻找突破
        for i in range(max_days):
            if i >= len(price_data_after):
                break
                
            current_data = price_data_after.iloc[i]
            
            # 检查价格突破
            breakthrough_price = None
            breakthrough_strength = 0
            
            if breakthrough_direction == 'up':
                resistance_level = key_levels['resistance']
                if current_data['high'] > resistance_level * (1 + min_breakthrough):
                    breakthrough_price = current_data['high']
                    breakthrough_strength = (breakthrough_price - resistance_level) / resistance_level
            else:  # down
                support_level = key_levels['support']
                if current_data['low'] < support_level * (1 - min_breakthrough):
                    breakthrough_price = current_data['low']
                    breakthrough_strength = (support_level - breakthrough_price) / support_level
            
            if breakthrough_price:
                # 成交量确认
                volume_ratio = self._check_volume_confirmation(price_data_after, i)
                
                return {
                    'date': current_data['timestamp'],
                    'price': breakthrough_price,
                    'direction': breakthrough_direction,
                    'strength': breakthrough_strength,
                    'volume_ratio': volume_ratio,
                    'day_index': i
                }
        
        return None
    
    def _determine_breakthrough_direction(self, pattern: PatternRecord) -> str:
        """确定预期突破方向"""
        # 基于旗杆方向确定突破方向
        return pattern.flagpole.direction  # 'up' or 'down'
    
    def _get_key_levels(self, pattern: PatternRecord) -> Dict[str, float]:
        """获取关键价位"""
        if not pattern.pattern_boundaries:
            # 如果没有边界线，使用旗杆价位
            if pattern.flagpole.direction == 'up':
                return {
                    'resistance': pattern.flagpole.end_price,
                    'support': pattern.flagpole.start_price
                }
            else:
                return {
                    'resistance': pattern.flagpole.start_price,
                    'support': pattern.flagpole.end_price
                }
        
        # 使用边界线的价位
        boundary_prices = []
        for boundary in pattern.pattern_boundaries:
            boundary_prices.extend([boundary.start_price, boundary.end_price])
        
        return {
            'resistance': max(boundary_prices),
            'support': min(boundary_prices)
        }
    
    def _check_volume_confirmation(self, price_data_after: pd.DataFrame, breakthrough_index: int) -> float:
        """检查成交量确认"""
        if breakthrough_index >= len(price_data_after):
            return 1.0
        
        # 计算突破时的成交量与平均成交量的比例
        breakthrough_volume = price_data_after.iloc[breakthrough_index]['volume']
        
        # 取前面几天的平均成交量作为基准
        lookback_days = min(10, breakthrough_index + len(price_data_after))
        if lookback_days > 0:
            recent_data = price_data_after.iloc[:lookback_days]
            avg_volume = recent_data['volume'].mean()
            volume_ratio = breakthrough_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        return volume_ratio
    
    def _comprehensive_analysis(self, pattern: PatternRecord, price_data_after: pd.DataFrame, 
                               breakthrough: Dict[str, Any]) -> Dict[str, float]:
        """综合多维度分析"""
        scores = {}
        
        try:
            # 1. 突破强度分析
            scores['breakthrough_strength'] = min(1.0, breakthrough['strength'] * 10)  # 归一化到0-1
            
            # 2. 成交量确认分析
            volume_threshold = self.config['breakthrough']['volume_confirmation_ratio']
            volume_score = min(1.0, breakthrough['volume_ratio'] / volume_threshold)
            scores['volume_confirmation'] = volume_score
            
            # 3. 技术指标确认
            scores['technical_indicators'] = self._analyze_technical_indicators(price_data_after, breakthrough['day_index'])
            
            # 4. 形态质量评分
            scores['pattern_quality'] = pattern.confidence_score
            
            # 5. 市场环境分析
            scores['market_context'] = self._analyze_market_context(price_data_after)
            
            # 6. 后续跟踪分析
            scores['follow_through'] = self._analyze_follow_through(price_data_after, breakthrough)
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # 提供默认评分
            scores = {
                'breakthrough_strength': 0.5,
                'volume_confirmation': 0.5,
                'technical_indicators': 0.5,
                'pattern_quality': pattern.confidence_score,
                'market_context': 0.5,
                'follow_through': 0.5
            }
        
        return scores
    
    def _analyze_technical_indicators(self, price_data_after: pd.DataFrame, breakthrough_index: int) -> float:
        """分析技术指标确认"""
        if len(price_data_after) < 14:  # 数据不足
            return 0.5
        
        try:
            scores = []
            
            # RSI分析
            rsi = TechnicalIndicators.rsi(price_data_after['close'], self.config['technical_indicators']['rsi_period'])
            if breakthrough_index < len(rsi) and not pd.isna(rsi.iloc[breakthrough_index]):
                rsi_value = rsi.iloc[breakthrough_index]
                # RSI在30-70之间较好，过度超买或超卖降低评分
                if 30 <= rsi_value <= 70:
                    rsi_score = 0.8
                elif 20 <= rsi_value <= 80:
                    rsi_score = 0.6
                else:
                    rsi_score = 0.4
                scores.append(rsi_score)
            
            # MACD分析
            macd_line, macd_signal, _ = TechnicalIndicators.macd(
                price_data_after['close'], 
                self.config['technical_indicators']['macd_fast'],
                self.config['technical_indicators']['macd_slow'],
                self.config['technical_indicators']['macd_signal']
            )
            
            if breakthrough_index < len(macd_line) and not pd.isna(macd_line.iloc[breakthrough_index]):
                macd_value = macd_line.iloc[breakthrough_index]
                signal_value = macd_signal.iloc[breakthrough_index]
                # MACD金叉银叉判断
                if macd_value > signal_value:
                    macd_score = 0.7
                else:
                    macd_score = 0.3
                scores.append(macd_score)
            
            # 布林带分析
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                price_data_after['close'], 
                self.config['technical_indicators']['bollinger_period']
            )
            
            if breakthrough_index < len(bb_upper):
                close_price = price_data_after.iloc[breakthrough_index]['close']
                bb_upper_val = bb_upper.iloc[breakthrough_index]
                bb_lower_val = bb_lower.iloc[breakthrough_index]
                
                # 价格在布林带内较好
                if not pd.isna(bb_upper_val) and not pd.isna(bb_lower_val):
                    if bb_lower_val <= close_price <= bb_upper_val:
                        bb_score = 0.8
                    else:
                        bb_score = 0.5
                    scores.append(bb_score)
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return 0.5
    
    def _analyze_market_context(self, price_data_after: pd.DataFrame) -> float:
        """分析市场环境"""
        if len(price_data_after) < 5:
            return 0.5
        
        try:
            # 分析整体趋势
            recent_trend = TrendAnalyzer.detect_trend_direction(price_data_after['close'].tail(10))
            trend_strength = TrendAnalyzer.calculate_trend_strength(price_data_after['close'].tail(10))
            
            # 趋势明确且强度高，给更高评分
            if recent_trend in ['uptrend', 'downtrend'] and trend_strength > 0.6:
                return 0.8
            elif recent_trend == 'sideways':
                return 0.4
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return 0.5
    
    def _analyze_follow_through(self, price_data_after: pd.DataFrame, breakthrough: Dict[str, Any]) -> float:
        """分析后续跟踪表现"""
        breakthrough_index = breakthrough['day_index']
        tracking_days = self.config['follow_through']['tracking_days']
        
        # 可用的跟踪天数
        available_days = min(tracking_days, len(price_data_after) - breakthrough_index - 1)
        
        if available_days < 3:  # 数据不足
            return 0.5
        
        try:
            # 获取突破后的数据
            follow_data = price_data_after.iloc[breakthrough_index + 1:breakthrough_index + 1 + available_days]
            breakthrough_price = breakthrough['price']
            
            # 计算后续表现
            if breakthrough['direction'] == 'up':
                # 向上突破，检查是否持续上涨
                max_high = follow_data['high'].max()
                performance = (max_high - breakthrough_price) / breakthrough_price * 100
            else:
                # 向下突破，检查是否持续下跌
                min_low = follow_data['low'].min()
                performance = (breakthrough_price - min_low) / breakthrough_price * 100
            
            # 根据表现给分
            continuation_threshold = self.config['follow_through']['continuation_threshold']
            reversal_threshold = abs(self.config['follow_through']['reversal_threshold'])
            
            if performance >= continuation_threshold:
                return 0.9  # 强烈持续
            elif performance >= continuation_threshold / 2:
                return 0.7  # 中等持续
            elif performance >= -reversal_threshold / 2:
                return 0.5  # 横盘
            else:
                return 0.2  # 反转
                
        except Exception as e:
            logger.error(f"Error analyzing follow through: {e}")
            return 0.5
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """计算加权总分"""
        weights = self.config['scoring_weights']
        total_score = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in scores:
                total_score += scores[component] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _classify_result(self, final_score: float, scores: Dict[str, float]) -> str:
        """分类结果类型"""
        # 基于综合得分和关键指标分类
        follow_through_score = scores.get('follow_through', 0.5)
        
        if final_score >= 0.75 and follow_through_score >= 0.7:
            return 'continuation'  # 强烈持续
        elif final_score >= 0.6 and follow_through_score >= 0.5:
            return 'continuation'  # 一般持续
        elif final_score <= 0.4 or follow_through_score <= 0.3:
            return 'reversal'     # 反转
        else:
            return 'sideways'     # 横盘
    
    def batch_analyze(self, patterns_with_data: List[tuple]) -> List[AnalysisResult]:
        """批量分析多个形态的突破"""
        results = []
        
        logger.info(f"Starting batch analysis of {len(patterns_with_data)} patterns")
        
        for i, (pattern, price_data_after) in enumerate(patterns_with_data):
            logger.info(f"Analyzing pattern {i+1}/{len(patterns_with_data)}: {pattern.id}")
            
            try:
                result = self.analyze_breakthrough(pattern, price_data_after)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch analysis for pattern {pattern.id}: {e}")
                # 创建错误结果
                error_result = AnalysisResult(
                    pattern=pattern,
                    breakthrough=None,
                    scores={'error': str(e)},
                    metadata={'analysis_date': datetime.now(), 'batch_error': True}
                )
                results.append(error_result)
        
        logger.info(f"Batch analysis completed. {len(results)} results generated")
        return results
    
    def generate_summary_statistics(self, analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """生成汇总统计"""
        if not analysis_results:
            return {}
        
        # 过滤出有效结果
        valid_results = [r for r in analysis_results if r.breakthrough is not None]
        
        if not valid_results:
            return {'valid_results': 0, 'total_results': len(analysis_results)}
        
        # 统计结果类型分布
        result_types = [r.breakthrough.result_type for r in valid_results]
        type_counts = {t: result_types.count(t) for t in set(result_types)}
        
        # 统计置信度分布
        confidence_scores = [r.breakthrough.confidence_score for r in valid_results]
        
        # 统计形态质量分布
        pattern_qualities = [r.pattern.pattern_quality for r in valid_results]
        quality_counts = {q: pattern_qualities.count(q) for q in set(pattern_qualities)}
        
        return {
            'total_patterns': len(analysis_results),
            'valid_breakthroughs': len(valid_results),
            'breakthrough_rate': len(valid_results) / len(analysis_results),
            'result_type_distribution': type_counts,
            'average_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores),
            'pattern_quality_distribution': quality_counts,
            'high_confidence_patterns': len([s for s in confidence_scores if s >= 0.75])
        }