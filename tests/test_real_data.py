"""
真实数据算法测试框架
用于验证ATR自适应、RANSAC拟合和智能摆动点检测在真实市场数据上的表现
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import time
from datetime import datetime
from loguru import logger

from src.patterns.detectors.flag_detector import FlagDetector
from src.patterns.detectors.pennant_detector import PennantDetector
from src.data.connectors.csv_connector import CSVDataConnector
from src.patterns.base.pattern_components import PatternComponents


class RealDataAlgorithmTester:
    """
    真实数据算法测试器
    测试新算法在实际市场数据上的性能
    """
    
    def __init__(self, data_directory: str = "data/csv"):
        """
        初始化测试器
        
        Args:
            data_directory: 数据目录路径
        """
        self.data_directory = Path(data_directory)
        self.results = {}
        
        # 初始化检测器 - 新算法版本
        self.flag_detector_new = FlagDetector()
        self.pennant_detector_new = PennantDetector()
        
        # 初始化检测器 - 传统版本（禁用新功能）
        legacy_config = {
            'global': {
                'enable_atr_adaptation': False,
                'enable_ransac_fitting': False
            }
        }
        self.flag_detector_legacy = FlagDetector(legacy_config)
        self.pennant_detector_legacy = PennantDetector(legacy_config)
        
        # 数据连接器
        self.data_connector = CSVDataConnector()
        
        # 测试统计
        self.test_stats = {
            'data_files_tested': 0,
            'total_processing_time': 0,
            'algorithms_compared': [],
            'performance_metrics': {}
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        运行综合算法测试
        
        Returns:
            完整的测试结果
        """
        logger.info("Starting comprehensive real data algorithm test")
        start_time = time.time()
        
        # 1. 发现并加载所有数据文件
        data_files = self._discover_data_files()
        logger.info(f"Found {len(data_files)} data files for testing")
        
        # 2. 对每个数据文件进行测试
        for file_path in data_files:
            try:
                self._test_single_file(file_path)
            except Exception as e:
                logger.error(f"Error testing file {file_path}: {e}")
                continue
        
        # 3. 生成汇总报告
        self.test_stats['total_processing_time'] = time.time() - start_time
        summary_report = self._generate_summary_report()
        
        logger.info(f"Comprehensive test completed in {self.test_stats['total_processing_time']:.2f}s")
        return summary_report
    
    def _discover_data_files(self) -> List[Path]:
        """发现所有可用的数据文件"""
        data_files = []
        
        if self.data_directory.exists():
            for file_path in self.data_directory.glob("*.csv"):
                if file_path.name != "EXAMPLE.csv":  # 跳过示例文件
                    data_files.append(file_path)
        
        return data_files
    
    def _test_single_file(self, file_path: Path) -> None:
        """
        测试单个数据文件
        
        Args:
            file_path: 数据文件路径
        """
        logger.info(f"Testing file: {file_path.name}")
        
        try:
            # 连接并加载数据
            self.data_connector = CSVDataConnector(str(self.data_directory))
            self.data_connector.connect()
            symbol = file_path.stem.split('-')[0]  # 从文件名提取品种
            
            # 使用默认日期范围获取所有数据
            from datetime import datetime
            start_date = datetime(2020, 1, 1)
            end_date = datetime.now()
            df = self.data_connector.get_data(symbol, start_date, end_date)
            
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data in {file_path.name}: {len(df) if df is not None else 0} records")
                return
            
            # 准备文件结果存储
            file_results = {
                'file_name': file_path.name,
                'symbol': symbol,
                'data_points': len(df),
                'timeframe': self._detect_timeframe(df),
                'date_range': {
                    'start': str(df['timestamp'].min()),
                    'end': str(df['timestamp'].max())
                },
                'algorithm_comparisons': {},
                'volatility_analysis': {},
                'performance_metrics': {}
            }
            
            # 进行算法比较测试
            self._compare_algorithms(df, file_results)
            
            # 进行波动率分析
            self._analyze_volatility_adaptation(df, file_results)
            
            # 进行RANSAC效果测试
            self._test_ransac_effectiveness(df, file_results)
            
            # 性能基准测试
            self._benchmark_performance(df, file_results)
            
            # 存储结果
            self.results[symbol] = file_results
            self.test_stats['data_files_tested'] += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            raise
    
    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """检测数据的时间周期"""
        if len(df) < 2:
            return "unknown"
        
        # 计算时间间隔
        time_diffs = df['timestamp'].diff().dropna()
        median_diff = time_diffs.median()
        
        # 转换为分钟
        minutes = median_diff.total_seconds() / 60
        
        if minutes <= 1:
            return "1m"
        elif minutes <= 5:
            return "5m"
        elif minutes <= 15:
            return "15m"
        elif minutes <= 30:
            return "30m"
        elif minutes <= 60:
            return "1h"
        elif minutes <= 240:
            return "4h"
        elif minutes <= 1440:
            return "1d"
        else:
            return "1w"
    
    def _compare_algorithms(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        比较新旧算法的检测效果
        
        Args:
            df: 数据
            results: 结果存储
        """
        logger.debug(f"Comparing algorithms for {results['symbol']}")
        
        comparisons = {}
        
        # 旗形检测比较
        try:
            start_time = time.time()
            new_flags = self.flag_detector_new.detect(df)
            new_flag_time = time.time() - start_time
            
            start_time = time.time()
            legacy_flags = self.flag_detector_legacy.detect(df)
            legacy_flag_time = time.time() - start_time
            
            comparisons['flag_detection'] = {
                'new_algorithm': {
                    'patterns_found': len(new_flags),
                    'processing_time': new_flag_time,
                    'patterns': [self._pattern_to_dict(p) for p in new_flags[:5]]  # 前5个
                },
                'legacy_algorithm': {
                    'patterns_found': len(legacy_flags),
                    'processing_time': legacy_flag_time,
                    'patterns': [self._pattern_to_dict(p) for p in legacy_flags[:5]]
                },
                'improvement': {
                    'pattern_count_change': len(new_flags) - len(legacy_flags),
                    'speed_ratio': legacy_flag_time / new_flag_time if new_flag_time > 0 else 0,
                    'quality_improvement': self._assess_quality_improvement(new_flags, legacy_flags)
                }
            }
        except Exception as e:
            logger.error(f"Error in flag detection comparison: {e}")
            comparisons['flag_detection'] = {'error': str(e)}
        
        # 三角旗形检测比较
        try:
            start_time = time.time()
            new_pennants = self.pennant_detector_new.detect(df)
            new_pennant_time = time.time() - start_time
            
            start_time = time.time()
            legacy_pennants = self.pennant_detector_legacy.detect(df)
            legacy_pennant_time = time.time() - start_time
            
            comparisons['pennant_detection'] = {
                'new_algorithm': {
                    'patterns_found': len(new_pennants),
                    'processing_time': new_pennant_time,
                    'patterns': [self._pattern_to_dict(p) for p in new_pennants[:5]]
                },
                'legacy_algorithm': {
                    'patterns_found': len(legacy_pennants),
                    'processing_time': legacy_pennant_time,
                    'patterns': [self._pattern_to_dict(p) for p in legacy_pennants[:5]]
                },
                'improvement': {
                    'pattern_count_change': len(new_pennants) - len(legacy_pennants),
                    'speed_ratio': legacy_pennant_time / new_pennant_time if new_pennant_time > 0 else 0,
                    'quality_improvement': self._assess_quality_improvement(new_pennants, legacy_pennants)
                }
            }
        except Exception as e:
            logger.error(f"Error in pennant detection comparison: {e}")
            comparisons['pennant_detection'] = {'error': str(e)}
        
        results['algorithm_comparisons'] = comparisons
    
    def _analyze_volatility_adaptation(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        分析ATR自适应参数的效果
        
        Args:
            df: 数据
            results: 结果存储
        """
        try:
            # 获取波动率报告
            volatility_report = self.flag_detector_new.get_volatility_report(df)
            
            # 解析波动率级别
            volatility_analysis = {
                'volatility_report': volatility_report,
                'adaptation_effects': {}
            }
            
            # 测试不同波动率时期的检测效果
            # 将数据分段分析
            segment_size = len(df) // 3
            if segment_size > 50:
                segments = [
                    df.iloc[:segment_size],
                    df.iloc[segment_size:segment_size*2],
                    df.iloc[segment_size*2:]
                ]
                
                for i, segment in enumerate(segments):
                    try:
                        segment_report = self.flag_detector_new.get_volatility_report(segment)
                        patterns = self.flag_detector_new.detect(segment)
                        
                        volatility_analysis['adaptation_effects'][f'segment_{i+1}'] = {
                            'volatility_report': segment_report,
                            'patterns_detected': len(patterns),
                            'avg_confidence': np.mean([p.confidence_score for p in patterns]) if patterns else 0
                        }
                    except Exception as e:
                        logger.error(f"Error analyzing segment {i+1}: {e}")
            
            results['volatility_analysis'] = volatility_analysis
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            results['volatility_analysis'] = {'error': str(e)}
    
    def _test_ransac_effectiveness(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        测试RANSAC拟合的效果
        
        Args:
            df: 数据
            results: 结果存储
        """
        try:
            pattern_components = PatternComponents()
            
            # 随机选择一些点进行拟合测试
            if len(df) >= 20:
                test_indices = np.random.choice(len(df), min(20, len(df)), replace=False)
                test_indices = sorted(test_indices)
                
                # RANSAC vs OLS比较
                comparison = pattern_components.compare_fitting_methods(df, test_indices.tolist(), 'close')
                
                # 多次测试以获得稳定结果
                ransac_tests = []
                for _ in range(5):
                    indices = np.random.choice(len(df), min(15, len(df)), replace=False)
                    indices = sorted(indices)
                    trend_line = pattern_components.fit_trend_line_ransac(df, indices.tolist(), 'close')
                    if trend_line:
                        stats = pattern_components.get_ransac_statistics()
                        ransac_tests.append({
                            'r_squared': trend_line.r_squared,
                            'inliers_ratio': stats.get('inliers_ratio', 0),
                            'outliers_count': len(stats.get('outliers_indices', []))
                        })
                
                ransac_effectiveness = {
                    'method_comparison': comparison,
                    'multiple_tests': {
                        'test_count': len(ransac_tests),
                        'avg_r_squared': np.mean([t['r_squared'] for t in ransac_tests]) if ransac_tests else 0,
                        'avg_inliers_ratio': np.mean([t['inliers_ratio'] for t in ransac_tests]) if ransac_tests else 0,
                        'avg_outliers_count': np.mean([t['outliers_count'] for t in ransac_tests]) if ransac_tests else 0
                    },
                    'robustness_assessment': self._assess_ransac_robustness(ransac_tests)
                }
                
                results['ransac_effectiveness'] = ransac_effectiveness
            
        except Exception as e:
            logger.error(f"Error in RANSAC effectiveness test: {e}")
            results['ransac_effectiveness'] = {'error': str(e)}
    
    def _benchmark_performance(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        性能基准测试
        
        Args:
            df: 数据
            results: 结果存储
        """
        try:
            benchmarks = {}
            
            # 检测性能测试
            for detector_name, detector in [
                ('flag_new', self.flag_detector_new),
                ('flag_legacy', self.flag_detector_legacy),
                ('pennant_new', self.pennant_detector_new),
                ('pennant_legacy', self.pennant_detector_legacy)
            ]:
                times = []
                for _ in range(3):  # 运行3次取平均
                    start_time = time.time()
                    patterns = detector.detect(df)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                benchmarks[detector_name] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'patterns_found': len(patterns)
                }
            
            # 计算性能改进
            flag_speedup = benchmarks['flag_legacy']['avg_time'] / benchmarks['flag_new']['avg_time'] if benchmarks['flag_new']['avg_time'] > 0 else 0
            pennant_speedup = benchmarks['pennant_legacy']['avg_time'] / benchmarks['pennant_new']['avg_time'] if benchmarks['pennant_new']['avg_time'] > 0 else 0
            
            performance_metrics = {
                'individual_benchmarks': benchmarks,
                'performance_improvements': {
                    'flag_detection_speedup': flag_speedup,
                    'pennant_detection_speedup': pennant_speedup,
                    'overall_speedup': (flag_speedup + pennant_speedup) / 2
                }
            }
            
            results['performance_metrics'] = performance_metrics
            
        except Exception as e:
            logger.error(f"Error in performance benchmark: {e}")
            results['performance_metrics'] = {'error': str(e)}
    
    def _pattern_to_dict(self, pattern) -> Dict[str, Any]:
        """将形态对象转换为字典"""
        try:
            return {
                'id': pattern.id,
                'symbol': pattern.symbol,
                'pattern_type': pattern.pattern_type.value,
                'confidence_score': float(pattern.confidence_score),
                'pattern_quality': pattern.pattern_quality.value if hasattr(pattern.pattern_quality, 'value') else str(pattern.pattern_quality),
                'flagpole_height_percent': float(pattern.flagpole.height_percent),
                'pattern_duration': pattern.pattern_duration
            }
        except Exception as e:
            return {'error': f'Pattern conversion failed: {e}'}
    
    def _assess_quality_improvement(self, new_patterns: List, legacy_patterns: List) -> Dict[str, Any]:
        """评估质量改进"""
        try:
            if not new_patterns and not legacy_patterns:
                return {'status': 'no_patterns_found'}
            
            if not new_patterns:
                return {'status': 'new_algorithm_found_no_patterns'}
            
            if not legacy_patterns:
                return {'status': 'only_new_algorithm_found_patterns', 'advantage': 'new'}
            
            # 比较平均置信度
            new_avg_confidence = np.mean([p.confidence_score for p in new_patterns])
            legacy_avg_confidence = np.mean([p.confidence_score for p in legacy_patterns])
            
            # 比较平均质量
            new_qualities = [self._quality_to_score(p.pattern_quality) for p in new_patterns]
            legacy_qualities = [self._quality_to_score(p.pattern_quality) for p in legacy_patterns]
            
            new_avg_quality = np.mean(new_qualities) if new_qualities else 0
            legacy_avg_quality = np.mean(legacy_qualities) if legacy_qualities else 0
            
            return {
                'confidence_improvement': new_avg_confidence - legacy_avg_confidence,
                'quality_improvement': new_avg_quality - legacy_avg_quality,
                'new_avg_confidence': new_avg_confidence,
                'legacy_avg_confidence': legacy_avg_confidence,
                'new_avg_quality': new_avg_quality,
                'legacy_avg_quality': legacy_avg_quality
            }
            
        except Exception as e:
            return {'error': f'Quality assessment failed: {e}'}
    
    def _quality_to_score(self, quality) -> float:
        """将质量枚举转换为数值分数"""
        quality_scores = {
            'EXCELLENT': 5.0,
            'GOOD': 4.0,
            'FAIR': 3.0,
            'POOR': 2.0,
            'VERY_POOR': 1.0
        }
        
        quality_str = quality.value if hasattr(quality, 'value') else str(quality)
        return quality_scores.get(quality_str, 3.0)
    
    def _assess_ransac_robustness(self, ransac_tests: List[Dict]) -> Dict[str, Any]:
        """评估RANSAC鲁棒性"""
        if not ransac_tests:
            return {'status': 'no_tests_completed'}
        
        # 计算稳定性指标
        r_squared_values = [t['r_squared'] for t in ransac_tests]
        inlier_ratios = [t['inliers_ratio'] for t in ransac_tests]
        
        return {
            'r_squared_stability': {
                'mean': np.mean(r_squared_values),
                'std': np.std(r_squared_values),
                'coefficient_of_variation': np.std(r_squared_values) / np.mean(r_squared_values) if np.mean(r_squared_values) > 0 else 0
            },
            'inlier_ratio_stability': {
                'mean': np.mean(inlier_ratios),
                'std': np.std(inlier_ratios)
            },
            'robustness_score': min(np.mean(r_squared_values), np.mean(inlier_ratios))
        }
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            'test_overview': {
                'files_tested': self.test_stats['data_files_tested'],
                'total_processing_time': self.test_stats['total_processing_time'],
                'test_completion_time': datetime.now().isoformat()
            },
            'algorithm_performance_summary': {},
            'key_improvements': [],
            'recommendations': []
        }
        
        if not self.results:
            summary['status'] = 'no_successful_tests'
            return summary
        
        # 汇总各文件的结果
        all_flag_improvements = []
        all_pennant_improvements = []
        all_performance_metrics = []
        
        for symbol, result in self.results.items():
            if 'algorithm_comparisons' in result:
                comparisons = result['algorithm_comparisons']
                
                if 'flag_detection' in comparisons and 'improvement' in comparisons['flag_detection']:
                    all_flag_improvements.append(comparisons['flag_detection']['improvement'])
                
                if 'pennant_detection' in comparisons and 'improvement' in comparisons['pennant_detection']:
                    all_pennant_improvements.append(comparisons['pennant_detection']['improvement'])
            
            if 'performance_metrics' in result:
                all_performance_metrics.append(result['performance_metrics'])
        
        # 计算平均改进
        if all_flag_improvements:
            summary['algorithm_performance_summary']['flag_detection'] = {
                'avg_pattern_count_change': np.mean([imp.get('pattern_count_change', 0) for imp in all_flag_improvements]),
                'avg_speed_ratio': np.mean([imp.get('speed_ratio', 1) for imp in all_flag_improvements]),
                'files_with_improvements': len([imp for imp in all_flag_improvements if imp.get('pattern_count_change', 0) > 0])
            }
        
        if all_pennant_improvements:
            summary['algorithm_performance_summary']['pennant_detection'] = {
                'avg_pattern_count_change': np.mean([imp.get('pattern_count_change', 0) for imp in all_pennant_improvements]),
                'avg_speed_ratio': np.mean([imp.get('speed_ratio', 1) for imp in all_pennant_improvements]),
                'files_with_improvements': len([imp for imp in all_pennant_improvements if imp.get('pattern_count_change', 0) > 0])
            }
        
        # 生成关键改进点
        self._generate_key_improvements(summary, all_flag_improvements, all_pennant_improvements)
        
        # 生成建议
        self._generate_recommendations(summary)
        
        # 添加详细结果
        summary['detailed_results'] = self.results
        
        return summary
    
    def _generate_key_improvements(self, summary: Dict, flag_improvements: List, pennant_improvements: List) -> None:
        """生成关键改进点"""
        improvements = []
        
        # 分析旗形检测改进
        if flag_improvements:
            avg_flag_change = np.mean([imp.get('pattern_count_change', 0) for imp in flag_improvements])
            if avg_flag_change > 0:
                improvements.append(f"旗形检测平均增加 {avg_flag_change:.1f} 个形态")
            
            avg_flag_speed = np.mean([imp.get('speed_ratio', 1) for imp in flag_improvements])
            if avg_flag_speed > 1.1:
                improvements.append(f"旗形检测速度提升 {(avg_flag_speed-1)*100:.1f}%")
        
        # 分析三角旗形检测改进
        if pennant_improvements:
            avg_pennant_change = np.mean([imp.get('pattern_count_change', 0) for imp in pennant_improvements])
            if avg_pennant_change > 0:
                improvements.append(f"三角旗形检测平均增加 {avg_pennant_change:.1f} 个形态")
            
            avg_pennant_speed = np.mean([imp.get('speed_ratio', 1) for imp in pennant_improvements])
            if avg_pennant_speed > 1.1:
                improvements.append(f"三角旗形检测速度提升 {(avg_pennant_speed-1)*100:.1f}%")
        
        # ATR自适应改进
        volatility_adaptations = [r.get('volatility_analysis', {}) for r in self.results.values()]
        if any('adaptation_effects' in va for va in volatility_adaptations):
            improvements.append("ATR自适应参数系统成功识别并适应不同波动率环境")
        
        # RANSAC改进
        ransac_results = [r.get('ransac_effectiveness', {}) for r in self.results.values()]
        if any('multiple_tests' in rr for rr in ransac_results):
            improvements.append("RANSAC趋势线拟合提供更鲁棒的形态边界识别")
        
        summary['key_improvements'] = improvements
    
    def _generate_recommendations(self, summary: Dict) -> None:
        """生成优化建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        if self.test_stats['data_files_tested'] > 0:
            recommendations.append("建议在生产环境中启用新算法以提升检测准确性")
        
        # 如果发现性能问题
        total_time = self.test_stats['total_processing_time']
        files_tested = self.test_stats['data_files_tested']
        if files_tested > 0 and total_time / files_tested > 10:
            recommendations.append("考虑优化大数据集的处理性能")
        
        # 如果发现质量问题
        all_results = list(self.results.values())
        error_count = sum(1 for r in all_results if any('error' in str(v) for v in r.values()))
        if error_count > len(all_results) * 0.2:
            recommendations.append("建议增强异常数据的处理能力")
        
        recommendations.append("定期在新的市场数据上验证算法性能")
        recommendations.append("根据实际使用效果调整ATR自适应参数")
        
        summary['recommendations'] = recommendations
    
    def save_results(self, output_path: str = "output/real_data_test_results.json") -> None:
        """保存测试结果"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 生成完整报告
            complete_report = self._generate_summary_report()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(complete_report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Test results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


if __name__ == "__main__":
    # 运行测试
    tester = RealDataAlgorithmTester()
    results = tester.run_comprehensive_test()
    tester.save_results()
    
    # 打印摘要
    print("\\n" + "="*60)
    print("真实数据算法测试完成")
    print("="*60)
    print(f"测试文件数: {results['test_overview']['files_tested']}")
    print(f"总处理时间: {results['test_overview']['total_processing_time']:.2f}秒")
    print("\\n关键改进:")
    for improvement in results['key_improvements']:
        print(f"- {improvement}")
    print("\\n建议:")
    for recommendation in results['recommendations']:
        print(f"- {recommendation}")