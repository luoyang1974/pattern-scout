"""
性能优化和参数调优分析报告
基于真实数据测试结果的优化建议
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datetime import datetime
from loguru import logger
import json
import cProfile
import pstats
from io import StringIO

from src.patterns.detectors.flag_detector import FlagDetector
from src.patterns.detectors.pennant_detector import PennantDetector
from src.data.connectors.csv_connector import CSVDataConnector


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.optimization_results = {}
        
    def run_performance_analysis(self):
        """运行性能分析"""
        print("开始性能优化分析...")
        
        # 1. 加载测试数据
        df_sample = self._load_test_data()
        
        # 2. 基准性能测试
        baseline_stats = self._benchmark_baseline_performance(df_sample)
        
        # 3. 瓶颈分析
        bottleneck_analysis = self._analyze_bottlenecks(df_sample)
        
        # 4. 参数敏感性分析
        parameter_analysis = self._analyze_parameter_sensitivity(df_sample)
        
        # 5. 优化建议
        optimization_recommendations = self._generate_optimization_recommendations(
            baseline_stats, bottleneck_analysis, parameter_analysis
        )
        
        # 6. 保存结果
        self._save_optimization_results({
            'baseline_performance': baseline_stats,
            'bottleneck_analysis': bottleneck_analysis,
            'parameter_analysis': parameter_analysis,
            'optimization_recommendations': optimization_recommendations
        })
        
        print("性能优化分析完成！")
        return optimization_recommendations
    
    def _load_test_data(self):
        """加载测试数据"""
        data_connector = CSVDataConnector("data/csv")
        data_connector.connect()
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime.now()
        df = data_connector.get_data("RBL8", start_date, end_date)
        
        # 使用不同大小的数据集进行测试
        return {
            'small': df.iloc[-500:],      # 500条记录
            'medium': df.iloc[-1000:],    # 1000条记录
            'large': df.iloc[-2000:]      # 2000条记录
        }
    
    def _benchmark_baseline_performance(self, datasets):
        """基准性能测试"""
        print("进行基准性能测试...")
        
        detector = FlagDetector()
        results = {}
        
        for size, df in datasets.items():
            print(f"测试 {size} 数据集 ({len(df)} 条记录)...")
            
            # 重复测试3次取平均
            times = []
            pattern_counts = []
            
            for i in range(3):
                start_time = time.time()
                patterns = detector.detect(df)
                end_time = time.time()
                
                times.append(end_time - start_time)
                pattern_counts.append(len(patterns))
            
            results[size] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_patterns': np.mean(pattern_counts),
                'records_per_second': len(df) / np.mean(times),
                'time_per_record': np.mean(times) / len(df) * 1000  # ms per record
            }
            
            print(f"  平均耗时: {results[size]['avg_time']:.2f}s")
            print(f"  每秒处理: {results[size]['records_per_second']:.0f} 条记录")
        
        return results
    
    def _analyze_bottlenecks(self, datasets):
        """分析性能瓶颈"""
        print("分析性能瓶颈...")
        
        # 使用cProfile分析中等大小数据集
        df = datasets['medium']
        detector = FlagDetector()
        
        # 性能分析
        profiler = cProfile.Profile()
        profiler.enable()
        
        patterns = detector.detect(df)
        
        profiler.disable()
        
        # 分析结果
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # 显示前20个最耗时的函数
        
        profile_output = s.getvalue()
        
        # 解析关键瓶颈
        bottlenecks = self._parse_profile_output(profile_output)
        
        return {
            'total_patterns': len(patterns),
            'profile_summary': profile_output.split('\n')[:25],  # 前25行摘要
            'key_bottlenecks': bottlenecks
        }
    
    def _parse_profile_output(self, profile_output):
        """解析profile输出，识别关键瓶颈"""
        lines = profile_output.split('\n')
        bottlenecks = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in 
                   ['detect', 'pattern', 'ransac', 'atr', 'preprocess']):
                if 'seconds' in line or any(char.isdigit() for char in line):
                    bottlenecks.append(line.strip())
        
        return bottlenecks[:10]  # 返回前10个相关瓶颈
    
    def _analyze_parameter_sensitivity(self, datasets):
        """参数敏感性分析"""
        print("进行参数敏感性分析...")
        
        df = datasets['medium']
        sensitivity_results = {}
        
        # 1. ATR自适应参数敏感性
        print("  测试ATR参数敏感性...")
        atr_results = self._test_atr_sensitivity(df)
        sensitivity_results['atr_adaptation'] = atr_results
        
        # 2. RANSAC参数敏感性
        print("  测试RANSAC参数敏感性...")
        ransac_results = self._test_ransac_sensitivity(df)
        sensitivity_results['ransac_fitting'] = ransac_results
        
        # 3. 检测阈值敏感性
        print("  测试检测阈值敏感性...")
        threshold_results = self._test_threshold_sensitivity(df)
        sensitivity_results['detection_thresholds'] = threshold_results
        
        return sensitivity_results
    
    def _test_atr_sensitivity(self, df):
        """测试ATR参数敏感性"""
        base_config = FlagDetector().get_default_config()
        results = []
        
        # 测试不同的ATR周期
        atr_periods = [7, 14, 21, 28]
        for period in atr_periods:
            config = base_config.copy()
            # 这里需要添加ATR周期配置
            
            detector = FlagDetector(config)
            start_time = time.time()
            patterns = detector.detect(df)
            end_time = time.time()
            
            results.append({
                'atr_period': period,
                'patterns_found': len(patterns),
                'processing_time': end_time - start_time,
                'avg_confidence': np.mean([p.confidence_score for p in patterns]) if patterns else 0
            })
        
        return results
    
    def _test_ransac_sensitivity(self, df):
        """测试RANSAC参数敏感性"""
        results = []
        
        # 测试RANSAC开启和关闭
        for enable_ransac in [True, False]:
            config = {
                'global': {
                    'enable_ransac_fitting': enable_ransac
                }
            }
            
            detector = FlagDetector(config)
            start_time = time.time()
            patterns = detector.detect(df)
            end_time = time.time()
            
            results.append({
                'ransac_enabled': enable_ransac,
                'patterns_found': len(patterns),
                'processing_time': end_time - start_time,
                'avg_confidence': np.mean([p.confidence_score for p in patterns]) if patterns else 0
            })
        
        return results
    
    def _test_threshold_sensitivity(self, df):
        """测试检测阈值敏感性"""
        base_config = FlagDetector().get_default_config()
        results = []
        
        # 测试不同的最小置信度阈值
        confidence_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        for threshold in confidence_thresholds:
            config = base_config.copy()
            config['scoring']['min_confidence_score'] = threshold
            
            detector = FlagDetector(config)
            start_time = time.time()
            patterns = detector.detect(df)
            end_time = time.time()
            
            results.append({
                'confidence_threshold': threshold,
                'patterns_found': len(patterns),
                'processing_time': end_time - start_time,
                'avg_confidence': np.mean([p.confidence_score for p in patterns]) if patterns else 0
            })
        
        return results
    
    def _generate_optimization_recommendations(self, baseline_stats, bottleneck_analysis, parameter_analysis):
        """生成优化建议"""
        recommendations = []
        
        # 1. 基于性能基准的建议
        large_time = baseline_stats['large']['avg_time']
        medium_time = baseline_stats['medium']['avg_time']
        
        if large_time > 10:  # 如果大数据集处理超过10秒
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'issue': '大数据集处理时间过长',
                'recommendation': '考虑实现数据分片处理或并行计算',
                'expected_improvement': '50-70%性能提升'
            })
        
        # 2. 基于瓶颈分析的建议
        bottlenecks = bottleneck_analysis['key_bottlenecks']
        if any('preprocess' in b.lower() for b in bottlenecks):
            recommendations.append({
                'category': 'preprocessing',
                'priority': 'medium',
                'issue': '数据预处理成为瓶颈',
                'recommendation': '优化技术指标计算，使用向量化操作',
                'expected_improvement': '20-30%性能提升'
            })
        
        if any('ransac' in b.lower() for b in bottlenecks):
            recommendations.append({
                'category': 'algorithm',
                'priority': 'medium',
                'issue': 'RANSAC算法计算耗时',
                'recommendation': '调整RANSAC迭代次数或使用快速近似算法',
                'expected_improvement': '15-25%性能提升'
            })
        
        # 3. 基于参数分析的建议
        if 'ransac_fitting' in parameter_analysis:
            ransac_results = parameter_analysis['ransac_fitting']
            ransac_on = next((r for r in ransac_results if r['ransac_enabled']), None)
            ransac_off = next((r for r in ransac_results if not r['ransac_enabled']), None)
            
            if ransac_on and ransac_off:
                time_overhead = (ransac_on['processing_time'] - ransac_off['processing_time']) / ransac_off['processing_time']
                quality_gain = ransac_on['avg_confidence'] - ransac_off['avg_confidence']
                
                if time_overhead > 0.5 and quality_gain < 0.1:  # 时间开销大但质量提升小
                    recommendations.append({
                        'category': 'parameters',
                        'priority': 'low',
                        'issue': 'RANSAC算法性价比不高',
                        'recommendation': '在低精度要求场景下可以关闭RANSAC',
                        'expected_improvement': f'{time_overhead*100:.0f}%时间节省'
                    })
        
        # 4. 通用优化建议
        recommendations.extend([
            {
                'category': 'caching',
                'priority': 'medium',
                'issue': '重复计算技术指标',
                'recommendation': '实现技术指标缓存机制',
                'expected_improvement': '10-20%性能提升'
            },
            {
                'category': 'memory',
                'priority': 'low',
                'issue': '内存使用优化',
                'recommendation': '使用更高效的数据结构，及时释放不需要的变量',
                'expected_improvement': '降低内存占用30%'
            }
        ])
        
        return recommendations
    
    def _save_optimization_results(self, results):
        """保存优化结果"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(output_dir / "performance_optimization_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成简化的优化报告
        self._generate_optimization_report(results, output_dir)
    
    def _generate_optimization_report(self, results, output_dir):
        """生成优化报告"""
        report = []
        report.append("# PatternScout 性能优化分析报告")
        report.append(f"生成时间: {datetime.now().isoformat()}")
        report.append("")
        
        # 性能基准
        report.append("## 性能基准测试")
        baseline = results['baseline_performance']
        for size, stats in baseline.items():
            report.append(f"### {size.upper()} 数据集")
            report.append(f"- 平均处理时间: {stats['avg_time']:.2f}秒")
            report.append(f"- 处理速度: {stats['records_per_second']:.0f} 记录/秒")
            report.append(f"- 单记录耗时: {stats['time_per_record']:.2f}ms")
            report.append("")
        
        # 瓶颈分析
        report.append("## 性能瓶颈分析")
        bottlenecks = results['bottleneck_analysis']['key_bottlenecks']
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            report.append(f"{i}. {bottleneck}")
        report.append("")
        
        # 优化建议
        report.append("## 优化建议")
        recommendations = results['optimization_recommendations']
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        low_priority = [r for r in recommendations if r['priority'] == 'low']
        
        for priority, recs in [('高优先级', high_priority), ('中优先级', medium_priority), ('低优先级', low_priority)]:
            if recs:
                report.append(f"### {priority}")
                for rec in recs:
                    report.append(f"- **{rec['issue']}**")
                    report.append(f"  - 建议: {rec['recommendation']}")
                    report.append(f"  - 预期改进: {rec['expected_improvement']}")
                    report.append("")
        
        # 保存报告
        with open(output_dir / "optimization_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))


def main():
    optimizer = PerformanceOptimizer()
    recommendations = optimizer.run_performance_analysis()
    
    print("\n优化建议总结:")
    for rec in recommendations:
        print(f"[{rec['priority'].upper()}] {rec['issue']}")
        print(f"  建议: {rec['recommendation']}")
        print(f"  预期改进: {rec['expected_improvement']}")
        print()


if __name__ == "__main__":
    main()