"""
简化的性能分析和优化建议
基于测试结果的性能报告
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datetime import datetime
import json

from src.patterns.detectors.flag_detector import FlagDetector
from src.data.connectors.csv_connector import CSVDataConnector


def quick_performance_analysis():
    """快速性能分析"""
    print("开始性能分析...")
    
    # 1. 加载数据
    data_connector = CSVDataConnector("data/csv")
    data_connector.connect()
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    df = data_connector.get_data("RBL8", start_date, end_date)
    
    # 2. 测试不同数据量的性能
    test_sizes = [500, 1000, 2000]
    performance_results = {}
    
    for size in test_sizes:
        print(f"\n测试 {size} 条记录...")
        df_sample = df.iloc[-size:]
        
        # 测试3次取平均
        times = []
        pattern_counts = []
        
        detector = FlagDetector()
        
        for i in range(3):
            start_time = time.time()
            patterns = detector.detect(df_sample)
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            pattern_counts.append(len(patterns))
            print(f"  第{i+1}次: {elapsed:.2f}s, {len(patterns)} 个形态")
        
        avg_time = np.mean(times)
        records_per_sec = size / avg_time
        
        performance_results[str(size)] = {
            'avg_time': avg_time,
            'std_time': np.std(times),
            'avg_patterns': np.mean(pattern_counts),
            'records_per_second': records_per_sec,
            'ms_per_record': (avg_time / size) * 1000
        }
        
        print(f"  平均: {avg_time:.2f}s, {records_per_sec:.0f} 记录/秒, {performance_results[str(size)]['ms_per_record']:.2f}ms/记录")
    
    # 3. 分析性能趋势
    print("\n性能趋势分析:")
    prev_efficiency = None
    for size in test_sizes:
        efficiency = performance_results[str(size)]['records_per_second']
        if prev_efficiency:
            change = (efficiency - prev_efficiency) / prev_efficiency * 100
            print(f"{size} 记录: {efficiency:.0f} 记录/秒 (变化: {change:+.1f}%)")
        else:
            print(f"{size} 记录: {efficiency:.0f} 记录/秒")
        prev_efficiency = efficiency
    
    # 4. 生成优化建议
    large_time = performance_results['2000']['avg_time']
    small_time = performance_results['500']['avg_time']
    scaling_factor = large_time / (small_time * 4)  # 理想情况下应该是1.0
    
    optimization_suggestions = []
    
    if large_time > 15:  # 超过15秒
        optimization_suggestions.append({
            'priority': 'HIGH',
            'issue': '大数据集处理时间过长',
            'suggestion': '实现数据分块处理或异步处理',
            'impact': '预计可减少50-70%处理时间'
        })
    
    if scaling_factor > 1.5:  # 扩展性不佳
        optimization_suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'算法复杂度过高 (扩展因子: {scaling_factor:.2f})',
            'suggestion': '优化核心算法，减少嵌套循环',
            'impact': '预计可提升30-50%性能'
        })
    
    avg_ms_per_record = np.mean([performance_results[str(size)]['ms_per_record'] for size in test_sizes])
    if avg_ms_per_record > 10:  # 每条记录超过10ms
        optimization_suggestions.append({
            'priority': 'MEDIUM',
            'issue': f'单记录处理时间过长 ({avg_ms_per_record:.1f}ms/记录)',
            'suggestion': '向量化操作，减少Python循环',
            'impact': '预计可提升20-40%性能'
        })
    
    # 基于之前测试结果的建议
    optimization_suggestions.extend([
        {
            'priority': 'MEDIUM',
            'issue': 'RANSAC算法计算开销',
            'suggestion': '在数据质量较好时可选择性关闭RANSAC',
            'impact': '预计节省15-25%计算时间'
        },
        {
            'priority': 'LOW',
            'issue': '技术指标重复计算',
            'suggestion': '实现指标缓存机制',
            'impact': '预计节省10-20%计算时间'
        },
        {
            'priority': 'LOW',
            'issue': '内存使用优化',
            'suggestion': '优化数据结构，及时释放临时变量',
            'impact': '减少30%内存占用'
        }
    ])
    
    # 5. 保存结果
    results = {
        'test_time': datetime.now().isoformat(),
        'performance_benchmarks': performance_results,
        'scaling_analysis': {
            'scaling_factor': scaling_factor,
            'linear_scaling_expected': 1.0,
            'performance_degradation': f"{((scaling_factor - 1.0) * 100):.1f}%"
        },
        'optimization_suggestions': optimization_suggestions
    }
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "performance_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 6. 生成报告
    generate_performance_report(results, output_dir)
    
    print(f"\n性能分析完成！结果已保存到 output/performance_analysis.json")
    print("详细报告: output/performance_report.md")
    
    return results


def generate_performance_report(results, output_dir):
    """生成性能报告"""
    report_lines = []
    report_lines.append("# PatternScout 性能分析报告")
    report_lines.append(f"生成时间: {results['test_time']}")
    report_lines.append("")
    
    # 性能基准
    report_lines.append("## 性能基准测试结果")
    report_lines.append("")
    benchmarks = results['performance_benchmarks']
    
    report_lines.append("| 数据量 | 平均耗时(s) | 处理速度(记录/s) | 单记录耗时(ms) | 检测形态数 |")
    report_lines.append("|--------|-------------|------------------|----------------|------------|")
    
    for size, stats in benchmarks.items():
        report_lines.append(f"| {size} | {stats['avg_time']:.2f} | {stats['records_per_second']:.0f} | {stats['ms_per_record']:.2f} | {stats['avg_patterns']:.1f} |")
    
    report_lines.append("")
    
    # 扩展性分析
    scaling = results['scaling_analysis']
    report_lines.append("## 扩展性分析")
    report_lines.append(f"- 扩展因子: {scaling['scaling_factor']:.2f} (理想值: 1.0)")
    report_lines.append(f"- 性能下降: {scaling['performance_degradation']}")
    
    if scaling['scaling_factor'] > 1.2:
        report_lines.append("- **分析**: 算法复杂度偏高，随数据量增长性能下降明显")
    else:
        report_lines.append("- **分析**: 扩展性良好，性能随数据量线性增长")
    
    report_lines.append("")
    
    # 优化建议
    report_lines.append("## 优化建议")
    suggestions = results['optimization_suggestions']
    
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        priority_suggestions = [s for s in suggestions if s['priority'] == priority]
        if priority_suggestions:
            report_lines.append(f"### {priority} 优先级")
            for i, suggestion in enumerate(priority_suggestions, 1):
                report_lines.append(f"{i}. **{suggestion['issue']}**")
                report_lines.append(f"   - 建议: {suggestion['suggestion']}")
                report_lines.append(f"   - 预期改进: {suggestion['impact']}")
                report_lines.append("")
    
    # 总结
    report_lines.append("## 总结")
    
    avg_time_2k = benchmarks['2000']['avg_time']
    if avg_time_2k < 5:
        performance_level = "优秀"
    elif avg_time_2k < 10:
        performance_level = "良好" 
    elif avg_time_2k < 20:
        performance_level = "一般"
    else:
        performance_level = "需要优化"
    
    report_lines.append(f"- 当前性能水平: **{performance_level}**")
    report_lines.append(f"- 2000条记录处理时间: {avg_time_2k:.2f}s")
    
    high_priority_count = len([s for s in suggestions if s['priority'] == 'HIGH'])
    if high_priority_count > 0:
        report_lines.append(f"- 发现 {high_priority_count} 个高优先级性能问题，建议优先解决")
    else:
        report_lines.append("- 未发现高优先级性能问题")
    
    report_lines.append("")
    report_lines.append("## 下一步行动")
    report_lines.append("1. 优先解决高优先级性能问题")
    report_lines.append("2. 在更大的数据集上进行测试验证")
    report_lines.append("3. 实施优化方案并重新测试")
    report_lines.append("4. 建立性能回归测试机制")
    
    with open(output_dir / "performance_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


def print_optimization_summary(results):
    """打印优化建议摘要"""
    print("\n=== 性能优化建议摘要 ===")
    
    suggestions = results['optimization_suggestions']
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        priority_suggestions = [s for s in suggestions if s['priority'] == priority]
        if priority_suggestions:
            print(f"\n[{priority}] 优先级建议:")
            for suggestion in priority_suggestions:
                print(f"  • {suggestion['issue']}")
                print(f"    建议: {suggestion['suggestion']}")
                print(f"    预期: {suggestion['impact']}")


if __name__ == "__main__":
    results = quick_performance_analysis()
    print_optimization_summary(results)