"""
快速真实数据测试
简化版本的算法验证
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from datetime import datetime
import json

from src.patterns.detectors.flag_detector import FlagDetector
from src.data.connectors.csv_connector import CSVDataConnector


def quick_test():
    """快速测试新算法"""
    print("开始快速真实数据测试...")
    
    # 1. 加载数据
    data_connector = CSVDataConnector("data/csv")
    data_connector.connect()
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    df = data_connector.get_data("RBL8", start_date, end_date)
    
    print(f"加载数据: {len(df)} 条记录")
    
    # 限制数据量进行快速测试
    df_sample = df.iloc[-1000:]  # 最后1000条记录
    print(f"使用样本数据: {len(df_sample)} 条记录")
    
    # 2. 测试新算法
    print("\n测试新算法...")
    flag_detector_new = FlagDetector()
    
    start_time = time.time()
    patterns_new = flag_detector_new.detect(df_sample)
    new_time = time.time() - start_time
    
    print(f"新算法结果: {len(patterns_new)} 个形态, 耗时: {new_time:.2f}秒")
    
    # 3. 测试传统算法
    print("\n测试传统算法...")
    legacy_config = {
        'global': {
            'enable_atr_adaptation': False,
            'enable_ransac_fitting': False
        },
        'timeframe_configs': {
            'short': {
                'flagpole': {
                    'min_bars': 4,
                    'max_bars': 10,
                    'min_height_percent': 1.5,
                    'max_height_percent': 10.0,
                    'volume_surge_ratio': 2.0,
                    'max_retracement': 0.3,
                    'min_trend_strength': 0.7
                },
                'flag': {
                    'min_bars': 8,
                    'max_bars': 30,
                    'min_slope_angle': 0.5,
                    'max_slope_angle': 10,
                    'retracement_range': (0.2, 0.6),
                    'volume_decay_threshold': 0.7,
                    'parallel_tolerance': 0.15,
                    'min_touches': 3
                }
            }
        },
        'scoring': {
            'min_confidence_score': 0.6
        }
    }
    flag_detector_legacy = FlagDetector(legacy_config)
    
    start_time = time.time()
    patterns_legacy = flag_detector_legacy.detect(df_sample)
    legacy_time = time.time() - start_time
    
    print(f"传统算法结果: {len(patterns_legacy)} 个形态, 耗时: {legacy_time:.2f}秒")
    
    # 4. 比较结果
    print("\n算法比较:")
    print(f"形态数量变化: {len(patterns_new) - len(patterns_legacy)}")
    if legacy_time > 0:
        speed_ratio = legacy_time / new_time
        print(f"速度比率: {speed_ratio:.2f}x")
    
    # 5. 分析置信度
    if patterns_new:
        new_confidences = [p.confidence_score for p in patterns_new]
        print(f"新算法平均置信度: {np.mean(new_confidences):.3f}")
    
    if patterns_legacy:
        legacy_confidences = [p.confidence_score for p in patterns_legacy]
        print(f"传统算法平均置信度: {np.mean(legacy_confidences):.3f}")
    
    # 6. 测试RANSAC效果
    print("\n测试RANSAC效果...")
    from src.patterns.base.pattern_components import PatternComponents
    components = PatternComponents()
    
    # 随机选择一些点测试RANSAC
    test_indices = list(range(0, len(df_sample), 50))[:10]  # 每50个点选一个
    if len(test_indices) >= 3:
        comparison = components.compare_fitting_methods(df_sample, test_indices, 'close')
        print("RANSAC vs OLS比较:")
        print(f"- RANSAC R2: {comparison.get('ransac', {}).get('r_squared', 0):.3f}")
        print(f"- OLS R2: {comparison.get('ols', {}).get('r_squared', 0):.3f}")
        print(f"- 异常值检测: {comparison.get('ransac', {}).get('outliers_count', 0)} 个")
    
    # 7. 保存简化结果
    results = {
        'test_time': datetime.now().isoformat(),
        'data_points': len(df_sample),
        'algorithm_comparison': {
            'new_patterns': len(patterns_new),
            'legacy_patterns': len(patterns_legacy),
            'improvement': len(patterns_new) - len(patterns_legacy),
            'speed_ratio': legacy_time / new_time if new_time > 0 else 0
        },
        'confidence_analysis': {
            'new_avg': np.mean([p.confidence_score for p in patterns_new]) if patterns_new else 0,
            'legacy_avg': np.mean([p.confidence_score for p in patterns_legacy]) if patterns_legacy else 0
        }
    }
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "quick_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_dir}/quick_test_results.json")
    print("快速测试完成!")
    
    return results


if __name__ == "__main__":
    quick_test()